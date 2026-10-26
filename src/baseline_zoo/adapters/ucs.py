"""UCS baseline: oracle-prompted iterative SAM refinement for curves.

UCS (``kylechuuuuu/UCS``, arXiv:2504.04034) is a curvilinear-structure SAM
variant.  The released inference path derives boxes/points from the GT mask and
samples refinement clicks by comparing predictions to GT.  It is therefore
**oracle-prompted**, not prompt-free; this adapter preserves that behaviour and
labels it in the paper table.

Training follows the official batch-1, Adam 1e-4 and MultiStepLR recipe,
with milestones scaled from the official 20-epoch [5,10] to the 200-epoch
3-shot budget [50,100].  Checkpoints are selected by training Dice.  The
shared runtime still owns data splits,
checkpointing, history, W&B and prediction export.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)


class UCSAdapter(BaselineAdapter):
    id = 'ucs'
    display_name = 'UCS'
    official_source = 'kylechuuuuu/UCS (self-contained copy)'
    official_entrypoint = 'https://github.com/kylechuuuuu/UCS'
    input_size = 1024
    official_epochs = 20
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Oracle-prompted: the official inference derives boxes/points from the '
        'GT mask and samples refinement clicks from prediction-vs-GT errors. '
        'UCS must not be reported as prompt-free.',
        'Uses the official batch-1, Adam 1e-4, MultiStepLR gamma 0.5 and '
        'iter_point=8 recipe with milestones scaled from [5,10] at 20 epochs '
        'to [50,100] at the 200-epoch 3-shot budget.',
        'Checkpoint selection uses training Dice (official), not the 6-image '
        'validation Dice used by other baselines; validation is still logged.',
        'Bypasses the UCS JSON dataset manifests and consumes the shared '
        '18/6 split Samples directly.',
        'Official train.py uses DataLoader workers=4; this port uses '
        'workers=0 and the shared seeded runtime, so data order is not '
        'bitwise identical.',
        'Ports HEAD cff2db1: the paper Sparse Adapter and HQ-feature '
        'compression were refactored out upstream; paper-faithful modules '
        'exist only in git history.',
        'The PCA decoder branch requires batch size 1 and runs sklearn PCA on '
        'CPU each forward; the hardcoded .cuda() was patched to the tensor '
        'device.',
        'No official UCS weights were found; training starts from SAM ViT-L.',
        'Evaluation prompt sampling is seeded from --seed (Python/NumPy/'
        'Torch) so repeated prediction runs are deterministic; upstream does '
        'not seed it. This is a reproducibility adaptation.',
        'The official evaluator reports IoU/Dice/F1/precision/recall; the '
        'shared evaluator reports Dice/IoU/clDice/HD95. official_metric_match '
        'is therefore unverified.',
        'No TTA beyond the official iterative prompt refinement.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._train_dice = -1.0
        self._prediction_seeded = False
        self.iter_point = 8
        self.point_num = 1
        self.mask_num = 1
        self.multimask = False

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument('--iter-point', type=int, default=8)
        parser.add_argument('--point-num', type=int, default=1)
        parser.add_argument('--mask-num', type=int, default=1)
        parser.add_argument('--point-list', type=int, nargs='+',
                            default=[1, 3, 5, 9])
        parser.add_argument('--multimask', action='store_true')

    def build_model(self, args):
        from baseline_zoo.models.ucs import build_sam_vit_l

        if int(args.input_size) != 1024:
            raise SystemExit('UCS expects --input-size 1024')
        self.input_size = int(args.input_size)
        self.iter_point = int(getattr(args, 'iter_point', 8))
        self.point_num = int(getattr(args, 'point_num', 1))
        if self.iter_point > 1:
            self.point_num = 1
        self.mask_num = int(getattr(args, 'mask_num', 1))
        self.multimask = bool(getattr(args, 'multimask', False))
        if self.mask_num != 1:
            raise SystemExit('UCS PCA decoder requires --mask-num 1')

        checkpoint = Path(args.sam_checkpoint)
        training = getattr(args, 'stage', 'train') in ('train', 'all')
        if training and not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-L checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_l_0b3195.pth')
        source = str(checkpoint) if checkpoint.is_file() else None
        if source:
            self._sam_sha = file_sha256(checkpoint)
        ns = SimpleNamespace(
            image_size=1024, sam_checkpoint=source, encoder_adapter=True)
        return build_sam_vit_l(ns)

    def configure_trainable(self, model):
        for parameter in model.prompt_encoder.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.image_encoder.named_parameters():
            parameter.requires_grad_(
                'Adapter' in name or 'neck' in name
                or 'prompt_generator' in name)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('UCS has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def make_scheduler(self, optimizer, args):
        # Official 20-epoch schedule is [5, 10] (25%, 50%).  The 3-shot
        # protocol uses 200 epochs, so keep the same proportions.
        milestones = [
            max(1, int(args.epochs) // 4),
            max(2, int(args.epochs) // 2),
        ]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)

    @staticmethod
    def _ucs_utils():
        from baseline_zoo.models.ucs import utils
        return utils

    def _prepare_sample(self, sample, train: bool, rng: random.Random):
        utils = self._ucs_utils()
        image = np.asarray(load_rgb(sample.image_path))
        image = (image - np.asarray(PIXEL_MEAN)) / np.asarray(PIXEL_STD)
        with Image.open(sample.mask_path) as handle:
            mask = np.asarray(handle.convert('L'))
        if mask.max() > 1:
            mask = mask / 255.0
        mask = mask.astype(np.float32)
        height, width = mask.shape
        transforms = (utils.train_transforms(self.input_size, height, width)
                      if train else
                      utils.test_transforms(self.input_size, height, width))
        augmented = transforms(image=image, mask=mask)
        image_tensor = augmented['image'].float()
        mask_tensor = (augmented['mask'] > 0.5).float()
        mask_np = mask_tensor.numpy()

        if train:
            if rng.random() > 0.5:
                boxes = utils.get_boxes_from_mask(
                    mask_np, max_pixel=5).float().unsqueeze(0)
                point_coords = None
                point_labels = None
            else:
                point_coords, point_labels = utils.init_point_sampling(
                    mask_np, 1)
                point_coords = point_coords.float().unsqueeze(0)
                point_labels = point_labels.int().unsqueeze(0)
                boxes = None
        else:
            boxes = utils.get_boxes_from_mask(
                mask_np, max_pixel=0).float().unsqueeze(0)
            point_coords, point_labels = utils.init_point_sampling(
                mask_np, self.point_num)
            point_coords = point_coords.float().unsqueeze(0)
            point_labels = point_labels.int().unsqueeze(0)

        return {
            'image': image_tensor.unsqueeze(0),
            'label': mask_tensor.unsqueeze(0).unsqueeze(0),
            'boxes': boxes,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_inputs': None,
            'original_size': (height, width),
        }

    def _to_device(self, batch):
        moved = {}
        for key, value in batch.items():
            if value is None or key == 'original_size':
                moved[key] = value
            elif isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _decode(self, model, batch, image_embeddings, interm_embeddings,
                prompt_no_grad=False):
        points = None
        if batch.get('point_coords') is not None:
            points = (batch['point_coords'], batch['point_labels'])
        context = torch.no_grad() if prompt_no_grad else nullcontext()
        with context:
            sparse, dense = model.prompt_encoder(
                points=points, boxes=batch.get('boxes'),
                masks=batch.get('mask_inputs'))
        low_res, iou = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=self.multimask,
            hq_token_only=False,
            interm_embeddings=interm_embeddings)
        masks = F.interpolate(
            low_res, (self.input_size, self.input_size),
            mode='bilinear', align_corners=False)
        return masks, low_res, iou

    @staticmethod
    def _dice(logits, labels):
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * labels).sum()
        return float((2 * inter + 1e-6)
                     / (pred.sum() + labels.sum() + 1e-6))

    def train_step(self, model, batch):
        raise RuntimeError('UCS uses a custom iterative train_epoch')

    def prepare_train_batch(self, samples, rng):
        return self._prepare_sample(samples[0], True, rng)

    def train_epoch(self, model, loader, optimizer, scheduler, args):
        utils = self._ucs_utils()
        criterion = utils.FocalDiceloss_IoULoss().to(self.device)
        model.train()
        losses, dices = [], []
        epoch = int(getattr(self, '_epoch', 1))

        for step, samples in enumerate(loader):
            if len(samples) != 1:
                raise RuntimeError('UCS requires batch-size 1')
            rng = random.Random(args.seed * 100003 + epoch * 10007 + step)
            batch = self._to_device(
                self._prepare_sample(samples[0], True, rng))
            labels = batch['label']
            image_embeddings, interm_embeddings = model.image_encoder(
                batch['image'])

            masks, low_res, iou = self._decode(
                model, batch, image_embeddings, interm_embeddings)
            loss = criterion(masks, labels, iou)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            point_num = rng.choice(args.point_list)
            batch = utils.generate_point(
                masks, labels, low_res, batch, point_num)
            batch = self._to_device(batch)
            image_embeddings = image_embeddings.detach().clone()
            interm_embeddings = [
                feature.detach().clone() for feature in interm_embeddings]

            if args.iter_point <= 2:
                init_mask_num = 1
            else:
                init_mask_num = rng.randint(1, args.iter_point - 1)
            for iteration in range(args.iter_point):
                if (iteration == init_mask_num
                        or iteration == args.iter_point - 1):
                    batch = utils.setting_prompt_none(batch)
                masks, low_res, iou = self._decode(
                    model, batch, image_embeddings, interm_embeddings,
                    prompt_no_grad=True)
                loss = criterion(masks, labels, iou)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if iteration != args.iter_point - 1:
                    point_num = rng.choice(args.point_list)
                    batch = utils.generate_point(
                        masks, labels, low_res, batch, point_num)
                    batch = self._to_device(batch)

            losses.append(float(loss.detach()))
            dices.append(self._dice(masks, labels))

        scheduler.step()
        self._train_dice = float(np.mean(dices)) if dices else -1.0
        return float(np.mean(losses)) if losses else 0.0

    def selection_score(self, metrics, per_dataset):
        return self._train_dice

    def _prediction_rng_active(self) -> bool:
        return bool(getattr(self, 'prediction_mode', False)
                    or getattr(self.args, 'stage', None) == 'predict')

    def _seed_prediction_rng(self):
        """Seed before the first eval prompt draw, not after it.

        ``_prepare_sample`` calls ``init_point_sampling``, which draws from
        the global NumPy RNG.  Seeding only in ``predict_sample`` would leave
        the initial point of the first image unseeded.  This is skipped for
        per-epoch validation so evaluation does not reset the training RNG.
        """
        if self._prediction_seeded or not self._prediction_rng_active():
            return
        seed = int(getattr(self.args, 'seed', 0) or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._prediction_seeded = True

    def prepare_eval_sample(self, sample, device):
        self._seed_prediction_rng()
        return self._prepare_sample(sample, False, random.Random(0))

    def predict_sample(self, model, prepared, sample):
        self._seed_prediction_rng()
        utils = self._ucs_utils()
        batch = self._to_device(prepared)
        labels = batch['label']
        image_embeddings, interm_embeddings = model.image_encoder(
            batch['image'])
        batch['boxes'] = None
        point_coords = [batch['point_coords']]
        point_labels = [batch['point_labels']]
        low_res = None
        for iteration in range(self.iter_point):
            masks, low_res, _ = self._decode(
                model, batch, image_embeddings, interm_embeddings)
            if iteration != self.iter_point - 1:
                batch = utils.generate_point(
                    masks, labels, low_res, batch, self.point_num)
                batch = self._to_device(batch)
                point_coords.append(batch['point_coords'])
                point_labels.append(batch['point_labels'])
                batch['point_coords'] = torch.cat(point_coords, dim=1)
                batch['point_labels'] = torch.cat(point_labels, dim=1)

        height, width = prepared['original_size']
        output = F.interpolate(
            low_res, (self.input_size, self.input_size),
            mode='bilinear', align_corners=False)
        if height < self.input_size and width < self.input_size:
            top = (self.input_size - height) // 2
            left = (self.input_size - width) // 2
            output = output[..., top:top + height, left:left + width]
        else:
            output = F.interpolate(
                output, (height, width), mode='bilinear',
                align_corners=False)
        return (torch.sigmoid(output)[0, 0].cpu().numpy() > 0.5)

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        return {
            'input_size': int(self.input_size),
            'iter_point': int(self.iter_point),
            'point_num': int(self.point_num),
            'mask_num': int(self.mask_num),
            'multimask': bool(self.multimask),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': self._sam_sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            args.input_size = int(config['input_size'])
        args.iter_point = int(config.get('iter_point', 8))
        args.point_num = int(config.get('point_num', 1))
        args.mask_num = int(config.get('mask_num', 1))
        args.multimask = bool(config.get('multimask', False))
        self.iter_point = args.iter_point
        self.point_num = args.point_num
        self.mask_num = args.mask_num
        self.multimask = args.multimask
        stored = config.get('sam_checkpoint_sha256')
        current = getattr(args, 'sam_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(f'WARNING: base SAM checkpoint differs from training: '
                      f'{current}; prediction uses the checkpoint weights')
