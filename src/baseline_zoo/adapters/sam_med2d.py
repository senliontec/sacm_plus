"""SAM-Med2D baseline with the official prompt-refinement training loop.

Official reference: https://github.com/OpenGVLab/SAM-Med2D (vendored snapshot
in ``baselines/sam-med2d``).  The SAM modeling package and ``utils.py``
(prompt generation, transforms, focal/dice/IoU loss) are copied into
``baseline_zoo.models.sam_med2d``.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.models.sam_med2d import build_sam_med2d
from baseline_zoo.models.sam_med2d.utils import (
    FocalDiceloss_IoULoss,
    generate_point,
    get_boxes_from_mask,
    init_point_sampling,
    setting_prompt_none,
    train_transforms,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter

PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class SamMed2DAdapter(BaselineAdapter):
    id = 'sam_med2d'
    display_name = 'SAM-Med2D'
    official_source = 'OpenGVLab/SAM-Med2D (self-contained copy)'
    official_entrypoint = 'upstream train.py (prompt refinement loop)'
    input_size = 256
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 2
    selection = 'dice'
    deviations = (
        'The official SAM-Med2D checkpoint is not present; this port uses '
        'vanilla SAM ViT-L weights with the official 256-input adapter and '
        'the official load_from interpolation. The paper requires the shared '
        'ViT-L encoder, but the decoder is initialized from SAM rather than '
        'SAM-Med2D.',
        'The official load_from() excludes mask_tokens, '
        'output_hypernetworks_mlps and iou_prediction_head. This port '
        'inherits every shape-compatible SAM tensor, so the full SAM '
        'decoder is loaded; only the SAM-Med2D Adapter stays randomly '
        'initialized for the controlled fine-tune-from-SAM protocol.',
        'Data protocol: the 18/6-shot split replaces the official JSON '
        'manifests; BGR + SAM-statistics preprocessing and the 256 '
        'pad/resize rule are preserved.',
        'The protocol datasets are binary, so the official mask_num=5 '
        'multi-mask sampler repeats the single available mask five times.',
        'Checkpoint selection uses the 6-image validation split by Dice; '
        'upstream selects the best training loss (there is no validation '
        'split).',
        'Predictions are postprocessed to the original GT resolution for the '
        'shared evaluator.',
        'Official Adam 1e-4, batch 2, mask_num 5, iter_point 8, point_list '
        '[1,3,5,9], 20*Focal+Dice+IoU loss, multimask max-IoU and 0.5 '
        'threshold are preserved; the official 15-epoch budget is replaced '
        'by 800 epochs for the 18-image 3-shot protocol.',
        'Official train.py uses workers=4 and unseeded global random/NumPy; '
        'this port uses workers=0 and the shared seeded runtime. The decoder '
        'initialization difference is documented above.',
        'The official Adapter has ~646M trainable parameters, whose Adam '
        'state plus the ViT-L model exceeds 11 GiB; official training needs '
        'a >=24 GiB GPU (e.g. RTX 3090). Prediction fits in 11 GiB.',
        'Per-block activation checkpointing is enabled for memory; it is '
        'disabled in eval and does not change numerics.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--mask-num', type=int, default=5)
        parser.add_argument('--iter-point', type=int, default=8)
        parser.add_argument('--point-list', default='1,3,5,9')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', self.input_size))
        if not args.sam_checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')
        return build_sam_med2d(args.sam_checkpoint, model_type='vit_l',
                               image_size=self.input_size,
                               encoder_adapter=True)

    def configure_trainable(self, model):
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for name, parameter in model.image_encoder.named_parameters():
            parameter.requires_grad_('Adapter' in name)
        model.image_encoder.grad_checkpointing = True

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        return {
            'input_size': int(self.input_size),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'mask_num': int(getattr(args, 'mask_num', 5)),
            'iter_point': int(getattr(args, 'iter_point', 8)),
            'point_list': str(getattr(args, 'point_list', '1,3,5,9')),
        }

    def restore_config(self, args, config):
        for key in ('mask_num', 'iter_point'):
            if key in config:
                setattr(args, key, int(config[key]))
        if 'point_list' in config:
            args.point_list = str(config['point_list'])
        if 'input_size' in config:
            args.input_size = int(config['input_size'])
            self.input_size = args.input_size
        stored_checkpoint = config.get('sam_checkpoint')
        if stored_checkpoint and Path(stored_checkpoint).is_file():
            args.sam_checkpoint = Path(stored_checkpoint)

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def _load(self, sample):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise SystemExit(f'cv2 could not read {sample.image_path}')
        image = (image - PIXEL_MEAN) / PIXEL_STD
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f'cv2 could not read {sample.mask_path}')
        if mask.max() == 255:
            mask = mask / 255.0
        height, width = mask.shape
        transform = train_transforms(self.input_size, height, width)
        augmented = transform(image=image, mask=mask)
        return (augmented['image'], augmented['mask'].to(torch.int64),
                height, width)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, labels, boxes, coords, point_labels = [], [], [], [], []
        mask_num = self.args.mask_num
        for sample in samples:
            image, mask, _, _ = self._load(sample)
            images.append(image)
            for _ in range(mask_num):
                labels.append(mask.unsqueeze(0))
                boxes.append(get_boxes_from_mask(mask))
                pc, pl = init_point_sampling(mask, 1)
                coords.append(pc)
                point_labels.append(pl)
        return {
            'image': torch.stack(images),
            'label': torch.stack(labels),
            'boxes': torch.stack(boxes),
            'point_coords': torch.stack(coords),
            'point_labels': torch.stack(point_labels),
        }

    @staticmethod
    def _to_device(batch, device):
        return {key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()}

    def _prompt_and_decoder(self, model, batch, image_embeddings,
                            decoder_iter=False):
        points = None
        if batch.get('point_coords') is not None:
            points = (batch['point_coords'], batch['point_labels'])
        if decoder_iter:
            with torch.no_grad():
                sparse, dense = model.prompt_encoder(
                    points=points, boxes=batch.get('boxes'),
                    masks=batch.get('mask_inputs'))
        else:
            sparse, dense = model.prompt_encoder(
                points=points, boxes=batch.get('boxes'),
                masks=batch.get('mask_inputs'))
        low_res, iou = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True)
        max_values, max_indexes = torch.max(iou, dim=1)
        iou = max_values.unsqueeze(1)
        selected = [low_res[i:i + 1, index]
                    for i, index in enumerate(max_indexes)]
        low_res = torch.stack(selected, dim=0)
        # Official prompt_and_decoder() upsamples to args.image_size, not a
        # hard-coded 256.
        masks = F.interpolate(
            low_res, (self.input_size, self.input_size), mode='bilinear',
            align_corners=False)
        return masks, low_res, iou

    def _postprocess(self, low_res, original_size):
        height, width = original_size
        masks = F.interpolate(
            low_res, (self.input_size, self.input_size), mode='bilinear',
            align_corners=False)
        if height < self.input_size and width < self.input_size:
            top = (self.input_size - height) // 2
            left = (self.input_size - width) // 2
            masks = masks[..., top:height + top, left:width + left]
        else:
            masks = F.interpolate(masks, original_size, mode='bilinear',
                                  align_corners=False)
        return masks

    def train_epoch(self, model, loader, optimizer, scheduler, args):
        if int(args.accum_steps) != 1 or float(args.grad_clip) != 0:
            raise SystemExit(
                'SAM-Med2D custom loop does not implement --accum-steps/'
                '--grad-clip')
        model.train()
        criterion = FocalDiceloss_IoULoss()
        point_list = [int(x) for x in args.point_list.split(',') if x]
        rng = random.Random(args.seed + self._epoch)
        total = 0.0
        batches = 0
        for samples in loader:
            batch = self._to_device(
                self.prepare_train_batch(samples, rng), self.device)
            images, labels = batch['image'], batch['label']

            for name, parameter in model.image_encoder.named_parameters():
                parameter.requires_grad_('Adapter' in name)
            image_embeddings = model.image_encoder(images)
            batch_size = images.shape[0]
            mask_num = labels.shape[0] // batch_size
            image_embeddings = torch.cat([
                image_embeddings[i].repeat(mask_num, 1, 1, 1)
                for i in range(batch_size)], dim=0)

            if rng.random() > 0.5:
                batch['point_coords'] = None
                batch['point_labels'] = None
            else:
                batch['boxes'] = None
            masks, low_res, iou = self._prompt_and_decoder(
                model, batch, image_embeddings)
            loss = criterion(masks, labels, iou)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            point_num = rng.choice(point_list)
            batch = generate_point(masks.detach(), labels, low_res.detach(),
                                   batch, point_num)
            batch = self._to_device(batch, self.device)
            image_embeddings = image_embeddings.detach()

            for name, parameter in model.named_parameters():
                parameter.requires_grad_(not name.startswith('image_encoder'))

            init_mask_num = rng.randint(1, max(args.iter_point - 2, 1))
            for iteration in range(args.iter_point):
                if (iteration == init_mask_num or
                        iteration == args.iter_point - 1):
                    batch = setting_prompt_none(batch)
                masks, low_res, iou = self._prompt_and_decoder(
                    model, batch, image_embeddings, decoder_iter=True)
                loss = criterion(masks, labels, iou)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if iteration != args.iter_point - 1:
                    point_num = rng.choice(point_list)
                    batch = generate_point(masks.detach(), labels,
                                           low_res.detach(), batch, point_num)
                    batch = self._to_device(batch, self.device)

            total += float(loss.detach())
            batches += 1
        if scheduler is not None:
            scheduler.step()
        # The official loop averages one loss per batch, not per sample.
        return total / max(batches, 1)

    def train_step(self, model, batch):
        raise RuntimeError('SAM-Med2D uses the custom train_epoch loop')

    def prepare_eval_sample(self, sample, device):
        image, mask, height, width = self._load(sample)
        boxes = get_boxes_from_mask(mask, max_pixel=0)
        return {
            'image': image.unsqueeze(0).to(device),
            'boxes': boxes.to(device),
            'original_size': (height, width),
        }

    def predict_sample(self, model, prepared, sample):
        with torch.no_grad():
            embeddings = model.image_encoder(prepared['image'])
            batch = {
                'boxes': prepared['boxes'],
                'point_coords': None,
                'point_labels': None,
            }
            _, low_res, _ = self._prompt_and_decoder(
                model, batch, embeddings)
        masks = self._postprocess(low_res, prepared['original_size'])
        return (torch.sigmoid(masks)[0, 0].cpu().numpy() > 0.5)
