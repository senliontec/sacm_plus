"""FSSAM baseline adapter (ICML 2025).

FSSAM is a support-based few-shot segmenter built on SAM 2 and DINOv2.
Each target image receives the protocol's support images (3 shots for seen
domains, a structurally matched donor for unseen/novel domains); no target
ground truth or prompts are used at inference.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.fssam import build_fssam, weighted_dice_loss
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256
from baseline_zoo.support import load_support_groups, select_supports

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255


class FSSAMAdapter(BaselineAdapter):
    id = 'fssam'
    display_name = 'FSSAM'
    official_source = 'FSSAM (self-contained SAM 2 + DINOv2 port)'
    official_entrypoint = 'https://github.com/Sam1224/FSSAM'
    input_size = 512
    official_lr = 1e-3
    official_weight_decay = 0.0
    train_batch_size = 2
    accumulation_steps = 4
    selection = 'dice'
    support_based = True
    deviations = (
        'FSSAM is support-based, not prompt-free: the protocol support '
        'images and masks (3 shots for seen domains, a donor domain for '
        'unseen/novel datasets) drive the pseudo query memories. The '
        'official paper samples supports from the same class; in the shared '
        '18-image split each seen training domain has only three images, so '
        'training episodes use up to two supports after excluding the query '
        'image, while evaluation uses the protocol support set.',
        'The official model uses SAM 2-S (46M) and DINOv2-B (86M). This '
        'port defaults to that official pair; DINOv2-L/14 remains an '
        'explicit ablation via --fssam-dino-size vitl. The local SAM 2 '
        'checkpoint must match the selected config.',
        'The released code runs under global bfloat16 autocast; this port '
        'runs in float32 for hardware independence and casts the memory '
        'features (which the upstream helper hard-codes to bf16) back to '
        'the model dtype. The official effective batch 8 is kept as batch '
        '2 x 4 accumulation steps.',
        'The official poly LR (power 0.9, no warmup) is stepped once per '
        'epoch because the shared runtime owns the loop; the released '
        'trainer steps it per iteration. The official 300-epoch (PASCAL) '
        'and 75-epoch (COCO) budgets are extended to the shared 800-epoch '
        'budget. The official per-epoch/half-epoch mIoU validation, '
        'patience-100 early stopping and test-split reporting are replaced '
        'by validation-Dice checkpoint selection on the shared 6-image '
        'validation split.',
        'The released FSSAM commit calls self.Weighted_GAP without '
        'defining it and overwrites the support-mask stack inside the shot '
        'loop, so its SCMA calibration is unusable for k>1. The vendored '
        'SAM 2 copy supplies the missing weighted-GAP helper and averages '
        'the per-shot support similarities as described in the paper.',
        'No official FSSAM trained weights are required; this port starts '
        'from the official SAM 2 and DINOv2 checkpoints. Images and '
        'supports are resized to 512 square and predictions are resized '
        'back to the original GT resolution. The DINOv2 prior path only '
        'requests the final block instead of all intermediate blocks; the '
        'consumed feature is identical.',
    )

    def __init__(self):
        super().__init__()
        self._support_groups = None
        self._support_root: Path | None = None
        self._sam2_sha: str | None = None
        self._dino_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument(
            '--fssam-sam2-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam2_hiera_small.pt',
            help='local SAM 2 checkpoint matching --fssam-sam2-config')
        parser.add_argument(
            '--fssam-sam2-config', default='sam2_hiera_s.yaml',
            choices=['sam2_hiera_s.yaml', 'sam2_hiera_b+.yaml',
                     'sam2_hiera_l.yaml'])
        parser.add_argument(
            '--fssam-dino-weights', type=Path, default=None,
            help='defaults to checkpoints/dinov2_vitl14_pretrain.pth or '
                 'dinov2_vitb14_pretrain.pth according to --fssam-dino-size')
        parser.add_argument('--fssam-dino-size', default='vitb',
                            choices=['vitl', 'vitb'])
        parser.add_argument('--fssam-num-refine', type=int, default=3)
        parser.add_argument('--fssam-ver-refine', default='v1',
                            choices=['v1', 'v2'])
        parser.add_argument('--fssam-shots', type=int, default=3)
        parser.add_argument('--fssam-support-dataset', default='auto')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 512))
        if self.input_size != 512:
            raise SystemExit('FSSAM uses the official 512x512 input')
        sam2_path = Path(getattr(
            args, 'fssam_sam2_checkpoint',
            REPO_ROOT / 'checkpoints' / 'sam2_hiera_small.pt'))
        dino_size = str(getattr(args, 'fssam_dino_size', 'vitb'))
        dino_name = ('dinov2_vitb14_pretrain.pth'
                     if dino_size == 'vitb'
                     else 'dinov2_vitl14_pretrain.pth')
        dino_path = getattr(args, 'fssam_dino_weights', None)
        if dino_path is None:
            dino_path = REPO_ROOT / 'checkpoints' / dino_name
            args.fssam_dino_weights = dino_path
        dino_path = Path(dino_path)
        for label, path in (('SAM 2', sam2_path), ('DINOv2', dino_path)):
            if not path.is_file():
                raise SystemExit(
                    f'missing {label} checkpoint: {path}\n'
                    'Upload the official weights under checkpoints/.')
        self._sam2_sha = file_sha256(sam2_path)
        self._dino_sha = file_sha256(dino_path)
        self._support_root = Path(getattr(args, 'data_root', ''))
        return build_fssam(
            sam2_checkpoint=sam2_path,
            sam2_config=str(getattr(
                args, 'fssam_sam2_config', 'sam2_hiera_s.yaml')),
            dino_weights=dino_path,
            dino_size=str(getattr(args, 'fssam_dino_size', 'vitb')),
            num_refine=int(getattr(args, 'fssam_num_refine', 3)),
            ver_refine=str(getattr(args, 'fssam_ver_refine', 'v1')),
            input_size=self.input_size,
        )

    def load_pretrained(self, model, args) -> None:
        print('loaded SAM 2 and DINOv2 base weights for FSSAM')

    def configure_trainable(self, model) -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('FSSAM has no trainable parameters')
        if any(p.requires_grad for p in model.sam2.image_encoder.parameters()):
            raise RuntimeError('FSSAM image encoder must stay frozen')
        if not any(p.requires_grad
                   for p in model.sam2.memory_attention.parameters()):
            raise RuntimeError('FSSAM memory attention is not trainable')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        # Official poly_learning_rate: lr = base * (1 - iter/max_iter)^0.9,
        # no warmup.  The shared runtime steps once per epoch.
        epochs = max(1, int(args.epochs))
        power = 0.9

        def factor(epoch: int) -> float:
            return max(0.0, 1.0 - epoch / epochs) ** power

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _groups(self):
        if self._support_groups is None:
            if self._support_root is None:
                raise RuntimeError('FSSAM data_root is not configured')
            self._support_groups = load_support_groups(self._support_root)
        return self._support_groups

    @staticmethod
    def _load_pair(sample):
        image = np.asarray(load_rgb(sample.image_path), dtype=np.float32)
        mask = load_mask(sample.mask_path).astype(np.float32)
        return image, mask

    def _augment(self, image, mask, rng: random.Random):
        """Official FSSAM training transform (cv2, seeded RNG)."""
        h, w = mask.shape
        scale = 0.9 + 0.2 * rng.random()
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)

        if rng.random() < 0.5:
            angle = -10 + 20 * rng.random()
            h, w = mask.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(
                image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=tuple(float(x) for x in _IMAGENET_MEAN))
            mask = cv2.warpAffine(
                mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        if rng.random() < 0.5:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        h, w = mask.shape
        pad_h = max(self.input_size - h, 0)
        pad_w = max(self.input_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            left = pad_w // 2
            image = cv2.copyMakeBorder(
                image, top, pad_h - top, left, pad_w - left,
                cv2.BORDER_CONSTANT,
                value=tuple(float(x) for x in _IMAGENET_MEAN))
            mask = cv2.copyMakeBorder(
                mask, top, pad_h - top, left, pad_w - left,
                cv2.BORDER_CONSTANT, value=255)
        h, w = mask.shape
        top = rng.randint(0, h - self.input_size)
        left = rng.randint(0, w - self.input_size)
        image = image[top:top + self.input_size,
                      left:left + self.input_size]
        mask = mask[top:top + self.input_size,
                    left:left + self.input_size]
        mask = (mask == 1).astype(np.float32)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask_t = torch.from_numpy(mask)[None]
        return image_t, mask_t

    def _eval_pair(self, sample):
        image = np.asarray(load_rgb(sample.image_path), dtype=np.float32)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = load_mask(sample.mask_path).astype(np.float32)
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_NEAREST)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        return image_t, torch.from_numpy(mask)[None]

    def _train_supports(self, sample, rng: random.Random, shots: int):
        groups = self._groups()
        candidates = [item for item in groups.get(sample.dataset, [])
                      if item.stem != sample.stem]
        if not candidates:
            candidates = [sample]
        count = min(int(shots), len(candidates))
        return rng.sample(candidates, count)

    def _eval_supports(self, dataset: str, shots: int):
        groups = self._groups()
        samples, donor = select_supports(
            dataset, groups, shots=shots,
            override=str(getattr(
                self.args, 'fssam_support_dataset', 'auto')))
        pairs = [self._eval_pair(sample) for sample in samples]
        images = torch.stack([item[0] for item in pairs])
        masks = torch.stack([item[1] for item in pairs])
        return images, masks, donor

    def prepare_train_batch(self, samples, rng: random.Random):
        shots = int(getattr(self.args, 'fssam_shots', 3))
        per_query = [self._train_supports(sample, rng, shots)
                     for sample in samples]
        count = min(len(items) for items in per_query)
        queries, query_masks = [], []
        support_images, support_masks = [], []
        for sample, supports in zip(samples, per_query):
            image, mask = self._augment(*self._load_pair(sample), rng)
            queries.append(image)
            query_masks.append(mask)
            sup_imgs, sup_masks = [], []
            for support in supports[:count]:
                sup_img, sup_mask = self._augment(
                    *self._load_pair(support), rng)
                sup_imgs.append(sup_img)
                sup_masks.append(sup_mask)
            support_images.append(torch.stack(sup_imgs))
            support_masks.append(torch.stack(sup_masks))
        return {
            'query': torch.stack(queries).to(self.device),
            'query_mask': torch.stack(query_masks).to(self.device),
            'supports': torch.stack(support_images).to(self.device),
            'support_masks': torch.stack(support_masks).to(self.device),
        }

    def train_step(self, model, batch):
        logits = model(batch['query'], batch['supports'],
                       batch['support_masks'])
        return weighted_dice_loss(logits, batch['query_mask'][:, 0])

    def prepare_eval_sample(self, sample, device):
        query, _ = self._eval_pair(sample)
        shots = int(getattr(self.args, 'fssam_shots', 3))
        supports, support_masks, _ = self._eval_supports(
            sample.dataset, shots)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'query': query.unsqueeze(0).to(device),
            'supports': supports.unsqueeze(0).to(device),
            'support_masks': support_masks.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits = model(prepared['query'], prepared['supports'],
                       prepared['support_masks'])
        width, height = prepared['size']
        logits = F.interpolate(
            logits.unsqueeze(1), size=(height, width), mode='bilinear',
            align_corners=True)
        return torch.sigmoid(logits[0, 0]).cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'sam2_checkpoint': str(
                getattr(args, 'fssam_sam2_checkpoint', '')),
            'sam2_config': str(
                getattr(args, 'fssam_sam2_config', 'sam2_hiera_s.yaml')),
            'dino_weights': str(getattr(args, 'fssam_dino_weights', '')),
            'dino_size': str(getattr(args, 'fssam_dino_size', 'vitb')),
            'num_refine': int(getattr(args, 'fssam_num_refine', 3)),
            'ver_refine': str(getattr(args, 'fssam_ver_refine', 'v1')),
            'shots': int(getattr(args, 'fssam_shots', 3)),
            'support_dataset': str(
                getattr(args, 'fssam_support_dataset', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
            'sam2_sha256': self._sam2_sha,
            'dino_sha256': self._dino_sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        for key, attribute in (
                ('sam2_checkpoint', 'fssam_sam2_checkpoint'),
                ('dino_weights', 'fssam_dino_weights')):
            if key in config:
                setattr(args, attribute, Path(config[key]))
        for key, attribute in (
                ('sam2_config', 'fssam_sam2_config'),
                ('dino_size', 'fssam_dino_size'),
                ('ver_refine', 'fssam_ver_refine'),
                ('support_dataset', 'fssam_support_dataset')):
            if key in config:
                setattr(args, attribute, str(config[key]))
        for key, attribute in (
                ('num_refine', 'fssam_num_refine'),
                ('shots', 'fssam_shots')):
            if key in config:
                setattr(args, attribute, int(config[key]))
        if 'data_root' in config:
            self._support_root = Path(config['data_root'])
