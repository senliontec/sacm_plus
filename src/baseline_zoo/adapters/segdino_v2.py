"""SegDINO v2 adapter (MICCAI 2026).

The official v2 repository keeps the DINOv3-S/16 backbone and replaces the v1
DPT neck with the TPA/SAD decoder: four token projections, four bilinear
resampling branches, residual depthwise refinement per branch and a top-down
fusion.  The official code uses 256-square ImageNet-normalized inputs,
BCEWithLogits, AdamW lr 1e-4, batch 4 and 50 epochs.  The paper states a frozen
DINOv3 encoder, CE and no augmentation; the released code trains the backbone,
uses BCE and flips/rotation/ColorJitter.  This port defaults to the paper's
frozen encoder, the released decoder/loss and the released augmentation, all
exposed as flags.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.segdino import find_dinov3_checkpoint
from baseline_zoo.models.segdino_v2 import build_segdino_v2
from baseline_zoo.runtime import BaselineAdapter, file_sha256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegDINOV2Adapter(BaselineAdapter):
    id = 'segdino_v2'
    display_name = 'SegDINO v2'
    family = 'dino'
    official_source = 'SegDINO v2 (MICCAI 2026; script-Yang/segdino_v2)'
    official_entrypoint = 'https://github.com/script-Yang/segdino_v2'
    input_size = 256
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 4
    accumulation_steps = 1
    selection = 'dice'
    deviations = (
        'The official v2 paper is not present under parsing/; this port was '
        'checked against the archival paper (arXiv 2606.17972) and the '
        'repository code.',
        'Paper says the DINOv3 encoder is frozen, CE is used and no '
        'augmentation is applied; the released code trains the backbone, uses '
        'BCEWithLogits and applies flips/rotation/ColorJitter. This port '
        'defaults to the paper frozen encoder, the released BCE and released '
        'augmentation; --segdino-v2-train-backbone and '
        '--segdino-v2-no-augment reproduce the other choices.',
        'Paper states weight decay 1e-4; the released trainer leaves the '
        'AdamW default 0.01. This port uses 1e-4.',
        'The official 50-epoch budget is extended to the shared 800-epoch '
        'budget with validation-Dice selection; the official code also '
        'selects best validation Dice.',
        'The official loader uses torch.hub with a local DINOv3 checkout; '
        'this port uses the timm DINOv3-S/16 backbone already vendored by the '
        'SegDINO v1 baseline and loads the same local weights.',
        'The official code reports metrics at 256; this port bilinearly '
        'upsamples logits to the exact GT resolution before thresholding.',
    )

    def __init__(self):
        super().__init__()
        self._dino_sha: str | None = None
        self._dino_ckpt: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=256)
        parser.add_argument('--segdino-v2-dino-size', default='small',
                            choices=['small', 'base', 'large'])
        parser.add_argument('--segdino-v2-dino-ckpt', type=Path, default=None)
        parser.add_argument('--segdino-v2-decoder-dim', type=int, default=256)
        parser.add_argument('--segdino-v2-train-backbone', action='store_true',
                            help='released-code behavior: train the backbone')
        parser.add_argument('--segdino-v2-no-augment', action='store_true')
        parser.add_argument('--segdino-v2-random-init', action='store_true',
                            help='smoke only: random DINOv3 weights')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 256))
        if self.input_size != 256:
            raise SystemExit('SegDINO v2 uses the official 256x256 input')
        random_init = bool(getattr(args, 'segdino_v2_random_init', False))
        size = getattr(args, 'segdino_v2_dino_size', 'small')
        ckpt = getattr(args, 'segdino_v2_dino_ckpt', None)
        if ckpt is None and not random_init and not self.prediction_mode:
            ckpt = find_dinov3_checkpoint(size)
        if ckpt is not None:
            ckpt = Path(ckpt)
            if not ckpt.is_file():
                raise SystemExit(f'missing DINOv3 checkpoint: {ckpt}')
            self._dino_ckpt = str(ckpt)
            self._dino_sha = file_sha256(ckpt)
        if not random_init and not self.prediction_mode and ckpt is None:
            raise SystemExit(
                'SegDINO v2 needs a local DINOv3 checkpoint under '
                'checkpoints/dinov3/ or --segdino-v2-dino-ckpt; '
                'the server is offline')
        return build_segdino_v2(
            dino_size=size,
            dino_ckpt=None if random_init or self.prediction_mode else ckpt,
            decoder_dim=int(getattr(args, 'segdino_v2_decoder_dim', 256)),
            random_init=random_init or self.prediction_mode,
        )

    def load_pretrained(self, model, args) -> None:
        if bool(getattr(args, 'segdino_v2_random_init', False)):
            print('SegDINO v2 smoke: random DINOv3 initialization')
        elif self.prediction_mode:
            return
        else:
            print('loaded local DINOv3 backbone weights')

    def configure_trainable(self, model) -> None:
        if bool(getattr(self.args, 'segdino_v2_train_backbone', False)):
            if not all(p.requires_grad for p in model.backbone.parameters()):
                raise RuntimeError('--segdino-v2-train-backbone requested but '
                                   'backbone is frozen')
        else:
            model.lock_backbone()
        if not any(p.requires_grad for p in model.decoder.parameters()):
            raise RuntimeError('SegDINO v2 decoder has no trainable parameters')
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f'SegDINO v2 parameters: {total:,} total, {trainable:,} '
              'trainable')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        # The official v2 trainer has no scheduler.
        return None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _resize_pair(self, sample: Sample, rng: random.Random | None,
                     train: bool):
        image = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path)
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode='L')
        size = (self.input_size, self.input_size)
        image = TF.resize(
            image, size, interpolation=TF.InterpolationMode.BILINEAR)
        mask_img = TF.resize(
            mask_img, size, interpolation=TF.InterpolationMode.NEAREST)
        if train and rng is not None and not bool(getattr(
                self.args, 'segdino_v2_no_augment', False)):
            if rng.random() < 0.5:
                image = TF.hflip(image)
                mask_img = TF.hflip(mask_img)
            if rng.random() < 0.5:
                image = TF.vflip(image)
                mask_img = TF.vflip(mask_img)
            if rng.random() < 0.5:
                angle = rng.randint(-15, 15)
                image = TF.rotate(
                    image, angle,
                    interpolation=TF.InterpolationMode.BILINEAR)
                mask_img = TF.rotate(
                    mask_img, angle,
                    interpolation=TF.InterpolationMode.NEAREST)
            if rng.random() < 0.5:
                image = TF.adjust_brightness(
                    image, 1 + rng.uniform(-0.2, 0.2))
                image = TF.adjust_contrast(
                    image, 1 + rng.uniform(-0.2, 0.2))
                image = TF.adjust_saturation(
                    image, 1 + rng.uniform(-0.1, 0.1))
                image = TF.adjust_hue(image, rng.uniform(-0.05, 0.05))
        image_t = TF.normalize(
            TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_t = TF.to_tensor(mask_img)
        return image_t, mask_t

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._resize_pair(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
        return {
            'images': torch.stack(images).to(self.device),
            'masks': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        logits = model(batch['images'])
        return F.binary_cross_entropy_with_logits(logits, batch['masks'])

    def prepare_eval_sample(self, sample: Sample, device):
        image, _ = self._resize_pair(sample, None, train=False)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits = model(prepared['image'])
        width, height = prepared['size']
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return logits.sigmoid()[0, 0].cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'dino_size': getattr(args, 'segdino_v2_dino_size', 'small'),
            'dino_ckpt': self._dino_ckpt,
            'dino_sha256': self._dino_sha,
            'decoder_dim': int(getattr(
                args, 'segdino_v2_decoder_dim', 256)),
            'train_backbone': bool(getattr(
                args, 'segdino_v2_train_backbone', False)),
            'augment': not bool(getattr(
                args, 'segdino_v2_no_augment', False)),
            'random_init': bool(getattr(
                args, 'segdino_v2_random_init', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'dino_size' in config:
            args.segdino_v2_dino_size = config['dino_size']
        if config.get('dino_ckpt'):
            args.segdino_v2_dino_ckpt = Path(config['dino_ckpt'])
            self._dino_ckpt = config['dino_ckpt']
        if 'dino_sha256' in config:
            self._dino_sha = config['dino_sha256']
        if 'decoder_dim' in config:
            args.segdino_v2_decoder_dim = config['decoder_dim']
        if 'train_backbone' in config:
            args.segdino_v2_train_backbone = config['train_backbone']
        if 'augment' in config:
            args.segdino_v2_no_augment = not config['augment']
        if 'random_init' in config:
            args.segdino_v2_random_init = config['random_init']
