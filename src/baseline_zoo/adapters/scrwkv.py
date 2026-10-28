"""SCRWKV baseline adapter (ICML 2026).

Ultra-compact Structure-Calibrated Vision-RWKV for topological crack
segmentation (1.22M parameters in the paper).  The model is trained from
scratch on the shared 18-image split with the official crack recipe:
512-square cubic resize, (0.5, 0.5, 0.5) normalization, 0.25 BCE + 0.75
Dice, AdamW lr 5e-4 / wd 0.01, PolyLR to 1e-6, batch 1.

The SCRWKV core uses a custom CUDA bidirectional WKV kernel; this port
vendors it and builds it on the target GPU (official sm_86 / RTX 3090).
The adapter fails with a clear error if the extension cannot be compiled.
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.scrwkv import build_scrwkv
from baseline_zoo.runtime import BaselineAdapter


class SCRWKVAdapter(BaselineAdapter):
    id = 'scrwkv'
    display_name = 'SCRWKV'
    official_source = 'zhxhzy/SCRWKV (self-contained port)'
    official_entrypoint = 'https://github.com/zhxhzy/SCRWKV'
    input_size = 512
    official_lr = 5e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    accumulation_steps = 1
    selection = 'dice'
    deviations = (
        'SCRWKV is trained from scratch on the shared 18-image split; no '
        'official weights are required. The official 50-epoch recipe on '
        'full crack datasets is extended to the shared 800-epoch budget '
        'with validation-Dice selection.',
        'The model core (SFE/SCIU/AMCM/CSHF/DySample) is vendored. The '
        'released SFE depends on mmcv/mmcls (not installed and not needed '
        'for the model); this port replaces build_norm_layer, ModuleList, '
        'AdaptivePadding, resize_pos_embed and BaseBackbone with local '
        'torch equivalents. The architecture and forward path are '
        'unchanged for the fixed 512 input (corner padding is zero).',
        'The official bi_wkv CUDA extension is vendored and compiled at '
        'import. The released code hardcodes sm_86; this port selects the '
        'current GPU capability or SCRWKV_CUDA_ARCH. Without a CUDA '
        'toolkit/nvcc the model can be built but forward fails with a '
        'clear error; the target RTX 3090 (sm_86) server has the official '
        'configuration.',
        'Official training uses 512-square cv2 INTER_CUBIC resize for both '
        'images and masks, thresholds masks at 127, normalizes with '
        '(0.5, 0.5, 0.5), and has no augmentation; all are kept.',
        'Official loss/optimizer/schedule: 0.25 BCEWithLogits + 0.75 Dice, '
        'AdamW lr 5e-4, wd 0.01, PolyLR to 1e-6, batch 1. The shared '
        'runtime steps the scheduler once per epoch; the released trainer '
        'also steps per epoch.',
        'Official test.py normalizes raw logits by each image maximum and '
        'thresholds at 0.2; this port keeps that inference rule and '
        'nearest-resizes the binary mask to the original GT resolution '
        '(the release evaluates at 512).',
        'The port builds ~2.04M parameters, whereas the paper reports '
        '1.22M for SCRWKV; the released Crack config and CSHF/DySample '
        'modules are kept verbatim, so the discrepancy is in the paper '
        'count/config rather than a deliberately enlarged model.',
    )

    def __init__(self):
        super().__init__()

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 512))
        if self.input_size != 512:
            raise SystemExit('SCRWKV uses the official 512x512 input')
        return build_scrwkv()

    def load_pretrained(self, model, args) -> None:
        print('initialized SCRWKV from scratch (official Crack config)')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        # Official PolyLR(eta_min=1e-6, end=epochs), stepped per epoch.
        epochs = max(1, int(args.epochs))
        floor = 1e-6 / max(float(args.lr), 1e-12)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: max(
                floor, 1.0 - min(epoch, epochs) / epochs),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _pair(self, sample):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.uint8)
        image = cv2.resize(
            image, (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(
            mask, (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC)
        mask = (mask > 127).astype(np.float32)
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        image_t = torch.from_numpy(
            image.transpose(2, 0, 1).copy()).float()
        return image_t, torch.from_numpy(mask)[None]

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._pair(sample)
            images.append(image)
            masks.append(mask)
        return {
            'images': torch.stack(images).to(self.device),
            'masks': torch.stack(masks).to(self.device),
        }

    @staticmethod
    def _bce_dice(logits, target, bce_ratio=0.25, dice_ratio=0.75):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = (-2, -1)
        tp = (prob * target).sum(dims)
        fp = (prob * (1 - target)).sum(dims)
        fn = ((1 - prob) * target).sum(dims)
        dice = 1 - (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
        return bce_ratio * bce + dice_ratio * dice.mean()

    def train_step(self, model, batch):
        logits = model(batch['images'])
        return self._bce_dice(logits, batch['masks'])

    def prepare_eval_sample(self, sample, device):
        image, _ = self._pair(sample)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits = model(prepared['image'])[0, 0]
        peak = float(logits.max())
        if peak > 0:
            logits = logits / peak
        pred = (logits >= 0.2).float()
        width, height = prepared['size']
        pred = F.interpolate(
            pred[None, None], size=(height, width), mode='nearest')[0, 0]
        return pred.cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'cuda_arch': str(os.environ.get(
                'SCRWKV_CUDA_ARCH', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
