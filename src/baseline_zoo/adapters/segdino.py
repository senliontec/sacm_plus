"""SegDINO baseline: DINOv3 backbone + official DPT decoder.

Official reference: vendored ``baselines/segdino``.  ``dpt.py`` and
``blocks.py`` are copied into ``baseline_zoo.models.segdino``.  DINOv3
weights are gated on Hugging Face; use ``HF_TOKEN`` or ``--dino-ckpt``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.segdino import build_segdino
from baseline_zoo.runtime import BaselineAdapter

PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SegDINOAdapter(BaselineAdapter):
    id = 'segdino'
    display_name = 'SegDINO'
    official_source = 'SegDINO (self-contained DPT + timm DINOv3)'
    official_entrypoint = 'upstream train_segdino.py'
    input_size = 256
    official_epochs = 50
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'Official checkpoint selection uses the test split; the protocol '
        'uses the 6-image validation split by Dice.',
        'The official code loads a local DINOv3 repository and gated DINOv3 '
        'weights. This port uses timm DINOv3 and accepts --dino-ckpt or an '
        'HF token; --random-init exists only for a code smoke test.',
        'The official README trains/evaluates at 256 while test_segdino.py '
        'defaults to 1024; this port follows the README 256 setting.',
        'Predictions are restored to the original GT resolution for the '
        'shared evaluator.',
        'The official no-augmentation, BCEWithLogits, AdamW 1e-4/1e-4, '
        '50-epoch, batch-4 recipe is preserved.',
    )

    def add_args(self, parser):
        parser.add_argument('--dino-size', default='small',
                            choices=['s', 'small', 'b', 'base', 'l', 'large'])
        default_ckpt = os.environ.get('DINO_CKPT')
        parser.add_argument('--dino-ckpt', type=Path,
                            default=Path(default_ckpt) if default_ckpt else None)
        parser.add_argument('--random-init', action='store_true',
                            help='code smoke only: do not load DINOv3 weights')

    def build_model(self, args):
        return build_segdino(
            args.dino_size,
            str(args.dino_ckpt) if args.dino_ckpt else None,
            args.random_init)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BICUBIC)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.NEAREST)
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
            target = torch.from_numpy(
                (np.asarray(mask) > 127).astype(np.float32).copy()
            ).unsqueeze(0)
            images.append(tensor)
            masks.append(target)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = model(images)
        return nn.BCEWithLogitsLoss()(logits, masks)

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        prob = torch.sigmoid(model(prepared))
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        prob = F.interpolate(prob, size=(height, width),
                             mode='bilinear', align_corners=False)
        return (prob[0, 0].cpu().numpy() > 0.5)
