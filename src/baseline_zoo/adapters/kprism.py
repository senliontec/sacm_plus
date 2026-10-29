"""K-Prism baseline: prompt-free semantic branch.

The upstream release supports semantic, in-context and interactive modes.
The benchmark is prompt/reference-free, so this adapter trains and evaluates
only the semantic query branch (``click_mode='0'``) with a zero dense prompt
map.  Backbone, pixel fuser, MoE decoder, loss, augmentation and resize/pad
inference are the vendored upstream implementations.

Official code: https://github.com/bangwayne/K-Prism (commit 78c3b1b, MIT).
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from baseline_zoo.models.kprism import build_kprism_semantic
from baseline_zoo.models.kprism.transforms import (
    build_train_transform,
    read_binary_mask,
    read_rgb,
    resize_pad_normalize,
)
from baseline_zoo.runtime import BaselineAdapter


class KPrismAdapter(BaselineAdapter):
    id = 'kprism'
    display_name = 'K-Prism'
    official_source = 'bangwayne/K-Prism (self-contained copy)'
    official_entrypoint = 'https://github.com/bangwayne/K-Prism'
    input_size = 512
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 2
    default_grad_clip = 3.0
    selection = 'dice'
    deviations = (
        'Only the semantic query branch is used (click_mode="0") with a '
        'zero 2-D dense prompt map. Upstream semantic evaluation derives '
        'initial points from the ground-truth mask and feeds them as the '
        'dense prompt; the benchmark is prompt-free, so no GT points, '
        'clicks, references or interactive iterations are used.',
        'The binary head uses num_classes=2 and queries label 1; upstream '
        'uses 33 class-query groups over 13 labels. Only one query group is '
        'trained, so class-query embeddings 2..33 do not exist here.',
        'Upstream training mixes click, reference and semantic modes with '
        'probabilities [0.3,0.4,0.3] and consumes support images/masks. This '
        'port trains the semantic branch alone on the 3-shot split, matching '
        'the prompt-free benchmark protocol.',
        'Training augmentation is the exact upstream albumentations 1.3.1 '
        'pipeline (LongestMaxSize 512, centre pad, flip, shift/scale/rotate, '
        'brightness/contrast, Gaussian blur/noise, grid distortion, '
        'normalize mean 0.26/std 0.30). The albumentations RNG is reseeded '
        'per batch for reproducibility; upstream uses the global unseeded '
        'RNG, so augmented pixels are not bitwise identical.',
        'Images are decoded with cv2 (BGR->RGB); upstream training uses '
        'torchvision.io.read_image. JPEG/PNG decode differences are minor '
        'and not bitwise identical.',
        'Optimizer AdamW lr 1e-4, weight decay 0, batch 2 and gradient '
        'clipping norm 3.0 are official. Upstream runs DDP over 4 GPUs at '
        'batch 2 per rank (effective 8) for 75 epochs; this port uses the '
        'shared single-GPU job runtime at batch 2 and the 800-epoch protocol '
        'budget, so effective batch and optimizer-step count differ.',
        'Official linear warmup is 10 epochs, but its scheduler sets the '
        'first epoch LR to exactly 0. This port starts at 1/10 of the base '
        'LR, reaches the base LR after warmup, then cosine-decays to '
        '0.1*base (official eta_min=1e-5).',
        'No official K-Prism checkpoint is released in the upstream tree; '
        'the model is trained from scratch on the 3-shot split.',
        'Checkpoint selection uses validation Dice on the 6-image val split; '
        'upstream periodically saves checkpoints and reports test metrics.',
    )

    def __init__(self):
        super().__init__()
        self._train_transform = None

    def add_args(self, parser):
        parser.add_argument(
            '--kprism-warmup-epochs', type=int, default=10,
            help='official linear warmup length in epochs')

    def build_model(self, args) -> nn.Module:
        self._train_transform = build_train_transform(self.input_size)
        return build_kprism_semantic(image_size=self.input_size)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        warmup = max(1, int(getattr(args, 'kprism_warmup_epochs', 10)))
        total = max(int(args.epochs), warmup + 1)
        eta_ratio = 0.1  # official end_lr_ratio=0.1

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return eta_ratio + (1.0 - eta_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def prepare_train_batch(self, samples, rng: random.Random):
        if self._train_transform is None:
            self._train_transform = build_train_transform(self.input_size)
        # Albumentations uses the global random/numpy RNG.  Reseed from the
        # runtime's per-epoch RNG so runs are reproducible without leaking
        # state across epochs.
        seed = rng.getrandbits(64)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        images, masks = [], []
        for sample in samples:
            image = read_rgb(sample.image_path)
            mask = read_binary_mask(sample.mask_path)
            augmented = self._train_transform(image=image, mask=mask)
            images.append(augmented['image'].float())
            masks.append(augmented['mask'].unsqueeze(0).float())
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        outputs = model(images)
        targets = [{'masks': mask} for mask in masks]
        losses = model.criterion(outputs, targets)
        return sum(losses.values())

    def prepare_eval_sample(self, sample, device):
        image, original_size, pad = resize_pad_normalize(
            read_rgb(sample.image_path), self.input_size)
        return {
            'image': image.unsqueeze(0).to(device),
            'original_size': original_size,
            'pad': pad,
        }

    def predict_sample(self, model, prepared, sample):
        with torch.no_grad():
            outputs = model(prepared['image'])
        logits = F.interpolate(
            outputs['pred_masks'],
            size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False)
        probability = torch.sigmoid(logits)[0, 0]
        pad_top, pad_bottom, pad_left, pad_right = prepared['pad']
        probability = probability[
            pad_top:self.input_size - pad_bottom,
            pad_left:self.input_size - pad_right,
        ]
        height, width = prepared['original_size']
        probability = F.interpolate(
            probability[None, None], size=(height, width),
            mode='bilinear', align_corners=False)[0, 0]
        return probability.cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'warmup_epochs': int(
                getattr(args, 'kprism_warmup_epochs', 10)),
            'num_classes': 2,
            'num_queries': 2,
            'dec_layers': 6,
            'num_experts': 5,
            'pretrained': False,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'warmup_epochs' in config:
            args.kprism_warmup_epochs = int(config['warmup_epochs'])
