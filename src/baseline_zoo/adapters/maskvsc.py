"""MaskVSC baseline with the official Fundus UNet training logic.

Official reference: https://github.com/Zhouyi-Zura/MaskVSC (vendored snapshot
in ``baselines/maskvsc``).  The UNet, Connect_Loss and masked-vessel
corruption are copied into ``baseline_zoo.models.maskvsc``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.maskvsc import (
    Connect_Loss,
    UNet,
    initialize_weights,
    mask_vessel_corrupt,
)
from baseline_zoo.runtime import BaselineAdapter


def mask_ratio(epoch: int, epochs: int) -> float:
    """Upstream epoch-0-based schedule, mapped to our 1-based epoch."""
    ratio = (epoch - 1) / max(epochs, 1)
    if ratio < 0.1 or ratio > 0.9:
        return 0.0
    if ratio <= 0.5:
        return 0.4 * (ratio - 0.1) / 0.4
    return 0.4 * (0.9 - ratio) / 0.4


class MaskVSCAdapter(BaselineAdapter):
    id = 'maskvsc'
    display_name = 'MaskVSC'
    official_source = 'Zhouyi-Zura/MaskVSC (self-contained copy)'
    official_entrypoint = 'https://github.com/Zhouyi-Zura/MaskVSC'
    input_size = 512
    official_epochs = 200
    official_lr = 2e-4
    official_weight_decay = 0.0
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'Data protocol: the 18/6-shot split replaces the original Train/Test '
        'folders; grayscale domains are converted to RGB because the official '
        'Fundus UNet expects three channels.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream evaluates the test split every epoch with a Dice > 0.5 '
        'gate; test-based selection would violate the protocol.',
        'Official test.py writes 512x512 masks; predictions are restored to '
        'the original GT resolution so every baseline shares one evaluator.',
        'The protocol seeds 0-4 replace upstream seed 1234 and deterministic '
        'cuDNN.',
        'The official corruption sampler still consumes RNG when ratio=0; this '
        'port early-returns before that draw, and also uses workers=0. '
        'Augmentation RNG order therefore differs from upstream.',
        'Connect_Loss uses the input device instead of a hardcoded .cuda(); '
        'the computation is identical on GPU.',
    )

    def build_model(self, args) -> nn.Module:
        model = UNet(n_channels=3, n_classes=1)
        initialize_weights(model)
        return model

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.5, 0.999),
            weight_decay=args.weight_decay)

    def prepare_train_batch(self, samples, rng: random.Random):
        ratio = mask_ratio(getattr(self, '_epoch', 1),
                           self.args.epochs if self.args else 200)
        images, labels = [], []
        for sample in samples:
            image = load_rgb(sample.image_path)
            label = Image.open(sample.mask_path).convert('L')
            if rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            image_np = np.array(image.resize([self.input_size,
                                              self.input_size]))
            label_np = np.array(label.resize([self.input_size,
                                              self.input_size]))
            if ratio > 0:
                image_np = mask_vessel_corrupt(image_np, label_np, ratio)

            if label_np.max() == 255:
                label_np = (label_np >= 128).astype(np.uint8) * 255
            elif label_np.max() == 1:
                label_np = (label_np >= 0.5).astype(np.uint8)

            image_t = torch.from_numpy(
                image_np.astype(np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            label_t = torch.from_numpy(
                label_np.astype(np.float32).copy()
            ).unsqueeze(0) / 255.0
            images.append(image_t)
            labels.append(label_t)
        return (torch.stack(images).to(self.device),
                torch.stack(labels).to(self.device))

    def train_step(self, model, batch):
        images, labels = batch
        pred = model(images)
        return (nn.BCELoss()(pred, labels) +
                Connect_Loss()(pred, labels))

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size))
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        pred = model(prepared)[0, 0].detach().cpu().numpy()
        value_range = pred.max() - pred.min()
        pred = (pred - pred.min()) / (value_range + 1e-10)
        binary = pred >= 0.5
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        binary = Image.fromarray((binary.astype(np.uint8) * 255)).resize(
            (width, height), Image.NEAREST)
        return np.asarray(binary) > 127
