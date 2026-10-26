"""U-Net baseline: vendored milesial/PyTorch-UNet architecture.

There is no canonical U-Net 3-shot training script.  The architecture and
loss (BCEWithLogits + Dice) come from the vendored upstream repository; the
few-shot training recipe is stated explicitly in ``deviations`` below.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.unet import UNet
from baseline_zoo.runtime import BaselineAdapter


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official milesial Dice loss (global over batch, eps=1e-6).

    Matches ``dice_coeff(..., reduce_batch_first=True)``: the empty-set
    branch reuses ``inter`` so two empty masks give Dice 1, and the
    epsilon is added to numerator and denominator.
    """
    prob = torch.sigmoid(logits)
    # Upstream calls dice_coeff(..., reduce_batch_first=True): one Dice
    # value over the whole batch/spatial extent, not a per-image mean.
    inter = 2.0 * (prob * target).sum()
    sets_sum = prob.sum() + target.sum()
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + 1e-6) / (sets_sum + 1e-6)
    return 1.0 - dice


class UNetAdapter(BaselineAdapter):
    id = 'unet'
    display_name = 'U-Net'
    official_source = 'milesial/PyTorch-UNet (self-contained copy)'
    official_entrypoint = 'https://github.com/milesial/Pytorch-UNet'
    input_size = 512
    official_epochs = 200
    official_lr = 1e-3
    official_weight_decay = 1e-4
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'No canonical U-Net 3-shot trainer exists; the vendored milesial '
        'architecture and BCE+Dice loss are used.',
        'Fixed 512x512 input instead of upstream img_scale=0.5 dynamic '
        'resize, so mixed-resolution domains can share batches.',
        'AdamW lr 1e-3/wd 1e-4/batch 4 + cosine over 200 epochs instead of '
        'upstream RMSprop lr 1e-5/wd 1e-8/batch 1 + ReduceLROnPlateau over '
        '5 epochs; the 3-shot budget and optimizer are the protocol choice '
        'used for every baseline.',
        'A one-class (BCE + foreground Dice) head is used; upstream '
        'train.py defaults to two classes (CE + multiclass Dice).',
        'Grayscale protocol images are converted to RGB because the '
        'vendored model expects three input channels; upstream keeps 2-D '
        'images single-channel and would assert.',
        'Random flips and 90-degree rotations are used for 3-shot training; '
        'upstream performs no augmentation.',
        'Checkpoint selected on the 6-image val split by Dice; upstream '
        'saves every epoch and evaluates a random 10% split.',
        'Official DataLoader workers=os.cpu_count(), shuffle=True and a global '
        'RNG; this port uses workers=0 and a per-epoch seeded RNG, so data '
        'order and floating-point accumulation are not bitwise identical.',
    )

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)

    def build_model(self, args) -> nn.Module:
        self.input_size = args.input_size
        return UNet(n_channels=3, n_classes=1)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1))

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BICUBIC)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.NEAREST)
            image = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(
                np.asarray(mask, dtype=np.float32).copy()
            ).unsqueeze(0)
            mask = (mask > 127.0).float()
            if rng.random() < 0.5:
                image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
            if rng.random() < 0.5:
                image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
            k = rng.randrange(4)
            if k:
                image = torch.rot90(image, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = model(images)
        return (nn.BCEWithLogitsLoss()(logits, masks) +
                dice_loss(logits, masks))

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        logits = model(prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        # Upstream predicts logits first, then upsamples, then applies
        # sigmoid; interpolating probabilities changes boundary decisions.
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
