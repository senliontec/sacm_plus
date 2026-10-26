"""CS2Net baseline with the official 2D CS-Net training/inference logic.

Official reference: https://github.com/ineedzx/CSNet (vendored snapshot in
``baselines/cs2net``).  The model is copied into
``baseline_zoo.models.cs2net`` so the runtime survives deleting ``baselines``.

Protocol-only deviations are listed in ``deviations``; every model-side
detail (geometry, augmentation, loss, optimizer, LR schedule, Otsu output)
follows the upstream code.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.cs2net import CSNet
from baseline_zoo.runtime import BaselineAdapter

DICE_EPS = 1e-4


def _center_square(image: Image.Image, mask: Image.Image, size: int):
    """Upstream ``Data.rescale``: centre-crop the min-side square, resize."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    box = (left, top, left + side, top + side)
    return (image.crop(box).resize((size, size)),
            mask.crop(box).resize((size, size)))


def _restore_center_square(pred: np.ndarray, width: int, height: int):
    """Place a 512 prediction back into the original centre square."""
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    resized = Image.fromarray((pred.astype(np.uint8) * 255)).resize(
        (side, side), Image.NEAREST)
    out = np.zeros((height, width), dtype=bool)
    out[top:top + side, left:left + side] = np.asarray(resized) > 127
    return out


def _official_dice_loss(pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Upstream ``dice_coeff_loss`` (per-image Dice, eps 1e-4)."""
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + DICE_EPS
    dice = (2 * inter + DICE_EPS) / union
    return (1 - dice).mean()


class CS2NetAdapter(BaselineAdapter):
    id = 'cs2net'
    display_name = 'CS2Net'
    official_source = 'ineedzx/CSNet (self-contained copy)'
    official_entrypoint = 'https://github.com/ineedzx/CSNet'
    input_size = 512
    official_lr = 1e-4
    official_weight_decay = 5e-4
    train_batch_size = 8
    selection = 'dice'
    deviations = (
        'Data protocol: the 18/6-shot split replaces the original DRIVE '
        'training/test folders; all 20 mixed-modality domains are converted '
        'to RGB because the official loader assumes 3-channel fundus images.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream evaluates the test split and its best-checkpoint branch is '
        'unreachable, so it never writes a best model; using test here would '
        'violate the protocol.',
        'The official predict.py depends on an undefined center_crop and is '
        'not runnable; inference reuses the data-loader center-square crop '
        'and places the Otsu mask back into that square at GT resolution.',
        'Official whole-model pickle checkpoints are replaced by the unified '
        'state_dict schema so every baseline shares one evaluator.',
        'Per-epoch validation runs once per epoch (upstream also tests every '
        'epoch but returns after the first image).',
        'Official train.py uses nn.DataParallel on two GPUs, DataLoader workers=2 '
        'and an unseeded global RNG; this port runs one GPU, workers=0 and a '
        'seeded runtime RNG. Training trajectories are not bitwise identical.',
    )

    def build_model(self, args) -> nn.Module:
        return CSNet(classes=1, channels=3)

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        max_epochs = max(args.epochs, 1)

        def lr_lambda(epoch):
            # Upstream calls adjust_lr(iter=epoch) *after* each 0-based epoch
            # and its first call (iter=0) is a no-op.  LambdaLR applies
            # lambda(0) at construction, so shift by one to avoid decaying
            # one epoch early.
            return (1.0 - max(epoch - 1, 0) / max_epochs) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

    @staticmethod
    def _enhance(image: Image.Image, rng: random.Random) -> Image.Image:
        value = rng.uniform(-2, 2)
        choice = rng.randint(1, 4)
        enhancer = {
            1: ImageEnhance.Brightness,
            2: ImageEnhance.Color,
            3: ImageEnhance.Contrast,
            4: ImageEnhance.Sharpness,
        }[choice]
        return enhancer(image).enhance(value)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path)
            mask = Image.open(sample.mask_path).convert('L')
            image, mask = _center_square(image, mask, self.input_size)

            angle = rng.randint(-40, 40)
            image = image.rotate(angle)
            mask = mask.rotate(angle)
            if rng.random() > 0.5:
                image = self._enhance(image, rng)

            # Upstream RandomCrop(512,512) after the 512 resize: a no-op kept
            # for exact sequencing of the random stream.
            side = self.input_size
            width, height = image.size
            left = rng.randint(0, width - side)
            top = rng.randint(0, height - side)
            image = image.crop((left, top, left + side, top + side))
            mask = mask.crop((left, top, left + side, top + side))

            if rng.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(
                np.asarray(mask, dtype=np.float32).copy()
            ).unsqueeze(0) / 255.0
            images.append(image_t)
            masks.append(mask_t)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        pred = model(images)
        return (nn.MSELoss()(pred, masks) +
                _official_dice_loss(pred, masks))

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path)
        mask = Image.open(sample.mask_path).convert('L')
        image, _ = _center_square(image, mask, self.input_size)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        pred = model(prepared)[0, 0].detach().cpu().numpy()
        arr = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        if arr.min() == arr.max():
            threshold = 0.0  # cv2 THRESH_OTSU returns 0 for constant images
        else:
            threshold = float(threshold_otsu(arr))
        binary = arr > threshold
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return _restore_center_square(binary, width, height)
