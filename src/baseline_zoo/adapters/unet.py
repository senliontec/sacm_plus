"""U-Net baseline: vendored milesial/PyTorch-UNet architecture.

The adapter follows the official training/inference recipe: per-image
``scale=0.5`` preprocessing (variable input size), batch size 1, RMSprop,
``ReduceLROnPlateau`` on validation Dice, no augmentation, BCE + the
official global Dice loss, and logit-space bilinear upsampling at
prediction.  The shared 3-shot protocol supplies 18 train / 6 val images
and 800 epochs; those protocol-level choices are listed in ``deviations``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

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
    # Official preprocessing is a per-image scale factor, so the model input
    # is variable-size; 0 marks that this adapter does not use a fixed size.
    input_size = 0
    scale = 0.5
    official_lr = 1e-5
    official_weight_decay = 1e-8
    train_batch_size = 1
    default_grad_clip = 1.0
    selection = 'dice'
    # ReduceLROnPlateau needs the validation metric, so the shared runtime
    # steps it after evaluate() instead of inside train_epoch().
    scheduler_after_validation = True
    deviations = (
        'There is no canonical U-Net 3-shot trainer; the vendored milesial '
        'architecture, BCE+Dice loss, RMSprop optimizer, '
        'ReduceLROnPlateau scheduler and gradient clipping norm 1.0 follow '
        'the official recipe.',
        'Official per-image scale=0.5 resize (variable input size) and batch '
        'size 1; the shared protocol supplies 18 train / 6 val images and '
        'extends the official 5-epoch default to 800 epochs.',
        'A one-class (BCE + foreground Dice) head is used; upstream train.py '
        'defaults to two classes (CE + multiclass Dice), but its 1-class '
        'branch is the official binary mode.',
        'Grayscale protocol images are converted to RGB because the vendored '
        'model expects three input channels; upstream keeps 2-D images '
        'single-channel and would assert.',
        'Checkpoint selected on the 6-image val split by Dice; upstream saves '
        'every epoch and evaluates a random 10% split.',
        'Validation Dice is computed at GT resolution by the shared '
        'evaluator; upstream evaluates at the scaled input resolution.',
        'ReduceLROnPlateau is stepped once per epoch on the shared 6-image '
        'validation split; upstream steps it every n_train // '
        '(5*batch_size) steps on a random 10% split.',
        'Official DataLoader workers=os.cpu_count(), shuffle=True and a global '
        'RNG; this port uses workers=0 and a per-epoch seeded RNG, so data '
        'order and floating-point accumulation are not bitwise identical.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--scale', type=float, default=self.scale,
            help='Official per-image downscaling factor (default 0.5)')
        parser.add_argument(
            '--momentum', type=float, default=0.999,
            help='RMSprop momentum (official 0.999)')

    def build_model(self, args) -> nn.Module:
        self.scale = float(getattr(args, 'scale', self.scale))
        return UNet(n_channels=3, n_classes=1)

    def make_optimizer(self, model, args):
        params = [p for p in model.parameters() if p.requires_grad]
        kwargs = dict(
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=float(getattr(args, 'momentum', 0.999)),
        )
        # The official trainer runs on CUDA with foreach=True; CPU tests use
        # the standard implementation.
        use_foreach = any(p.is_cuda for p in params)
        try:
            return torch.optim.RMSprop(params, foreach=use_foreach, **kwargs)
        except TypeError:  # pragma: no cover - torch without foreach
            return torch.optim.RMSprop(params, **kwargs)

    def make_scheduler(self, optimizer, args):
        mode = ('min' if getattr(args, 'selection', 'dice') == 'hd95'
                else 'max')
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=5)

    def step_scheduler(self, scheduler, metrics: dict[str, float]) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            key = (self.args.selection if self.args is not None
                   else self.selection)
            scheduler.step(metrics[key])
        else:
            # Generic epoch-based override (--scheduler cosine/poly/...).
            scheduler.step()

    def _resize_pair(self, sample):
        """Official BasicDataset.preprocess at ``scale`` (image/mask pair)."""
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert('RGB')
            width, height = image.size
            with Image.open(sample.mask_path) as mask_file:
                mask = mask_file.convert('L')
                if mask.size != image.size:
                    raise RuntimeError(
                        f'{self.id}: image/mask size mismatch for '
                        f'{sample.image_path}: {image.size} vs {mask.size}')
                new_w = int(self.scale * width)
                new_h = int(self.scale * height)
                if new_w <= 0 or new_h <= 0:
                    raise RuntimeError(
                        f'{self.id}: scale={self.scale} produces an empty '
                        f'image for {sample.image_path} '
                        f'({width}x{height})')
                image = image.resize((new_w, new_h), Image.BICUBIC)
                mask = mask.resize((new_w, new_h), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32).copy()
        if (image_np > 1).any():  # official uint8 -> [0, 1]
            image_np /= 255.0
        # Every protocol mask is {0, 255}; this is equivalent to the
        # official sorted mask_values mapping and keeps the binary head.
        mask_np = (np.asarray(mask, dtype=np.float32).copy() > 127.0)
        return image_np, mask_np.astype(np.float32)

    def prepare_train_batch(self, samples, rng: random.Random):
        del rng  # the official U-Net trainer has no augmentation
        images, masks = [], []
        for sample in samples:
            image_np, mask_np = self._resize_pair(sample)
            images.append(torch.from_numpy(image_np).permute(2, 0, 1))
            masks.append(torch.from_numpy(mask_np).unsqueeze(0))
        shapes = {tuple(tensor.shape) for tensor in images}
        if len(shapes) > 1:
            raise RuntimeError(
                f'{self.id}: official scale=0.5 preprocessing produces '
                f'variable input sizes ({sorted(shapes)}); use --batch-size 1')
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = model(images)
        return (nn.BCEWithLogitsLoss()(logits, masks) +
                dice_loss(logits, masks))

    def prepare_eval_sample(self, sample, device):
        image_np, _ = self._resize_pair(sample)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
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

    def save_config(self, args):
        return {'scale': float(self.scale)}

    def restore_config(self, args, config):
        if 'scale' in config:
            self.scale = float(config['scale'])
            if hasattr(args, 'scale'):
                args.scale = self.scale
