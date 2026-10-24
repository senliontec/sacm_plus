"""BCUNet baseline with the official ConvNeXt+U-Net training logic.

Official reference: vendored ``baselines/bcunet`` (upstream URL is not
recorded there).  The model is copied into ``baseline_zoo.models.bcunet``
and the undefined ``OutConv.upconv2x2`` is repaired as a 1x1 logits
projection; everything else follows the official training script.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from baseline_zoo.models.bcunet import BCUNet
from baseline_zoo.runtime import BaselineAdapter


class BCUNetAdapter(BaselineAdapter):
    id = 'bcunet'
    display_name = 'BCUNet'
    official_source = 'BCU-Net (self-contained copy)'
    official_entrypoint = 'upstream code/main.py (vendored snapshot)'
    input_size = 512
    official_epochs = 50
    official_lr = 1.5e-4
    official_weight_decay = 0.0
    train_batch_size = 1
    accumulation_steps = 2
    selection = 'dice'
    deviations = (
        'Upstream batch size 2 does not fit an 11 GiB card at 512x512; the '
        'adapter uses batch 1 with 2-step gradient accumulation, preserving '
        'the effective batch. On >=24 GiB pass --batch-size 2 '
        '--accum-steps 1.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream tracks IOU on the test list, which would leak test data.',
        'Upstream preprocessing is BGR [0,255] with no normalization '
        '(cv2.imread + INTER_AREA); preserved exactly.',
        'Upstream OutConv referenced an undefined upconv2x2; repaired as a '
        '1x1 logits projection at full resolution. Upstream main.py also '
        'imports a missing warmlearnrate.py and cannot run; this adapter '
        'supplies the official loss/optimizer loop instead.',
        'The official 50 epochs, Adam 1.5e-4, no scheduler and no '
        'augmentation are preserved.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--no-pretrained', dest='pretrained', action='store_false',
            help='do not load ImageNet-22k ConvNeXt weights')

    def build_model(self, args) -> nn.Module:
        return BCUNet(n_channels=3, n_classes=2, bilinear=False,
                      pretrained=args.pretrained, in_22k=True)

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def _load_image(self, sample):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise SystemExit(f'cv2 could not read {sample.image_path}')
        return cv2.resize(image, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_AREA)

    def _load_mask(self, sample):
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f'cv2 could not read {sample.mask_path}')
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_AREA)
        mask = mask.astype(np.int64)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def prepare_train_batch(self, samples, rng: random.Random):
        images = []
        masks = []
        for sample in samples:
            image = self._load_image(sample)
            images.append(torch.from_numpy(
                np.transpose(image, [2, 0, 1]).copy()).float())
            masks.append(torch.from_numpy(self._load_mask(sample)))
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        out1, out2, fused = model(images)
        criterion = nn.CrossEntropyLoss()
        return (criterion(fused, masks) + criterion(out1, masks) +
                criterion(out2, masks))

    def prepare_eval_sample(self, sample, device):
        image = self._load_image(sample)
        tensor = torch.from_numpy(
            np.transpose(image, [2, 0, 1]).copy()).float()
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        _, _, fused = model(prepared)
        pred = fused.argmax(dim=1)[0].detach().cpu().numpy()
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        pred = Image.fromarray((pred.astype(np.uint8) * 255)).resize(
            (width, height), Image.NEAREST)
        return np.asarray(pred) > 127
