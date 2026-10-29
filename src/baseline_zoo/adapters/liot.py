"""LIOT + IterNet adapter (IEEE TIP 2022).

The official pipeline converts a retinal/crack image to a 4-channel local
intensity-order representation and feeds it to IterNet.  This port keeps
that representation and the 32-channel IterNet, trains on 128x128 patches
with the release's augmentation family, and uses whole-image inference for
the shared validation/test protocol.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import RandomSampler
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.liot import (
    Iternet,
    local_intensity_order,
    polarity_for_dataset,
)
from baseline_zoo.runtime import BaselineAdapter


class LIOTAdapter(BaselineAdapter):
    id = 'liot'
    display_name = 'LIOT-IterNet'
    family = 'cnn'
    official_source = 'LIOT + IterNet (IEEE TIP 2022)'
    official_entrypoint = 'https://github.com/TY-Shi/LIOT'
    input_size = 128
    official_lr = 1e-3
    official_weight_decay = 0.0
    train_batch_size = 8
    accumulation_steps = 4
    default_grad_clip = 0.0
    selection = 'dice'
    deviations = (
        'The release README describes 1000 epochs and the included '
        'configuration.txt uses 170; this port uses the shared 800-epoch '
        'budget with validation-Dice selection.',
        'Official batch 32 is replaced by batch 8 with 4-step accumulation '
        'on the 18-image split, preserving an effective batch of 32.  The '
        'release repeats each training image 30 times per epoch; this port '
        'keeps that repeat (--liot-repeat 30) so each epoch still exposes '
        '540 random crops.',
        'The optional post-150-epoch topological loss is not used: it '
        'imports a hard-coded /home/sty/... PersistencePython module and is '
        'not part of the LIOT representation contribution.  The official '
        'cross-entropy objective is kept.',
        'LIOT assumes a dark curvilinear structure.  The retinal datasets '
        '(DRIVE, CHASEDB1, FIVES, HRF, LES-AV, OCTA-500, STARE) keep the '
        'official dark polarity; bright-foreground protocol datasets (CORN, '
        'DeepGlobe, ROAD, TopoMortar, WIRE) are inverted before LIOT at '
        'both training and inference.  The bright set can be overridden '
        'with --liot-polarity; unknown datasets default to the paper '
        'polarity.',
        'The release predicts overlapping 128x128 patches with stride 8 and '
        'sweeps thresholds for F1; this port performs a whole-image forward '
        'pass, thresholds at 0.5 for the shared boolean protocol, and '
        'returns the exact GT resolution.',
        'The official HDF5/DRIVE data pipeline and torchvision augmentation '
        'are replaced by original-resolution protocol loading and a '
        'torchvision reimplementation of the release rotation/shear/flip/'
        'shift/zoom parameters.',
        'No official LIOT weights are loaded; training starts from random '
        'initialization.',
    )

    def __init__(self):
        super().__init__()
        self._liot_cache: dict[tuple[str, bool], np.ndarray] = {}
        self._polarity_mode = 'auto'

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=128)
        parser.add_argument(
            '--liot-polarity', choices=['auto', 'dark', 'bright'],
            default='auto',
            help='LIOT foreground polarity; auto uses the protocol table')
        parser.add_argument(
            '--liot-no-augment', action='store_true',
            help='disable the release augmentation (smoke/tests only)')
        parser.add_argument(
            '--liot-repeat', type=int, default=30,
            help='release repeat factor per training image (default 30)')
        parser.add_argument(
            '--liot-random-init', action='store_true',
            help='compatibility flag; LIOT always starts from random')

    def begin_prediction(self) -> None:
        super().begin_prediction()
        # Validation keeps the 18+6 arrays warm during a training run, but a
        # prediction pass over many test images must not retain them.
        self._liot_cache.clear()

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 128))
        if self.input_size != 128:
            raise SystemExit('LIOT uses the official 128x128 training patch')
        self._polarity_mode = str(getattr(args, 'liot_polarity', 'auto'))
        return Iternet(n_channels=4, n_classes=2)

    def load_pretrained(self, model, args) -> None:
        print('LIOT: random initialization (no official weights loaded)')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        print(f'LIOT-IterNet parameters: {total:,} total, {total:,} trainable')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
        )

    def make_scheduler(self, optimizer, args):
        # The release uses a constant Adam learning rate and has no scheduler.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0)

    def make_train_sampler(self, samples, args):
        repeat = max(1, int(getattr(args, 'liot_repeat', 30)))
        if repeat == 1 or not samples:
            return None
        generator = torch.Generator()
        generator.manual_seed(int(getattr(args, 'seed', 0)))
        return RandomSampler(
            samples, replacement=True,
            num_samples=len(samples) * repeat, generator=generator)

    # ------------------------------------------------------------------
    # LIOT input
    # ------------------------------------------------------------------
    def _invert(self, sample: Sample) -> bool:
        return polarity_for_dataset(sample.dataset, self._polarity_mode)

    def _liot_array(self, sample: Sample) -> np.ndarray:
        invert = self._invert(sample)
        key = (str(sample.image_path), invert)
        cached = self._liot_cache.get(key)
        if cached is not None:
            return cached
        rgb = np.asarray(load_rgb(sample.image_path))
        array = local_intensity_order(rgb, invert=invert)
        # Keep the cache bounded so a long prediction pass does not retain
        # every full-resolution LIOT image.
        if len(self._liot_cache) >= 32:
            self._liot_cache.pop(next(iter(self._liot_cache)))
        self._liot_cache[key] = array
        return array

    @staticmethod
    def _mask_tensor(sample: Sample) -> torch.Tensor:
        mask = load_mask(sample.mask_path).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

    def _resize_if_needed(self, image: torch.Tensor,
                          mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        tile = self.input_size
        if height >= tile and width >= tile:
            return image, mask
        scale = max(tile / height, tile / width)
        new_h = max(tile, int(round(height * scale)))
        new_w = max(tile, int(round(width * scale)))
        image = F.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
            align_corners=False)[0]
        mask = F.interpolate(
            mask.unsqueeze(0), size=(new_h, new_w), mode='nearest')[0]
        return image, mask

    def _augment(self, image: torch.Tensor, mask: torch.Tensor,
                 rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(getattr(self.args, 'liot_no_augment', False)):
            return image, mask
        height, width = image.shape[-2:]
        if rng.random() < 0.5:
            angle = rng.uniform(-180.0, 180.0)
            image = TF.rotate(
                image, angle, interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0)
            mask = TF.rotate(
                mask, angle, interpolation=TF.InterpolationMode.NEAREST,
                fill=0.0)
        if rng.random() < 0.05:
            shear = rng.uniform(-5.0, 5.0)
            image = TF.affine(
                image, 0.0, (0.0, 0.0), 1.0, shear,
                interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(
                mask, 0.0, (0.0, 0.0), 1.0, shear,
                interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        if rng.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if rng.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if rng.random() < 0.05:
            tx = rng.uniform(-0.1, 0.1) * width
            ty = rng.uniform(-0.1, 0.1) * height
            image = TF.affine(
                image, 0.0, (tx, ty), 1.0, 0.0,
                interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(
                mask, 0.0, (tx, ty), 1.0, 0.0,
                interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        if rng.random() < 0.05:
            scale = rng.uniform(0.8, 1.2)
            image = TF.affine(
                image, 0.0, (0.0, 0.0), scale, 0.0,
                interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(
                mask, 0.0, (0.0, 0.0), scale, 0.0,
                interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        return image, (mask > 0.5).float()

    def _train_pair(self, sample: Sample, rng: random.Random):
        array = self._liot_array(sample)
        image = torch.from_numpy(
            np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
        mask = self._mask_tensor(sample)
        image, mask = self._resize_if_needed(image, mask)
        top = rng.randrange(0, image.shape[-2] - self.input_size + 1)
        left = rng.randrange(0, image.shape[-1] - self.input_size + 1)
        image = image[:, top:top + self.input_size,
                      left:left + self.input_size]
        mask = mask[:, top:top + self.input_size,
                    left:left + self.input_size]
        image, mask = self._augment(image, mask, rng)
        return image.contiguous(), mask[0].long()

    # ------------------------------------------------------------------
    # Data/loss
    # ------------------------------------------------------------------
    def prepare_train_batch(self, samples: Sequence[Sample], rng):
        rng = rng or random.Random(0)
        images, masks = [], []
        for sample in samples:
            image, mask = self._train_pair(sample, rng)
            images.append(image)
            masks.append(mask)
        return {
            'image': torch.stack(images).to(self.device),
            'gt': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        logits = model(batch['image'])
        return F.cross_entropy(logits, batch['gt'])

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        array = self._liot_array(sample)
        image = torch.from_numpy(
            np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.to(device),
            'size': (int(height), int(width)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        model.eval()
        image = prepared['image']
        height, width = image.shape[-2:]
        pad_h = max(0, self.input_size - height)
        pad_w = max(0, self.input_size - width)
        if pad_h or pad_w:
            image = F.pad(
                image.unsqueeze(0), (0, pad_w, 0, pad_h),
                mode='constant', value=0.0)
        else:
            image = image.unsqueeze(0)
        logits = model(image)
        probability = torch.softmax(logits, dim=1)[0, 1]
        probability = probability[:height, :width]
        return (probability > 0.5).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'polarity': str(getattr(args, 'liot_polarity', 'auto')),
            'augment': not bool(getattr(args, 'liot_no_augment', False)),
            'repeat': int(getattr(args, 'liot_repeat', 30)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'polarity' in config:
            args.liot_polarity = str(config['polarity'])
            self._polarity_mode = args.liot_polarity
        if 'repeat' in config:
            args.liot_repeat = int(config['repeat'])
        if config.get('augment') is False:
            args.liot_no_augment = True
