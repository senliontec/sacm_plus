"""JTFN baseline: topology-aware multi-task curvilinear segmentation.

Official reference: ``3rd/JTFN`` (no upstream license file).  JTFN uses a
5-stage CNN with gated skip connections and a recurrent mask/boundary
decoder.  It is not a SAM method; it is a from-scratch non-SAM baseline.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from torch import nn

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.jtfn import build_jtfn
from baseline_zoo.runtime import BaselineAdapter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class JTFNAdapter(BaselineAdapter):
    id = 'jtfn'
    display_name = 'JTFN'
    official_source = 'zkl20061823/JTFN (self-contained copy)'
    official_entrypoint = 'https://github.com/zkl20061823/JTFN'
    input_size = 256
    official_lr = 1e-3
    official_weight_decay = 5e-4
    train_batch_size = 2
    selection = 'dice'
    deviations = (
        'The repository has separate DRIVE/base64, roads/resbase32 and '
        'Crack500/resbase32 recipes; this port uses the DRIVE base64/steps=3 '
        'configuration for one shared 3-shot model. --jtfn-backbone and '
        '--jtfn-steps expose the alternatives.',
        'The official validation uses the dataset test split and selects by '
        'F1; this port selects on the shared 6-image validation Dice.',
        'The official 1000-epoch DRIVE recipe is kept as the default; users '
        'can override --epochs for shorter runs. The per-iteration StepLR '
        '(500, gamma 0.9) is approximated per epoch from the 18-image '
        'iterations-per-epoch.',
        'The official arbitrary-angle/color augmentation is simplified to '
        'flips, 90-degree rotations, +/-20-degree rotations and '
        'brightness/contrast jitter before the 256 crop.',
        'Official DataLoader workers and global RNG are not preserved: this '
        'port uses workers=0 and the shared per-epoch RNG; augmentation is '
        'also simplified (see above).',
        'Boundary targets are generated from the binary GT masks with the '
        'official 3x3 max-pool difference; the upstream boundary PNGs are '
        'not required.',
        'Inference pads the original image to a multiple of 32 for shape '
        'safety and crops the prediction back; the official test runs '
        'full-resolution images without padding. Images whose longest side '
        'exceeds --jtfn-eval-max-side (default 1024) are downscaled before '
        'the forward pass and the prediction is upsampled to GT resolution; '
        'this memory adaptation can change high-resolution metrics.',
        'No official JTFN checkpoints are present; training starts from '
        'scratch.',
        'No test-time augmentation.',
    )

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=256)
        parser.add_argument(
            '--jtfn-backbone', default='base64',
            choices=['base64', 'base32', 'resbase32', 'resbase64'])
        parser.add_argument('--jtfn-steps', type=int, default=3)
        parser.add_argument(
            '--jtfn-eval-max-side', type=int, default=1024,
            help='downscale eval images whose longest side exceeds this '
                 'value before the forward pass (memory safety)')

    def build_model(self, args) -> nn.Module:
        self.input_size = int(args.input_size)
        return build_jtfn(
            backbone=args.jtfn_backbone,
            steps=int(args.jtfn_steps),
        )

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=1e-8)

    def make_scheduler(self, optimizer, args):
        limit = int(getattr(args, 'limit_train', 0) or 18)
        steps_per_epoch = max(1, math.ceil(limit / max(args.batch_size, 1)))
        step_size = max(1, round(500 / steps_per_epoch))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.9)

    @staticmethod
    def _boundary(mask: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        boundary = (tensor - F.max_pool2d(tensor, 3, 1, 1)).abs()
        return (boundary[0, 0].numpy() >= 0.5).astype(np.float32)

    @staticmethod
    def _crop(image: np.ndarray, mask: np.ndarray, boundary: np.ndarray,
              size: int, rng: random.Random):
        h, w = image.shape[:2]
        if h < size or w < size:
            image = np.asarray(Image.fromarray(image).resize(
                (size, size), Image.BICUBIC))
            mask = np.asarray(Image.fromarray(mask).resize(
                (size, size), Image.NEAREST))
            boundary = np.asarray(Image.fromarray(boundary).resize(
                (size, size), Image.NEAREST))
            return image, mask, boundary
        top = rng.randrange(0, h - size + 1)
        left = rng.randrange(0, w - size + 1)
        return (image[top:top + size, left:left + size],
                mask[top:top + size, left:left + size],
                boundary[top:top + size, left:left + size])

    def _preprocess(self, sample, rng: random.Random, train: bool):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.float32)
        boundary = self._boundary(mask)

        if train:
            if rng.random() < 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
                boundary = np.flip(boundary, axis=1)
            if rng.random() < 0.5:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
                boundary = np.flip(boundary, axis=0)
            if rng.random() < 0.5:
                k = rng.randrange(4)
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
                boundary = np.rot90(boundary, k)
            elif rng.random() < 0.3:
                angle = rng.uniform(-20, 20)
                image = nd_rotate(image, angle, order=0, reshape=False)
                mask = nd_rotate(mask, angle, order=0, reshape=False)
                boundary = nd_rotate(
                    boundary, angle, order=0, reshape=False)
            if rng.random() < 0.5:
                image = np.clip(
                    image.astype(np.float32) * rng.uniform(0.8, 1.4),
                    0, 255).astype(np.uint8)
            image, mask, boundary = self._crop(
                image, mask, boundary, self.input_size, rng)

        image = torch.from_numpy(
            image.astype(np.float32).copy()).permute(2, 0, 1) / 255.0
        mean = torch.from_numpy(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.from_numpy(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std
        mask_t = torch.from_numpy(mask.copy()).unsqueeze(0)
        boundary_t = torch.from_numpy(boundary.copy()).unsqueeze(0)
        return image, mask_t, boundary_t

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks, boundaries = [], [], []
        for sample in samples:
            image, mask, boundary = self._preprocess(
                sample, rng, train=True)
            images.append(image)
            masks.append(mask)
            boundaries.append(boundary)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device),
                torch.stack(boundaries).to(self.device))

    def train_step(self, model, batch):
        images, masks, boundaries = batch
        outputs = model({'img': images})
        losses = model.compute_objective(
            outputs, {'anno_mask': masks, 'anno_boundary': boundaries})
        return losses['total_loss']

    def prepare_eval_sample(self, sample, device):
        image, _, _ = self._preprocess(
            sample, random.Random(0), train=False)
        height, width = image.shape[-2:]
        max_side = max(int(getattr(self.args, 'jtfn_eval_max_side', 1024)), 32)
        model_h, model_w = height, width
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            model_h = max(32, int(round(height * scale)))
            model_w = max(32, int(round(width * scale)))
            image = F.interpolate(
                image.unsqueeze(0), size=(model_h, model_w),
                mode='bilinear', align_corners=False)[0]
        pad_h = (-model_h) % 32
        pad_w = (-model_w) % 32
        if pad_h or pad_w:
            image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h))
        else:
            image = image.unsqueeze(0)
        return {'image': image.to(device),
                'original_size': (height, width),
                'model_size': (model_h, model_w)}

    def predict_sample(self, model, prepared, sample):
        output = model({'img': prepared['image']})['output']
        height, width = prepared['original_size']
        model_h, model_w = prepared['model_size']
        output = output[..., :model_h, :model_w]
        if (model_h, model_w) != (height, width):
            output = F.interpolate(
                output, size=(height, width), mode='bilinear',
                align_corners=False)
        return output[0, 0].detach().cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'jtfn_backbone': str(getattr(args, 'jtfn_backbone', 'base64')),
            'jtfn_steps': int(getattr(args, 'jtfn_steps', 3)),
            'jtfn_eval_max_side': int(
                getattr(args, 'jtfn_eval_max_side', 1024)),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'jtfn_backbone', 'jtfn_steps',
                    'jtfn_eval_max_side'):
            if key in config:
                setattr(args, key, config[key])
        self.input_size = int(getattr(args, 'input_size', 256))
