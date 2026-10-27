"""MedCAGD baseline: context-aware gated decoder for thin binary structures.

Official reference: ``3rd/MedCAGD`` (ICML 2026, MIT).  The upstream model is
trained on 256x256 DRIVE patches with deep supervision and edge supervision.
This adapter keeps the architecture and objective but trains it on the shared
18-image 3-shot split and evaluates full images with overlapping 256 patches,
because the protocol test sets are not pre-patched.

Deviations from the upstream recipe are listed in ``deviations`` and in the
model's ``UPSTREAM.md``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.medcagd import build_medcagd
from baseline_zoo.models.medcagd.losses import MedCAGDLoss
from baseline_zoo.runtime import BaselineAdapter


class MedCAGDAdapter(BaselineAdapter):
    id = 'medcagd'
    display_name = 'MedCAGD'
    official_source = 'saadwazir/MedCAGD (self-contained copy)'
    official_entrypoint = 'https://github.com/saadwazir/MedCAGD'
    input_size = 256
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 8
    selection = 'dice'
    deviations = (
        'Upstream trains on pre-patched DRIVE 256x256 images for 8 epochs '
        'per xrun iteration (up to 16 iterations). This port uses the shared '
        '18-image 3-shot split and the project 800-epoch budget; checkpoint '
        'selection is validation Dice.',
        'The protocol test images are full-resolution. This port tiles each '
        'image with overlapping 256 patches (default stride 128) and averages '
        'probabilities before thresholding, instead of using the upstream '
        'pre-patched DRIVE test directory. The upstream patch/unpatch scripts '
        'and CuPy evaluation path are not used.',
        'The upstream Albumentations/CutMix recipe is replaced by seeded '
        'random crops, foreground-biased crops, flips and 90-degree rotations '
        'to avoid a new dependency and to keep augmentation identical across '
        'the 3-shot protocol.',
        'The official BCEWithLogits + normalized deep supervision + edge BCE '
        'objective is preserved; the encoder is timm PVTv2-B2 with ImageNet '
        'pretraining during training and random initialization followed by '
        'checkpoint loading during prediction.',
        'The official model is binary (num_Classes=2 in the upstream config, '
        'one foreground channel here). No multiclass head is exposed.',
        'The official per-iteration fresh/resume xrun scheduler is replaced '
        'by the shared epoch budget and a cosine warm-restart scheduler; no '
        'test-time augmentation is used.',
    )

    def __init__(self):
        super().__init__()
        self._criterion = MedCAGDLoss()
        self._eval_stride = 128
        self._eval_batch = 8
        self._threshold = 0.5

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=256)
        parser.add_argument(
            '--medcagd-encoder', default='pvt_v2_b2',
            help='timm encoder name used by the official MedCAGD config')
        parser.add_argument(
            '--medcagd-pretrained', type=int, choices=[0, 1], default=1,
            help='load timm ImageNet encoder weights during training')
        parser.add_argument(
            '--medcagd-eval-stride', type=int, default=128,
            help='stride of overlapping 256 patches at inference')
        parser.add_argument(
            '--medcagd-eval-batch', type=int, default=8,
            help='number of patches per inference forward')
        parser.add_argument(
            '--medcagd-edge-weight', type=float, default=1.0,
            help='weight of the edge BCE term added to the mask objective')
        parser.add_argument(
            '--medcagd-threshold', type=float, default=0.5,
            help='probability threshold applied to the stitched prediction')

    def build_model(self, args) -> nn.Module:
        self.input_size = int(getattr(args, 'input_size', 256))
        self._eval_stride = max(1, int(getattr(args, 'medcagd_eval_stride', 128)))
        self._eval_batch = max(1, int(getattr(args, 'medcagd_eval_batch', 8)))
        self._threshold = float(getattr(args, 'medcagd_threshold', 0.5))
        self._criterion = MedCAGDLoss(
            edge_weight=float(getattr(args, 'medcagd_edge_weight', 1.0)))
        training = getattr(args, 'stage', 'train') in ('train', 'all')
        return build_medcagd(
            encoder_name=str(getattr(args, 'medcagd_encoder', 'pvt_v2_b2')),
            input_size=self.input_size,
            num_classes=1,
            pretrained=bool(int(getattr(args, 'medcagd_pretrained', 1)))
            and training,
        )

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('MedCAGD has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def make_scheduler(self, optimizer, args):
        # Official xrun uses CosineAnnealingWarmRestarts; the shared runtime
        # steps once per epoch, so T_0 is in epochs.
        t0 = max(1, min(int(getattr(args, 'epochs', 1)), 20))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=2, eta_min=1e-6)

    @staticmethod
    def _crop_origin(mask: np.ndarray, patch: int,
                     rng: random.Random) -> tuple[int, int]:
        height, width = mask.shape
        if height <= patch and width <= patch:
            return 0, 0
        # Half of the crops are centred on a foreground pixel.  This is a
        # thin-structure adaptation to the 3-shot split; it does not change
        # the model or the loss.
        if rng.random() < 0.5:
            ys, xs = np.nonzero(mask)
            if len(ys):
                cy = int(ys[rng.randrange(len(ys))])
                cx = int(xs[rng.randrange(len(xs))])
                top = min(max(0, cy - patch // 2), max(0, height - patch))
                left = min(max(0, cx - patch // 2), max(0, width - patch))
                return top, left
        top = rng.randint(0, max(0, height - patch))
        left = rng.randint(0, max(0, width - patch))
        return top, left

    def _train_pair(self, sample, rng: random.Random):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.uint8) * 255
        patch = self.input_size
        height, width = mask.shape
        if height < patch or width < patch:
            pad_h = max(0, patch - height)
            pad_w = max(0, patch - width)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)),
                           mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            height, width = mask.shape
        top, left = self._crop_origin(mask, patch, rng)
        image = image[top:top + patch, left:left + patch]
        mask = mask[top:top + patch, left:left + patch]

        if rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if rng.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        k = rng.randrange(4)
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Mild photometric jitter; no new dependency (Albumentations is not
        # part of the baseline environment).
        if rng.random() < 0.5:
            image = np.clip(
                image.astype(np.float32) * rng.uniform(0.85, 1.15)
                + rng.uniform(-12.0, 12.0), 0, 255).astype(np.uint8)

        image_t = torch.from_numpy(
            image.astype(np.float32).copy()).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(
            (mask > 127).astype(np.float32).copy()).unsqueeze(0)
        return image_t, mask_t

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._train_pair(sample, rng)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        main, auxiliaries, edges = model(images)
        return self._criterion(main, auxiliaries, edges, masks)

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1).unsqueeze(0) / 255.0
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return tensor.to(device), (int(width), int(height))

    def _tile_positions(self, total: int, patch: int, stride: int) -> list[int]:
        if total <= patch:
            return [0]
        positions = list(range(0, total - patch + 1, max(1, stride)))
        last = total - patch
        if positions[-1] != last:
            positions.append(last)
        return positions

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, (width, height) = prepared
        tensor = image
        orig_h, orig_w = tensor.shape[-2:]
        patch = self.input_size
        # If the image is smaller than the patch, pad it rather than
        # rescaling, so native thin-structure pixels are preserved.
        if orig_h < patch or orig_w < patch:
            pad_h = max(0, patch - orig_h)
            pad_w = max(0, patch - orig_w)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        img_h, img_w = tensor.shape[-2:]
        ys = self._tile_positions(img_h, patch, self._eval_stride)
        xs = self._tile_positions(img_w, patch, self._eval_stride)
        prob = torch.zeros((img_h, img_w), dtype=torch.float32)
        count = torch.zeros((img_h, img_w), dtype=torch.float32)
        patches = []
        coords = []
        for top in ys:
            for left in xs:
                patches.append(tensor[:, :, top:top + patch, left:left + patch])
                coords.append((top, left))
        for start in range(0, len(patches), self._eval_batch):
            batch = torch.cat(patches[start:start + self._eval_batch], dim=0)
            main, _, _ = model(batch)
            probs = torch.sigmoid(main[:, 0]).cpu()
            for (top, left), p in zip(coords[start:start + self._eval_batch], probs):
                prob[top:top + patch, left:left + patch] += p
                count[top:top + patch, left:left + patch] += 1.0
        prob = prob / count.clamp_min(1.0)
        prob = prob[:orig_h, :orig_w]
        prob = F.interpolate(
            prob[None, None], size=(height, width),
            mode='bilinear', align_corners=False)[0, 0]
        return prob.cpu().numpy() > self._threshold

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'encoder': str(getattr(args, 'medcagd_encoder', 'pvt_v2_b2')),
            'num_classes': 1,
            'eval_stride': int(getattr(args, 'medcagd_eval_stride', 128)),
            'eval_batch': int(getattr(args, 'medcagd_eval_batch', 8)),
            'threshold': float(getattr(args, 'medcagd_threshold', 0.5)),
            'edge_weight': float(getattr(args, 'medcagd_edge_weight', 1.0)),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'encoder' in config:
            args.medcagd_encoder = str(config['encoder'])
        if 'eval_stride' in config:
            args.medcagd_eval_stride = int(config['eval_stride'])
        if 'eval_batch' in config:
            args.medcagd_eval_batch = int(config['eval_batch'])
        if 'threshold' in config:
            args.medcagd_threshold = float(config['threshold'])
        if 'edge_weight' in config:
            args.medcagd_edge_weight = float(config['edge_weight'])
        self._eval_stride = int(getattr(args, 'medcagd_eval_stride', 128))
        self._eval_batch = int(getattr(args, 'medcagd_eval_batch', 8))
        self._threshold = float(getattr(args, 'medcagd_threshold', 0.5))
