"""MedCAGD baseline: context-aware gated decoder for thin binary structures.

Official reference: ``3rd/MedCAGD`` (ECCV 2026, MIT).  The upstream model is
trained on 256x256 DRIVE patches with deep supervision and edge supervision.
The paper uses 256/128 patches for DRIVE/FIVES and 224 full images for its
large-object benchmarks; because this protocol is entirely thin curvilinear
structures, the default task adaptation uses the patch recipe for every
dataset.  ``--medcagd-input-policy paper`` restores the paper dataset
mapping.  It trains on the shared 18-image 3-shot split; the protocol test
sets are not pre-patched, so inference tiles images with overlapping 256
patches.

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
    # The paper uses 224x224 full images for its large-object benchmarks and
    # 256/128 overlapping patches for DRIVE and FIVES.  Every dataset in this
    # protocol is a thin curvilinear structure (vessels, cracks, neurons,
    # roads), so the task-adapted default applies the DRIVE/FIVES patch
    # recipe to all datasets.  ``--medcagd-input-policy paper`` reproduces
    # the paper's dataset mapping.
    PATCH_DATASETS = frozenset({'DRIVE', 'FIVES'})

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
        'Upstream trains on pre-patched DRIVE 256x256 images. The paper '
        'uses 256x256/128 patches for DRIVE/FIVES and 224x224 full images '
        'for its large-object benchmarks. All protocol datasets are thin '
        'curvilinear structures, so the task-adapted default applies the '
        'DRIVE/FIVES patch recipe to every dataset; '
        '--medcagd-input-policy paper reproduces the paper dataset mapping. '
        'Batch 8 and the shared 800-epoch budget replace the paper batch 16 '
        'and >300 epochs.',
        'In the default patch policy, every test image is tiled with '
        'overlapping 256 patches (default stride 128) and probabilities are '
        'averaged before thresholding. In paper mode, non-DRIVE/FIVES '
        'datasets are resized to 224x224 and their prediction is resized '
        'back to GT resolution. The upstream patch/unpatch scripts and CuPy '
        'evaluation path are not used.',
        'The upstream Albumentations/CutMix recipe is replaced by seeded '
        'random crops, foreground-biased crops, flips and 90-degree rotations '
        'to avoid a new dependency and to keep augmentation identical across '
        'the 3-shot protocol.',
        'The official BCEWithLogits + normalized deep supervision + edge BCE '
        'objective is preserved; the encoder is timm PVTv2-B2 with ImageNet '
        'pretraining during training and random initialization followed by '
        'checkpoint loading during prediction.',
        'The upstream loader feeds OpenCV BGR images scaled to [0,1] without '
        'ImageNet mean/std normalization. This port uses RGB PIL images with '
        'the same [0,1] scaling; RGB matches the ImageNet-pretrained '
        'encoder and the paper, while preserving the upstream normalization.',
        'The official model is binary (num_Classes=2 in the upstream config, '
        'one foreground channel here). No multiclass head is exposed.',
        'The paper excludes the raw-image Stage 0 path from the final model. '
        'This port defaults to use_stage0=0 (paper final); set '
        '--medcagd-use-stage0 1 to reproduce the upstream code variant.',
        'The official per-iteration fresh/resume xrun scheduler is replaced '
        'by the shared epoch budget and CosineAnnealingWarmRestarts with the '
        'upstream T_0=5, T_mult=1; no test-time augmentation is used.',
    )

    def __init__(self):
        super().__init__()
        self._criterion = MedCAGDLoss()
        self._full_size = 224
        self._input_policy = 'patches'
        self._eval_stride = 128
        self._eval_batch = 8
        self._threshold = 0.5

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=256,
                            help='patch size for the patch input policy')
        parser.add_argument(
            '--medcagd-full-size', type=int, default=224,
            help='full-image size for non-patch datasets in paper mode')
        parser.add_argument(
            '--medcagd-input-policy', choices=['patches', 'paper'],
            default='patches',
            help='patches: 256/128 patch recipe for every dataset (task '
                 'adaptation for thin structures); paper: 224 full images '
                 'for non-DRIVE/FIVES datasets')
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
        parser.add_argument(
            '--medcagd-use-stage0', type=int, choices=[0, 1], default=0,
            help='include the upstream raw-image Stage 0 path; the paper '
                 'excludes it from the final model')

    def build_model(self, args) -> nn.Module:
        self.input_size = int(getattr(args, 'input_size', 256))
        self._full_size = int(getattr(args, 'medcagd_full_size', 224))
        self._input_policy = str(
            getattr(args, 'medcagd_input_policy', 'patches'))
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
            use_stage0=bool(
                int(getattr(args, 'medcagd_use_stage0', 0))),
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
        # Official xrun uses CosineAnnealingWarmRestarts(T_0=5, T_mult=1)
        # for its >10-epoch recipe; the shared runtime steps once per epoch.
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6)

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

    def _uses_patches(self, sample) -> bool:
        if self._input_policy == 'patches':
            return True
        return sample.dataset in self.PATCH_DATASETS

    def _train_pair(self, sample, rng: random.Random):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.uint8) * 255
        if self._uses_patches(sample):
            # Paper: DRIVE/FIVES use 256x256 patches (stride 128), while all
            # other datasets use 224x224 full images.
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
        else:
            size = self._full_size
            image = np.asarray(
                Image.fromarray(image).resize((size, size), Image.BILINEAR))
            mask = np.asarray(
                Image.fromarray(mask).resize((size, size), Image.NEAREST))

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
        # The paper uses 256 patches for DRIVE/FIVES and 224 full images for
        # the other datasets, so a shuffled batch may mix two tensor sizes.
        # Group by input mode; train_step averages the per-sample losses.
        groups: dict[bool, list] = {}
        for sample in samples:
            image, mask = self._train_pair(sample, rng)
            key = self._uses_patches(sample)
            groups.setdefault(key, []).append((image, mask))
        batches = []
        for pairs in groups.values():
            images = torch.stack([pair[0] for pair in pairs]).to(self.device)
            masks = torch.stack([pair[1] for pair in pairs]).to(self.device)
            batches.append((images, masks, len(pairs)))
        return batches

    def train_step(self, model, batch):
        total = 0.0
        seen = 0
        for images, masks, count in batch:
            main, auxiliaries, edges = model(images)
            total = total + self._criterion(
                main, auxiliaries, edges, masks) * count
            seen += count
        return total / max(seen, 1)

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
        if not self._uses_patches(sample):
            size = self._full_size
            tensor = F.interpolate(
                image, size=(size, size), mode='bilinear',
                align_corners=False)
            main, _, _ = model(tensor)
            prob = torch.sigmoid(main[:, 0])
            prob = F.interpolate(
                prob[:, None], size=(height, width),
                mode='bilinear', align_corners=False)[0, 0]
            return prob.cpu().numpy() > self._threshold
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
            'full_size': int(self._full_size),
            'input_policy': str(
                getattr(args, 'medcagd_input_policy', 'patches')),
            'encoder': str(getattr(args, 'medcagd_encoder', 'pvt_v2_b2')),
            'num_classes': 1,
            'eval_stride': int(getattr(args, 'medcagd_eval_stride', 128)),
            'eval_batch': int(getattr(args, 'medcagd_eval_batch', 8)),
            'threshold': float(getattr(args, 'medcagd_threshold', 0.5)),
            'edge_weight': float(getattr(args, 'medcagd_edge_weight', 1.0)),
            'use_stage0': int(getattr(args, 'medcagd_use_stage0', 0)),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'full_size' in config:
            self._full_size = int(config['full_size'])
            args.medcagd_full_size = self._full_size
        if 'input_policy' in config:
            self._input_policy = str(config['input_policy'])
            args.medcagd_input_policy = self._input_policy
        else:
            # Checkpoints written before the policy switch used the paper
            # dataset mapping (224 full images outside DRIVE/FIVES).
            self._input_policy = 'paper'
            args.medcagd_input_policy = 'paper'
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
        if 'use_stage0' in config:
            args.medcagd_use_stage0 = int(config['use_stage0'])
        else:
            # Checkpoints written before the Stage 0 switch existed contain
            # the upstream Stage 0 keys; rebuild that variant to load them.
            args.medcagd_use_stage0 = 1
        self._eval_stride = int(getattr(args, 'medcagd_eval_stride', 128))
        self._eval_batch = int(getattr(args, 'medcagd_eval_batch', 8))
        self._threshold = float(getattr(args, 'medcagd_threshold', 0.5))
