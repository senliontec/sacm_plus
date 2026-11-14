"""CurviMix-SAM adapter.

New-model contract
------------------
- Base: the frozen DCD four-candidate scaffold and the SPA-SAM mild
  augmentation, imported through the public ``SpaSamAdapter`` API.
- Method: **Structure-aware Copy-Paste (SCP)**.  With probability
  ``--copy-paste-prob``, paste an intact curvilinear structure from another
  training image into the current image.  Geometry is restricted to flips,
  90-degree rotations and translation; no crop/elastic deformation is used.
- Inference: unchanged fixed candidate-3 deployment.  Copy-paste is a
  training-only regularizer and adds zero inference cost.
- No file under ``baseline_zoo/models`` or ``baseline_zoo/adapters`` is
  modified by this package.

The copied structure is taken from a *training* image only; validation and
test images are never used as donors.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.adapters.spa_sam import SpaSamAdapter
from baseline_zoo.data import load_mask, load_rgb, load_split_samples


class CurviMixAdapter(SpaSamAdapter):
    id = 'curvimix'
    display_name = 'CurviMix-SAM'
    official_source = 'DCD/SPA-SAM + structure-aware copy-paste augmentation'
    official_entrypoint = (
        'DCD/SPA-SAM fixed3 recipe with real curvilinear structure '
        'copy-paste during training')
    deviations = SpaSamAdapter.deviations + (
        'Training adds a structure-aware copy-paste view: an intact '
        'curvilinear structure from another training image is pasted with '
        'flips / 90-degree rotations / translation only; no crop or elastic '
        'warp is applied to the pasted structure.',
        'Inference is unchanged fixed candidate-3 deployment; the new '
        'augmentation adds no parameters and no inference cost.',
    )

    def __init__(self):
        super().__init__()
        self.copy_paste_prob = 0.5
        self.copy_paste_intensity_match = True
        self.tta_flip = False
        self.tta_select = False
        self._source_bank = None

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--copy-paste-prob', type=float, default=0.5,
            help='probability of applying structure-aware copy-paste to a '
                 'training batch')
        parser.add_argument(
            '--copy-paste-intensity-match', type=int, choices=[0, 1],
            default=1,
            help='match the donor foreground intensity to the recipient '
                 'foreground (1) or paste raw donor pixels (0)')
        parser.add_argument(
            '--tta-flip', action='store_true',
            help='inference-only horizontal/vertical/hv flip logit averaging')
        parser.add_argument(
            '--tta-select', action='store_true',
            help='with --tta-flip, choose the candidate whose binary mask is '
                 'most stable across the four views; label-free selector')

    def _read_copy_paste(self, args) -> None:
        self.copy_paste_prob = float(getattr(
            args, 'copy_paste_prob', 0.5))
        self.copy_paste_intensity_match = bool(int(getattr(
            args, 'copy_paste_intensity_match', 1)))
        if not 0.0 <= self.copy_paste_prob <= 1.0:
            raise SystemExit('--copy-paste-prob must be in [0, 1]')

    def build_model(self, args):
        model = super().build_model(args)
        self._read_copy_paste(args)
        self.tta_flip = bool(getattr(args, 'tta_flip', False))
        self.tta_select = bool(getattr(args, 'tta_select', False))
        return model

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'copy_paste_prob': self.copy_paste_prob,
            'copy_paste_intensity_match': int(
                self.copy_paste_intensity_match),
            'tta_flip': int(self.tta_flip),
            'tta_select': int(self.tta_select),
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        # ``tta_flip`` is an evaluation switch: the CLI must be able to
        # enable it even when the trained checkpoint recorded 0.
        for key in ('copy_paste_prob', 'copy_paste_intensity_match'):
            if key in config:
                setattr(args, key, config[key])

    def _load_source_bank(self):
        """Cache the 18 training images/masks at model resolution.

        Only the training split is eligible as a donor.  The bank is CPU
        resident (18 x 1024^2 is small) and selected donors are moved to the
        training device per batch.
        """
        if self._source_bank is not None:
            return self._source_bank
        if self.args is None or self.args.data_root is None:
            self._source_bank = []
            return self._source_bank
        size = int(self.input_size)
        samples = load_split_samples(Path(self.args.data_root), 'train')
        bank = []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (size, size), Image.BILINEAR)
            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask_bool = load_mask(sample.mask_path)
            mask_image = Image.fromarray(
                (mask_bool.astype(np.uint8) * 255))
            mask_image = mask_image.resize((size, size), Image.BILINEAR)
            mask_t = torch.from_numpy(
                np.asarray(mask_image, dtype=np.float32).copy()
            ).unsqueeze(0) / 255.0
            bank.append((image_t, mask_t))
        self._source_bank = bank
        return bank

    @staticmethod
    def _match_intensity(source, source_mask, target, target_mask):
        if float(source_mask.max()) <= 0.5:
            return source
        source_fg = (source_mask > 0.5).expand_as(source)
        target_fg = (target_mask > 0.5).expand_as(target)
        source_mean = source[source_fg].mean()
        target_mean = (target[target_fg].mean() if bool(target_fg.any())
                       else target.mean())
        return (source + (target_mean - source_mean)).clamp(0.0, 1.0)

    def _copy_paste(self, images, masks, rng: random.Random):
        bank = self._load_source_bank()
        if not bank:
            return images, masks
        source_image, source_mask = bank[rng.randrange(len(bank))]
        source_image = source_image.unsqueeze(0).to(self.device)
        source_mask = source_mask.unsqueeze(0).to(self.device)

        # Only topology-preserving geometry for the donor.
        if rng.random() < 0.5:
            source_image = torch.flip(source_image, dims=[3])
            source_mask = torch.flip(source_mask, dims=[3])
        if rng.random() < 0.5:
            source_image = torch.flip(source_image, dims=[2])
            source_mask = torch.flip(source_mask, dims=[2])
        k = int(rng.randrange(4))
        if k:
            source_image = torch.rot90(source_image, k, dims=[2, 3])
            source_mask = torch.rot90(source_mask, k, dims=[2, 3])

        if self.copy_paste_intensity_match:
            source_image = self._match_intensity(
                source_image, source_mask, images, masks)

        height, width = images.shape[-2:]
        dy = rng.randrange(-height // 2, height // 2 + 1)
        dx = rng.randrange(-width // 2, width // 2 + 1)
        y1, y2 = max(0, dy), min(height, dy + height)
        x1, x2 = max(0, dx), min(width, dx + width)
        sy1 = max(0, -dy)
        sy2 = sy1 + (y2 - y1)
        sx1 = max(0, -dx)
        sx2 = sx1 + (x2 - x1)
        if y2 <= y1 or x2 <= x1:
            return images, masks

        donor = source_image[:, :, sy1:sy2, sx1:sx2]
        donor_mask = source_mask[:, :, sy1:sy2, sx1:sx2]
        recipient = images[:, :, y1:y2, x1:x2]
        recipient_mask = masks[:, :, y1:y2, x1:x2]
        images[:, :, y1:y2, x1:x2] = (
            recipient * (1.0 - donor_mask) + donor * donor_mask)
        masks[:, :, y1:y2, x1:x2] = torch.maximum(
            recipient_mask, donor_mask)
        return images.clamp(0.0, 1.0), masks.clamp(0.0, 1.0)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = super().prepare_train_batch(samples, rng)
        if self.copy_paste_prob <= 0.0:
            return images, masks
        if rng.random() >= self.copy_paste_prob:
            return images, masks
        return self._copy_paste(images, masks, rng)

    def _deploy_logits(self, model, images):
        stage2, _ = self._forward(model, images, return_stage1=False)
        if self.head_fusion:
            return self._fused_heads(model, stage2)
        index = self._selected_stage2_index()
        return stage2[:, index:index + 1]

    def predict_sample(self, model, prepared, sample):
        if not self.tta_flip:
            return super().predict_sample(model, prepared, sample)
        views = (
            (prepared, None),
            (torch.flip(prepared, dims=[3]), (3,)),
            (torch.flip(prepared, dims=[2]), (2,)),
            (torch.flip(prepared, dims=[2, 3]), (2, 3)),
        )
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        if not self.tta_select:
            total = None
            for view, dims in views:
                logits = self._deploy_logits(model, view)
                if dims is not None:
                    logits = torch.flip(logits, dims=list(dims))
                total = logits if total is None else total + logits
            logits = total / len(views)
            logits = F.interpolate(
                logits, size=(height, width), mode='bilinear',
                align_corners=False)
            return (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold)

        candidates = []
        for view, dims in views:
            stage2, _ = self._forward(model, view, return_stage1=False)
            if dims is not None:
                stage2 = torch.flip(stage2, dims=list(dims))
            candidates.append(stage2)
        stacked = torch.stack(candidates, dim=0).squeeze(1)
        # [V, 4, 256, 256] -> interpolate each candidate/view to GT.
        v, c, h, w = stacked.shape
        interp = F.interpolate(
            stacked.reshape(v * c, 1, h, w), size=(height, width),
            mode='bilinear', align_corners=False).reshape(
                v, c, height, width)
        probs = torch.sigmoid(interp)
        masks = probs > threshold
        scores = []
        for k in range(c):
            mask_k = masks[:, k]
            if not bool(mask_k.any()):
                scores.append(-1.0)
                continue
            pairwise = []
            for i in range(v):
                for j in range(i + 1, v):
                    a = mask_k[i].float()
                    b = mask_k[j].float()
                    inter = (a * b).sum()
                    union = a.sum() + b.sum()
                    pairwise.append(
                        float((2.0 * inter + 1e-6) /
                              (union + 1e-6)))
            scores.append(float(sum(pairwise) / len(pairwise)))
        best = int(max(range(c), key=lambda k: scores[k]))
        avg = probs.mean(dim=0)[best] > threshold
        return avg.cpu().numpy()
