"""TopoStrand-SAM: independent structure-aware copy-paste baseline.

SPA-SAM is a self-contained DCD-family baseline.  The architecture, four
candidate heads, losses and fixed candidate-3 deployment are copied into
``methods.topostrand.models.topostrand`` so that changes to DCD cannot affect this
model.  The only method change is a structure-preserving augmentation
policy applied during training:

* ``none``: no augmentation (DCD control);
* ``flip``: synchronized horizontal / vertical flips;
* ``photo``: mild brightness / contrast / gamma / noise;
* ``mild``: flips + photometric (default);
* ``strong``: mild + 90-degree rotations + mild blur;
* ``crop`` / ``elastic`` / ``cutout`` / ``standard``: deliberately
  destructive negative controls for ICCV ablations.  They are never used by
  the canonical model; they exist to show that topology-preserving
  augmentation is necessary on thin curvilinear structures.

All DCD structural ablations remain available: ``--specialized-heads 0``,
``--head-fusion 0``, zero topology weights, ``--stage2-loss-scope`` and
``--topology-head``.  The canonical recipe excludes crops and elastic warps.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_mask, load_rgb, load_split_samples
from methods.topostrand.models.topostrand import build_sam_vit_l
from methods.topostrand.models.topostrand.fusion import HeadFusion
from methods.topostrand.models.topostrand.topology_losses import (
    SkeletonRecallLoss,
    SoftclDiceLoss,
    TverskyLoss,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter
from core.losses.cape import CAPELoss

AUG_MODES = (
    'none', 'flip', 'photo', 'mild', 'strong',
    # Deliberately destructive negative controls, ablation-only.
    'crop', 'elastic', 'cutout', 'standard',
)


def _photometric(images: torch.Tensor, rng: random.Random,
                 strength: float = 1.0,
                 noise_std: float = 0.01) -> torch.Tensor:
    if rng.random() < 0.4:
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        contrast = 1.0 + (rng.random() - 0.5) * 0.4 * strength
        brightness = 1.0 + (rng.random() - 0.5) * 0.4 * strength
        images = ((images - mean) * contrast + mean) * brightness
    if rng.random() < 0.3:
        gamma = 1.0 + (rng.random() - 0.5) * 0.4 * strength
        images = images.clamp(0.0, 1.0).pow(gamma)
    if rng.random() < 0.2:
        images = images + torch.randn_like(images) * noise_std * strength
    return images.clamp(0.0, 1.0)


def _flips(images: torch.Tensor, masks: torch.Tensor, rng: random.Random,
           flip_prob: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    if rng.random() < flip_prob:
        images = torch.flip(images, dims=[3])
        masks = torch.flip(masks, dims=[3])
    if rng.random() < flip_prob:
        images = torch.flip(images, dims=[2])
        masks = torch.flip(masks, dims=[2])
    return images, masks


def _random_resized_crop(images: torch.Tensor, masks: torch.Tensor,
                         rng: random.Random, min_scale: float = 0.5
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Destructive control: random scale/aspect crop, then resize back."""
    out_images, out_masks = [], []
    height, width = images.shape[-2:]
    for index in range(images.shape[0]):
        scale = min_scale + (1.0 - min_scale) * rng.random()
        aspect = math.exp((2.0 * rng.random() - 1.0) * math.log(4.0 / 3.0))
        area = scale * height * width
        crop_h = min(height, max(8, int(round(math.sqrt(area / aspect)))))
        crop_w = min(width, max(8, int(round(math.sqrt(area * aspect)))))
        y0 = rng.randrange(0, height - crop_h + 1)
        x0 = rng.randrange(0, width - crop_w + 1)
        image = images[index:index + 1, :, y0:y0 + crop_h, x0:x0 + crop_w]
        mask = masks[index:index + 1, :, y0:y0 + crop_h, x0:x0 + crop_w]
        out_images.append(F.interpolate(
            image, (height, width), mode='bilinear', align_corners=False))
        out_masks.append(F.interpolate(
            mask, (height, width), mode='bilinear', align_corners=False))
    return torch.cat(out_images), torch.cat(out_masks)


def _random_rotate(images: torch.Tensor, masks: torch.Tensor,
                   rng: random.Random, max_angle: float = 25.0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Destructive control: arbitrary-angle rotation with border padding."""
    batch = images.shape[0]
    device, dtype = images.device, images.dtype
    angles = torch.tensor([
        (2.0 * rng.random() - 1.0) * max_angle * math.pi / 180.0
        for _ in range(batch)], device=device, dtype=dtype)
    cos, sin = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(batch, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return (F.grid_sample(images, grid, align_corners=False,
                          padding_mode='border'),
            F.grid_sample(masks, grid, align_corners=False,
                          padding_mode='border'))


def _elastic(images: torch.Tensor, masks: torch.Tensor, rng: random.Random,
             alpha: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Destructive control: smooth random elastic displacement."""
    del rng  # the displacement field uses the deterministic torch RNG
    batch, _, height, width = images.shape
    device, dtype = images.device, images.dtype
    displacement = torch.randn(
        batch, 2, 8, 8, device=device, dtype=dtype)
    displacement = F.interpolate(
        displacement, size=(height, width), mode='bilinear',
        align_corners=False).clamp(-2.0, 2.0) * (2.0 * alpha)
    ys = torch.linspace(-1.0, 1.0, height, device=device,
                        dtype=dtype).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=device,
                        dtype=dtype).view(1, 1, 1, width)
    ys = ys.expand(batch, 1, height, width)
    xs = xs.expand(batch, 1, height, width)
    grid = torch.cat([xs + displacement[:, 0:1],
                      ys + displacement[:, 1:2]], dim=1)
    grid = grid.permute(0, 2, 3, 1)
    return (F.grid_sample(images, grid, align_corners=False,
                          padding_mode='border'),
            F.grid_sample(masks, grid, align_corners=False,
                          padding_mode='border'))


def _cutout(images: torch.Tensor, masks: torch.Tensor,
            rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Destructive control: erase one random image rectangle."""
    height, width = images.shape[-2:]
    for index in range(images.shape[0]):
        cut_h = max(8, int(height * (0.10 + 0.20 * rng.random())))
        cut_w = max(8, int(width * (0.10 + 0.20 * rng.random())))
        y0 = rng.randrange(0, max(1, height - cut_h + 1))
        x0 = rng.randrange(0, max(1, width - cut_w + 1))
        images[index:index + 1, :, y0:y0 + cut_h, x0:x0 + cut_w] = 0.0
    return images, masks


def augment_batch(images: torch.Tensor, masks: torch.Tensor,
                  rng: random.Random, mode: str = 'mild', *,
                  flip_prob: float = 0.5, photo_strength: float = 1.0,
                  noise_std: float = 0.01, crop_min_scale: float = 0.5,
                  elastic_alpha: float = 0.05
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one augmentation variant; destructive modes are ablation-only."""
    if mode not in AUG_MODES:
        raise ValueError(f'unsupported augmentation mode: {mode!r}')
    if mode == 'none':
        return images, masks
    if mode in ('flip', 'mild', 'strong'):
        images, masks = _flips(images, masks, rng, flip_prob)
    if mode in ('mild', 'strong', 'photo'):
        images = _photometric(images, rng, photo_strength, noise_std)
    if mode == 'strong':
        k = int(rng.randrange(4))
        if k:
            images = torch.rot90(images, k, dims=[2, 3])
            masks = torch.rot90(masks, k, dims=[2, 3])
        if rng.random() < 0.2:
            images = F.avg_pool2d(images, 3, 1, 1)
        images = images.clamp(0.0, 1.0)
    elif mode == 'crop':
        images, masks = _random_resized_crop(
            images, masks, rng, crop_min_scale)
    elif mode == 'elastic':
        images, masks = _elastic(images, masks, rng, elastic_alpha)
    elif mode == 'cutout':
        images, masks = _cutout(images, masks, rng)
    elif mode == 'standard':
        images, masks = _random_rotate(images, masks, rng, 25.0)
        images, masks = _random_resized_crop(images, masks, rng, 0.6)
        images = _photometric(images, rng, 1.5, 0.02)
        if rng.random() < 0.3:
            images = F.avg_pool2d(images, 3, 1, 1)
        images = images.clamp(0.0, 1.0)
    return images, masks


class _TopoStrandBase(BaselineAdapter):
    id = '_topostrand_base'
    display_name = 'TopoStrand base'
    official_source = 'DCD + structure-preserving augmentation'
    official_entrypoint = (
        'DCD deterministic candidate decoding with topology-preserving '
        'training augmentation')
    input_size = 1024
    official_lr = 3e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Deployment-aligned Stage-2 aggregation: the deployed head keeps '
        'full loss weight and the three auxiliary heads contribute a '
        '0.25-weight mean term.  The legacy equal-weight mean is removed.',
        'Topology losses consume the native Stage-2 resolution before '
        'resampling, avoiding the 256->1024->512 blur.',
        'SAM normalization is applied before the image encoder '
        '(scale [0,1] -> [0,255], then Sam.preprocess); upstream passed '
        '[0,1] directly into the frozen encoder.',
        'Prompt-free decoding uses the learned no_mask_embed dense anchor '
        'from the frozen prompt encoder instead of the upstream zero map.',
        'Adapter-E uses one LayerNorm per block (paper Eq. 3) instead of '
        'one shared LayerNorm.',
        'Adapter-Fusion implements Eq. 6 as Softmax(FFN(Concat(z_l))) with '
        'per-layer 32-d projections; the upstream Sigmoid-before-softmax '
        'compressed the layer weights into roughly [0.039, 0.106].',
        'Stage-2 consumes the Stage-1-ranked descriptors (paper Eq. 10) '
        'and is not gathered back; per-head IoU predictions are computed '
        'from the ranked descriptors so output slot j is bound to Stage-2 '
        'head j.',
        'All four dense Stage-2 candidates and all four Stage-1 coarse '
        'heads are supervised.  Skeleton Recall and soft-clDice are '
        'consistency terms on the evaluated masks, not auxiliary heads.',
        'Training uses the official soft bilinear targets; foreground-'
        'weighted BCE (pos_weight 2) compensates the low foreground ratio.',
        'AdamW excludes LayerNorm and bias parameters from weight decay; '
        'upstream applied 0.01 to every parameter.',
        'The 1024x1024 resize, structure-preserving augmentation, 3e-4 '
        'learning rate, cosine to 1e-6, gradient clipping 1.0, BCE 0.7 + '
        'Dice 0.3, frozen encoder except adapters, and the validation-Dice '
        'checkpoint selection are preserved.',
        'Predictions are exported at GT resolution and thresholded after '
        'sigmoid (> 0.5); upstream thresholded raw logits at 0.5.',
        'Training uses a structure-preserving augmentation policy: '
        'synchronized flips, optional 90-degree rotations, mild blur and '
        'photometric perturbations.  The canonical recipe excludes crops '
        'and elastic warps because they destroy thin curvilinear structures; '
        'crop/elastic/cutout/standard are destructive ablation controls.',
        'Inference is unchanged: fixed candidate 3 with fusion disabled, '
        'single-pass, threshold 0.5.  This model is physically independent '
        'of DCD: it owns its adapter and model package.',
    )

    def __init__(self):
        super().__init__()
        self._cl_loss = None
        self._skeleton_recall_loss = None
        self._specialized_cl_loss = None
        self._specialized_tversky = None
        self._cape_loss = None
        self._deployed_tversky = None
        self.cape_weight = 0.0
        self.cape_resolution = 128
        self.cape_pairs = 4
        self.cape_warmup_epochs = 10
        self.deployed_tversky_weight = 0.0
        self.area_loss_weight = 0.0
        self.area_loss_mode = 'two_sided'
        self.deploy_fusion = 'none'
        self.deployment_head = 3
        self.specialized_heads = True
        self.specialized_head_weight = 0.1
        self.specialized_topology_weight = 0.1
        self.head_fusion = False
        self.fusion_loss_weight = 1.0
        self.head_fusion_mode = 'linear'
        self.head_fusion_warmup_epochs = 0
        self.head_select = 'gating'
        self.stage2_loss_scope = 'all'
        self.topology_head = 'all'
        self.topology_source = 'native'
        self.cl_dice_weight = 0.1
        self.skeleton_recall_weight = 0.1
        self.topology_resolution = 512
        self.topology_warmup_epochs = 5
        self.deep_sup_weight = 0.3
        self.train_aug = 'mild'
        self.aug_flip_prob = 0.5
        self.aug_photo_strength = 1.0
        self.aug_noise_std = 0.01
        self.aug_crop_min_scale = 0.5
        self.aug_elastic_alpha = 0.05

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--bce-weight', type=float, default=0.7)
        parser.add_argument('--dice-weight', type=float, default=0.3)
        parser.add_argument(
            '--bce-pos-weight', type=float, default=2.0,
            help='foreground weight for BCE in the mask criterion')
        parser.add_argument(
            '--binary-targets', type=int, choices=[0, 1], default=0,
            help='binarize resized training masks at 0.5 (1) or keep the '
                 'official soft bilinear targets (0)')
        parser.add_argument(
            '--deploy-threshold', type=float, default=0.5,
            help='probability threshold used to export the mask')
        parser.add_argument(
            '--deep-sup-weight', type=float, default=0.3,
            help='weight of the Stage-1 all-head mask loss')
        parser.add_argument(
            '--specialized-heads', type=int, choices=[0, 1], default=1,
            help='role-specialised losses on the three auxiliary Stage-2 '
                 'heads; 0 is the native four-head control')
        parser.add_argument(
            '--specialized-head-weight', type=float, default=0.1,
            help='weight of the boundary and recall head losses')
        parser.add_argument(
            '--specialized-topology-weight', type=float, default=0.1,
            help='weight of the topology head loss; set 0 for the '
                 'no-topology control')
        parser.add_argument(
            '--head-fusion', type=int, choices=[0, 1], default=0,
            help='train the zero-initialised linear fusion as an auxiliary '
                 'output; deployment should disable it')
        parser.add_argument(
            '--fusion-loss-weight', type=float, default=1.0,
            help='weight of the fused-head dense mask loss')
        parser.add_argument(
            '--stage2-loss-scope', choices=['all', 'selected'],
            default='all',
            help='dense mask loss on every candidate (all) or only the '
                 'deployment-selected slot (selected)')
        parser.add_argument(
            '--topology-head', choices=['deployment', 'selected', 'all'],
            default='all',
            help='apply topology losses to the fixed deployment head, the '
                 'deployment-selected slot, or all four heads')
        parser.add_argument(
            '--cl-dice-weight', type=float, default=0.1,
            help='binary soft-clDice weight on the topology heads')
        parser.add_argument(
            '--skeleton-recall-weight', type=float, default=0.1,
            help='skeleton-recall weight on the topology heads')
        parser.add_argument(
            '--topology-resolution', type=int, default=512,
            help='resolution used to compute topology losses')
        parser.add_argument(
            '--topology-warmup-epochs', type=int, default=5,
            help='linearly ramp topology losses in over this many epochs')
        parser.add_argument(
            '--cape-weight', type=float, default=0.0,
            help='connectivity-aware path-enforcement loss on slot 3; '
                 '0 disables (CurviMix control)')
        parser.add_argument(
            '--cape-resolution', type=int, default=128,
            help='work resolution for the GT-skeleton path loss')
        parser.add_argument(
            '--cape-pairs', type=int, default=4,
            help='number of connected endpoint pairs per training image')
        parser.add_argument(
            '--cape-warmup-epochs', type=int, default=10,
            help='linearly ramp CAPE in over this many epochs')
        parser.add_argument(
            '--deployed-tversky-weight', type=float, default=0.0,
            help='precision-heavy Tversky (alpha=.7,beta=.3) on the '
                 'deployment slot 3; targets the observed width inflation')
        parser.add_argument(
            '--area-loss-weight', type=float, default=0.0,
            help='L1 foreground-area calibration on slot 3')
        parser.add_argument(
            '--area-loss-mode', choices=['two_sided', 'over'],
            default='two_sided',
            help='penalise both over- and under-segmentation (two_sided) '
                 'or only over-segmentation (over), which preserves thin '
                 'under-segmented branches')
        parser.add_argument(
            '--head-select', choices=['gating', 'fixed'], default=None,
            help='training uses gating; the derived deployment checkpoint '
                 'sets fixed when the CLI leaves this unset')
        parser.add_argument(
            '--deployment-head', type=int, default=3,
            help='Stage-2 head used by fixed deployment')
        parser.add_argument(
            '--deploy-fusion', choices=['none', 'mean'], default='none',
            help='deployment output: the selected Stage-2 head (none) or '
                 'the mean logit of all four Stage-2 candidates (mean)')
        parser.add_argument(
            '--clip-grad-norm', dest='grad_clip', type=float,
            help='gradient clipping norm (official default: 1.0)')
        parser.add_argument(
            '--train-aug', choices=list(AUG_MODES), default='mild',
            help='structure-preserving augmentation variant: none (DCD '
                 'control), flip, photo, mild (flips + photometric, default) '
                 'or strong (mild + rot90 + mild blur); crop, elastic, cutout '
                 'and standard are destructive ICCV ablation controls')
        parser.add_argument(
            '--aug-flip-prob', type=float, default=0.5,
            help='probability of each synchronized flip')
        parser.add_argument(
            '--aug-photo-strength', type=float, default=1.0,
            help='multiplier on photometric perturbation strength')
        parser.add_argument(
            '--aug-noise-std', type=float, default=0.01,
            help='Gaussian noise standard deviation at strength 1.0')
        parser.add_argument(
            '--aug-crop-min-scale', type=float, default=0.5,
            help='minimum random-resized-crop area scale for the destructive '
                 'crop control')
        parser.add_argument(
            '--aug-elastic-alpha', type=float, default=0.05,
            help='elastic displacement magnitude for the destructive control')
        parser.set_defaults(grad_clip=1.0)

    def _read_aug(self, args) -> None:
        self.train_aug = str(getattr(args, 'train_aug', 'mild'))
        if self.train_aug not in AUG_MODES:
            raise SystemExit(
                f'unsupported train-aug: {self.train_aug!r}')
        self.aug_flip_prob = float(getattr(
            args, 'aug_flip_prob', 0.5))
        self.aug_photo_strength = float(getattr(
            args, 'aug_photo_strength', 1.0))
        self.aug_noise_std = float(getattr(args, 'aug_noise_std', 0.01))
        self.aug_crop_min_scale = float(getattr(
            args, 'aug_crop_min_scale', 0.5))
        self.aug_elastic_alpha = float(getattr(
            args, 'aug_elastic_alpha', 0.05))
        if not 0.0 <= self.aug_flip_prob <= 1.0:
            raise SystemExit('--aug-flip-prob must be in [0, 1]')
        if self.aug_photo_strength < 0:
            raise SystemExit('--aug-photo-strength must be non-negative')
        if self.aug_noise_std < 0:
            raise SystemExit('--aug-noise-std must be non-negative')
        if not 0.0 < self.aug_crop_min_scale <= 1.0:
            raise SystemExit('--aug-crop-min-scale must be in (0, 1]')
        if not 0.0 <= self.aug_elastic_alpha <= 0.5:
            raise SystemExit('--aug-elastic-alpha must be in [0, 0.5]')

    @staticmethod
    def _resolved_head_select(args):
        requested = getattr(args, 'head_select', None) or 'gating'
        if requested not in ('gating', 'fixed'):
            raise SystemExit(
                f'unsupported head-select {requested!r}; expected gating or '
                'fixed')
        return requested

    def _selected_stage2_index(self):
        """Return the post-gating Stage-2 slot used for deployment."""
        if self.head_select == 'gating':
            return 0
        if self.head_select == 'fixed':
            return int(self.deployment_head)
        raise ValueError(
            f'unsupported head-select {self.head_select!r}')

    def _stage2_loss_indices(self, num_heads):
        """Indices of Stage-2 candidates receiving the dense mask loss."""
        scope = str(getattr(
            getattr(self, 'args', None), 'stage2_loss_scope', 'all'))
        if scope == 'all':
            return list(range(int(num_heads)))
        if scope == 'selected':
            index = self._selected_stage2_index()
            if not 0 <= index < int(num_heads):
                raise ValueError(
                    f'selected Stage-2 index {index} outside [0, '
                    f'{num_heads})')
            return [index]
        raise ValueError(f'unknown stage2-loss-scope: {scope!r}')

    def _topology_indices(self, num_heads):
        """Indices of Stage-2 candidates receiving structural losses."""
        head = str(getattr(
            getattr(self, 'args', None), 'topology_head', 'all'))
        if head == 'all':
            return list(range(int(num_heads)))
        if head == 'selected':
            index = self._selected_stage2_index()
        elif head == 'deployment':
            index = int(self.deployment_head)
        else:
            raise ValueError(f'unknown topology-head: {head!r}')
        if not 0 <= index < int(num_heads):
            raise ValueError(
                f'topology index {index} outside [0, {num_heads})')
        return [index]

    def _deployment_anchor_index(self):
        """Head that keeps full loss weight under deployment aggregation."""
        if self.head_select == 'gating':
            return 0
        return int(self.deployment_head)

    @staticmethod
    def _aggregate_head_losses(losses, indices, anchor):
        """Keep the deployed head at full weight and average the rest.

        The auxiliary heads enter at weight 0.25 (the canonical recipe).
        With a single selected head this is just that head's loss.
        """
        values = [losses[index] for index in indices]
        if len(values) == 1 or anchor not in indices:
            return torch.stack(values).mean()
        auxiliary = [
            losses[index] for index in indices if index != anchor]
        if not auxiliary:
            return losses[anchor]
        return losses[anchor] + 0.25 * torch.stack(auxiliary).mean()

    def selected_candidate_index(self):
        """Public deployment-slot index for candidate diagnostics."""
        return self._selected_stage2_index()

    def build_model(self, args):
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict'
            or getattr(args, 'checkpoint', None) is not None)
        checkpoint = None
        if not predicting:
            if not args.sam_checkpoint.is_file():
                raise SystemExit(
                    f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')
            checkpoint = str(args.sam_checkpoint)

        self.args = args
        self._read_aug(args)
        self.head_select = self._resolved_head_select(args)
        self.deployment_head = int(getattr(args, 'deployment_head', 3))
        self.specialized_heads = bool(
            int(getattr(args, 'specialized_heads', 1)))
        self.specialized_head_weight = float(
            getattr(args, 'specialized_head_weight', 0.1))
        self.specialized_topology_weight = float(
            getattr(args, 'specialized_topology_weight', 0.1))
        if self.specialized_topology_weight < 0:
            raise SystemExit(
                '--specialized-topology-weight must be non-negative')
        self.head_fusion = bool(int(getattr(args, 'head_fusion', 0)))
        self.fusion_loss_weight = float(
            getattr(args, 'fusion_loss_weight', 1.0))
        self.head_fusion_mode = 'linear'
        self.head_fusion_warmup_epochs = 0
        self.stage2_loss_scope = str(
            getattr(args, 'stage2_loss_scope', 'all'))
        self.topology_head = str(
            getattr(args, 'topology_head', 'all'))
        self.topology_source = 'native'
        self.cl_dice_weight = float(
            getattr(args, 'cl_dice_weight', 0.1))
        self.skeleton_recall_weight = float(
            getattr(args, 'skeleton_recall_weight', 0.1))
        self.topology_resolution = int(
            getattr(args, 'topology_resolution', 512))
        self.topology_warmup_epochs = int(
            getattr(args, 'topology_warmup_epochs', 5))
        self.deep_sup_weight = float(
            getattr(args, 'deep_sup_weight', 0.3))
        self.cape_weight = float(getattr(args, 'cape_weight', 0.0))
        self.cape_resolution = int(
            getattr(args, 'cape_resolution', 128))
        self.cape_pairs = int(getattr(args, 'cape_pairs', 4))
        self.cape_warmup_epochs = int(
            getattr(args, 'cape_warmup_epochs', 10))
        self.deployed_tversky_weight = float(
            getattr(args, 'deployed_tversky_weight', 0.0))
        self.area_loss_weight = float(
            getattr(args, 'area_loss_weight', 0.0))
        self.area_loss_mode = str(
            getattr(args, 'area_loss_mode', 'two_sided'))
        self.deploy_fusion = str(
            getattr(args, 'deploy_fusion', 'none'))
        if self.area_loss_mode not in ('two_sided', 'over'):
            raise SystemExit(
                '--area-loss-mode must be two_sided or over')
        if self.deploy_fusion not in ('none', 'mean'):
            raise SystemExit('--deploy-fusion must be none or mean')
        if self.cape_weight < 0:
            raise SystemExit('--cape-weight must be non-negative')
        if self.cape_resolution <= 0:
            raise SystemExit('--cape-resolution must be positive')
        if self.cape_pairs <= 0:
            raise SystemExit('--cape-pairs must be positive')
        if self.deployed_tversky_weight < 0:
            raise SystemExit(
                '--deployed-tversky-weight must be non-negative')
        if self.area_loss_weight < 0:
            raise SystemExit('--area-loss-weight must be non-negative')

        model = build_sam_vit_l(
            checkpoint=checkpoint, use_adapter=True,
            adapter_dim_ratio=args.adapter_dim_ratio)
        decoder = getattr(model, 'mask_decoder', None)
        if (decoder is not None
                and not 0 <= self.deployment_head
                < decoder.num_mask_tokens):
            raise SystemExit(
                f'deployment-head must be in [0, '
                f'{decoder.num_mask_tokens}), got {self.deployment_head}')
        if self.head_fusion:
            model.head_fusion = HeadFusion()
        return model

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'adapter_dim_ratio': float(
                getattr(args, 'adapter_dim_ratio', 0.1)),
            'bce_weight': float(getattr(args, 'bce_weight', 0.7)),
            'dice_weight': float(getattr(args, 'dice_weight', 0.3)),
            'bce_pos_weight': float(
                getattr(args, 'bce_pos_weight', 2.0)),
            'binary_targets': int(getattr(args, 'binary_targets', 0)),
            'deploy_threshold': float(
                getattr(args, 'deploy_threshold', 0.5)),
            'deep_sup_weight': self.deep_sup_weight,
            'specialized_heads': int(self.specialized_heads),
            'specialized_head_weight': self.specialized_head_weight,
            'specialized_topology_weight': self.specialized_topology_weight,
            'head_fusion': int(self.head_fusion),
            'fusion_loss_weight': self.fusion_loss_weight,
            'head_fusion_mode': self.head_fusion_mode,
            'head_fusion_warmup_epochs': self.head_fusion_warmup_epochs,
            'head_select': self.head_select,
            'deployment_head': int(self.deployment_head),
            'stage2_loss_scope': self.stage2_loss_scope,
            'topology_head': self.topology_head,
            'topology_source': self.topology_source,
            'cl_dice_weight': self.cl_dice_weight,
            'skeleton_recall_weight': self.skeleton_recall_weight,
            'topology_resolution': self.topology_resolution,
            'topology_warmup_epochs': self.topology_warmup_epochs,
            'cape_weight': self.cape_weight,
            'cape_resolution': self.cape_resolution,
            'cape_pairs': self.cape_pairs,
            'cape_warmup_epochs': self.cape_warmup_epochs,
            'deployed_tversky_weight': self.deployed_tversky_weight,
            'area_loss_weight': self.area_loss_weight,
            'area_loss_mode': self.area_loss_mode,
            'train_aug': self.train_aug,
            'aug_flip_prob': self.aug_flip_prob,
            'aug_photo_strength': self.aug_photo_strength,
            'aug_noise_std': self.aug_noise_std,
            'aug_crop_min_scale': self.aug_crop_min_scale,
            'aug_elastic_alpha': self.aug_elastic_alpha,
        }

    def restore_config(self, args, config):
        for key in (
                'adapter_dim_ratio', 'bce_weight', 'dice_weight',
                'bce_pos_weight', 'binary_targets', 'deploy_threshold',
                'deep_sup_weight', 'specialized_heads',
                'specialized_head_weight', 'specialized_topology_weight',
                'head_fusion', 'fusion_loss_weight', 'head_fusion_mode',
                'head_fusion_warmup_epochs', 'deployment_head',
                'stage2_loss_scope', 'topology_head', 'topology_source',
                'cl_dice_weight', 'skeleton_recall_weight',
                'topology_resolution', 'topology_warmup_epochs',
                'cape_weight', 'cape_resolution', 'cape_pairs',
                'cape_warmup_epochs', 'deployed_tversky_weight',
                'area_loss_weight', 'area_loss_mode',
                'train_aug', 'aug_flip_prob', 'aug_photo_strength',
                'aug_noise_std', 'aug_crop_min_scale',
                'aug_elastic_alpha'):
            if key in config:
                setattr(args, key, config[key])
        saved_size = config.get('input_size')
        if saved_size is not None:
            self.input_size = int(saved_size)
            args.input_size = self.input_size
        if (getattr(args, 'head_select', None) is None
                and config.get('head_select') in ('gating', 'fixed')):
            args.head_select = config['head_select']

    def load_state_dict(self, model, state) -> None:
        model.load_state_dict(state)

    def configure_trainable(self, model):
        for name, parameter in model.image_encoder.named_parameters():
            parameter.requires_grad_('adapter' in name)
        for parameter in model.prompt_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad_(True)
        model.image_encoder.grad_checkpointing = True

    def make_optimizer(self, model, args):
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith('.bias'):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.AdamW([
            {'params': decay, 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        ])

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    @staticmethod
    def _sam_preprocess(model, images):
        """Apply SAM normalization to [0,1] inputs (upstream omitted it)."""
        lo = images.amin()
        hi = images.amax()
        if bool(lo >= 0.0) and bool(hi <= 1.0 + 1e-6):
            images = images * 255.0
        return model.preprocess(images)

    def _forward(self, model, images, return_stage1=False):
        images = self._sam_preprocess(model, images)
        batch = images.shape[0]
        sparse, dense = model.prompt_encoder(
            points=None, boxes=None, masks=None)
        sparse = sparse.expand(batch, -1, -1)
        dense = dense.expand(batch, -1, -1, -1)
        encoder_output = model.image_encoder(images, return_features=True)
        if len(encoder_output) == 3:
            embeddings, _, adapter_features = encoder_output
        else:
            embeddings, _ = encoder_output
            adapter_features = None
        return model.mask_decoder.predict_masks(
            image_embeddings=embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            adapter_features=adapter_features,
            return_stage1=return_stage1,
        )

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BILINEAR)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.BILINEAR)
            images.append(torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0)
            mask_np = np.asarray(mask, dtype=np.float32).copy()
            if mask_np.size and float(mask_np.max()) <= 1.0:
                mask_np *= 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) / 255.0
            if int(getattr(self.args, 'binary_targets', 0)) == 1:
                mask_tensor = (mask_tensor > 0.5).float()
            masks.append(mask_tensor)
        images = torch.stack(images).to(self.device)
        masks = torch.stack(masks).to(self.device)
        return augment_batch(
            images, masks, rng, self.train_aug,
            flip_prob=self.aug_flip_prob,
            photo_strength=self.aug_photo_strength,
            noise_std=self.aug_noise_std,
            crop_min_scale=self.aug_crop_min_scale,
            elastic_alpha=self.aug_elastic_alpha)

    def _mask_criterion(self, logits, target, bce_weight, dice_weight):
        pos_weight = float(getattr(
            getattr(self, 'args', None), 'bce_pos_weight', 2.0))
        if pos_weight > 0 and pos_weight != 1.0:
            bce = F.binary_cross_entropy_with_logits(
                logits, target,
                pos_weight=logits.new_tensor(pos_weight))
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return bce_weight * bce + dice_weight * dice.mean()

    def _topology_loss(self, stage2, masks):
        """Soft-clDice and Skeleton Recall on the selected Stage-2 heads."""
        cl_w = float(self.cl_dice_weight)
        sk_w = float(self.skeleton_recall_weight)
        if max(cl_w, sk_w) <= 0:
            return stage2.new_zeros(())

        if self._cl_loss is None:
            self._cl_loss = SoftclDiceLoss()
        if sk_w > 0 and self._skeleton_recall_loss is None:
            self._skeleton_recall_loss = SkeletonRecallLoss()

        resolution = int(self.topology_resolution)
        if resolution <= 0:
            raise ValueError('topology-resolution must be positive')
        target = F.interpolate(
            masks, size=(resolution, resolution), mode='nearest')
        indices = self._topology_indices(stage2.shape[1])

        head_totals = {}
        for index in indices:
            logits = F.interpolate(
                stage2[:, index:index + 1],
                size=(resolution, resolution),
                mode='bilinear', align_corners=False)
            per_head = stage2.new_zeros(())
            if cl_w > 0:
                per_head = per_head + cl_w * self._cl_loss(logits, target)
            if sk_w > 0:
                per_head = per_head + sk_w * self._skeleton_recall_loss(
                    logits, target)
            head_totals[index] = per_head
        anchor = (self._deployment_anchor_index()
                  if self.topology_head == 'all'
                  else indices[0])
        total = self._aggregate_head_losses(head_totals, indices, anchor)

        epoch = int(getattr(self, '_epoch', 1))
        warmup = max(0, int(self.topology_warmup_epochs))
        ramp = 1.0 if warmup <= 0 else min(
            1.0, max(0.0, (epoch - warmup) / float(warmup)))
        return total * ramp

    @staticmethod
    def _boundary_weighted_loss(logits, target):
        """Boundary-emphasised full-mask loss for the boundary head."""
        eroded = -F.max_pool2d(
            -target, kernel_size=5, stride=1, padding=2)
        boundary = (target - eroded).clamp(0.0, 1.0)
        weight = 1.0 + 4.0 * boundary
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return bce + dice.mean()

    def _specialized_head_loss(self, stage2, masks):
        """Role losses on the three auxiliary Stage-2 heads.

        Head 1: soft-clDice topology.  Head 2: boundary-weighted dense
        loss.  Head 3: recall-heavy Tversky dense loss.  All heads keep the
        canonical all-dense loss; these terms only add a different emphasis.
        """
        weight = float(self.specialized_head_weight)
        topology_weight = float(self.specialized_topology_weight)
        if weight <= 0 and topology_weight <= 0:
            return stage2.new_zeros(())
        resolution = int(self.topology_resolution)
        target = F.interpolate(
            masks, (resolution, resolution), mode='bilinear',
            align_corners=False)
        heads = F.interpolate(
            stage2, (resolution, resolution), mode='bilinear',
            align_corners=False)
        topology = stage2.new_zeros(())
        if topology_weight > 0:
            if self._specialized_cl_loss is None:
                self._specialized_cl_loss = SoftclDiceLoss()
            topology = self._specialized_cl_loss(heads[:, 1:2], target)
        if self._specialized_tversky is None:
            self._specialized_tversky = TverskyLoss(alpha=0.3, beta=0.7)
        boundary = self._boundary_weighted_loss(heads[:, 2:3], target)
        recall = self._specialized_tversky(heads[:, 3:4], target)
        return (weight * (boundary + recall)
                + topology_weight * topology)

    def _fused_heads(self, model, stage2):
        fusion = getattr(model, 'head_fusion', None)
        if fusion is None:
            raise RuntimeError(
                'head-fusion=1 but model.head_fusion is missing')
        return fusion(stage2)

    def train_step(self, model, batch):
        images, masks = batch
        stage2, _, stage1 = self._forward(
            model, images, return_stage1=True)
        # Keep the native Stage-2 tensor for topology losses; the dense loss
        # uses the 256 -> 1024 training geometry.
        stage2_native = stage2
        masks = F.interpolate(masks, (1024, 1024), mode='bilinear',
                              align_corners=False)
        stage1 = F.interpolate(stage1, (1024, 1024), mode='bilinear',
                               align_corners=False)
        stage2 = F.interpolate(stage2, (1024, 1024), mode='bilinear',
                               align_corners=False)

        bce_weight = float(self.args.bce_weight)
        dice_weight = float(self.args.dice_weight)
        head_indices = self._stage2_loss_indices(stage2.shape[1])
        head_losses = {
            k: self._mask_criterion(stage2[:, k:k + 1], masks,
                                    bce_weight, dice_weight)
            for k in head_indices
        }
        loss = self._aggregate_head_losses(
            head_losses, head_indices, self._deployment_anchor_index())

        if self.deep_sup_weight > 0:
            ds_losses = [
                self._mask_criterion(stage1[:, k:k + 1], masks,
                                     bce_weight, dice_weight)
                for k in range(stage1.shape[1])
            ]
            loss = loss + self.deep_sup_weight * torch.stack(ds_losses).mean()

        loss = loss + self._topology_loss(stage2_native, masks)
        if self.cape_weight > 0:
            if self._cape_loss is None:
                self._cape_loss = CAPELoss(
                    resolution=self.cape_resolution,
                    num_pairs=self.cape_pairs)
            epoch = int(getattr(self, '_epoch', 1))
            warmup = max(0, int(self.cape_warmup_epochs))
            ramp = 1.0 if warmup <= 0 else min(
                1.0, max(0.0, (epoch - warmup) / float(warmup)))
            if ramp > 0:
                loss = loss + self.cape_weight * ramp * self._cape_loss(
                    stage2_native[:, 3:4], masks)
        if self.deployed_tversky_weight > 0:
            if self._deployed_tversky is None:
                self._deployed_tversky = TverskyLoss(alpha=0.7, beta=0.3)
            loss = loss + self.deployed_tversky_weight * self._deployed_tversky(
                stage2[:, 3:4], masks)
        if self.area_loss_weight > 0:
            pred_area = torch.sigmoid(stage2[:, 3:4]).mean(
                dim=(1, 2, 3))
            true_area = masks.mean(dim=(1, 2, 3))
            area_delta = pred_area - true_area
            if self.area_loss_mode == 'over':
                area_delta = area_delta.clamp_min(0.0)
            loss = loss + self.area_loss_weight * area_delta.abs().mean()
        if self.specialized_heads:
            loss = loss + self._specialized_head_loss(
                stage2_native, masks)
        if self.head_fusion:
            fused = F.interpolate(
                self._fused_heads(model, stage2_native),
                (1024, 1024), mode='bilinear', align_corners=False)
            loss = loss + self.fusion_loss_weight * self._mask_criterion(
                fused, masks, bce_weight, dice_weight)
        return loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        stage2, _ = self._forward(
            model, prepared, return_stage1=False)
        logits = self._deploy_from_stage2(model, stage2)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold)

    @torch.no_grad()
    def predict_candidates(self, model, prepared, sample):
        """Return all four post-gating Stage-2 candidates at GT resolution."""
        if not sample.mask_path:
            raise ValueError('candidate diagnostics require a GT mask')
        stage2, _ = self._forward(
            model, prepared, return_stage1=False)
        if stage2.shape[1] != 4:
            raise RuntimeError(
                f'expected four Stage-2 candidates, got {stage2.shape[1]}')
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(
            stage2, size=(height, width), mode='bilinear',
            align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return probs > threshold


class TopoStrandAdapter(_TopoStrandBase):
    id = 'topostrand'
    display_name = 'TopoStrand-SAM'
    official_source = 'DCD/SPA-SAM + structure-aware copy-paste augmentation'
    official_entrypoint = (
        'DCD/SPA-SAM fixed3 recipe with real curvilinear structure '
        'copy-paste during training')
    deviations = _TopoStrandBase.deviations + (
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
        self.copy_paste_scale_jitter = 0.0
        self.tta_flip = False
        self.tta_rot90 = False
        self.tta_select = False
        self.selection_mix = 0.0
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
            '--copy-paste-scale-jitter', type=float, default=0.0,
            help='maximum independent x/y scale jitter for a pasted donor; '
                 '0 disables (CurviMix control), e.g. 0.3 means [0.7,1.3]')
        parser.add_argument(
            '--tta-flip', action='store_true',
            help='inference-only horizontal/vertical/hv flip logit averaging')
        parser.add_argument(
            '--tta-select', action='store_true',
            help='with --tta-flip, choose the candidate whose binary mask is '
                 'most stable across the four views; label-free selector')
        parser.add_argument(
            '--selection-mix', type=float, default=0.0,
            help='checkpoint selection score: (1-mix)*val Dice + '
                 'mix*val clDice; 0 keeps validation Dice')

    def _read_copy_paste(self, args) -> None:
        self.copy_paste_prob = float(getattr(
            args, 'copy_paste_prob', 0.5))
        self.copy_paste_intensity_match = bool(int(getattr(
            args, 'copy_paste_intensity_match', 1)))
        self.copy_paste_scale_jitter = float(getattr(
            args, 'copy_paste_scale_jitter', 0.0))
        if not 0.0 <= self.copy_paste_prob <= 1.0:
            raise SystemExit('--copy-paste-prob must be in [0, 1]')
        if not 0.0 <= self.copy_paste_scale_jitter < 1.0:
            raise SystemExit(
                '--copy-paste-scale-jitter must be in [0, 1)')

    def build_model(self, args):
        model = super().build_model(args)
        self._read_copy_paste(args)
        self.tta_flip = bool(getattr(args, 'tta_flip', False))
        self.tta_select = bool(getattr(args, 'tta_select', False))
        self.selection_mix = float(getattr(args, 'selection_mix', 0.0))
        if not 0.0 <= self.selection_mix <= 1.0:
            raise SystemExit('--selection-mix must be in [0, 1]')
        return model

    def selection_score(self, metrics, per_dataset):
        """Optionally balance validation Dice and clDice at checkpoint time."""
        if self.selection_mix > 0.0:
            dice = float(metrics.get('dice', float('nan')))
            cldice = float(metrics.get('cldice', float('nan')))
            if math.isfinite(dice) and math.isfinite(cldice):
                return ((1.0 - self.selection_mix) * dice
                        + self.selection_mix * cldice)
        return super().selection_score(metrics, per_dataset)

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'copy_paste_prob': self.copy_paste_prob,
            'copy_paste_intensity_match': int(
                self.copy_paste_intensity_match),
            'copy_paste_scale_jitter': self.copy_paste_scale_jitter,
            'tta_flip': int(self.tta_flip),
            'tta_select': int(self.tta_select),
            'selection_mix': self.selection_mix,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        # ``tta_flip`` is an evaluation switch: the CLI must be able to
        # enable it even when the trained checkpoint recorded 0.
        for key in (
                'copy_paste_prob', 'copy_paste_intensity_match',
                'copy_paste_scale_jitter', 'selection_mix'):
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
    def _fit_center(tensor, height: int, width: int):
        h, w = tensor.shape[-2:]
        if h >= height:
            y0 = (h - height) // 2
            tensor = tensor[:, :, y0:y0 + height]
        else:
            top = (height - h) // 2
            tensor = F.pad(
                tensor, (0, 0, top, height - h - top), value=0.0)
        h, w = tensor.shape[-2:]
        if w >= width:
            x0 = (w - width) // 2
            tensor = tensor[:, :, :, x0:x0 + width]
        else:
            left = (width - w) // 2
            tensor = F.pad(
                tensor, (left, width - w - left, 0, 0), value=0.0)
        return tensor

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

        height, width = images.shape[-2:]
        if self.copy_paste_scale_jitter > 0:
            jitter = float(self.copy_paste_scale_jitter)
            sy = 1.0 + (2.0 * rng.random() - 1.0) * jitter
            sx = 1.0 + (2.0 * rng.random() - 1.0) * jitter
            nh = max(16, int(round(height * sy)))
            nw = max(16, int(round(width * sx)))
            source_image = F.interpolate(
                source_image, size=(nh, nw), mode='bilinear',
                align_corners=False)
            source_mask = F.interpolate(
                source_mask, size=(nh, nw), mode='nearest')
            source_image = self._fit_center(source_image, height, width)
            source_mask = self._fit_center(source_mask, height, width)

        if self.copy_paste_intensity_match:
            source_image = self._match_intensity(
                source_image, source_mask, images, masks)

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

    def _deploy_from_stage2(self, model, stage2):
        """Map the four Stage-2 candidates to the deployed logit map."""
        if self.deploy_fusion == 'mean':
            return stage2.mean(dim=1, keepdim=True)
        if self.head_fusion:
            return self._fused_heads(model, stage2)
        index = self._selected_stage2_index()
        return stage2[:, index:index + 1]

    def _deploy_logits(self, model, images):
        stage2, _ = self._forward(model, images, return_stage1=False)
        return self._deploy_from_stage2(model, stage2)

    def predict_sample(self, model, prepared, sample):
        if not (self.tta_flip or self.tta_rot90):
            return super().predict_sample(model, prepared, sample)
        if self.tta_select and self.tta_rot90:
            raise RuntimeError('--tta-rot90 is not supported with --tta-select')

        if self.tta_select:
            views = (
                (prepared, None),
                (torch.flip(prepared, dims=[3]), (3,)),
                (torch.flip(prepared, dims=[2]), (2,)),
                (torch.flip(prepared, dims=[2, 3]), (2, 3)),
            )
        else:
            # Each view stores its forward rotation and flip so the logits can
            # be mapped back before averaging.
            rotations = (0, 1, 2, 3) if self.tta_rot90 else (0,)
            flips = (None, (3,), (2,), (2, 3)) if self.tta_flip else (None,)
            views = []
            for rotation in rotations:
                rotated = prepared if rotation == 0 else torch.rot90(
                    prepared, rotation, dims=[2, 3])
                for flip in flips:
                    view = rotated if flip is None else torch.flip(
                        rotated, dims=list(flip))
                    views.append((view, rotation, flip))

        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        if not self.tta_select:
            total = None
            for view, rotation, flip in views:
                logits = self._deploy_logits(model, view)
                if flip is not None:
                    logits = torch.flip(logits, dims=list(flip))
                if rotation:
                    logits = torch.rot90(
                        logits, -rotation, dims=[2, 3])
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
