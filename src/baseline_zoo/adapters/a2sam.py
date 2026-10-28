"""A2SAM adapter (Medical Image Analysis 2026).

A2SAM adapts SAM ViT-B with a structural prompt generator and contrastive
parameter selection.  The released repository only supplies SAM ViT-B, a
detached uncertainty/logit fusion network and the CPS selection code; the
probabilistic U-Net prior and the paper's topological SPG are reimplemented
in ``baseline_zoo.models.a2sam``.  Training has two stages on the shared
18-image split: a probabilistic prior for 100 epochs, then A2SAM + SPG with
CPS on the top 3.5% of SAM elements for the remaining shared budget.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.a2sam import (
    apply_cps_masks,
    build_a2sam,
    contrastive_scores,
    select_contrastive_masks,
)
from baseline_zoo.models.a2sam.loss import IOU, ThicknessAwareLoss
from baseline_zoo.runtime import BaselineAdapter, file_sha256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAM_PREFIXES = ('image_encoder.', 'prompt_encoder.', 'mask_decoder.')


class A2SAMAdapter(BaselineAdapter):
    id = 'a2sam'
    display_name = 'A2SAM'
    family = 'sam'
    official_source = 'A2SAM (Medical Image Analysis 2026)'
    official_entrypoint = 'https://github.com/YongChen-Exact/A2SAM'
    input_size = 1024
    official_lr = 2e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    accumulation_steps = 1
    default_grad_clip = 0.0
    selection = 'dice'
    deviations = (
        'The released A2SAM repository does not contain the paper prior or '
        'the topological SPG: it reads precomputed uncertainty/coarse-logit '
        '.npy files and its PromptGenerator only fuses them, with the '
        'fusion output detached from the loss. This port trains a '
        'probabilistic U-Net prior, extracts persistence-pruned skeleton '
        'branches, learns branch gates and fuses the structural map, '
        'likelihood and uncertainty without detaching the prompt.',
        'The SPG topology is a graph-based discrete-Morse/persistence '
        'approximation on the likelihood skeleton (junction/endpoint '
        'critical cells, geodesic filtration, persistence pruning); the '
        'paper exact cell decomposition is not available in the release.',
        'The prior is trained inside the run for 100 epochs on the shared '
        '18-image split; the release expects precomputed maps produced by a '
        'separate U-Net.',
        'The prior is sampled K=5 times at 512 resolution; the release '
        'uncertainty/logit maps are precomputed at the network input size.',
        'CPS influence statistics use the 18 training images (the released '
        'code uses its validation loader); this avoids validation-label '
        'leakage in checkpoint selection. CPS keeps the top 3.5% elements '
        'with alpha 0.4 and freezes norms/position embeddings/IoU head as in '
        'the release.',
        'The official AdamW lr 2e-4, cosine to 1e-7, batch 1, 1024-square '
        'ImageNet-normalized input and horizontal-flip-only augmentation '
        'are preserved; the official 50 epochs become the shared 800-epoch '
        'budget (100 prior + 700 A2SAM) with validation-Dice selection.',
        'The thickness-aware term uses the same plus-kernel opening and '
        'EDT/skeleton weights as the release, but divides by the foreground '
        'pixel count (the release divides by all non-zero pixels after the '
        'sigmoid) and implements the opening locally instead of using '
        'Kornia.',
        'The released prompt network receives a coarse logit; this port '
        'uses the prior mean probability and its variance-based uncertainty, '
        'and the SPG output is a soft dense prompt rather than a detached '
        'mask.',
        'SAM ViT-B weights come from the local '
        'checkpoints/sam_vit_b_01ec64.pth file; there are no official '
        'A2SAM weights. Prediction exports a 0/255 mask at exact GT '
        'resolution.',
        'The official model is built at a fixed 1024x1024 input and its '
        'postprocess uses a fixed original size per dataset; this port '
        'resizes the GT to the actual protocol image size and exports at '
        'that resolution.',
    )

    def __init__(self):
        super().__init__()
        self._stage = 'sam'
        self._stage2_ready = False
        self._prior_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._structural_cache: dict[str, tuple] = {}
        self._prior_optimizer = None
        self._sam_ckpt: str | None = None
        self._sam_sha: str | None = None
        self._cps_stats: dict[str, float] = {}
        self._bce = nn.BCEWithLogitsLoss()
        self._iou = IOU()
        self._ta = ThicknessAwareLoss()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument(
            '--a2sam-sam-checkpoint', type=Path,
            default=Path('checkpoints/sam_vit_b_01ec64.pth'))
        parser.add_argument('--a2sam-prior-epochs', type=int, default=100)
        parser.add_argument('--a2sam-prior-lr', type=float, default=1e-3)
        parser.add_argument('--a2sam-prior-beta', type=float, default=1e-5)
        parser.add_argument('--a2sam-prior-base', type=int, default=32)
        parser.add_argument('--a2sam-latent-dim', type=int, default=16)
        parser.add_argument('--a2sam-prior-samples', type=int, default=5)
        parser.add_argument('--a2sam-keep-ratio', type=float, default=0.035)
        parser.add_argument('--a2sam-cps-alpha', type=float, default=0.4)
        parser.add_argument(
            '--a2sam-cps-batches', type=int, default=0,
            help='training images used for CPS statistics (0 = all 18)')
        parser.add_argument(
            '--a2sam-thickness-weight', type=float, default=0.7)
        parser.add_argument('--a2sam-spg-points', type=int, default=512)
        parser.add_argument('--a2sam-spg-stride', type=int, default=3)
        parser.add_argument('--a2sam-spg-hidden', type=int, default=32)
        parser.add_argument('--a2sam-persistence', type=float, default=5.0)
        parser.add_argument(
            '--a2sam-relative-persistence', type=float, default=0.1)
        parser.add_argument('--a2sam-lr-min', type=float, default=1e-7)
        parser.add_argument('--a2sam-random-init', action='store_true',
                            help='smoke only: skip the SAM ViT-B checkpoint')
        parser.add_argument('--a2sam-no-augment', action='store_true')

    def _reset_run_state(self) -> None:
        self._stage = 'sam'
        self._stage2_ready = False
        self._prior_cache = {}
        self._structural_cache = {}
        self._prior_optimizer = None
        self._cps_stats = {}

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 1024))
        if self.input_size != 1024:
            raise SystemExit('A2SAM uses the official 1024x1024 input')
        prior_epochs = int(getattr(args, 'a2sam_prior_epochs', 100))
        epochs = int(getattr(args, 'epochs', 800))
        if not self.prediction_mode and prior_epochs >= epochs:
            raise SystemExit(
                '--a2sam-prior-epochs must be smaller than --epochs')
        self._reset_run_state()
        return build_a2sam(
            image_size=1024,
            prompt_size=256,
            prior_base=int(getattr(args, 'a2sam_prior_base', 32)),
            latent_dim=int(getattr(args, 'a2sam_latent_dim', 16)),
            spg_hidden=int(getattr(args, 'a2sam_spg_hidden', 32)),
            spg_points=int(getattr(args, 'a2sam_spg_points', 512)),
            spg_stride=int(getattr(args, 'a2sam_spg_stride', 3)),
            persistence=float(getattr(args, 'a2sam_persistence', 5.0)),
            relative_persistence=float(getattr(
                args, 'a2sam_relative_persistence', 0.1)),
        )

    def load_pretrained(self, model, args) -> None:
        if self.prediction_mode:
            return
        if bool(getattr(args, 'a2sam_random_init', False)):
            print('A2SAM smoke: random SAM ViT-B initialization')
            return
        checkpoint = Path(getattr(
            args, 'a2sam_sam_checkpoint',
            'checkpoints/sam_vit_b_01ec64.pth'))
        counts = model.load_sam_checkpoint(checkpoint)
        self._sam_ckpt = str(checkpoint)
        self._sam_sha = file_sha256(checkpoint)
        print(f'loaded SAM ViT-B weights: {counts}')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        groups = {
            'prior': 0, 'image_encoder': 0, 'prompt_encoder': 0,
            'mask_decoder': 0, 'spg': 0,
        }
        for name, parameter in model.named_parameters():
            for prefix in groups:
                if name.startswith(prefix + '.'):
                    groups[prefix] += parameter.numel()
                    break
        print('A2SAM parameters: ' + ', '.join(
            f'{key}={value / 1e6:.1f}M' for key, value in groups.items()))

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        stage_epochs = max(
            1, int(args.epochs) - int(args.a2sam_prior_epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs,
            eta_min=float(getattr(args, 'a2sam_lr_min', 1e-7)))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_image(self, sample: Sample, size: int) -> torch.Tensor:
        image = load_rgb(sample.image_path)
        image = TF.resize(
            image, [size, size],
            interpolation=TF.InterpolationMode.BILINEAR)
        return TF.normalize(
            TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

    def _load_mask(self, sample: Sample, size: int | None = None) -> torch.Tensor:
        mask = load_mask(sample.mask_path)
        if size is not None:
            image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            image = TF.resize(
                image, [size, size],
                interpolation=TF.InterpolationMode.NEAREST)
            mask = np.asarray(image) > 127
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    def _prepare_batch(self, samples: Sequence[Sample],
                       rng: random.Random, stage: str,
                       augment: bool = True):
        images, prior_images, gts, prior_gts, probs, uncs = (
            [], [], [], [], [], [])
        structural_batch = []
        no_augment = bool(getattr(self.args, 'a2sam_no_augment', False))
        flip = bool(
            augment and not no_augment and len(samples) and rng.random() < 0.5)
        for sample in samples:
            image = self._load_image(sample, 1024)
            prior_image = self._load_image(sample, 512)
            gt = self._load_mask(sample)
            gt_prior = self._load_mask(sample, 512)
            if flip:
                image = torch.flip(image, dims=[-1]).contiguous()
                prior_image = torch.flip(
                    prior_image, dims=[-1]).contiguous()
                gt = torch.flip(gt, dims=[-1]).contiguous()
                gt_prior = torch.flip(gt_prior, dims=[-1]).contiguous()
            images.append(image)
            prior_images.append(prior_image)
            gts.append(gt)
            prior_gts.append(gt_prior)
            if stage == 'sam':
                cached = self._prior_cache.get(sample.stem)
                if cached is None:
                    raise RuntimeError(
                        f'A2SAM stage 2 has no prior cache for '
                        f'{sample.stem}; call _prepare_stage2 first')
                probability, uncertainty = cached
                if flip:
                    probability = torch.flip(
                        probability, dims=[-1]).contiguous()
                    uncertainty = torch.flip(
                        uncertainty, dims=[-1]).contiguous()
                probs.append(probability)
                uncs.append(uncertainty)
                cached_structural = self._structural_cache.get(sample.stem)
                if cached_structural is not None:
                    structural_batch.append(
                        cached_structural[1 if flip else 0])
        batch = {
            'image': torch.stack(images).to(self.device),
            'prior_image': torch.stack(prior_images).to(self.device),
            'gt': torch.stack(gts).to(self.device),
            'gt_prior': torch.stack(prior_gts).to(self.device),
        }
        if stage == 'sam':
            batch['prob'] = torch.stack(probs).to(self.device)
            batch['uncertainty'] = torch.stack(uncs).to(self.device)
            if len(structural_batch) == len(samples):
                batch['structural'] = structural_batch
        return batch

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        return self._prepare_batch(samples, rng, self._stage)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    @staticmethod
    def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = logits.sigmoid()
        intersection = (probability * target).sum()
        return 1 - (2 * intersection + 1.0) / (
            probability.sum() + target.sum() + 1.0)

    def _prior_loss(self, model, batch):
        logits, mu, logvar = model.prior(
            batch['prior_image'], batch['gt_prior'])
        beta = float(getattr(self.args, 'a2sam_prior_beta', 1e-5))
        return (self._bce(logits, batch['gt_prior'])
                + self._dice_loss(logits, batch['gt_prior'])
                + beta * model.prior.kl_divergence(mu, logvar))

    def _sam_loss(self, model, batch):
        original_size = (batch['gt'].shape[-2], batch['gt'].shape[-1])
        structural = batch.get('structural')
        if structural is None:
            structural = model.spg.prepare(
                batch['prob'], batch['uncertainty'])
        masks, _, _ = model(
            batch['image'], batch['prob'], batch['uncertainty'],
            original_size, structural=structural)
        weight = float(getattr(self.args, 'a2sam_thickness_weight', 0.7))
        return (self._bce(masks, batch['gt'])
                + self._iou(masks, batch['gt'])
                + weight * self._ta(masks, batch['gt']))

    def train_step(self, model, batch):
        if self._stage == 'prior':
            return self._prior_loss(model, batch)
        return self._sam_loss(model, batch)

    # ------------------------------------------------------------------
    # Two-stage training
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prior_outputs(self, prior, image, cpu: bool = False):
        k = int(getattr(self.args, 'a2sam_prior_samples', 5))
        samples = prior.sample_prior(image, k=k)
        probabilities = torch.sigmoid(samples)
        mean = probabilities.mean(dim=0)
        uncertainty = (
            probabilities.var(dim=0, unbiased=False) * 4.0).clamp(0.0, 1.0)
        if cpu:
            return mean.cpu(), uncertainty.cpu()
        return mean, uncertainty

    def _prepare_stage2(self, model, loader, args) -> None:
        model.eval()
        model.prior.eval()
        for parameter in model.prior.parameters():
            parameter.requires_grad_(False)
        samples = list(loader.dataset.samples)
        self._prior_cache = {}
        self._structural_cache = {}
        for sample in samples:
            prior_image = self._load_image(
                sample, 512).unsqueeze(0).to(self.device)
            probability, uncertainty = self._prior_outputs(
                model.prior, prior_image, cpu=True)
            # Cache per-sample [1, H, W]; batch preparation stacks the
            # sample dimension back to [B, 1, H, W].
            self._prior_cache[sample.stem] = (
                probability[0].contiguous(), uncertainty[0].contiguous())
            flipped_probability = torch.flip(
                probability, dims=[-1]).contiguous()
            flipped_uncertainty = torch.flip(
                uncertainty, dims=[-1]).contiguous()
            self._structural_cache[sample.stem] = (
                model.spg.prepare(probability, uncertainty)[0],
                model.spg.prepare(
                    flipped_probability, flipped_uncertainty)[0],
            )

        # CPS statistics: accumulate gradients over the training split.
        model.train()
        model.spg.eval()
        limit = int(getattr(args, 'a2sam_cps_batches', 0)) or len(samples)
        for sample in samples[:limit]:
            batch = self._prepare_batch(
                [sample], random.Random(0), 'sam', augment=False)
            self._sam_loss(model, batch).backward()
        sam_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name.startswith(SAM_PREFIXES)
        }
        sam_grads = {
            name: parameter.grad.detach()
            for name, parameter in sam_parameters.items()
            if parameter.grad is not None
        }
        scores = contrastive_scores(
            sam_grads, sam_parameters,
            alpha=float(getattr(args, 'a2sam_cps_alpha', 0.4)))
        masks = select_contrastive_masks(
            scores, sam_parameters,
            keep_ratio=float(getattr(args, 'a2sam_keep_ratio', 0.035)))
        self._cps_stats = apply_cps_masks(model, masks)
        model.zero_grad(set_to_none=True)
        print(
            'A2SAM CPS: kept '
            f'{self._cps_stats["kept_elements"]:,}/'
            f'{self._cps_stats["total_elements"]:,} SAM elements '
            f'({self._cps_stats["kept_fraction"] * 100:.2f}%)')
        self._stage2_ready = True
        model.train()

    def _train_prior_epoch(self, model, loader, args) -> float:
        model.train()
        model.prior.train()
        if self._prior_optimizer is None:
            self._prior_optimizer = torch.optim.Adam(
                [p for p in model.prior.parameters() if p.requires_grad],
                lr=float(getattr(args, 'a2sam_prior_lr', 1e-3)))
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        running = 0.0
        seen = 0
        for samples in loader:
            batch = self._prepare_batch(samples, rng, 'prior')
            loss = self._prior_loss(model, batch)
            self._prior_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._prior_optimizer.step()
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        return running / max(seen, 1)

    def _train_sam_epoch(self, model, loader, optimizer, scheduler,
                         args) -> float:
        model.train()
        model.prior.eval()
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        accum = max(int(args.accum_steps), 1)
        pending = 0
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for samples in loader:
            batch = self._prepare_batch(samples, rng, 'sam')
            loss = self._sam_loss(model, batch)
            (loss / accum).backward()
            pending += 1
            if pending == accum:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        if pending:
            if pending < accum:
                scale = accum / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        return running / max(seen, 1)

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        prior_epochs = int(getattr(args, 'a2sam_prior_epochs', 100))
        epoch = int(getattr(self, '_epoch', 1))
        if epoch <= prior_epochs:
            self._stage = 'prior'
            return self._train_prior_epoch(model, loader, args)
        self._stage = 'sam'
        if not self._stage2_ready:
            self._prepare_stage2(model, loader, args)
        return self._train_sam_epoch(model, loader, optimizer, scheduler, args)

    def selection_score(self, metrics, per_dataset):
        if self._stage == 'prior':
            # Avoid selecting a random-SAM stage-1 checkpoint, but keep the
            # value finite so the history JSON remains strict.
            return -1e6
        return super().selection_score(metrics, per_dataset)

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        image = self._load_image(sample, 1024).unsqueeze(0).to(device)
        prior_image = self._load_image(sample, 512).unsqueeze(0).to(device)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image,
            'prior_image': prior_image,
            'size': (int(height), int(width)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        model.eval()
        model.prior.eval()
        probability, uncertainty = self._prior_outputs(
            model.prior, prepared['prior_image'])
        masks, _, _ = model(
            prepared['image'], probability, uncertainty, prepared['size'])
        return (masks.sigmoid()[0, 0].cpu().numpy() > 0.5)

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'prior_epochs': int(getattr(args, 'a2sam_prior_epochs', 100)),
            'prior_lr': float(getattr(args, 'a2sam_prior_lr', 1e-3)),
            'prior_beta': float(getattr(args, 'a2sam_prior_beta', 1e-5)),
            'prior_base': int(getattr(args, 'a2sam_prior_base', 32)),
            'latent_dim': int(getattr(args, 'a2sam_latent_dim', 16)),
            'prior_samples': int(getattr(args, 'a2sam_prior_samples', 5)),
            'keep_ratio': float(getattr(args, 'a2sam_keep_ratio', 0.035)),
            'cps_alpha': float(getattr(args, 'a2sam_cps_alpha', 0.4)),
            'cps_batches': int(getattr(args, 'a2sam_cps_batches', 0)),
            'thickness_weight': float(getattr(
                args, 'a2sam_thickness_weight', 0.7)),
            'spg_points': int(getattr(args, 'a2sam_spg_points', 512)),
            'spg_stride': int(getattr(args, 'a2sam_spg_stride', 3)),
            'spg_hidden': int(getattr(args, 'a2sam_spg_hidden', 32)),
            'persistence': float(getattr(args, 'a2sam_persistence', 5.0)),
            'relative_persistence': float(getattr(
                args, 'a2sam_relative_persistence', 0.1)),
            'sam_checkpoint': self._sam_ckpt,
            'sam_sha256': self._sam_sha,
            'augment': not bool(getattr(
                args, 'a2sam_no_augment', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        mappings = {
            'input_size': 'input_size',
            'prior_epochs': 'a2sam_prior_epochs',
            'prior_lr': 'a2sam_prior_lr',
            'prior_beta': 'a2sam_prior_beta',
            'prior_base': 'a2sam_prior_base',
            'latent_dim': 'a2sam_latent_dim',
            'prior_samples': 'a2sam_prior_samples',
            'keep_ratio': 'a2sam_keep_ratio',
            'cps_alpha': 'a2sam_cps_alpha',
            'cps_batches': 'a2sam_cps_batches',
            'thickness_weight': 'a2sam_thickness_weight',
            'spg_points': 'a2sam_spg_points',
            'spg_stride': 'a2sam_spg_stride',
            'spg_hidden': 'a2sam_spg_hidden',
            'persistence': 'a2sam_persistence',
            'relative_persistence': 'a2sam_relative_persistence',
        }
        for key, attribute in mappings.items():
            if key in config:
                setattr(args, attribute, config[key])
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if config.get('sam_checkpoint'):
            self._sam_ckpt = config['sam_checkpoint']
            args.a2sam_sam_checkpoint = Path(config['sam_checkpoint'])
        if 'sam_sha256' in config:
            self._sam_sha = config['sam_sha256']
        if 'augment' in config:
            args.a2sam_no_augment = not config['augment']
