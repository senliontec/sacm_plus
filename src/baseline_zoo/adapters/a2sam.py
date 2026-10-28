"""A2SAM adapter (Medical Image Analysis 2026).

Three-stage shared-budget training:

* epochs 1..U: BCE-only coarse U-Net;
* epochs U+1..U+P: global persistence-threshold prior;
* remaining epochs: frozen U-Net/prior + SAM + official prompt fusion + CPS.

Official SAM/prompt/CPS/TA semantics are used where the release implements
them.  The U-Net, prior, persistence/F_MS and uncertainty map are paper-level
reconstructions because the release does not contain them.  See
``docs/A2SAM_FIDELITY.md``.
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.a2sam import (
    IOU,
    ThicknessAwareLoss,
    build_a2sam,
    build_morse_complex,
    contrastive_scores,
    dice_optimal_xi,
    f_ms,
    sam_cps_candidates,
    select_freeze_masks,
    uncertainty_from_samples,
    zero_cps_gradients,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SAM = REPO_ROOT / 'checkpoints/sam_vit_b_01ec64.pth'
TA_WEIGHT = 0.7


def _sample_seed(seed: int, stem: str, purpose: str = 'eval') -> int:
    payload = f'{purpose}:{stem}'.encode('utf-8')
    return (int(seed) * 1_000_003
            + (zlib.crc32(payload) & 0xFFFFFFFF)) & 0xFFFFFFFF


class A2SAMAdapter(BaselineAdapter):
    id = 'a2sam'
    display_name = 'A2SAM (MIA 2026)'
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
        'The official release has no U-Net, prior network, DMT/persistence '
        'sampling, uncertainty-map generator or full SPG. It reads '
        'precomputed uncertainty/logit .npy maps and only supplies the SAM '
        'modules, two-branch prompt fusion, CPS and TA loss.',
        'The coarse U-Net and global persistence-threshold prior are '
        'paper-level reconstructions. Persistence is exact cripser H0; the '
        'DMT gradient field and stable manifolds are reconstructed by a '
        'join-tree/watershed backend, not a bit-exact dipha DMT port.',
        'Shared 800-epoch budget: 100 U-Net + 100 prior + 600 A2SAM, batch 1, '
        'instead of the release 50/100 A2SAM epochs. Validation-Dice '
        'selection instead of the release validation S-measure.',
        'The U-Net uses BCE with foreground pos_weight=10 because the shared '
        '3-shot split is strongly foreground-imbalanced; plain BCE collapsed '
        'to all-background in the local 100-epoch diagnostic. Set '
        '--a2sam-unet-pos-weight 1 for the paper plain-BCE variant.',
        'CPS statistics use the 18 training images (paper Algorithm 1) '
        'rather than the release validation loader, avoiding label leakage. '
        'The official freeze-mask semantics (IoU head excluded, image-encoder '
        'norm/pos/neck frozen, zero gradients, no requires_grad=False) are '
        'used by default; a paper-scoring mode is available.',
        'The official SAM checkpoint allowlist excludes mask_tokens, '
        'output_hypernetworks_mlps and iou_prediction_head (31 tensors / '
        '692,868 parameters remain random), matching train.py load_from.',
        'Prompt fusion uses the official two-branch architecture. The '
        'release detaches its output; this port trains it by default '
        '(paper semantics) and exposes a detach parity switch. The default '
        'input mode is release-style (1-uncertainty, coarse logit); '
        'paper-style (u, y_c) is available as an ablation.',
        'The thickness loss defaults to paper Eq. 24-30 (log W, foreground '
        'denominator); a release mode reproduces the reciprocal/all-pixel '
        'behavior. AdamW is the default (paper/config); a release Adam mode '
        'is available.',
        'The prior is frozen for A2SAM training and sampled with per-sample '
        'deterministic generators. Prediction is reproducible from the '
        'checkpoint and --seed.',
        'There are no official A2SAM weights; only local SAM ViT-B weights '
        'are used. Predictions are 0/255 masks at exact GT resolution.',
    )

    def __init__(self):
        super().__init__()
        self._stage = 'unet'
        self._unet_frozen = False
        self._prior_frozen = False
        self._stage3_ready = False
        self._unet_cache: dict[str, dict] = {}
        self._stage3_cache: dict[str, dict] = {}
        self._eval_cache: dict[str, dict] = {}
        self._unet_optimizer = None
        self._prior_optimizer = None
        self._cps_masks: dict[str, torch.Tensor] = {}
        self._cps_stats: dict[str, float] = {}
        self._sam_ckpt: str | None = None
        self._sam_sha: str | None = None
        self._bce = nn.BCEWithLogitsLoss()
        self._iou = IOU()
        self._ta = ThicknessAwareLoss(mode='paper')

    # ------------------------------------------------------------------
    # CLI / construction
    # ------------------------------------------------------------------
    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument('--a2sam-sam-checkpoint', type=Path,
                            default=DEFAULT_SAM)
        parser.add_argument('--a2sam-unet-epochs', type=int, default=100)
        parser.add_argument('--a2sam-prior-epochs', type=int, default=100)
        parser.add_argument('--a2sam-unet-lr', type=float, default=1e-3)
        parser.add_argument('--a2sam-unet-pos-weight', type=float, default=10.0,
                            help='foreground weight for the U-Net BCE; 1 = plain BCE')
        parser.add_argument('--a2sam-prior-lr', type=float, default=1e-3)
        parser.add_argument('--a2sam-prior-samples', type=int, default=5)
        parser.add_argument('--a2sam-prior-min-sigma', type=float, default=0.05)
        parser.add_argument('--a2sam-keep-ratio', type=float, default=0.035)
        parser.add_argument('--a2sam-cps-alpha', type=float, default=0.4)
        parser.add_argument('--a2sam-cps-mode', choices=('official', 'paper'),
                            default='official')
        parser.add_argument('--a2sam-ta-mode', choices=('paper', 'release'),
                            default='paper')
        parser.add_argument('--a2sam-fusion-inputs',
                            choices=('release', 'paper'), default='release')
        parser.add_argument('--a2sam-prompt-detach', action='store_true')
        parser.add_argument('--a2sam-topology-backend',
                            choices=('auto', 'cripser', 'gudhi', 'join_tree'),
                            default='auto')
        parser.add_argument('--a2sam-coarse-threshold', type=float,
                            default=0.5)
        parser.add_argument('--a2sam-min-persistence', type=float,
                            default=1e-3)
        parser.add_argument('--a2sam-max-branches', type=int, default=256)
        parser.add_argument('--a2sam-prior-grid-size', type=int, default=32)
        parser.add_argument('--a2sam-optimizer', choices=('adamw', 'adam'),
                            default='adamw')
        parser.add_argument('--a2sam-lr-min', type=float, default=1e-7)
        parser.add_argument('--a2sam-release-parity', action='store_true',
                            help='set release detach/TA/inputs/Adam modes')
        parser.add_argument('--a2sam-random-init', action='store_true')
        parser.add_argument('--a2sam-no-augment', action='store_true')

    def _stage_epochs(self) -> tuple[int, int]:
        args = self.args
        unet = max(1, int(getattr(args, 'a2sam_unet_epochs', 100)))
        prior = max(1, int(getattr(args, 'a2sam_prior_epochs', 100)))
        total = int(getattr(args, 'epochs', 800))
        if unet + prior >= total:
            raise SystemExit(
                'A2SAM needs epochs > unet_epochs + prior_epochs')
        return unet, prior

    def _fusion_mode(self) -> str:
        if bool(getattr(self.args, 'a2sam_release_parity', False)):
            return 'release'
        return str(getattr(self.args, 'a2sam_fusion_inputs', 'release'))

    def _ta_mode(self) -> str:
        if bool(getattr(self.args, 'a2sam_release_parity', False)):
            return 'release'
        return str(getattr(self.args, 'a2sam_ta_mode', 'paper'))

    def _cps_mode(self) -> str:
        if bool(getattr(self.args, 'a2sam_release_parity', False)):
            return 'official'
        return str(getattr(self.args, 'a2sam_cps_mode', 'official'))

    def _prompt_detach(self) -> bool:
        return bool(getattr(self.args, 'a2sam_release_parity', False)
                    or getattr(self.args, 'a2sam_prompt_detach', False))

    def _optimizer_name(self) -> str:
        if bool(getattr(self.args, 'a2sam_release_parity', False)):
            return 'adam'
        return str(getattr(self.args, 'a2sam_optimizer', 'adamw'))

    def _reset_run_state(self):
        self._stage = 'unet'
        self._unet_frozen = False
        self._prior_frozen = False
        self._stage3_ready = False
        self._unet_cache = {}
        self._stage3_cache = {}
        self._eval_cache = {}
        self._unet_optimizer = None
        self._prior_optimizer = None
        self._cps_masks = {}
        self._cps_stats = {}

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 1024))
        if self.input_size != 1024:
            raise SystemExit('A2SAM uses the official 1024x1024 input')
        self._reset_run_state()
        self._ta = ThicknessAwareLoss(mode=self._ta_mode())
        self._unet_pos_weight = float(getattr(
            args, 'a2sam_unet_pos_weight', 10.0))
        if self._unet_pos_weight < 1.0:
            raise SystemExit('--a2sam-unet-pos-weight must be >= 1.0')
        if self.prediction_mode:
            # Only the eligible stage-3 checkpoint is saved as best_model.
            self._stage = 'a2sam'
        return build_a2sam(
            image_size=1024, prompt_size=256,
            prior_min_sigma=float(getattr(
                args, 'a2sam_prior_min_sigma', 0.05)))

    def load_pretrained(self, model, args) -> None:
        if self.prediction_mode:
            return
        if bool(getattr(args, 'a2sam_random_init', False)):
            print('A2SAM smoke: random SAM ViT-B initialization')
            return
        checkpoint = Path(getattr(args, 'a2sam_sam_checkpoint', DEFAULT_SAM))
        info = model.load_sam_checkpoint(checkpoint)
        self._sam_ckpt = str(checkpoint)
        self._sam_sha = file_sha256(checkpoint)
        random_params = sum(
            int(item['parameters']) for item in info['random_init'].values())
        random_keys = sum(
            len(item['keys']) for item in info['random_init'].values())
        print(
            'loaded SAM ViT-B weights: '
            f'{info["loaded"]}; random-init {random_keys} tensors / '
            f'{random_params:,} params')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        groups = {}
        for name, parameter in model.named_parameters():
            prefix = name.split('.', 1)[0]
            groups[prefix] = groups.get(prefix, 0) + parameter.numel()
        print('A2SAM parameters: ' + ', '.join(
            f'{key}={value / 1e6:.1f}M'
            for key, value in sorted(groups.items())))

    def make_optimizer(self, model, args):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if self._optimizer_name() == 'adam':
            return torch.optim.Adam(parameters, lr=float(args.lr))
        return torch.optim.AdamW(
            parameters, lr=float(args.lr),
            weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        unet_epochs, prior_epochs = self._stage_epochs()
        stage_epochs = int(args.epochs) - unet_epochs - prior_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage_epochs),
            eta_min=float(getattr(args, 'a2sam_lr_min', 1e-7)))

    # ------------------------------------------------------------------
    # Image/mask helpers
    # ------------------------------------------------------------------
    def _load_image(self, sample: Sample, size: int = 1024) -> torch.Tensor:
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

    def _flip_allowed(self) -> bool:
        return not bool(getattr(self.args, 'a2sam_no_augment', False))

    # ------------------------------------------------------------------
    # Stage 1: U-Net batches
    # ------------------------------------------------------------------
    def _prepare_unet_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, targets = [], []
        flip = self._flip_allowed() and rng.random() < 0.5
        for sample in samples:
            image = self._load_image(sample, 1024)
            target = self._load_mask(sample, 1024)
            if flip:
                image = torch.flip(image, dims=[-1]).contiguous()
                target = torch.flip(target, dims=[-1]).contiguous()
            images.append(image)
            targets.append(target)
        return {
            'image': torch.stack(images).to(self.device),
            'target': torch.stack(targets).to(self.device),
        }

    def _unet_loss(self, model, batch):
        logits = model.unet(batch['image'])
        target = batch['target']
        if self._unet_pos_weight == 1.0:
            return self._bce(logits, target)
        pos_weight = torch.tensor(
            [self._unet_pos_weight], device=target.device,
            dtype=target.dtype)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight)

    def _train_unet_epoch(self, model, loader, args) -> float:
        model.train()
        model.unet.train()
        if self._unet_optimizer is None:
            self._unet_optimizer = torch.optim.Adam(
                [p for p in model.unet.parameters() if p.requires_grad],
                lr=float(getattr(args, 'a2sam_unet_lr', 1e-3)))
        rng = random.Random(int(args.seed) + int(getattr(self, '_epoch', 0)))
        running = 0.0
        seen = 0
        for samples in loader:
            batch = self._prepare_unet_batch(samples, rng)
            loss = self._unet_loss(model, batch)
            self._unet_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._unet_optimizer.step()
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        return running / max(seen, 1)

    # ------------------------------------------------------------------
    # Stage 2: prior batches/cache
    # ------------------------------------------------------------------
    def _ensure_unet_cache(self, model, samples: Sequence[Sample]):
        if self._unet_cache:
            return
        was_training = model.training
        model.eval()
        model.unet.eval()
        backend = str(getattr(
            self.args, 'a2sam_topology_backend', 'cripser'))
        min_persistence = float(getattr(
            self.args, 'a2sam_min_persistence', 1e-3))
        max_branches = int(getattr(
            self.args, 'a2sam_max_branches', 256))
        coarse_threshold = float(getattr(
            self.args, 'a2sam_coarse_threshold', 0.5))
        grid_size = int(getattr(
            self.args, 'a2sam_prior_grid_size', 32))
        dice_values = []
        branch_values = []
        xi_values = []
        for sample in samples:
            image = self._load_image(sample, 1024).unsqueeze(0).to(self.device)
            target = self._load_mask(sample, 1024).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, probability, coarse = model.coarse(image)
            prob_np = probability[0, 0].detach().cpu().numpy()
            gt_np = target[0, 0].detach().cpu().numpy() > 0.5
            pred = coarse[0, 0].detach().cpu().numpy() > 0.5
            inter = np.logical_and(pred, gt_np).sum()
            denom = pred.sum() + gt_np.sum()
            dice_values.append(
                1.0 if denom == 0 else 2.0 * inter / denom)
            morse = build_morse_complex(
                prob_np, backend=backend,
                coarse_threshold=coarse_threshold,
                min_persistence=min_persistence,
                max_branches=max_branches)
            xi_star = dice_optimal_xi(morse, gt_np, grid_size=grid_size)
            branch_values.append(morse.n_branches)
            xi_values.append(float(xi_star))
            self._unet_cache[sample.stem] = {
                'logits': logits[0].detach().cpu(),
                'probability': probability[0].detach().cpu(),
                'morse': morse,
                'xi_star': float(xi_star),
            }
        if dice_values:
            print(
                'A2SAM prior cache: U-Net Dice='
                f'{float(np.mean(dice_values)):.4f}, branches='
                f'{float(np.mean(branch_values)):.1f}, xi='
                f'{float(np.mean(xi_values)):.4f}+/-'
                f'{float(np.std(xi_values)):.4f}')
        if was_training:
            model.train()

    def _prepare_prior_batch(self, samples: Sequence[Sample],
                             rng: random.Random):
        images, probs, targets = [], [], []
        flip = self._flip_allowed() and rng.random() < 0.5
        for sample in samples:
            cached = self._unet_cache.get(sample.stem)
            if cached is None:
                raise RuntimeError(
                    f'A2SAM prior cache missing for {sample.stem}')
            image = self._load_image(sample, 1024)
            probability = cached['probability']
            if flip:
                image = torch.flip(image, dims=[-1]).contiguous()
                probability = torch.flip(probability, dims=[-1]).contiguous()
            images.append(image)
            probs.append(probability)
            targets.append(float(cached['xi_star']))
        return {
            'image': torch.stack(images).to(self.device),
            'probability': torch.stack(probs).to(self.device),
            'target': torch.tensor(targets, device=self.device),
        }

    def _prior_loss(self, model, batch):
        mu, log_var = model.prior_stats(
            batch['image'], batch['probability'])
        return model.prior.nll_loss(mu, log_var, batch['target'])

    def _train_prior_epoch(self, model, loader, args) -> float:
        model.train()
        model.unet.eval()
        model.prior.train()
        if not self._unet_frozen:
            for parameter in model.unet.parameters():
                parameter.requires_grad_(False)
            self._unet_frozen = True
            self._ensure_unet_cache(model, list(loader.dataset.samples))
        if self._prior_optimizer is None:
            self._prior_optimizer = torch.optim.Adam(
                [p for p in model.prior.parameters() if p.requires_grad],
                lr=float(getattr(args, 'a2sam_prior_lr', 1e-3)))
        rng = random.Random(int(args.seed) + int(getattr(self, '_epoch', 0)))
        running = 0.0
        seen = 0
        for samples in loader:
            batch = self._prepare_prior_batch(samples, rng)
            loss = self._prior_loss(model, batch)
            self._prior_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._prior_optimizer.step()
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        return running / max(seen, 1)

    # ------------------------------------------------------------------
    # Stage 3: prompt cache / CPS / SAM training
    # ------------------------------------------------------------------
    def _sample_uncertainty(self, model, image, probability,
                            seed: int, morse=None):
        if morse is None:
            prob_np = probability[0, 0].detach().cpu().numpy()
            morse = build_morse_complex(
                prob_np,
                backend=str(getattr(
                    self.args, 'a2sam_topology_backend', 'cripser')),
                coarse_threshold=float(getattr(
                    self.args, 'a2sam_coarse_threshold', 0.5)),
                min_persistence=float(getattr(
                    self.args, 'a2sam_min_persistence', 1e-3)),
                max_branches=int(getattr(
                    self.args, 'a2sam_max_branches', 256)))
        mu, log_var = model.prior_stats(image, probability)
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        z = int(getattr(self.args, 'a2sam_prior_samples', 5))
        xi = model.prior.sample_xi(mu, log_var, k=z, generator=generator)
        candidates = np.stack([
            f_ms(morse, float(value)) for value in xi[:, 0].detach().cpu()
        ])
        uncertainty = uncertainty_from_samples(candidates)
        return torch.from_numpy(uncertainty).unsqueeze(0), morse, xi

    def _ensure_stage3_cache(self, model, samples: Sequence[Sample]):
        if self._stage3_cache:
            return
        model.eval()
        model.unet.eval()
        model.prior.eval()
        for parameter in model.prior.parameters():
            parameter.requires_grad_(False)
        self._prior_frozen = True
        uncertainty_means = []
        uncertainty_max = []
        branch_values = []
        for sample in samples:
            cached_unet = self._unet_cache.get(sample.stem)
            if cached_unet is None:
                raise RuntimeError(
                    f'A2SAM stage-3 cache missing for {sample.stem}')
            image = self._load_image(sample, 1024).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, probability, y_c = model.coarse(image)
                uncertainty, _, _ = self._sample_uncertainty(
                    model, image, probability,
                    _sample_seed(self.args.seed, sample.stem, 'cache'),
                    morse=cached_unet['morse'])
            uncertainty_means.append(float(uncertainty.mean()))
            uncertainty_max.append(float(uncertainty.max()))
            branch_values.append(cached_unet['morse'].n_branches)
            self._stage3_cache[sample.stem] = {
                'logits': logits[0].detach().cpu(),
                'y_c': y_c[0].float().detach().cpu(),
                'uncertainty': uncertainty[0].detach().cpu(),
                'gt': self._load_mask(sample, None),
            }
        if uncertainty_means:
            print(
                'A2SAM stage-3 cache: u mean/max='
                f'{float(np.mean(uncertainty_means)):.4f}/'
                f'{float(np.max(uncertainty_max)):.4f}, branches='
                f'{float(np.mean(branch_values)):.1f}')
        model.train()

    def _prepare_a2sam_batch(self, samples: Sequence[Sample],
                             rng: random.Random, augment: bool = True):
        images, uncertainties, coarse_maps, gts = [], [], [], []
        flip = augment and self._flip_allowed() and rng.random() < 0.5
        mode = self._fusion_mode()
        for sample in samples:
            cached = self._stage3_cache.get(sample.stem)
            if cached is None:
                raise RuntimeError(
                    f'A2SAM stage-3 cache missing for {sample.stem}')
            image = self._load_image(sample, 1024)
            uncertainty = cached['uncertainty']
            coarse = cached['logits'] if mode == 'release' else cached['y_c']
            gt = cached['gt']
            if flip:
                image = torch.flip(image, dims=[-1]).contiguous()
                uncertainty = torch.flip(
                    uncertainty, dims=[-1]).contiguous()
                coarse = torch.flip(coarse, dims=[-1]).contiguous()
                gt = torch.flip(gt, dims=[-1]).contiguous()
            images.append(image)
            uncertainties.append(uncertainty)
            coarse_maps.append(coarse)
            gts.append(gt)
        return {
            'image': torch.stack(images).to(self.device),
            'uncertainty': torch.stack(uncertainties).to(self.device),
            'coarse': torch.stack(coarse_maps).to(self.device),
            'gt': torch.stack(gts).to(self.device),
        }

    def _prompt_from_batch(self, model, batch):
        return model.make_prompt(
            batch['uncertainty'], batch['coarse'],
            mode=self._fusion_mode(), detach=self._prompt_detach())

    def _sam_loss(self, model, batch):
        prompt = self._prompt_from_batch(model, batch)
        original_size = (batch['gt'].shape[-2], batch['gt'].shape[-1])
        masks, _, _ = model(batch['image'], prompt, original_size)
        return (self._bce(masks, batch['gt'])
                + self._iou(masks, batch['gt'])
                + TA_WEIGHT * self._ta(masks, batch['gt']))

    def _apply_cps(self, model, samples: Sequence[Sample]):
        model.eval()
        model.zero_grad(set_to_none=True)
        for sample in samples:
            batch = self._prepare_a2sam_batch(
                [sample], random.Random(0), augment=False)
            self._sam_loss(model, batch).backward()
        candidates = sam_cps_candidates(model)
        grads = {
            name: parameter.grad.detach()
            for name, parameter in candidates.items()
            if parameter.grad is not None
        }
        scores = contrastive_scores(
            grads, candidates,
            alpha=float(getattr(self.args, 'a2sam_cps_alpha', 0.4)),
            mode=self._cps_mode())
        masks = select_freeze_masks(
            scores, candidates,
            keep_ratio=float(getattr(self.args, 'a2sam_keep_ratio', 0.035)),
            mode=self._cps_mode(),
            force_image_encoder=self._cps_mode() == 'official')
        self._cps_stats = zero_cps_gradients(model, masks)
        self._cps_masks = {
            name: mask.detach().bool().to(self.device)
            for name, mask in masks.items()
        }
        model.zero_grad(set_to_none=True)
        model.train()
        print(
            'A2SAM CPS: trainable '
            f'{self._cps_stats["trainable_elements"]:,}/'
            f'{self._cps_stats["total_elements"]:,} SAM elements '
            f'({self._cps_stats["trainable_fraction"] * 100:.2f}%)')

    def _train_a2sam_epoch(self, model, loader, optimizer, scheduler,
                           args) -> float:
        model.train()
        model.unet.eval()
        model.prior.eval()
        rng = random.Random(int(args.seed) + int(getattr(self, '_epoch', 0)))
        accum = max(int(args.accum_steps), 1)
        pending = 0
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for samples in loader:
            batch = self._prepare_a2sam_batch(samples, rng)
            loss = self._sam_loss(model, batch)
            (loss / accum).backward()
            zero_cps_gradients(model, self._cps_masks)
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

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------
    def _stage_for_epoch(self, epoch: int) -> str:
        unet_epochs, prior_epochs = self._stage_epochs()
        if epoch <= unet_epochs:
            return 'unet'
        if epoch <= unet_epochs + prior_epochs:
            return 'prior'
        return 'a2sam'

    def prepare_train_batch(self, samples, rng):
        if self._stage == 'unet':
            return self._prepare_unet_batch(samples, rng)
        if self._stage == 'prior':
            return self._prepare_prior_batch(samples, rng)
        return self._prepare_a2sam_batch(samples, rng)

    def train_step(self, model, batch):
        if self._stage == 'unet':
            return self._unet_loss(model, batch)
        if self._stage == 'prior':
            return self._prior_loss(model, batch)
        return self._sam_loss(model, batch)

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        epoch = int(getattr(self, '_epoch', 1))
        stage = self._stage_for_epoch(epoch)
        self._stage = stage
        if stage == 'unet':
            return self._train_unet_epoch(model, loader, args)
        if stage == 'prior':
            return self._train_prior_epoch(model, loader, args)
        samples = list(loader.dataset.samples)
        if not self._unet_frozen:
            for parameter in model.unet.parameters():
                parameter.requires_grad_(False)
            self._unet_frozen = True
        if not self._unet_cache:
            self._ensure_unet_cache(model, samples)
        if not self._stage3_ready:
            self._ensure_stage3_cache(model, samples)
            self._apply_cps(model, samples)
            self._stage3_ready = True
        return self._train_a2sam_epoch(
            model, loader, optimizer, scheduler, args)

    def checkpoint_eligible(self, epoch: int) -> bool:
        del epoch
        return self._stage == 'a2sam'

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def begin_prediction(self):
        super().begin_prediction()
        self._eval_cache = {}

    def prepare_eval_sample(self, sample: Sample, device):
        image = self._load_image(sample, 1024).unsqueeze(0).to(device)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image,
            'size': (int(height), int(width)),
            'seed': _sample_seed(
                getattr(self.args, 'seed', 42), sample.stem, 'predict'),
        }

    def _predict_coarse(self, model, prepared):
        with torch.no_grad():
            _, probability, _ = model.coarse(prepared['image'])
        probability = F.interpolate(
            probability, size=prepared['size'], mode='bilinear',
            align_corners=False)
        return (probability[0, 0] > 0.5).cpu().numpy()

    def _predict_a2sam(self, model, prepared, sample):
        cached = None if self.prediction_mode else self._eval_cache.get(
            sample.stem)
        if cached is None:
            image = prepared['image']
            with torch.no_grad():
                logits, probability, y_c = model.coarse(image)
                uncertainty, _, _ = self._sample_uncertainty(
                    model, image, probability, prepared['seed'])
            cached = {
                'logits': logits[0].detach().cpu(),
                'y_c': y_c[0].float().detach().cpu(),
                'uncertainty': uncertainty[0].detach().cpu(),
            }
            if not self.prediction_mode:
                self._eval_cache[sample.stem] = cached
        mode = self._fusion_mode()
        coarse = cached['logits'] if mode == 'release' else cached['y_c']
        with torch.no_grad():
            prompt = model.make_prompt(
                cached['uncertainty'].unsqueeze(0).to(self.device),
                coarse.unsqueeze(0).to(self.device),
                mode=mode, detach=False)
            masks, _, _ = model(
                prepared['image'], prompt, prepared['size'])
        return (masks.sigmoid()[0, 0] > 0.5).cpu().numpy()

    def predict_sample(self, model, prepared, sample):
        model.eval()
        if self._stage != 'a2sam':
            return self._predict_coarse(model, prepared)
        return self._predict_a2sam(model, prepared, sample)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'unet_epochs': int(getattr(args, 'a2sam_unet_epochs', 100)),
            'prior_epochs': int(getattr(args, 'a2sam_prior_epochs', 100)),
            'unet_lr': float(getattr(args, 'a2sam_unet_lr', 1e-3)),
            'unet_pos_weight': float(getattr(
                args, 'a2sam_unet_pos_weight', 10.0)),
            'prior_lr': float(getattr(args, 'a2sam_prior_lr', 1e-3)),
            'prior_samples': int(getattr(args, 'a2sam_prior_samples', 5)),
            'prior_min_sigma': float(getattr(
                args, 'a2sam_prior_min_sigma', 0.05)),
            'keep_ratio': float(getattr(args, 'a2sam_keep_ratio', 0.035)),
            'cps_alpha': float(getattr(args, 'a2sam_cps_alpha', 0.4)),
            'cps_mode': str(getattr(args, 'a2sam_cps_mode', 'official')),
            'ta_mode': str(getattr(args, 'a2sam_ta_mode', 'paper')),
            'fusion_inputs': str(getattr(
                args, 'a2sam_fusion_inputs', 'release')),
            'prompt_detach': bool(getattr(
                args, 'a2sam_prompt_detach', False)),
            'topology_backend': str(getattr(
                args, 'a2sam_topology_backend', 'cripser')),
            'coarse_threshold': float(getattr(
                args, 'a2sam_coarse_threshold', 0.5)),
            'min_persistence': float(getattr(
                args, 'a2sam_min_persistence', 1e-3)),
            'max_branches': int(getattr(
                args, 'a2sam_max_branches', 256)),
            'prior_grid_size': int(getattr(
                args, 'a2sam_prior_grid_size', 32)),
            'optimizer_mode': str(getattr(
                args, 'a2sam_optimizer', 'adamw')),
            'release_parity': bool(getattr(
                args, 'a2sam_release_parity', False)),
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
            'unet_epochs': 'a2sam_unet_epochs',
            'prior_epochs': 'a2sam_prior_epochs',
            'unet_lr': 'a2sam_unet_lr',
            'unet_pos_weight': 'a2sam_unet_pos_weight',
            'prior_lr': 'a2sam_prior_lr',
            'prior_samples': 'a2sam_prior_samples',
            'prior_min_sigma': 'a2sam_prior_min_sigma',
            'keep_ratio': 'a2sam_keep_ratio',
            'cps_alpha': 'a2sam_cps_alpha',
            'cps_mode': 'a2sam_cps_mode',
            'ta_mode': 'a2sam_ta_mode',
            'fusion_inputs': 'a2sam_fusion_inputs',
            'topology_backend': 'a2sam_topology_backend',
            'coarse_threshold': 'a2sam_coarse_threshold',
            'min_persistence': 'a2sam_min_persistence',
            'max_branches': 'a2sam_max_branches',
            'prior_grid_size': 'a2sam_prior_grid_size',
            'optimizer_mode': 'a2sam_optimizer',
        }
        for key, attribute in mappings.items():
            if key in config:
                setattr(args, attribute, config[key])
        if 'prompt_detach' in config:
            args.a2sam_prompt_detach = bool(config['prompt_detach'])
        if 'release_parity' in config:
            args.a2sam_release_parity = bool(config['release_parity'])
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if config.get('sam_checkpoint'):
            self._sam_ckpt = config['sam_checkpoint']
            args.a2sam_sam_checkpoint = Path(config['sam_checkpoint'])
        if 'sam_sha256' in config:
            self._sam_sha = config['sam_sha256']
        if 'augment' in config:
            args.a2sam_no_augment = not config['augment']
