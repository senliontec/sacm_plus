"""ACO-MoE-LoRA adapter (ICML 2026).

The released code trains a frozen SAM2 Hiera-L encoder with dynamic LoRA in
the attention projections and a four-level ACO-ConvLoRA decoder.  An ant-colony
search chooses LoRA rank/alpha and decoder top-k during training.  The official
script uses 352-square ImageNet-normalized inputs, weighted BCE + weighted IoU,
a gate/MoE auxiliary loss, AdamW 1e-3, cosine decay, 50 epochs and batch 12.

This adapter keeps the official model, loss, freeze policy and ACO search.  It
uses the shared 18/6 split, validation-Dice checkpoint selection, and a
4-microbatch x 3-accumulation approximation of the official batch 12.
"""

from __future__ import annotations

import contextlib
import copy
import io
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.aco_moe_lora import ACOMoELoRA, AdaptiveTrainingScheduler
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Official weighted BCE + weighted IoU (``train.py``)."""
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2, 3))
    union = ((prob + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.5) -> float:
    """Official training IoU (used for ACO pheromone feedback)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum(dim=(2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(2, 3))
    return float(((inter + 1e-6) / (union + 1e-6)).mean())


class ACOMoeLoRAAdapter(BaselineAdapter):
    id = 'aco_moe_lora'
    display_name = 'ACO-MoE-LoRA'
    family = 'sam'
    official_source = 'ACO-MoE-LoRA (ICML 2026; lky0234/ACO-MoE-LoRA)'
    official_entrypoint = 'https://github.com/lky0234/ACO-MoE-LoRA'
    input_size = 352
    official_lr = 1e-3
    official_weight_decay = 1e-4
    train_batch_size = 4
    accumulation_steps = 3
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'Official checkpoint selection is the minimum training main loss and '
        'the pheromone feedback uses training IoU; this port keeps the '
        'official training feedback but selects the checkpoint by validation '
        'Dice, as required by the shared protocol.',
        'Official batch size is 12; this port uses batch 4 with 3 gradient '
        'accumulation steps (effective batch 12) for 11 GiB local smoke and '
        'the shared runner.',
        'Official masks are resized with BICUBIC and kept soft for the '
        'weighted BCE/IoU loss; this is retained.',
        'The official 50-epoch budget is extended to the shared 800-epoch '
        'budget by default; the ACO search total_epochs tracks the actual '
        'budget so its 98% argmax phase is consistent.',
        'The released ACO repository has no LICENSE file; the vendored SAM2 '
        'Hiera files are Apache-2.0 (Meta SAM2).',
        'The official model constructs full SAM2 and deletes the unused '
        'prompt/mask decoder, memory modules and FPN neck. This port builds '
        'the same Hiera-L trunk directly; the used modules and weights are '
        'identical.',
        'At prediction the saved checkpoint contains the full model state; '
        'the base SAM2 file is only required during training.',
    )

    def __init__(self):
        super().__init__()
        self._sam2_sha: str | None = None
        self._last_hyperparams: dict | None = None
        self._aco_best_hyperparams: dict | None = None
        self._aco_scheduler: AdaptiveTrainingScheduler | None = None
        self._performance_history: list[float] = []
        self._epoch_performances: list[dict] = []

    # ------------------------------------------------------------------
    # CLI / construction
    # ------------------------------------------------------------------
    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=352)
        parser.add_argument(
            '--aco-sam2-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt')
        parser.add_argument('--aco-num-experts', type=int, default=8)
        parser.add_argument(
            '--aco-use-multi-start', action='store_true',
            help='enable the official MultiStartACOOptimizer')
        parser.add_argument('--aco-num-ants', type=int, default=20)
        parser.add_argument('--aco-diversification-interval', type=int,
                            default=50)
        parser.add_argument('--aco-loss-weight', type=float, default=1.0)
        parser.add_argument(
            '--aco-random-init', action='store_true',
            help='smoke only: random Hiera-L weights')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 352))
        if self.input_size != 352:
            raise SystemExit('ACO-MoE-LoRA uses the official 352x352 input')
        random_init = bool(getattr(args, 'aco_random_init', False))
        path = Path(getattr(
            args, 'aco_sam2_checkpoint',
            REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt'))
        weights = None
        if not random_init and not self.prediction_mode:
            if not path.is_file():
                raise SystemExit(
                    f'missing SAM2 Hiera-L checkpoint: {path}\n'
                    'place the official sam2_hiera_large.pt at '
                    'checkpoints/sam2_hiera_large.pt')
            weights = path
            self._sam2_sha = file_sha256(path)
        aco_config = {
            'rank_candidates': [4, 8, 16, 32],
            'alpha_candidates': [4.0, 8.0, 16.0, 32.0, 64.0],
            'top_k_candidates': [1, 2, 3, 4],
            'num_ants': int(getattr(args, 'aco_num_ants', 20)),
            'evaporation_rate': 0.3,
            'diversification_interval': int(getattr(
                args, 'aco_diversification_interval', 50)),
            'mutation_prob': 0.2,
            'total_epochs': max(1, int(getattr(args, 'epochs', 800))),
        }
        # The released ACO optimizers print their search decisions; keep the
        # shared one-row-per-epoch terminal table clean.
        with contextlib.redirect_stdout(io.StringIO()):
            model = ACOMoELoRA(
                checkpoint_path=weights,
                num_experts=int(getattr(args, 'aco_num_experts', 8)),
                use_multi_start=bool(getattr(
                    args, 'aco_use_multi_start', False)),
                aco_optimizer_config=aco_config,
            )
        return model

    def load_pretrained(self, model, args) -> None:
        if bool(getattr(args, 'aco_random_init', False)):
            print('ACO smoke: random Hiera-L initialization')
        elif self.prediction_mode:
            return
        else:
            print('loaded frozen SAM2 Hiera-L trunk from the local checkpoint')

    def configure_trainable(self, model) -> None:
        frozen_violations = [
            name for name, parameter in model.encoder.named_parameters()
            if parameter.requires_grad and '.lora.' not in name
        ]
        if frozen_violations:
            raise RuntimeError(
                'ACO encoder must be frozen except for LoRA: '
                f'{frozen_violations[:5]}')
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('ACO model has no trainable parameters')
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f'ACO parameters: {total:,} total, {trainable:,} trainable')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        return CosineAnnealingLR(
            optimizer, T_max=max(1, int(args.epochs)), eta_min=1e-6)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _resize_pair(self, sample: Sample, train: bool,
                     rng: random.Random | None):
        image = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path)
        mask_img = Image.fromarray(
            (mask.astype(np.uint8) * 255), mode='L')
        size = (self.input_size, self.input_size)
        image = image.resize(size, Image.BILINEAR)
        mask_img = mask_img.resize(size, Image.BICUBIC)
        if train and rng is not None:
            if rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / (
            np.asarray(IMAGENET_STD, dtype=np.float32))
        mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(
            image_np.transpose(2, 0, 1).copy()).float()
        # Official ToTensor + DataLoader produces [B, 1, H, W]; keep the
        # single foreground channel so the weighted BCE/IoU dims match.
        mask_t = torch.from_numpy(mask_np[None, :, :].copy()).float()
        return image_t, mask_t

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._resize_pair(sample, True, rng)
            images.append(image)
            masks.append(mask)
        return {
            'images': torch.stack(images).to(self.device),
            'masks': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        outputs, aco_loss = model(batch['images'], update_hyperparams=False)
        weight = float(getattr(self.args, 'aco_loss_weight', 1.0))
        return structure_loss(outputs, batch['masks']) + weight * aco_loss

    def prepare_eval_sample(self, sample: Sample, device):
        image, _ = self._resize_pair(sample, False, None)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits, _ = model(prepared['image'], update_hyperparams=False)
        width, height = prepared['size']
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return logits.sigmoid()[0, 0].cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Official ACO training loop
    # ------------------------------------------------------------------
    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        model.train()
        model.set_epoch(int(getattr(self, '_epoch', 0)))
        if self._aco_scheduler is None:
            self._aco_scheduler = AdaptiveTrainingScheduler(
                initial_aco_weight=float(
                    getattr(args, 'aco_loss_weight', 1.0)),
                cooldown_epochs=10)
        rng = random.Random(int(args.seed) + int(getattr(self, '_epoch', 0)))
        accum = max(int(args.accum_steps), 1)
        running = 0.0
        seen = 0
        pending = 0
        batch_ious: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for index, samples in enumerate(loader):
            batch = self.prepare_train_batch(samples, rng)
            with contextlib.redirect_stdout(io.StringIO()):
                outputs, aco_loss = model(
                    batch['images'], update_hyperparams=(index == 0))
            main_loss = structure_loss(outputs, batch['masks'])
            batch_iou = calculate_iou(outputs, batch['masks'])
            batch_ious.append(batch_iou)
            aco_weight = self._aco_scheduler.step(
                batch_iou, self._performance_history)
            total_loss = main_loss + aco_weight * aco_loss
            (total_loss / accum).backward()
            pending += 1
            if pending == accum or index == len(loader) - 1:
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
                pending = 0
            running += float(total_loss.detach()) * len(samples)
            seen += len(samples)

        mean_iou = sum(batch_ious) / max(len(batch_ious), 1)
        self._performance_history.append(mean_iou)
        if len(self._performance_history) > 20:
            self._performance_history.pop(0)
        snapshot = copy.deepcopy(model.get_current_hyperparams())
        self._last_hyperparams = snapshot
        self._epoch_performances.append({
            'epoch': int(getattr(self, '_epoch', 0)),
            'mean_iou': mean_iou,
            'hyperparams': snapshot,
        })
        solution = (snapshot['encoder'], snapshot['decoder'])
        with contextlib.redirect_stdout(io.StringIO()):
            model.aco_optimizer.update_pheromone([(solution, mean_iou)])
        best = model.aco_optimizer.get_best_hyperparams()
        if isinstance(best, tuple) and len(best) == 2:
            self._aco_best_hyperparams = {
                'encoder': best[0], 'decoder': best[1]}
        if scheduler is not None:
            scheduler.step()
        return running / max(seen, 1)

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def state_dict(self, model):
        hyperparams = self._last_hyperparams or model.get_current_hyperparams()
        return {
            'model': model.state_dict(),
            'aco_hyperparams': copy.deepcopy(hyperparams),
        }

    def load_state_dict(self, model, state) -> None:
        hyperparams = None
        if isinstance(state, dict) and isinstance(state.get('model'), dict):
            hyperparams = state.get('aco_hyperparams')
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        if isinstance(hyperparams, dict):
            encoder = hyperparams.get('encoder')
            decoder = hyperparams.get('decoder')
            if isinstance(encoder, dict):
                model._apply_encoder_hyperparams(encoder)
            if isinstance(decoder, dict):
                for layer in model.aco_layers:
                    layer.set_hyperparams(
                        rank=decoder['rank'], alpha=decoder['alpha'],
                        top_k=decoder['top_k'])
            model.current_encoder_params = encoder
            model.current_decoder_params = decoder
        model.set_deterministic_mode()

    def save_config(self, args):
        path = getattr(args, 'aco_sam2_checkpoint', None)
        sha = self._sam2_sha
        if sha is None and path is not None and Path(path).is_file():
            sha = file_sha256(Path(path))
        return {
            'input_size': int(self.input_size),
            'sam2_checkpoint': str(path) if path else None,
            'sam2_sha256': sha,
            'num_experts': int(getattr(args, 'aco_num_experts', 8)),
            'use_multi_start': bool(
                getattr(args, 'aco_use_multi_start', False)),
            'num_ants': int(getattr(args, 'aco_num_ants', 20)),
            'diversification_interval': int(getattr(
                args, 'aco_diversification_interval', 50)),
            'aco_loss_weight': float(
                getattr(args, 'aco_loss_weight', 1.0)),
            'random_init': bool(getattr(args, 'aco_random_init', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if config.get('sam2_checkpoint'):
            args.aco_sam2_checkpoint = Path(config['sam2_checkpoint'])
        if 'sam2_sha256' in config:
            self._sam2_sha = config['sam2_sha256']
        for key, attr in (
                ('num_experts', 'aco_num_experts'),
                ('use_multi_start', 'aco_use_multi_start'),
                ('num_ants', 'aco_num_ants'),
                ('diversification_interval',
                 'aco_diversification_interval'),
                ('aco_loss_weight', 'aco_loss_weight'),
                ('random_init', 'aco_random_init')):
            if key in config:
                setattr(args, attr, config[key])
