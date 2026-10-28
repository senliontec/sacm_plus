"""FlowSDF adapter (IJCV 2025).

Image-guided conditional flow matching on truncated signed distance maps.
The official released code is followed for the velocity target, RRDB/UNet
architecture, Adam recipe, EMA, Euler sampling and the ``sdf <= 3e-2``
threshold.  The missing ``SimulationHelper`` submodule, TensorBoard and
``torchdiffeq`` are replaced by the shared runtime and a local Euler loop.
"""

from __future__ import annotations

import copy
import random
import zlib
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.flowsdf import (
    build_flowsdf,
    predict_tiled,
    predict_whole,
    signed_distance_map,
)
from baseline_zoo.runtime import BaselineAdapter


class FlowSDFAdapter(BaselineAdapter):
    id = 'flowsdf'
    display_name = 'FlowSDF'
    family = 'flow_matching'
    official_source = 'FlowSDF (IJCV 2025)'
    official_entrypoint = 'https://github.com/leabogensperger/FlowSDF'
    input_size = 128
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 4
    accumulation_steps = 1
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'The official data loader reads SDF targets precomputed by the '
        'companion generative-segmentation-sdf notebook. This port computes '
        'the exact Euclidean truncated SDF on the fly from each GT mask '
        '(delta=15, divided by delta).',
        'Training follows the official MoNuSeg recipe: random 128x128 crops '
        'and random horizontal/vertical flips. Images are read as RGB; the '
        'released OpenCV loader yields BGR, which is immaterial for '
        'from-scratch training.',
        'The official Vahadane colour normalization and 500x500 resize are '
        'performed by the companion preprocessing notebook and are not part '
        'of the released training code; this port uses the raw protocol '
        'images.',
        'Final prediction uses overlapping 128x128 tiles (stride 64) '
        'averaged in SDF space, following the MoNuSeg patch-averaging '
        'strategy; the released sampler only evaluates a single test batch. '
        'Validation resizes the whole image to 128 (the official GlaS path) '
        'with K=1 so the shared every-epoch validation is affordable; final '
        'prediction uses K=4 (MMSE, as in the paper).',
        'The official checkpoint is the lowest training loss. This port '
        'selects by validation Dice of the EMA weights (shared protocol).',
        'The released config uses batch 4 while the paper reports batch 16; '
        'this port keeps the released batch 4 on the 18-image split.',
        'The official 30k-40k epoch budget is replaced by the shared '
        '800-epoch budget.',
        'The official EMA decay is 0.999 in the released config and 0.9999 '
        'in the paper; this port defaults to 0.999 and exposes '
        '--flowsdf-ema-decay.',
        'The released sampler passes torch.linspace(0, 1, 4) to odeint '
        '(three Euler steps) although the config labels it "4 NFE"; this '
        'port runs four equal Euler steps as in Algorithm 2.',
        'The empty SimulationHelper submodule, TensorBoard writer and '
        'torchdiffeq dependency are not used; checkpointing/logging live in '
        'the shared runtime and Euler integration is local.',
        'The official sampler evaluates a four-sample test batch as four '
        'separate predictions; this port averages K samples per image '
        '(MMSE) as described in the paper.',
    )

    def __init__(self):
        super().__init__()
        self.ema_model: torch.nn.Module | None = None
        self._sdf_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=128)
        parser.add_argument('--flowsdf-model-channels', type=int, default=128)
        parser.add_argument('--flowsdf-rrdb-blocks', type=int, default=12)
        parser.add_argument('--flowsdf-sigma-min', type=float, default=1e-5)
        parser.add_argument('--flowsdf-ema-decay', type=float, default=0.999)
        parser.add_argument('--flowsdf-ode-steps', type=int, default=4)
        parser.add_argument('--flowsdf-k-samples', type=int, default=4)
        parser.add_argument('--flowsdf-val-k-samples', type=int, default=1)
        parser.add_argument('--flowsdf-sdf-delta', type=float, default=15.0)
        parser.add_argument('--flowsdf-threshold', type=float, default=0.03)
        parser.add_argument('--flowsdf-tile-stride', type=int, default=64)
        parser.add_argument('--flowsdf-tile-chunk', type=int, default=4)
        parser.add_argument('--flowsdf-no-augment', action='store_true')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 128))
        if self.input_size != 128:
            raise SystemExit('FlowSDF uses the official 128x128 input')
        return build_flowsdf(
            model_channels=int(getattr(args, 'flowsdf_model_channels', 128)),
            rrdb_blocks=int(getattr(args, 'flowsdf_rrdb_blocks', 12)),
            image_channels=3,
        )

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        print(f'FlowSDF parameters: {total:,} total, {total:,} trainable')

    def make_optimizer(self, model, args):
        # Official: Adam(lr=1e-4) with no weight decay.
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), weight_decay=float(args.weight_decay),
        )

    def make_scheduler(self, optimizer, args):
        # The official trainer has no scheduler.
        return None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _mask_sdf(self, sample: Sample, delta: float,
                  resized: tuple[int, int] | None) -> np.ndarray:
        mask = load_mask(sample.mask_path)
        if resized is not None:
            mask_img = Image.fromarray(
                (mask.astype(np.uint8) * 255), mode='L')
            mask_img = TF.resize(
                mask_img, resized,
                interpolation=TF.InterpolationMode.NEAREST)
            mask = np.asarray(mask_img) > 127
        key = (str(sample.mask_path), float(delta), mask.shape)
        cached = self._sdf_cache.get(key)
        if cached is None:
            cached = signed_distance_map(mask, delta=delta)
            self._sdf_cache[key] = cached
        return cached

    def _train_pair(self, sample: Sample, rng: random.Random):
        image = load_rgb(sample.image_path)
        width, height = image.size
        patch = self.input_size
        resized = None
        if height < patch or width < patch:
            scale = max(patch / height, patch / width)
            new_h = max(patch, int(round(height * scale)))
            new_w = max(patch, int(round(width * scale)))
            resized = (new_w, new_h)
            image = TF.resize(
                image, resized,
                interpolation=TF.InterpolationMode.BILINEAR)

        delta = float(getattr(self.args, 'flowsdf_sdf_delta', 15.0))
        sdf = self._mask_sdf(sample, delta, resized)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        height, width = image_np.shape[:2]

        top = rng.randrange(0, height - patch + 1)
        left = rng.randrange(0, width - patch + 1)
        image_crop = image_np[top:top + patch, left:left + patch]
        sdf_crop = sdf[top:top + patch, left:left + patch]

        if not bool(getattr(self.args, 'flowsdf_no_augment', False)):
            if rng.random() < 0.5:
                image_crop = image_crop[:, ::-1]
                sdf_crop = sdf_crop[:, ::-1]
            if rng.random() < 0.5:
                image_crop = image_crop[::-1]
                sdf_crop = sdf_crop[::-1]

        image_crop = np.ascontiguousarray(image_crop * 2.0 - 1.0)
        sdf_crop = np.ascontiguousarray(sdf_crop, dtype=np.float32)
        image_t = torch.from_numpy(image_crop).permute(2, 0, 1)
        sdf_t = torch.from_numpy(sdf_crop).unsqueeze(0)
        return image_t, sdf_t

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, sdfs = [], []
        for sample in samples:
            image, sdf = self._train_pair(sample, rng)
            images.append(image)
            sdfs.append(sdf)
        return {
            'image': torch.stack(images).to(self.device),
            'sdf': torch.stack(sdfs).to(self.device),
        }

    def train_step(self, model, batch):
        """Official conditional flow-matching velocity loss."""
        image = batch['image']
        sdf = batch['sdf']
        n = sdf.shape[0]
        sigma_min = float(getattr(self.args, 'flowsdf_sigma_min', 1e-5))
        t = torch.rand(n, device=sdf.device)
        eta = torch.randn_like(sdf)

        sigma_t = 1 - (1 - sigma_min) * t
        mu_t = t[:, None, None, None] * sdf
        m_t = mu_t + sigma_t[:, None, None, None] * eta
        denominator = 1 - (1 - sigma_min) * t[:, None, None, None]
        target = (sdf - (1 - sigma_min) * m_t) / denominator
        velocity = model(m_t, t.reshape(n, 1), img_cond=image)
        return ((velocity - target) ** 2).mean()

    # ------------------------------------------------------------------
    # EMA-aware training loop
    # ------------------------------------------------------------------
    def _ensure_ema(self, model) -> torch.nn.Module:
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        return self.ema_model

    @torch.no_grad()
    def _update_ema(self, model) -> None:
        ema = self.ema_model
        if ema is None:
            raise RuntimeError('EMA model not initialized')
        decay = float(getattr(self.args, 'flowsdf_ema_decay', 0.999))
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1 - decay)

    def _optimizer_step(self, model, optimizer, args) -> None:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip, norm_type=float('inf'))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._update_ema(model)

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        model.train()
        self._ensure_ema(model)
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        accum = max(int(args.accum_steps), 1)
        running = 0.0
        seen = 0
        pending = 0
        optimizer.zero_grad(set_to_none=True)
        for samples in loader:
            batch = self.prepare_train_batch(samples, rng)
            loss = self.train_step(model, batch)
            (loss / accum).backward()
            pending += 1
            if pending == accum:
                self._optimizer_step(model, optimizer, args)
                pending = 0
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        if pending:
            if pending < accum:
                scale = accum / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            self._optimizer_step(model, optimizer, args)
        if scheduler is not None:
            scheduler.step()
        return running / max(seen, 1)

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        image = load_rgb(sample.image_path)
        width, height = image.size
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.permute(2, 0, 1) * 2.0 - 1.0
        seed = (
            int(getattr(self.args, 'seed', 42)) * 1_000_003
            + (zlib.crc32(sample.stem.encode('utf-8')) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        return {
            'image': tensor.to(device),
            'size': (int(height), int(width)),
            'seed': seed,
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        net = self.ema_model if self.ema_model is not None else model
        net.eval()
        image = prepared['image']
        parameter = next(net.parameters())
        if parameter.device != image.device:
            net.to(image.device)
        k_samples = int(
            getattr(self.args, 'flowsdf_k_samples', 4)
            if self.prediction_mode
            else getattr(self.args, 'flowsdf_val_k_samples', 1)
        )
        common = dict(
            k_samples=max(1, k_samples),
            nfe=int(getattr(self.args, 'flowsdf_ode_steps', 4)),
            sigma_min=float(getattr(self.args, 'flowsdf_sigma_min', 1e-5)),
            seed=int(prepared['seed']),
        )
        if self.prediction_mode:
            sdf = predict_tiled(
                net,
                image,
                patch=self.input_size,
                stride=int(getattr(self.args, 'flowsdf_tile_stride', 64)),
                tile_chunk=int(getattr(
                    self.args, 'flowsdf_tile_chunk', 4)),
                **common,
            )
        else:
            # Official GlaS-style whole-image pass; keeps the shared
            # every-epoch validation affordable.
            sdf = predict_whole(
                net, image, size=self.input_size, **common)
        threshold = float(getattr(self.args, 'flowsdf_threshold', 0.03))
        return (sdf <= threshold).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def state_dict(self, model):
        ema = self._ensure_ema(model)
        return {'model': model.state_dict(), 'ema': ema.state_dict()}

    def load_state_dict(self, model, state):
        if 'model' in state and 'ema' in state:
            model.load_state_dict(state['model'])
            ema = copy.deepcopy(model)
            ema.load_state_dict(state['ema'])
            ema.requires_grad_(False)
            ema.eval()
            self.ema_model = ema.to(self.device)
        else:
            model.load_state_dict(state)
            self.ema_model = None

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'model_channels': int(getattr(
                args, 'flowsdf_model_channels', 128)),
            'rrdb_blocks': int(getattr(args, 'flowsdf_rrdb_blocks', 12)),
            'sigma_min': float(getattr(args, 'flowsdf_sigma_min', 1e-5)),
            'ema_decay': float(getattr(args, 'flowsdf_ema_decay', 0.999)),
            'ode_steps': int(getattr(args, 'flowsdf_ode_steps', 4)),
            'k_samples': int(getattr(args, 'flowsdf_k_samples', 4)),
            'val_k_samples': int(getattr(
                args, 'flowsdf_val_k_samples', 1)),
            'sdf_delta': float(getattr(args, 'flowsdf_sdf_delta', 15.0)),
            'threshold': float(getattr(args, 'flowsdf_threshold', 0.03)),
            'tile_stride': int(getattr(args, 'flowsdf_tile_stride', 64)),
            'tile_chunk': int(getattr(args, 'flowsdf_tile_chunk', 4)),
            'augment': not bool(getattr(
                args, 'flowsdf_no_augment', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        mappings = {
            'input_size': 'input_size',
            'model_channels': 'flowsdf_model_channels',
            'rrdb_blocks': 'flowsdf_rrdb_blocks',
            'sigma_min': 'flowsdf_sigma_min',
            'ema_decay': 'flowsdf_ema_decay',
            'ode_steps': 'flowsdf_ode_steps',
            'k_samples': 'flowsdf_k_samples',
            'val_k_samples': 'flowsdf_val_k_samples',
            'sdf_delta': 'flowsdf_sdf_delta',
            'threshold': 'flowsdf_threshold',
            'tile_stride': 'flowsdf_tile_stride',
            'tile_chunk': 'flowsdf_tile_chunk',
        }
        for key, attribute in mappings.items():
            if key in config:
                setattr(args, attribute, config[key])
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if 'augment' in config:
            args.flowsdf_no_augment = not config['augment']
