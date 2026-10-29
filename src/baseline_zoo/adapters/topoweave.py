"""TopoWeave adapter: a single-stage topology-aware curvilinear model.

TopoWeave is a project extension derived from MaskVSC (TMI 2025), not an
official MaskVSC port and not an official baseline.  The adapter uses the
shared single-stage runtime, so checkpointing, validation, W&B and prediction
are identical to the rest of the zoo.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.topoweave import (
    TopoWeaveLoss,
    TopoWeaveNet,
    augment_pair,
    topo_ratio,
    width_aware_topomask,
)
from baseline_zoo.runtime import BaselineAdapter


class TopoWeaveAdapter(BaselineAdapter):
    id = 'topoweave'
    display_name = 'TopoWeave'
    family = 'cnn'
    official_source = (
        'project extension derived from MaskVSC (TMI 2025); '
        'not an official baseline')
    official_entrypoint = 'https://github.com/Zhouyi-Zura/MaskVSC'
    input_size = 512
    official_lr = 2e-4
    official_weight_decay = 0.01
    train_batch_size = 4
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'TopoWeave is a project extension derived from MaskVSC (TMI 2025), '
        'not an official port and not an official baseline. No official '
        'weights exist for this variant and no parity with MaskVSC is claimed.',
        'Single-stage joint training replaces MaskVSC\'s frozen stage-1 + '
        'ViG two-stage pipeline. The gated graph refiner consumes image '
        'features and a detached coarse probability map, so it can use '
        'image evidence and receives gradients during the same training run.',
        'Connect_Loss is replaced by the verified bounded soft-clDice loss '
        'plus Dice and BCE. There is no morphological opening, no connected '
        'component count and no per-foreground rescaling; sparse domains '
        'therefore cannot receive the 10-100x weight explosion of paper-mode '
        'MaskVSC L_C.',
        'TopoMask is width-aware: contiguous skeleton paths are erased with '
        'a radius taken from the label distance transform, using an explicit '
        'random.Random and a biphasic ratio curriculum. MaskVSC used a fixed '
        '7x7 dilation, global RNG and a retinal-tuned schedule.',
        'The backbone uses GroupNorm and the new method uses strong geometry '
        'and intensity augmentation, AdamW and cosine warm-up. MaskVSC used '
        'BatchNorm, flip-only augmentation, fixed Adam lr and no scheduler.',
        'The refiner warm-up is a single-model training schedule, not a '
        'separate frozen stage; the final checkpoint is selected by the '
        'shared 6-image validation Dice.',
        'TopoWeave is trained on the pooled 18-image / 6-seed-domain split and '
        'evaluated with the shared 20-domain evaluator at original GT '
        'resolution. Input is 512x512 RGB [0,1]; predictions are resized '
        'back with nearest interpolation.',
    )

    def __init__(self):
        super().__init__()
        self._refiner = 'graph'
        self._graph_blocks = 4
        self._graph_dim = 128
        self._graph_k = 9
        self._graph_stride = 16
        self._base_width = 32
        self._refiner_warmup = 0
        self._mask_ratio = 0.35
        self._mask_jitter = 0.05
        self._dice_weight = 1.0
        self._topo_weight = 0.5
        self._coarse_weight = 0.4
        self._topo_iters = 10
        self._strong_augment = True
        self._criterion = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument(
            '--topoweave-refiner', choices=('graph', 'none'),
            default='graph')
        parser.add_argument(
            '--topoweave-graph-blocks', type=int, default=4)
        parser.add_argument(
            '--topoweave-graph-dim', type=int, default=128)
        parser.add_argument(
            '--topoweave-graph-k', type=int, default=9)
        parser.add_argument(
            '--topoweave-graph-stride', type=int, default=16)
        parser.add_argument(
            '--topoweave-base-width', type=int, default=32)
        parser.add_argument(
            '--topoweave-refiner-warmup', type=int, default=0)
        parser.add_argument(
            '--topoweave-mask-ratio', type=float, default=0.35)
        parser.add_argument(
            '--topoweave-mask-jitter', type=float, default=0.05)
        parser.add_argument(
            '--topoweave-loss-dice', type=float, default=1.0)
        parser.add_argument(
            '--topoweave-loss-topo', type=float, default=0.5)
        parser.add_argument(
            '--topoweave-coarse-weight', type=float, default=0.4)
        parser.add_argument(
            '--topoweave-topo-iters', type=int, default=10)
        parser.add_argument(
            '--topoweave-no-strong-augment', action='store_true',
            help='use flips only (MaskVSC-style ablation)')

    def _apply_arch(self, args):
        self._refiner = str(getattr(args, 'topoweave_refiner', 'graph'))
        if self._refiner not in ('graph', 'none'):
            raise SystemExit('--topoweave-refiner must be graph or none')
        self._graph_blocks = int(
            getattr(args, 'topoweave_graph_blocks', 4))
        self._graph_dim = int(getattr(args, 'topoweave_graph_dim', 128))
        self._graph_k = int(getattr(args, 'topoweave_graph_k', 9))
        self._graph_stride = int(
            getattr(args, 'topoweave_graph_stride', 16))
        self._base_width = int(
            getattr(args, 'topoweave_base_width', 32))
        self._refiner_warmup = int(
            getattr(args, 'topoweave_refiner_warmup', 0))
        self._mask_ratio = float(
            getattr(args, 'topoweave_mask_ratio', 0.35))
        self._mask_jitter = float(
            getattr(args, 'topoweave_mask_jitter', 0.05))
        self._dice_weight = float(
            getattr(args, 'topoweave_loss_dice', 1.0))
        self._topo_weight = float(
            getattr(args, 'topoweave_loss_topo', 0.5))
        self._coarse_weight = float(
            getattr(args, 'topoweave_coarse_weight', 0.4))
        self._topo_iters = int(
            getattr(args, 'topoweave_topo_iters', 10))
        self._strong_augment = not bool(
            getattr(args, 'topoweave_no_strong_augment', False))
        self.input_size = int(getattr(args, 'input_size', self.input_size))
        if self._graph_stride != 16:
            raise SystemExit(
                'TopoWeave graph tokens are taken at stride 16')
        if self.input_size % self._graph_stride:
            raise SystemExit(
                '--input-size must be divisible by the graph stride')
        if self._refiner_warmup < 0:
            raise SystemExit('--topoweave-refiner-warmup must be >= 0')
        epochs = int(getattr(args, 'epochs', 0) or 0)
        if epochs and self._refiner_warmup >= epochs:
            raise SystemExit(
                '--topoweave-refiner-warmup must be smaller than --epochs')
        if self._graph_blocks < 1 or self._graph_k < 1:
            raise SystemExit('TopoWeave graph blocks and k must be positive')
        if self._graph_dim % 4:
            raise SystemExit('--topoweave-graph-dim must be divisible by 4')
        if not 0 <= self._mask_ratio <= 0.5:
            raise SystemExit('--topoweave-mask-ratio must be in [0, 0.5]')

    def build_model(self, args):
        self._apply_arch(args)
        model = TopoWeaveNet(
            input_size=self.input_size,
            base_width=self._base_width,
            refiner=self._refiner,
            graph_blocks=self._graph_blocks,
            graph_dim=self._graph_dim,
            graph_k=self._graph_k,
            graph_stride=self._graph_stride,
        )
        self._criterion = TopoWeaveLoss(
            dice_weight=self._dice_weight,
            topo_weight=self._topo_weight,
            coarse_weight=self._coarse_weight,
            topo_iters=self._topo_iters,
        )
        return model

    def _prepare_pair(self, sample: Sample, rng: random.Random):
        image = load_rgb(sample.image_path)
        label = Image.fromarray(
            (load_mask(sample.mask_path).astype(np.uint8) * 255), mode='L')
        image, label = augment_pair(
            image, label, rng, strong=self._strong_augment)
        size = (self.input_size, self.input_size)
        image = image.resize(size, resample=Image.BILINEAR)
        label = label.resize(size, resample=Image.NEAREST)
        image_np = np.asarray(image, dtype=np.uint8)
        label_np = np.asarray(label) > 0
        return image_np, label_np

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        epoch = max(1, int(getattr(self, '_epoch', 1)))
        epochs = max(1, int(getattr(self.args, 'epochs', 800)))
        ratio = topo_ratio(
            epoch, epochs, peak=self._mask_ratio,
            jitter=self._mask_jitter, rng=rng)
        images, labels = [], []
        for sample in samples:
            image_np, label_np = self._prepare_pair(sample, rng)
            if ratio > 0:
                image_np = width_aware_topomask(
                    image_np, label_np, rng, ratio)
            image_t = torch.from_numpy(
                image_np.astype(np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            label_t = torch.from_numpy(
                label_np.astype(np.float32).copy()).unsqueeze(0)
            images.append(image_t)
            labels.append(label_t)
        return (torch.stack(images).to(self.device),
                torch.stack(labels).to(self.device))

    def train_step(self, model, batch):
        images, labels = batch
        if self._refiner == 'graph':
            epoch = int(getattr(self, '_epoch', 1))
            model.set_refiner_enabled(epoch > self._refiner_warmup)
        outputs = model(images, return_aux=True)
        if self._criterion is None:
            raise RuntimeError('TopoWeave criterion is not initialized')
        return self._criterion(outputs, labels)

    def checkpoint_eligible(self, epoch: int) -> bool:
        return (self._refiner != 'graph'
                or int(epoch) > self._refiner_warmup)

    def prepare_eval_sample(self, sample: Sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), resample=Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        if self._refiner == 'graph':
            enabled = (bool(getattr(self, 'prediction_mode', False))
                       or int(getattr(self, '_epoch', 0))
                       > self._refiner_warmup)
            model.set_refiner_enabled(enabled)
        with torch.no_grad():
            logits = model(prepared)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        binary = probability >= 0.5
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        binary = Image.fromarray(
            (binary.astype(np.uint8) * 255), mode='L').resize(
                (width, height), resample=Image.NEAREST)
        return np.asarray(binary) > 127

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'refiner': self._refiner,
            'graph_blocks': self._graph_blocks,
            'graph_dim': self._graph_dim,
            'graph_k': self._graph_k,
            'graph_stride': self._graph_stride,
            'base_width': self._base_width,
            'refiner_warmup': self._refiner_warmup,
            'mask_ratio': self._mask_ratio,
            'mask_jitter': self._mask_jitter,
            'loss_dice': self._dice_weight,
            'loss_topo': self._topo_weight,
            'coarse_weight': self._coarse_weight,
            'topo_iters': self._topo_iters,
            'strong_augment': self._strong_augment,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            setattr(args, 'input_size', self.input_size)
        self._refiner = str(config.get('refiner', 'graph'))
        self._graph_blocks = int(config.get('graph_blocks', 4))
        self._graph_dim = int(config.get('graph_dim', 128))
        self._graph_k = int(config.get('graph_k', 9))
        self._graph_stride = int(config.get('graph_stride', 16))
        self._base_width = int(config.get('base_width', 32))
        self._refiner_warmup = int(config.get('refiner_warmup', 0))
        self._mask_ratio = float(config.get('mask_ratio', 0.35))
        self._mask_jitter = float(config.get('mask_jitter', 0.05))
        self._dice_weight = float(config.get('loss_dice', 1.0))
        self._topo_weight = float(config.get('loss_topo', 0.5))
        self._coarse_weight = float(config.get('coarse_weight', 0.4))
        self._topo_iters = int(config.get('topo_iters', 10))
        self._strong_augment = bool(config.get('strong_augment', True))
        setattr(args, 'topoweave_refiner', self._refiner)
        setattr(args, 'topoweave_graph_blocks', self._graph_blocks)
        setattr(args, 'topoweave_graph_dim', self._graph_dim)
        setattr(args, 'topoweave_graph_k', self._graph_k)
        setattr(args, 'topoweave_graph_stride', self._graph_stride)
        setattr(args, 'topoweave_base_width', self._base_width)
        setattr(args, 'topoweave_refiner_warmup', self._refiner_warmup)
        setattr(args, 'topoweave_mask_ratio', self._mask_ratio)
        setattr(args, 'topoweave_mask_jitter', self._mask_jitter)
        setattr(args, 'topoweave_loss_dice', self._dice_weight)
        setattr(args, 'topoweave_loss_topo', self._topo_weight)
        setattr(args, 'topoweave_coarse_weight', self._coarse_weight)
        setattr(args, 'topoweave_topo_iters', self._topo_iters)
        setattr(args, 'topoweave_no_strong_augment',
                not self._strong_augment)
