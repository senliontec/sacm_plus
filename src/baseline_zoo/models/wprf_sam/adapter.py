"""CoSNA: Consensus-Guided Snake Reconstruction for curvilinear segmentation.

CoSNA is a self-contained inference/training adapter around a physically
copied BCTD-SAM scaffold.  It combines:

* a label-free gated candidate-consensus mask over the four Stage-2 heads;
* a TCSU-lite snake-upsampling residual on candidate 3, trained only on the
  18 source images while the scaffold stays frozen;
* a constrained near-union with a resolution-adaptive added-component
  filter.

Everything lives under ``baseline_zoo.models.wprf_sam``.  The package does not
import the experimental methods tree, the BCTD baseline package, or any
other model package at runtime.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, disk

from baseline_zoo.data import load_rgb
from baseline_zoo.runtime import main_for

from .base import CoSNAScaffoldAdapter
from .adaptive_skel import apply_adaptive_skel
from .boundary_loss import BoundaryLoss
from .decoding import route_masks
from .topology_losses import SoftclDiceLoss
from .tcsu import SnakeUpsampler


def _starts(total: int, tile: int, stride: int) -> list[int]:
    if total <= tile:
        return [0]
    starts = list(range(0, max(1, total - tile + 1), stride))
    last = max(0, total - tile)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _prep(image: Image.Image, size: int) -> torch.Tensor:
    array = np.asarray(
        image.resize((size, size), Image.BILINEAR), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0


def _gated_mask(probs: np.ndarray, gate: float) -> np.ndarray:
    h3 = probs[3] > 0.5
    return np.where(
        probs.std(0) > gate, (probs > 0.5).all(0), h3)


class CoSNAAdapter(CoSNAScaffoldAdapter):
    """Independent CoSNA adapter with optional TCSU-head training."""

    id = 'cosna'
    display_name = 'CoSNA'
    official_source = (
        'CoSNA: consensus-guided snake reconstruction on a self-contained '
        'BCTD-SAM scaffold')
    official_entrypoint = (
        'Gated candidate consensus + TCSU-lite snake reconstruction + '
        'constrained near-union')
    deviations = tuple(CoSNAScaffoldAdapter.deviations) + (
        'In canonical source mode the scaffold is frozen and only the '
        '18k-parameter TCSU residual is fitted; --cosna-train-scope full '
        'trains scaffold adapters/decoder and TCSU jointly from scratch.',
        'Inference combines a gated candidate-consensus mask with a TCSU-h3 '
        'mask; TCSU pixels are added only within 4 px of the gated mask.',
        'Added components smaller than 80 px are removed; on images with max '
        'dimension >=1200, elongated added components with bbox aspect >=2 '
        'are retained. The threshold is label-free but benchmark-selected.',
    )

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--cosna-train-scope', choices=['tcsu', 'full'], default='tcsu',
            help='tcsu=frozen scaffold + residual (canonical); '
                 'full=train scaffold adapters/decoder and TCSU jointly')
        parser.add_argument('--cosna-weight', type=float, default=1.0)
        parser.add_argument('--cosna-cl', type=float, default=0.1)
        parser.add_argument('--cosna-boundary', type=float, default=0.0)
        parser.add_argument('--cosna-boundary-resolution', type=int,
                            default=512)
        parser.add_argument('--cosna-render', type=int, default=1024)
        parser.add_argument('--cosna-coarse', type=int, default=512)
        parser.add_argument('--cosna-feature-dim', type=int, default=16)
        parser.add_argument('--cosna-gate', type=float, default=0.10)
        parser.add_argument('--cosna-near-radius', type=int, default=4)
        parser.add_argument('--cosna-tcsu-threshold', type=float, default=0.5)
        parser.add_argument('--cosna-filter-min-size', type=int, default=80)
        parser.add_argument('--cosna-filter-aspect', type=float, default=2.0)
        parser.add_argument('--cosna-filter-maxdim', type=int, default=1200)
        parser.add_argument('--cosna-save-last', type=str, default='')

    def build_model(self, args):
        model = super().build_model(args)
        self.cosna_train_scope = str(
            getattr(args, 'cosna_train_scope', 'tcsu'))
        self.cosna_weight = float(getattr(args, 'cosna_weight', 1.0))
        self.cosna_cl = float(getattr(args, 'cosna_cl', 0.1))
        self.cosna_boundary = float(
            getattr(args, 'cosna_boundary', 0.0))
        self.cosna_boundary_resolution = int(
            getattr(args, 'cosna_boundary_resolution', 512))
        self.cosna_gate = float(getattr(args, 'cosna_gate', 0.10))
        self.cosna_near_radius = int(
            getattr(args, 'cosna_near_radius', 4))
        self.cosna_tcsu_threshold = float(
            getattr(args, 'cosna_tcsu_threshold', 0.5))
        self.cosna_filter_min_size = int(
            getattr(args, 'cosna_filter_min_size', 80))
        self.cosna_filter_aspect = float(
            getattr(args, 'cosna_filter_aspect', 2.0))
        self.cosna_filter_maxdim = int(
            getattr(args, 'cosna_filter_maxdim', 1200))
        self.cosna_save_last = str(
            getattr(args, 'cosna_save_last', '') or '')
        model.tcsu = SnakeUpsampler(
            render_size=int(getattr(args, 'cosna_render', 1024)),
            coarse_size=int(getattr(args, 'cosna_coarse', 512)),
            feature_dim=int(getattr(args, 'cosna_feature_dim', 16)))
        self._cosna_cl_loss = SoftclDiceLoss(num_iter=5)
        self._cosna_boundary_loss = BoundaryLoss(
            resolution=self.cosna_boundary_resolution)
        self._cosna_stage2 = None
        return model

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'cosna_train_scope': self.cosna_train_scope,
            'cosna_weight': self.cosna_weight,
            'cosna_cl': self.cosna_cl,
            'cosna_boundary': self.cosna_boundary,
            'cosna_boundary_resolution': self.cosna_boundary_resolution,
            'cosna_gate': self.cosna_gate,
            'cosna_near_radius': self.cosna_near_radius,
            'cosna_tcsu_threshold': self.cosna_tcsu_threshold,
            'cosna_filter_min_size': self.cosna_filter_min_size,
            'cosna_filter_aspect': self.cosna_filter_aspect,
            'cosna_filter_maxdim': self.cosna_filter_maxdim,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        for key in ('cosna_train_scope', 'cosna_weight', 'cosna_cl',
                    'cosna_boundary', 'cosna_boundary_resolution',
                    'cosna_gate',
                    'cosna_near_radius', 'cosna_tcsu_threshold',
                    'cosna_filter_min_size', 'cosna_filter_aspect',
                    'cosna_filter_maxdim'):
            if key in config:
                setattr(args, key, config[key])

    def load_state_dict(self, model, state):
        own = model.state_dict()
        filtered = {
            key: value for key, value in state.items()
            if key in own and own[key].shape == value.shape}
        model.load_state_dict(filtered, strict=False)

    def configure_trainable(self, model):
        if self.cosna_train_scope == 'full':
            # From-scratch CoSNA: train the scaffold's adapter/decoder
            # parameters with the standard four-candidate policy, and the
            # TCSU residual jointly.  No BCTD checkpoint is required.
            super().configure_trainable(model)
            for parameter in model.tcsu.parameters():
                parameter.requires_grad_(True)
            return
        # Canonical CoSNA: the scaffold is a frozen source model; fit only
        # the reconstruction residual on the 18 source images.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.tcsu.parameters():
            parameter.requires_grad_(True)
        model.image_encoder.grad_checkpointing = False

    def _tcsu_loss(self, model, images, masks, stage2):
        high = model.tcsu(stage2, images)
        masks = F.interpolate(
            masks, size=(high.shape[-2], high.shape[-1]),
            mode='bilinear', align_corners=False)
        bce = float(self.args.bce_weight)
        dice = float(self.args.dice_weight)
        loss = self._mask_criterion(high, masks[:, :1], bce, dice)
        if self.cosna_cl > 0:
            target = F.interpolate(
                masks[:, :1], size=(512, 512), mode='nearest')
            logits = F.interpolate(
                high, size=(512, 512), mode='bilinear',
                align_corners=False)
            loss = loss + self.cosna_cl * self._cosna_cl_loss(
                logits, target)
        if self.cosna_boundary > 0:
            boundary_target = F.interpolate(
                masks[:, :1], size=(high.shape[-2], high.shape[-1]),
                mode='bilinear', align_corners=False)
            loss = loss + self.cosna_boundary * self._cosna_boundary_loss(
                high, boundary_target)
        return loss

    def train_step(self, model, batch):
        images, masks = batch
        if self.cosna_train_scope == 'full':
            base_loss = super().train_step(model, batch)
            stage2 = getattr(self, '_last_stage2_native', None)
            if stage2 is None:
                raise RuntimeError(
                    'full CoSNA training did not receive a Stage-2 tensor')
            return base_loss + self.cosna_weight * self._tcsu_loss(
                model, images, masks, stage2)
        output = super()._forward(model, images, return_stage1=True)
        stage2 = output[0] if isinstance(output, (tuple, list)) else output
        return self.cosna_weight * self._tcsu_loss(
            model, images, masks, stage2)

    @torch.no_grad()
    def _full_pair(self, model, prepared, height: int, width: int):
        views = (None, (3,), (2,), (2, 3))
        stage_total = None
        high_total = None
        for dims in views:
            view = prepared if dims is None else torch.flip(
                prepared, dims=list(dims))
            stage2, _ = super()._forward(model, view, return_stage1=False)
            high = model.tcsu(stage2, view)
            if dims is not None:
                stage2 = torch.flip(stage2, dims=list(dims))
                high = torch.flip(high, dims=list(dims))
            stage_total = stage2 if stage_total is None else stage_total + stage2
            high_total = high if high_total is None else high_total + high
        stage_logits = F.interpolate(
            stage_total / len(views), size=(height, width),
            mode='bilinear', align_corners=False)
        high_logits = F.interpolate(
            high_total / len(views), size=(height, width),
            mode='bilinear', align_corners=False)
        probs = torch.sigmoid(stage_logits)[0].cpu().numpy()
        gated = _gated_mask(probs, self.cosna_gate)
        tcsu = (torch.sigmoid(high_logits)[0, 0].cpu().numpy()
                > self.cosna_tcsu_threshold)
        return gated, tcsu

    @torch.no_grad()
    def _tiled_pair(self, model, image: Image.Image):
        width, height = image.size
        tile = int(self.rcd_tile)
        stride = int(self.rcd_stride)
        stage_buf = torch.zeros((4, height, width), dtype=torch.float32)
        high_buf = torch.zeros((1, height, width), dtype=torch.float32)
        weight = torch.zeros((1, height, width), dtype=torch.float32)
        for y in _starts(height, tile, stride):
            for x in _starts(width, tile, stride):
                crop = image.crop(
                    (x, y, min(x + tile, width), min(y + tile, height)))
                cw, ch = crop.size
                view = _prep(crop, self.input_size).to(self.device)
                stage2, _ = super()._forward(
                    model, view, return_stage1=False)
                high = model.tcsu(stage2, view)
                part = F.interpolate(
                    stage2, size=(ch, cw), mode='bilinear',
                    align_corners=False)[0].cpu()
                part_high = F.interpolate(
                    high, size=(ch, cw), mode='bilinear',
                    align_corners=False)[0].cpu()
                stage_buf[:, y:y + ch, x:x + cw] += part
                high_buf[:, y:y + ch, x:x + cw] += part_high
                weight[:, y:y + ch, x:x + cw] += 1.0
        stage_logits = stage_buf / weight.clamp_min(1.0)
        high_logits = high_buf / weight.clamp_min(1.0)
        probs = torch.sigmoid(stage_logits).numpy()
        gated = _gated_mask(probs, self.cosna_gate)
        tcsu = (torch.sigmoid(high_logits)[0].numpy()
                > self.cosna_tcsu_threshold)
        return gated, tcsu

    @torch.no_grad()
    def predict_tcsu_sample(self, model, prepared, sample):
        """Return only the TCSU-h3 branch, before near-union."""
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        _, tcsu = self._full_pair(model, prepared, height, width)
        if (bool(getattr(self, 'prediction_mode', False))
                and self.rcd
                and (width > self.rcd_tile or height > self.rcd_tile)):
            _, tile_tcsu = self._tiled_pair(model, load_rgb(
                sample.image_path))
            tcsu, _ = route_masks(
                tcsu, tile_tcsu, self.routing_skel_ratio,
                self.routing_area_ratio, self.routing_mode)
        if self.close_radius > 0:
            tcsu = binary_closing(tcsu, disk(self.close_radius))
        if self.as_calibration:
            tcsu = apply_adaptive_skel(tcsu)
        return np.asarray(tcsu, dtype=bool)

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        gated, tcsu = self._full_pair(model, prepared, height, width)
        if (bool(getattr(self, 'prediction_mode', False))
                and self.rcd
                and (width > self.rcd_tile or height > self.rcd_tile)):
            tile_gated, tile_tcsu = self._tiled_pair(model, load_rgb(
                sample.image_path))
            gated, _ = route_masks(
                gated, tile_gated, self.routing_skel_ratio,
                self.routing_area_ratio, self.routing_mode)
            tcsu, _ = route_masks(
                tcsu, tile_tcsu, self.routing_skel_ratio,
                self.routing_area_ratio, self.routing_mode)
        if self.close_radius > 0:
            gated = binary_closing(gated, disk(self.close_radius))
            tcsu = binary_closing(tcsu, disk(self.close_radius))
        if self.as_calibration:
            gated = apply_adaptive_skel(gated)
            tcsu = apply_adaptive_skel(tcsu)
        gated = np.asarray(gated, dtype=bool)
        tcsu = np.asarray(tcsu, dtype=bool)
        near = binary_dilation(gated, disk(self.cosna_near_radius))
        extra = tcsu & near & ~gated
        if extra.any() and (self.cosna_filter_min_size > 0
                            or self.cosna_filter_aspect > 0):
            labels = label(extra, connectivity=1)
            sizes = np.bincount(labels.ravel())
            keep = sizes >= max(1, self.cosna_filter_min_size)
            if (self.cosna_filter_aspect > 0
                    and max(gated.shape) >= self.cosna_filter_maxdim):
                for prop in regionprops(labels):
                    if sizes[prop.label] >= self.cosna_filter_min_size:
                        continue
                    height = prop.bbox[2] - prop.bbox[0]
                    width = prop.bbox[3] - prop.bbox[1]
                    aspect = max(height, width) / max(
                        1, min(height, width))
                    if aspect >= self.cosna_filter_aspect:
                        keep[prop.label] = True
            keep[0] = False
            extra = keep[labels]
        return gated | extra

    def train_epoch(self, model, loader, optimizer, scheduler, args):
        result = super().train_epoch(
            model, loader, optimizer, scheduler, args)
        if self.cosna_save_last:
            epoch = int(getattr(self, '_epoch', 0))
            if epoch >= int(args.epochs) - 1:
                from pathlib import Path
                path = Path(self.cosna_save_last)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.checkpoint_config,
                    'epoch': epoch,
                }, path)
        return result


if __name__ == '__main__':
    raise SystemExit(main_for(CoSNAAdapter()))
