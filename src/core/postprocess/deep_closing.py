"""Deep Closing post-processing refiner.

This is a clean, self-contained port of the core method in

    Qian Wu, Yufei Chen, Wei Liu, Xiaodong Yue, Xiahai Zhuang,
    "Deep Closing: Enhancing Topological Connectivity in Medical Tubular
    Segmentation", IEEE Transactions on Medical Imaging, 2024.
    https://github.com/5k5000/DeepClosing (MIT)

The upstream repository is a research prototype: it only ships a DRIVE
training entry point and its ``DeepClosing`` inference method has several
runtime bugs (undefined ``transform`` for tensor inputs, a broken binary
assertion, batch-1 assumptions).  This module preserves the paper's core
algorithm while making it usable:

1. Masked Image Modeling (MIM) learns a tubular shape prior from training
   masks, with a recurrent tube scaler and a topology-aware reconstruction
   loss.
2. Deep Dilation uses that prior to propose disconnected regions.
3. Simple Component Erosion (SCE) keeps only topologically non-simple
   proposed components and adds them to the original prediction.

The refiner is inference-only with respect to the segmentation model; it
does not change logits and requires no ground truth at inference time.
This is an intentional post-processing adapter, not a new segmentation
backbone.  See ``UPSTREAM.md`` for the complete modification list.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as ndi_label
from torch import nn

# ---------------------------------------------------------------------------
# Simple points
# ---------------------------------------------------------------------------

_FG_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
_BG_STRUCTURE = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8,
)


def _component_count(mask: np.ndarray, structure: np.ndarray) -> int:
    _, count = ndi_label(mask, structure=structure)
    return int(count)


def is_simple_patch(patch: np.ndarray) -> bool:
    """Return whether the centre of a 3x3 binary patch is a simple point.

    Official criterion (``Simple_Point_Erosion_Module.py``): compare the
    number of 8-connected foreground and 4-connected background components
    before and after flipping the centre pixel.
    """
    patch = np.asarray(patch, dtype=np.uint8)
    if patch.shape != (3, 3):
        raise ValueError(f"expected a 3x3 patch, got {patch.shape}")
    fg_before = _component_count(patch, _FG_STRUCTURE)
    bg_before = _component_count(1 - patch, _BG_STRUCTURE)
    flipped = patch.copy()
    flipped[1, 1] = 1 - flipped[1, 1]
    fg_after = _component_count(flipped, _FG_STRUCTURE)
    bg_after = _component_count(1 - flipped, _BG_STRUCTURE)
    return fg_before == fg_after and bg_before == bg_after


def _build_simple_lut() -> torch.Tensor:
    """Enumerate all 512 3x3 binary patches and build the SPLT."""
    lut = torch.zeros(512, dtype=torch.float32)
    for index in range(512):
        bits = np.array(
            [(index >> (8 - bit)) & 1 for bit in range(9)],
            dtype=np.uint8,
        ).reshape(3, 3)
        if is_simple_patch(bits):
            lut[index] = 1.0
    return lut


_SIMPLE_LUT = _build_simple_lut()


class SimplePointErosion:
    """Parallel, topology-preserving simple-point erosion.

    The implementation follows Algorithm 2 of the Deep Closing paper:
    four checkerboard order masks decouple simultaneous erosion, and the
    process is repeated until convergence (or ``max_iters``).

    This class is deliberately not an ``nn.Module``: it has no learnable
    state and can be shared by training and inference.
    """

    def __init__(self, device: Optional[torch.device | str] = None):
        self.device = torch.device(device) if device is not None else None
        self._lut_cache: dict[torch.device, torch.Tensor] = {}
        self._order_cache: dict[tuple[int, int, torch.device, torch.dtype],
                                tuple[torch.Tensor, ...]] = {}

    def _lut(self, device: torch.device) -> torch.Tensor:
        cached = self._lut_cache.get(device)
        if cached is None or cached.device != device:
            cached = _SIMPLE_LUT.to(device=device)
            self._lut_cache[device] = cached
        return cached

    def _order_masks(self, height: int, width: int, device: torch.device,
                     dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        key = (height, width, device, dtype)
        cached = self._order_cache.get(key)
        if cached is not None:
            return cached
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij',
        )
        masks = (
            ((yy % 2 == 0) & (xx % 2 == 0)),
            ((yy % 2 == 0) & (xx % 2 == 1)),
            ((yy % 2 == 1) & (xx % 2 == 0)),
            ((yy % 2 == 1) & (xx % 2 == 1)),
        )
        masks = tuple(
            m.to(dtype=dtype).unsqueeze(0).unsqueeze(0) for m in masks)
        self._order_cache[key] = masks
        return masks

    def patch_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """Serialize every 3x3 neighbourhood to a 9-bit index."""
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(
                f"mask must have shape [B,1,H,W], got {tuple(mask.shape)}")
        patches = F.unfold(mask, kernel_size=3, padding=1, stride=1)
        # [B, 9, H*W] -> [B*H*W, 9]
        bits = patches.permute(0, 2, 1).reshape(-1, 9).round()
        bits = bits.to(torch.long)
        powers = (2 ** torch.arange(
            8, -1, -1, device=bits.device, dtype=torch.long))
        return (bits * powers).sum(dim=1)

    def simple_points(self, mask: torch.Tensor) -> torch.Tensor:
        """Return a 0/1 simple-point label map for a binary mask."""
        if mask.ndim != 4:
            raise ValueError(
                f"mask must have shape [B,1,H,W], got {tuple(mask.shape)}")
        batch, _, height, width = mask.shape
        indices = self.patch_indices(mask)
        labels = self._lut(mask.device)[indices]
        return labels.view(batch, 1, height, width)

    def rspe(self, mask: torch.Tensor, region: torch.Tensor,
             max_iters: int | float | None = 1000) -> torch.Tensor:
        """Recurrent Simple Point Erosion within ``region``.

        Args:
            mask: binary structure, shape [B,1,H,W].
            region: binary update region, broadcastable to mask.
            max_iters: iteration cap; ``None``/``inf`` means until
                convergence.  The official implementation uses infinity.
        """
        mask = (mask > 0.5).to(dtype=torch.float32)
        region = (region > 0.5).to(device=mask.device, dtype=mask.dtype)
        if region.shape != mask.shape:
            region = torch.broadcast_to(region, mask.shape).to(mask.dtype)
        if max_iters is None or math.isinf(float(max_iters)):
            iterations = 1_000_000
        else:
            iterations = max(1, int(max_iters))
        batch, _, height, width = mask.shape
        order_masks = self._order_masks(
            height, width, mask.device, mask.dtype)
        for _ in range(iterations):
            before = float(mask.sum().item())
            for order in order_masks:
                simple = self.simple_points(mask)
                remove = simple * mask * region * order
                mask = mask - remove
            if float(mask.sum().item()) == before:
                break
        return mask

    def simple_component_erosion(
            self, original: torch.Tensor, dilated: torch.Tensor,
            max_iters: int | float | None = 1000) -> torch.Tensor:
        """Apply Deep Closing's Simple Component Erosion.

        ``original`` is the pre-segmentation I and ``dilated`` is Deep
        Dilation's T.  We retain only the components of ``T - I`` that
        survive complete simple-point erosion; simple components are
        removed.  The original prediction is never removed.
        """
        if original.shape != dilated.shape or original.ndim != 4:
            raise ValueError(
                'original and dilated must both have shape [B,1,H,W]')
        original = (original > 0.5).to(dtype=torch.float32)
        dilated = (dilated > 0.5).to(dtype=torch.float32)
        result = original.clone()
        region = torch.clamp(dilated - original, min=0.0, max=1.0)
        for batch_index in range(original.shape[0]):
            region_np = (
                region[batch_index, 0].detach().cpu().numpy() > 0.5)
            if not region_np.any():
                continue
            labels, count = ndi_label(
                region_np, structure=np.ones((3, 3), dtype=np.uint8))
            if count == 0:
                continue
            eroded = self.rspe(
                dilated[batch_index:batch_index + 1],
                region[batch_index:batch_index + 1],
                max_iters=max_iters,
            )
            eroded_np = (
                eroded[0, 0].detach().cpu().numpy() > 0.5)
            retained = np.unique(labels[eroded_np & region_np])
            retained = retained[retained != 0]
            for label_value in retained.tolist():
                result[batch_index, 0] = torch.maximum(
                    result[batch_index, 0],
                    torch.from_numpy(labels == label_value).to(
                        device=result.device, dtype=result.dtype),
                )
        return result


# ---------------------------------------------------------------------------
# Masked shape reconstruction
# ---------------------------------------------------------------------------

def random_block_mask(
        mask: torch.Tensor, rng: random.Random,
        patch_range: tuple[int, int] = (2, 8),
        ratio_range: tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """Generate the paper's dynamic block mask (1 = masked)."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"mask must have 2-4 dims, got {tuple(mask.shape)}")
    batch, _, height, width = mask.shape
    patch_h = rng.randint(int(patch_range[0]), int(patch_range[1]))
    patch_w = rng.randint(int(patch_range[0]), int(patch_range[1]))
    patch_h = max(1, patch_h)
    patch_w = max(1, patch_w)
    rows = math.ceil(height / patch_h)
    cols = math.ceil(width / patch_w)
    total = rows * cols
    ratio = rng.uniform(float(ratio_range[0]), float(ratio_range[1]))
    ratio = min(max(ratio, 0.0), 1.0)
    keep = int(round(total * (1.0 - ratio)))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(rng.randrange(2 ** 31 - 1))
    permutation = torch.randperm(total, generator=generator)
    blocks = torch.ones(total, dtype=torch.float32)
    blocks[permutation[:keep]] = 0.0
    blocks = blocks.view(rows, cols)
    blocks = blocks.repeat_interleave(patch_h, dim=0)
    blocks = blocks.repeat_interleave(patch_w, dim=1)
    blocks = blocks[:height, :width]
    return blocks.unsqueeze(0).unsqueeze(0).expand(
        batch, 1, height, width).to(mask.device)


def recurrent_tube_scale(
        mask: torch.Tensor, spe: SimplePointErosion, rng: random.Random,
        max_k: int = 2) -> torch.Tensor:
    """Official Recurrent TubeScaler augmentation.

    The released code calls ``RSPE(T=x, M_T=x, max_K=k_r)`` (the paper
    writes the region as zero); we follow the executable code: the whole
    foreground is the erosion region.  Each sample receives an independent
    random k in ``[0, max_k]``.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    outputs = []
    for index in range(mask.shape[0]):
        sample = (mask[index:index + 1] > 0.5).float()
        k = rng.randint(0, max(0, int(max_k)))
        if k == 0:
            outputs.append(sample)
        else:
            outputs.append(spe.rspe(sample, sample, max_iters=k))
    return torch.cat(outputs, dim=0)


def mim_loss(
        prediction: torch.Tensor, target: torch.Tensor,
        masked: torch.Tensor, spe: SimplePointErosion,
        topo_weight: float = 0.1) -> torch.Tensor:
    """Masked shape reconstruction + topology-aware loss (paper Eq. 4)."""
    if prediction.shape != target.shape or target.shape != masked.shape:
        raise ValueError(
            'prediction, target and masked must have identical shapes')
    target = (target > 0.5).float()
    masked = (masked > 0.5).float()
    reconstruction = F.mse_loss(prediction, target)
    # Non-simple points within the masked region remain after RSPE; these
    # are the topology-critical pixels that receive extra supervision.
    critical = spe.rspe(target, masked, max_iters=10) * masked
    topology = F.mse_loss(
        prediction * critical, target * critical)
    return reconstruction + float(topo_weight) * topology


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetMIM(nn.Module):
    """Compact 2D U-Net used as the MIM shape-prior autoencoder.

    The official implementation uses MONAI's U-Net with channels
    (32, 64, 128, 256, 512).  This dependency-free version defaults to
    (16, 32, 64, 128, 256) for efficient 512x512 training on 18 masks;
    the channel tuple is stored in the checkpoint and can be overridden.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: Sequence[int] = (16, 32, 64, 128, 256)):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) < 2:
            raise ValueError('channels must contain at least two levels')
        self.channels = channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.inc = ConvBlock(in_channels, channels[0])
        self.downs = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.ups = nn.ModuleList([
            UpBlock(channels[i + 1], channels[i], channels[i])
            for i in range(len(channels) - 2, -1, -1)
        ])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        x = features[-1]
        for index, up in enumerate(self.ups):
            x = up(x, features[-2 - index])
        logits = self.out_conv(x)
        logits = F.interpolate(
            logits, size=input_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class DeepClosingRefiner:
    """Inference wrapper for Deep Dilation + Simple Component Erosion."""

    def __init__(self, model: UNetMIM, resolution: int = 512,
                 dilation_threshold: float = 0.2,
                 max_iters: int | float | None = 1000,
                 device: Optional[torch.device | str] = None):
        self.device = torch.device(
            device if device is not None else
            ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device).eval()
        self.resolution = max(1, int(resolution))
        self.dilation_threshold = float(dilation_threshold)
        self.max_iters = max_iters
        self.spe = SimplePointErosion(self.device)

    @classmethod
    def from_checkpoint(
            cls, checkpoint: str | os.PathLike,
            device: Optional[torch.device | str] = None,
            resolution: Optional[int] = None,
            dilation_threshold: Optional[float] = None,
            max_iters: Optional[int | float] = None) -> 'DeepClosingRefiner':
        payload = torch.load(
            checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or 'model_state_dict' not in payload:
            raise ValueError(
                f'{checkpoint} is not a Deep Closing shape-prior checkpoint')
        config = payload.get('config', {}) or {}
        model = UNetMIM(**config.get('model', {}))
        model.load_state_dict(payload['model_state_dict'])
        return cls(
            model,
            resolution=(
                int(resolution) if resolution and int(resolution) > 0
                else int(config.get('resolution', 512))),
            dilation_threshold=(
                float(dilation_threshold)
                if dilation_threshold is not None
                else float(config.get('dilation_threshold', 0.2))),
            max_iters=(
                max_iters if max_iters is not None
                else config.get('max_iters', 1000)),
            device=device,
        )

    @torch.no_grad()
    def deep_dilation(self, mask: torch.Tensor) -> torch.Tensor:
        """Predict and threshold the shape prior at the working size."""
        probability = self.model(mask)
        return (probability >= self.dilation_threshold).float()

    def refine(self, mask: np.ndarray,
               threshold: float = 0.5) -> np.ndarray:
        """Refine one binary mask at its original resolution.

        Args:
            mask: [H,W], [1,H,W] or [1,1,H,W] binary/float mask.
            threshold: binarization threshold for the input mask.
        Returns:
            A boolean [H,W] mask.  Empty inputs are returned unchanged
            (the shape prior must not hallucinate foreground from nothing).
        """
        array = np.asarray(mask)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(
                f'mask must squeeze to [H,W], got {array.shape}')
        original = array > threshold
        if not original.any():
            return np.zeros_like(original, dtype=bool)
        height, width = original.shape
        tensor = torch.from_numpy(
            original.astype(np.float32)).to(self.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        if (height, width) != (self.resolution, self.resolution):
            tensor = F.interpolate(
                tensor, size=(self.resolution, self.resolution),
                mode='nearest')
        dilated = self.deep_dilation(tensor)
        refined = self.spe.simple_component_erosion(
            tensor, dilated, max_iters=self.max_iters)
        if (height, width) != (self.resolution, self.resolution):
            refined = F.interpolate(
                refined, size=(height, width), mode='nearest')
        return (refined[0, 0] > 0.5).detach().cpu().numpy().astype(bool)

    def __call__(self, mask: np.ndarray,
                 threshold: float = 0.5) -> np.ndarray:
        return self.refine(mask, threshold=threshold)


def save_checkpoint(
        path: str | os.PathLike, model: nn.Module,
        config: dict, epoch: int, val_loss: float) -> Path:
    """Atomically save a shape-prior checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': int(epoch),
        'val_loss': float(val_loss),
        'source': 'DeepClosing (TMI 2024) clean post-processing port',
        'upstream': 'https://github.com/5k5000/DeepClosing',
    }
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


__all__ = [
    'DeepClosingRefiner',
    'SimplePointErosion',
    'UNetMIM',
    'is_simple_patch',
    'mim_loss',
    'random_block_mask',
    'recurrent_tube_scale',
    'save_checkpoint',
]
