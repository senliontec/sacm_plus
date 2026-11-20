"""Resolution-consensus decoding (RCD) for BCTD-SAM.

For images larger than the SAM input, run a full-image view and overlapping
native-resolution tiles.  The routed output uses only prediction statistics;
no labels or dataset names are used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, stride))
    if out[-1] != length - tile:
        out.append(length - tile)
    return out


def _prep(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(
        np.asarray(image, dtype=np.float32).copy()
    ).permute(2, 0, 1).unsqueeze(0) / 255.0


@torch.no_grad()
def tile_mask(adapter, model, image: Image.Image, tile: int, stride: int,
              flip: bool, rot90: bool, threshold: float) -> np.ndarray:
    """Predict overlapping native-resolution tiles and stitch logits."""
    width, height = image.size
    logits = torch.zeros((1, 1, height, width), dtype=torch.float32)
    weight = torch.zeros((1, 1, height, width), dtype=torch.float32)
    rotations = (0, 1, 2, 3) if rot90 else (0,)
    flips = (None, (3,), (2,), (2, 3)) if flip else (None,)
    for y in _starts(height, tile, stride):
        for x in _starts(width, tile, stride):
            crop = image.crop(
                (x, y, min(x + tile, width), min(y + tile, height)))
            cw, ch = crop.size
            view = _prep(crop, adapter.input_size).to(adapter.device)
            parts = []
            for rotation in rotations:
                rotated = view if rotation == 0 else torch.rot90(
                    view, rotation, dims=[2, 3])
                for dims in flips:
                    item = rotated if dims is None else torch.flip(
                        rotated, dims=list(dims))
                    part = adapter._deploy_logits(model, item)
                    if dims is not None:
                        part = torch.flip(part, dims=list(dims))
                    if rotation:
                        part = torch.rot90(part, -rotation, dims=[2, 3])
                    parts.append(part)
            part = torch.stack(parts).mean(0)
            part = F.interpolate(
                part, size=(ch, cw), mode='bilinear', align_corners=False)
            logits[0, 0, y:y + ch, x:x + cw] += part[0, 0].cpu()
            weight[0, 0, y:y + ch, x:x + cw] += 1.0
    logits = logits / weight.clamp_min(1.0)
    return (torch.sigmoid(logits)[0, 0].numpy() > threshold)


def route_masks(full: np.ndarray, tiled: np.ndarray,
                skel_ratio: float = 1.0,
                area_ratio: float = 0.7,
                mode: str = 'auto') -> tuple[np.ndarray, str]:
    """Choose full, tiled, AND or UNION from prediction statistics."""
    full = np.asarray(full, dtype=bool)
    tiled = np.asarray(tiled, dtype=bool)
    if mode == 'full':
        return full, 'full'
    if mode == 'tile':
        return tiled, 'tile'
    if mode == 'and':
        return full & tiled, 'and'
    if mode == 'union':
        return full | tiled, 'union'
    if mode != 'auto':
        raise ValueError(f'unknown RCD routing mode: {mode!r}')

    full_skel = float(skeletonize(full).sum())
    tile_skel = float(skeletonize(tiled).sum())
    foreground = float(full.sum())
    skel_change = tile_skel / max(full_skel, 1.0)
    area_change = float(tiled.sum()) / max(foreground, 1.0)
    if skel_change <= skel_ratio:
        return full, 'full'
    if area_change < area_ratio:
        return tiled, 'tile'
    return full & tiled, 'and'
