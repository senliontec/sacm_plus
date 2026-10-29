"""Width-aware skeleton-guided TopoMask for TopoWeave.

MaskVSC deletes skeleton branches and dilates them with a fixed 7x7 kernel
using global RNG.  This version samples contiguous skeleton paths and erases
a disk whose radius follows the label distance transform, so thin cracks and
thick vessels are both masked realistically.  The label is never modified.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.draw import disk
from skimage.morphology import skeletonize

_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def topo_ratio(epoch: int, epochs: int, peak: float = 0.35,
               jitter: float = 0.05,
               rng: random.Random | None = None) -> float:
    """Biphasic masking ratio with optional per-sample jitter."""
    progress = (int(epoch) - 1) / max(int(epochs), 1)
    if progress < 0.1 or progress > 0.9:
        ratio = 0.0
    elif progress <= 0.5:
        ratio = float(peak) * (progress - 0.1) / 0.4
    else:
        ratio = float(peak) * (0.9 - progress) / 0.4
    if rng is not None and ratio > 0:
        ratio += rng.uniform(-float(jitter), float(jitter))
    return float(np.clip(ratio, 0.0, 0.5))


def _sample_paths(skeleton: np.ndarray, target: int,
                  rng: random.Random) -> list[list[tuple[int, int]]]:
    coords = np.argwhere(skeleton)
    if len(coords) == 0 or target <= 0:
        return []
    available = {tuple(int(v) for v in xy) for xy in coords}
    visited: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []
    remaining = min(int(target), len(available))
    starts = [tuple(int(v) for v in xy) for xy in coords]
    rng.shuffle(starts)

    for start in starts:
        if remaining <= 0:
            break
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        remaining -= 1
        current = start
        max_len = min(remaining, rng.randint(4, 24))
        for _ in range(max_len):
            candidates = []
            for dy, dx in _NEIGHBOURS:
                nxt = (current[0] + dy, current[1] + dx)
                if nxt in available and nxt not in visited:
                    candidates.append(nxt)
            if not candidates:
                break
            current = candidates[rng.randrange(len(candidates))]
            visited.add(current)
            path.append(current)
            remaining -= 1
        paths.append(path)
    return paths


def width_aware_topomask(image: np.ndarray, label: np.ndarray,
                         rng: random.Random,
                         ratio: float) -> np.ndarray:
    """Erase width-aware skeleton paths from ``image``; ``label`` is unchanged.

    Args:
        image: [H, W, 3] uint8 RGB image.
        label: [H, W] binary or 0/255 mask.
        rng: explicit random source for reproducibility.
        ratio: fraction of skeleton pixels to remove, clipped to [0, 0.5].

    Returns:
        A new uint8 image; the original array is not modified.
    """
    image = np.asarray(image, dtype=np.uint8)
    label_bool = np.asarray(label) > 0
    ratio = float(np.clip(ratio, 0.0, 0.5))
    if ratio <= 0 or not label_bool.any():
        return image.copy()

    skeleton = skeletonize(label_bool)
    skeleton_count = int(skeleton.sum())
    if skeleton_count == 0:
        return image.copy()

    target = max(1, int(round(ratio * skeleton_count)))
    paths = _sample_paths(skeleton, target, rng)
    if not paths:
        return image.copy()

    distance = distance_transform_edt(label_bool)
    erase = np.zeros(label_bool.shape, dtype=bool)
    height, width = label_bool.shape
    for path in paths:
        for row, col in path:
            radius = int(np.clip(round(float(distance[row, col])), 1, 8))
            rr, cc = disk((row, col), radius, shape=(height, width))
            erase[rr, cc] = True
    if not erase.any():
        return image.copy()

    smooth = gaussian_filter(
        image.astype(np.float32), sigma=(5.0, 5.0, 0.0))
    noise_rng = np.random.default_rng(rng.randrange(0, 2 ** 32))
    noise = noise_rng.normal(0.0, 8.0, size=image.shape)
    fill = np.clip(smooth + noise, 0, 255).astype(np.uint8)
    result = image.copy()
    result[erase] = fill[erase]
    return result
