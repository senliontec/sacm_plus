"""Hilbert-curve traversal used by TPSegformer's LCCM.

The released code imports the external ``hilbertcurve`` package, which is not
available offline.  This module implements the standard iterative
``xy2d`` Hilbert index and caches the resulting permutation for every square
feature size (32x32 for the official 512 input).  Coordinates follow the
release's loops ``for i in range(H): for j in range(W)``, i.e. the first
coordinate is the row.
"""

from __future__ import annotations

import numpy as np
import torch

_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _xy2d(size: int, x: int, y: int) -> int:
    distance = 0
    step = size // 2
    while step > 0:
        rx = 1 if (x & step) > 0 else 0
        ry = 1 if (y & step) > 0 else 0
        distance += step * step * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = size - 1 - x
                y = size - 1 - y
            x, y = y, x
        step //= 2
    return distance


def hilbert_permutation(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, inverse)`` for row-major -> Hilbert order."""
    if size < 2 or size & (size - 1) != 0:
        raise ValueError(f'size must be a power of two >= 2, got {size}')
    cached = _CACHE.get(size)
    if cached is not None:
        return cached
    distances = np.empty((size, size), dtype=np.int64)
    for row in range(size):
        for column in range(size):
            distances[row, column] = _xy2d(size, row, column)
    indices = np.argsort(distances.reshape(-1), kind='stable')
    inverse = np.argsort(indices, kind='stable')
    _CACHE[size] = (indices, inverse)
    return indices, inverse


def flatten_hilbert(x: torch.Tensor,
                    indices: np.ndarray | None = None) -> torch.Tensor:
    """Flatten ``[B,C,H,W]`` in Hilbert order to ``[B,C,H*W]``."""
    if x.ndim != 4 or x.shape[-2] != x.shape[-1]:
        raise ValueError(f'expected a square [B,C,H,W] tensor, got {x.shape}')
    size = x.shape[-1]
    if indices is None:
        indices, _ = hilbert_permutation(size)
    order = torch.as_tensor(indices, device=x.device)
    return x.flatten(2).index_select(2, order)


def expand_hilbert(x: torch.Tensor, inverse: np.ndarray,
                   size: int) -> torch.Tensor:
    """Expand ``[B,C,H*W]`` Hilbert order back to ``[B,C,H,W]``."""
    order = torch.as_tensor(inverse, device=x.device)
    return x.index_select(2, order).view(
        x.shape[0], x.shape[1], size, size)
