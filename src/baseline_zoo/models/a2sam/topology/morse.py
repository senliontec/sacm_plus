"""Reconstructed Morse/branch geometry for A2SAM.

Persistence values come from ``persistence.py`` (exact cripser H0).  This
module reconstructs spatial branches: each retained persistence bar is a
peak/basin, and a deterministic watershed over the coarse support assigns
pixels to those basins.  It is deliberately called a reconstruction: the
release does not contain the DMT gradient vector field or manifolds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.segmentation import watershed

from .persistence import PersistenceBars, compute_persistence


@dataclass(frozen=True)
class MorseComplex:
    labels: np.ndarray              # int32 [H,W], 0 = background
    label_persistence: np.ndarray   # float32 [M+1], index 0 unused
    peak_rc: np.ndarray             # int32 [M+1,2]
    peak_prob: np.ndarray           # float32 [M+1]
    coarse: np.ndarray              # bool [H,W]
    n_branches: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.labels.shape)

    @property
    def branch_ids(self) -> np.ndarray:
        return np.arange(1, self.n_branches + 1, dtype=np.int32)

    def retained_ids(self, xi: float) -> np.ndarray:
        if self.n_branches == 0:
            return np.zeros(0, np.int32)
        return np.flatnonzero(
            self.label_persistence[1:] >= float(xi)).astype(np.int32) + 1


def _empty_complex(prob: np.ndarray,
                   coarse_threshold: float) -> MorseComplex:
    coarse = prob > coarse_threshold
    return MorseComplex(
        labels=np.zeros(prob.shape, np.int32),
        label_persistence=np.zeros(1, np.float32),
        peak_rc=np.zeros((1, 2), np.int32),
        peak_prob=np.zeros(1, np.float32),
        coarse=coarse,
        n_branches=0,
    )


def _fallback_single(prob: np.ndarray, coarse: np.ndarray) -> MorseComplex:
    """One basin when persistence gives no usable marker."""
    if not coarse.any():
        return _empty_complex(prob, 0.5)
    flat = np.argmax(np.where(coarse, prob, -np.inf))
    rc = np.unravel_index(int(flat), prob.shape)
    markers = np.zeros(prob.shape, np.int32)
    markers[rc] = 1
    labels = watershed(-prob.astype(np.float32), markers, mask=coarse)
    peak_rc = np.asarray([[rc[0], rc[1]]], np.int32)
    return MorseComplex(
        labels=labels.astype(np.int32),
        label_persistence=np.asarray([0.0, np.inf], dtype=np.float32),
        peak_rc=np.vstack([np.zeros((1, 2), np.int32), peak_rc]),
        peak_prob=np.asarray([0.0, prob[rc]], np.float32),
        coarse=coarse,
        n_branches=1,
    )


def build_morse_complex(
    prob: np.ndarray,
    *,
    backend: str = 'cripser',
    coarse_threshold: float = 0.5,
    min_persistence: float = 1e-3,
    max_branches: int = 256,
) -> MorseComplex:
    """Build branch basins from exact persistence bars.

    Bars with birth pixels outside the coarse support are discarded.  The
    essential global-maximum basin is always retained; finite branches are
    limited to ``max_branches`` by persistence.  This limit is a reconstruction
    hyperparameter, not a paper value.
    """
    array = np.asarray(prob, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'probability map must be 2-D, got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('probability map contains NaN/inf')
    if max_branches < 1:
        raise ValueError('max_branches must be positive')
    coarse = array > float(coarse_threshold)
    if not coarse.any():
        return _empty_complex(array, coarse_threshold)

    bars = compute_persistence(
        array, backend=backend, min_persistence=min_persistence)
    if len(bars) == 0:
        return _fallback_single(array, coarse)

    order = np.lexsort((
        -bars.persistence.astype(np.float64),
        -bars.essential.astype(np.int32),
    ))
    bars = PersistenceBars(
        birth_prob=bars.birth_prob[order],
        death_prob=bars.death_prob[order],
        persistence=bars.persistence[order],
        birth_rc=bars.birth_rc[order],
        death_rc=bars.death_rc[order],
        essential=bars.essential[order],
    )

    markers = np.zeros(array.shape, np.int32)
    peak_rc = [(0, 0)]
    peak_prob = [0.0]
    label_persistence = [0.0]
    used: set[tuple[int, int]] = set()
    for index, (rc, persistence) in enumerate(zip(
            bars.birth_rc, bars.persistence), start=1):
        if len(used) >= max_branches:
            break
        row, col = int(rc[0]), int(rc[1])
        if not (0 <= row < array.shape[0] and 0 <= col < array.shape[1]):
            continue
        if not coarse[row, col]:
            continue
        key = (row, col)
        if key in used:
            continue
        used.add(key)
        label = len(used)
        markers[row, col] = label
        peak_rc.append((row, col))
        peak_prob.append(float(array[row, col]))
        label_persistence.append(
            np.inf if bars.essential[index - 1] else float(persistence))

    if not used:
        return _fallback_single(array, coarse)

    labels = watershed(-array, markers, mask=coarse)
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return _fallback_single(array, coarse)

    # Watershed may drop markers enclosed by another marker.  Remap surviving
    # labels to compact IDs 1..M.
    remap = np.zeros(int(labels.max()) + 1, np.int32)
    new_persistence = np.zeros(present.size + 1, np.float32)
    new_peak_rc = np.zeros((present.size + 1, 2), np.int32)
    new_peak_prob = np.zeros(present.size + 1, np.float32)
    for new_id, old_id in enumerate(present, start=1):
        remap[old_id] = new_id
        new_persistence[new_id] = label_persistence[old_id]
        new_peak_rc[new_id] = peak_rc[old_id]
        new_peak_prob[new_id] = peak_prob[old_id]
    labels = remap[labels]
    labels[~coarse] = 0
    return MorseComplex(
        labels=labels.astype(np.int32),
        label_persistence=new_persistence,
        peak_rc=new_peak_rc,
        peak_prob=new_peak_prob,
        coarse=coarse,
        n_branches=int(present.size),
    )
