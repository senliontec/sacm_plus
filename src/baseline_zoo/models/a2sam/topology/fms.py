"""Persistence-threshold reconstruction ``F_MS`` and uncertainty."""

from __future__ import annotations

import numpy as np

from .morse import MorseComplex


def f_ms(complex_: MorseComplex, xi: float) -> np.ndarray:
    """Return the union of basins with persistence at least ``xi``.

    The essential global-maximum basin is assigned infinite persistence and
    is therefore always retained.  If no basin is retained, the result is an
    empty mask; this is the documented reconstruction behavior.
    """
    if complex_.n_branches == 0:
        return np.zeros(complex_.shape, bool)
    keep = complex_.retained_ids(float(xi))
    if keep.size == 0:
        return np.zeros(complex_.shape, bool)
    return np.isin(complex_.labels, keep)


def uncertainty_from_samples(samples: np.ndarray) -> np.ndarray:
    """Pixel-wise population variance of binary candidate segmentations."""
    array = np.asarray(samples)
    if array.ndim != 3:
        raise ValueError(
            f'samples must be [Z,H,W], got {array.shape}')
    if array.shape[0] < 1:
        raise ValueError('at least one sample is required')
    variance = array.astype(np.float32).var(axis=0, ddof=0)
    return np.clip(variance, 0.0, 0.25)


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred, target).sum()) / denom


def dice_optimal_xi(
    complex_: MorseComplex,
    target: np.ndarray,
    *,
    grid_size: int = 32,
) -> float:
    """Dice-optimal global persistence threshold for prior supervision.

    The paper does not specify the prior-network training target.  We use the
    threshold that maximizes coarse overlap with the training GT and train a
    Gaussian NLL around it.  This uses training labels only; never pass
    validation/test masks here.
    """
    target = np.asarray(target, dtype=bool)
    if target.shape != complex_.shape:
        raise ValueError(
            f'target shape {target.shape} != complex shape {complex_.shape}')
    if not target.any() or complex_.n_branches == 0:
        return 0.0
    values = complex_.label_persistence[1:]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if finite.size > grid_size:
        candidates = np.quantile(
            finite, np.linspace(0.0, 1.0, grid_size)).astype(np.float32)
    else:
        candidates = np.unique(finite)
    candidates = np.unique(np.concatenate([
        np.asarray([0.0], np.float32), candidates]))
    best_xi = 0.0
    best_dice = -1.0
    for xi in candidates:
        score = _dice(f_ms(complex_, float(xi)), target)
        if score > best_dice:
            best_dice = score
            best_xi = float(xi)
    return best_xi
