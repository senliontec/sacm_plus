"""Persistence-based topology metrics — official engines vendored in the
losses package, evaluated at 256 resolution (the pure-Python engines are
impractically slow at 1024; the official repos train on small crops)."""

import numpy as np

from .common import to_numpy


def betti_matching_score(pred_logits, gt, resolution=256):
    """Betti matching error (persistence-based, Stucki et al.), computed
    with the official engine vendored in losses/betti_matching.py.

    Inputs are downsampled to `resolution` (default 256). The computation
    at the chosen resolution is the official code. Returns NaN on empty
    masks, missing scipy, or missing torch.
    """
    try:
        from losses.betti_matching import betti_number_error_metric
        from scipy.ndimage import zoom
    except ImportError:
        return float("nan")

    probs = 1.0 / (1.0 + np.exp(-to_numpy(pred_logits).squeeze().astype(np.float64)))
    gt_bin = (to_numpy(gt).squeeze() > 0.5).astype(np.float64)
    if probs.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")

    if max(probs.shape) > resolution:
        f = resolution / max(probs.shape)
        probs = zoom(probs, f, order=1)
        gt_bin = zoom(gt_bin, f, order=0)
    return betti_number_error_metric(probs, gt_bin)


def topograph_error_score(pred_logits, gt, resolution=256):
    """Topograph topological error count (official engine vendored in
    losses/topograph.py; the monai wrapper is replaced by a plain call —
    computation is the official code). Evaluated at `resolution` (256
    default; the per-image networkx graph is CPU-heavy). Returns NaN on
    empty masks or missing dependencies (networkx/scipy).
    """
    try:
        from losses.topograph import topograph_error_metric
        from scipy.ndimage import zoom
    except ImportError:
        return float("nan")

    probs = 1.0 / (1.0 + np.exp(-to_numpy(pred_logits).squeeze().astype(np.float64)))
    pred_bin = (probs > 0.5)
    gt_bin = (to_numpy(gt).squeeze() > 0.5)
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")

    if max(pred_bin.shape) > resolution:
        f = resolution / max(pred_bin.shape)
        pred_bin = zoom(pred_bin.astype(np.float64), f, order=0) > 0.5
        gt_bin = zoom(gt_bin.astype(np.float64), f, order=0) > 0.5

    return float(topograph_error_metric(pred_bin, gt_bin))
