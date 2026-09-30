"""Topology metrics: clDice (official clDice_metric/clDice.py, Shit et
al., no smoothing, NaN on empty skeletons) and Betti number error
(project extension, count-based; the persistence-aware Betti matching is
available in metrics/persistence.py)."""

import numpy as np

from .common import _bool, skeletonize


def _cl_score(v, s):
    """Official clDice cl_score: sum(v*s)/sum(s)."""
    return np.sum(v * s) / np.sum(s)


def cldice_score(pred, gt):
    """Official clDice metric (clDice_metric/clDice.py), 2D:
    tprec = cl_score(v_p, skel(v_l)), tsens = cl_score(v_l, skel(v_p)),
    clDice = 2*tprec*tsens/(tprec+tsens).

    NO smoothing: an empty skeleton yields NaN (as in the official 0/0).
    """
    v_p = _bool(pred)
    v_l = _bool(gt)
    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)
    if np.count_nonzero(s_p) == 0 or np.count_nonzero(s_l) == 0:
        return float("nan")
    tprec = _cl_score(v_p, s_l)
    tsens = _cl_score(v_l, s_p)
    return float(2.0 * tprec * tsens / (tprec + tsens))


def betti_error(pred, gt, connectivity=2):
    """Betti number error: |b0_pred - b0_gt| + |b1_pred - b1_gt|.

    b0 = number of foreground connected components (8-connectivity,
    scipy.ndimage.label). b1 = number of holes = background components
    not touching the image border. Count-based version; the
    persistence-aware Betti matching (Stucki et al.) is a strictly
    stronger refinement available as an optional upgrade.

    Returns NaN if either mask is empty, or if scipy is not installed.
    """
    try:
        from scipy.ndimage import generate_binary_structure, label
    except ImportError:
        return float("nan")

    pred = _bool(pred)
    gt = _bool(gt)
    if np.count_nonzero(pred) == 0 or np.count_nonzero(gt) == 0:
        return float("nan")

    struct = generate_binary_structure(2, connectivity)

    def components(m):
        return label(m, struct)[1]

    def holes(m):
        lab, n = label(1 - m, struct)
        border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        border = border[border != 0]
        return max(n - len(border), 0)

    return float(
        abs(components(pred) - components(gt)) + abs(holes(pred) - holes(gt))
    )
