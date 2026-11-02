"""Connected-component / fragmentation metrics (diagnostic extensions).

These metrics are not part of the official SACM metric suite.  They expose a
failure mode that Dice/clDice/HD95 hide: a prediction can have good overlap
but hundreds of tiny isolated components (CREMI, ROAD, DeepGlobe, retinal
datasets).  All values are computed from binary masks at the original GT
resolution.
"""

import numpy as np
from scipy.ndimage import label

COMPONENT_KEYS = (
    'components_pred',
    'components_gt',
    'component_count_error',
    'small_components_pred',
    'small_component_rate',
    'small_components_gt',
    'largest_component_ratio',
    'fp_area_ratio',
    'fn_area_ratio',
)


def _component_sizes(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(0, dtype=np.int64)
    lab, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(lab.ravel())[1:]


def component_metrics(pred, gt, small_px=20):
    """Return component counts and fragmentation/area ratios.

    ``small_components_*`` counts 8-connected components with fewer than
    ``small_px`` pixels.  ``fp_area_ratio`` and ``fn_area_ratio`` are
    normalised by the GT foreground area (clipped at one pixel).
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    sizes_p = _component_sizes(pred)
    sizes_g = _component_sizes(gt)
    n_p = len(sizes_p)
    n_g = len(sizes_g)
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    small_p = int(np.count_nonzero(sizes_p < small_px)) if n_p else 0
    small_g = int(np.count_nonzero(sizes_g < small_px)) if n_g else 0
    largest_p = int(sizes_p.max()) if n_p else 0
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    gt_denom = max(gt_area, 1)
    return {
        'components_pred': n_p,
        'components_gt': n_g,
        'component_count_error': abs(n_p - n_g),
        'small_components_pred': small_p,
        'small_component_rate': (small_p / n_p) if n_p else 0.0,
        'small_components_gt': small_g,
        'largest_component_ratio': (largest_p / pred_area)
        if pred_area else 0.0,
        'fp_area_ratio': fp / gt_denom,
        'fn_area_ratio': fn / gt_denom,
    }
