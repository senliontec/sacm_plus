"""Optional component-level cleanup for fragmentation.

The audit found a large excess of tiny predicted components (CREMI ~1027
vs GT ~12; ROAD ~400 vs ~6; retinal ~27-111 vs ~3-12).  This module removes
components that are BOTH small and short in skeleton length, while always
keeping the largest component(s).  It is intentionally opt-in and must be
calibrated on validation/support data, not on test labels.

``relative`` mode scales the thresholds with the image dimensions; absolute
mode uses pixel counts directly.
"""

import numpy as np
from scipy.ndimage import find_objects, label

try:
    from skimage.morphology import skeletonize as _skeletonize
except ImportError:  # pragma: no cover - skimage is a project dependency
    _skeletonize = None


def _skeleton_length(mask, bbox):
    if _skeletonize is None:
        return int(mask.sum())
    y0, y1, x0, x1 = bbox
    crop = mask[y0:y1, x0:x1]
    if not crop.any():
        return 0
    return int(_skeletonize(crop).sum())


def prune_small_components(mask, min_area=0, min_skeleton=0,
                           rel_area=0.0, rel_skeleton=0.0,
                           keep_largest=1):
    """Remove components that are both small and short.

    Args:
        mask: bool/uint8 [H, W] binary prediction.
        min_area: absolute area threshold in pixels.
        min_skeleton: absolute skeleton-length threshold in pixels.
        rel_area: image-area-relative area threshold (e.g. 1e-5).
        rel_skeleton: image-size-relative skeleton threshold (e.g. 1e-2).
        keep_largest: always retain this many largest components.

    Returns:
        (pruned_mask, stats_dict)
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    stats = {
        'components_before': 0,
        'components_after': 0,
        'removed_components': 0,
        'removed_pixels': 0,
        'area_threshold': max(float(min_area), float(rel_area) * H * W),
        'skeleton_threshold': max(
            float(min_skeleton), float(rel_skeleton) * max(H, W)),
    }
    if not mask.any():
        return mask.copy(), stats

    lab, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    stats['components_before'] = int(n)
    if n == 0:
        return mask.copy(), stats

    sizes = np.bincount(lab.ravel())
    order = np.argsort(sizes[1:])[::-1] + 1
    keep_ids = set(int(i) for i in order[:max(int(keep_largest), 0)])
    area_thr = stats['area_threshold']
    skel_thr = stats['skeleton_threshold']

    keep = mask.copy()
    if area_thr <= 0 and skel_thr <= 0:
        stats['components_after'] = int(n)
        return keep, stats

    remove_ids = []
    slices = find_objects(lab)
    for comp_id in range(1, n + 1):
        area = int(sizes[comp_id])
        if comp_id in keep_ids or area >= area_thr:
            continue
        sl = slices[comp_id - 1]
        if sl is None:
            continue
        local = lab[sl] == comp_id
        if _skeleton_length(local, (0, local.shape[0], 0, local.shape[1])) < skel_thr:
            remove_ids.append(comp_id)

    if remove_ids:
        remove = np.isin(lab, remove_ids)
        keep[remove] = False
        stats['removed_components'] = len(remove_ids)
        stats['removed_pixels'] = int(sum(int(sizes[i]) for i in remove_ids))

    # Always preserve at least one component if the rule would erase all.
    if not keep.any() and mask.any():
        keep = mask.copy()
        stats['removed_components'] = 0
        stats['removed_pixels'] = 0

    stats['components_after'] = int(
        label(keep, structure=np.ones((3, 3), dtype=np.uint8))[1]
        if keep.any() else 0)
    return keep, stats
