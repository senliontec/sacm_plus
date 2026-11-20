"""Adaptive skeleton calibration (AS) for BCTD-SAM.

AS is an inference-only, label-free postprocess.  It expands an
under-segmented prediction along its skeleton, gated by the predicted width
and foreground density:

* medium-width, non-dense structures: leave the mask unchanged;
* dense thick structures: dilate the full mask by ``thick_radius``;
* dense thin structures: dilate the full mask by ``dense_thin_radius``;
* otherwise: dilate the skeleton by ``radius`` and union with the mask.

The width proxy is ``2 * foreground_area / skeleton_length``.  The thresholds
are the ones used for the reported BCTD four-seed result; they are label-free
but benchmark-selected, so validate them on a held-out split before a paper
claim.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import binary_dilation, disk, skeletonize


def apply_adaptive_skel(
    mask: np.ndarray,
    *,
    width_low: float = 7.0,
    width_high: float = 15.0,
    fg_min: float = 0.04,
    fg_dense: float = 0.09,
    fg_thick: float = 0.15,
    thick_radius: int = 2,
    fg_dense_thin: float = 0.09,
    width_dense_thin: float = 8.0,
    dense_thin_radius: int = 1,
    radius: int = 1,
) -> np.ndarray:
    """Return the BCTD adaptive-skeleton calibrated binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    skeleton = skeletonize(mask)
    length = float(skeleton.sum())
    if length <= 0:
        return mask

    width = 2.0 * float(mask.sum()) / length
    foreground = float(mask.mean())
    medium = (width_low <= width <= width_high
              and foreground < fg_dense)
    apply = ((foreground >= fg_min and not medium)
             or foreground >= fg_thick)
    if not apply:
        return mask

    if foreground >= fg_thick and width <= width_high:
        return mask | binary_dilation(mask, disk(thick_radius))
    if (foreground >= fg_dense_thin
            and width <= width_dense_thin):
        return mask | binary_dilation(mask, disk(dense_thin_radius))
    return mask | binary_dilation(skeleton, disk(radius))
