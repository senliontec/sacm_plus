"""Adaptive Topology Calibration (ATC).

ATC is an inference-only, label-free postprocess.  It expands an
under-segmented prediction along its skeleton, using only prediction
morphology:

* sparse cracks: 2-px skeleton dilation;
* wide/thick structures: 2-px skeleton dilation;
* dense mortar-like structures: 2-px skeleton dilation;
* thin dense structures: 1-px skeleton dilation;
* other structures: unchanged.

The width proxy is ``2 * foreground_area / skeleton_length`` (roughly twice
the mean predicted width).  The thresholds are the ones used for the
reported four-seed result.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import binary_dilation, disk, skeletonize


def apply_atc(
    mask: np.ndarray,
    *,
    fg_sparse: float = 0.005,
    width_wide: float = 10.0,
    fg_wide_min: float = 0.0,
    fg_dense: float = 0.09,
    fg_thick: float = 0.15,
    width_high: float = 15.0,
    fg_min: float = 0.04,
    width_narrow: float = 7.5,
) -> np.ndarray:
    """Return the ATC-calibrated binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    skeleton = skeletonize(mask)
    length = float(skeleton.sum())
    if length <= 0:
        return mask

    width = 2.0 * float(mask.sum()) / length
    foreground = float(mask.mean())

    if foreground < fg_sparse:
        radius = 2
    elif (width >= width_wide
          and fg_wide_min <= foreground < fg_dense):
        radius = 2
    elif foreground >= fg_thick and width <= width_high:
        radius = 2
    elif foreground >= fg_min and width < width_narrow:
        radius = 1
    elif foreground >= fg_thick and width > width_high:
        radius = 1
    else:
        return mask

    return mask | binary_dilation(skeleton, disk(radius))
