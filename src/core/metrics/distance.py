"""Boundary-distance metrics — strict medpy conventions (verified against
medpy/metric/binary.py): 1-px surface borders (binary erosion,
connectivity 1), pixel units, pooled aggregation for hd95/assd.
NaN on empty masks (medpy raises RuntimeError; NaN is the batch-eval
friendly equivalent)."""

import numpy as np

from .common import _bool


def _surface_distances_medpy(result, reference):
    """medpy `__surface_distances` (2D, pixel spacing): distances from
    result border pixels to the nearest reference border pixel.

    Border = mask XOR 1-px binary erosion (connectivity 1). Raises the
    same RuntimeError as medpy on empty inputs; callers turn it into NaN.
    """
    from scipy.ndimage import (
        binary_erosion,
        distance_transform_edt,
        generate_binary_structure,
    )

    result = _bool(result)
    reference = _bool(reference)
    if np.count_nonzero(result) == 0:
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if np.count_nonzero(reference) == 0:
        raise RuntimeError("The second supplied array does not contain any binary object.")

    footprint = generate_binary_structure(result.ndim, 1)
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    dt = distance_transform_edt(~reference_border)
    return dt[result_border]


def hausdorff_score(pred, gt):
    """medpy `hd`: max symmetric surface distance."""
    try:
        hd1 = _surface_distances_medpy(pred, gt).max()
        hd2 = _surface_distances_medpy(gt, pred).max()
        return float(max(hd1, hd2))
    except RuntimeError:
        return float("nan")
    except ImportError:
        return float("nan")


def hd95_score(pred, gt):
    """medpy `hd95`: pooled 95th percentile of both directional surface
    distance sets. NaN on empty masks or missing scipy."""
    try:
        hd1 = _surface_distances_medpy(pred, gt)
        hd2 = _surface_distances_medpy(gt, pred)
    except RuntimeError:
        return float("nan")
    except ImportError:
        return float("nan")
    return float(np.percentile(np.hstack((hd1, hd2)), 95))


def assd_score(pred, gt):
    """medpy `assd`: mean of the pooled bidirectional surface distances.
    NaN on empty masks or missing scipy."""
    try:
        sd1 = _surface_distances_medpy(pred, gt)
        sd2 = _surface_distances_medpy(gt, pred)
    except RuntimeError:
        return float("nan")
    except ImportError:
        return float("nan")
    return float(np.concatenate((sd1, sd2)).mean())


def asd_score(pred, gt):
    """medpy `asd` (directed): mean distance from PRED border to GT
    border. NaN on empty masks or missing scipy."""
    try:
        sd = _surface_distances_medpy(pred, gt)
    except RuntimeError:
        return float("nan")
    except ImportError:
        return float("nan")
    return float(sd.mean())


def ravd_score(pred, gt):
    """medpy `ravd`: relative absolute volume difference (vol1-vol2)/vol2.
    NaN when the reference (GT) is empty (medpy raises RuntimeError)."""
    pred = _bool(pred)
    gt = _bool(gt)
    vol1 = np.count_nonzero(pred)
    vol2 = np.count_nonzero(gt)
    if vol2 == 0:
        return float("nan")
    return float((vol1 - vol2) / float(vol2))
