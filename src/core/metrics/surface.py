"""NSD (surface Dice at tolerance) — strict DeepMind surface-distance
implementation, verified against surface_distance/metrics.py and
lookup_tables.py (Bertels et al., MICCAI 2020). Surfel areas come from
the 2x2 neighbourhood-code lookup table with the given spacing, so even
at unit spacing diagonal edges contribute sqrt(2)-weighted areas.
Spacing is in pixel units (2D images, no physical metadata); tolerance
therefore in pixels. NaN when both masks are empty (official 0/0)."""

import numpy as np

from .common import _bool

_ENCODE_NEIGHBOURHOOD_2D_KERNEL = np.array([[8, 4], [2, 1]])


def create_table_neighbour_code_to_contour_length(spacing_mm):
    """DeepMind lookup_tables.create_table_neighbour_code_to_contour_length."""
    table = np.zeros([16])
    vertical = spacing_mm[0]
    horizontal = spacing_mm[1]
    diag = 0.5 * np.sqrt(spacing_mm[0] ** 2 + spacing_mm[1] ** 2)
    table[int("0001", 2)] = diag
    table[int("0010", 2)] = diag
    table[int("0011", 2)] = horizontal
    table[int("0100", 2)] = diag
    table[int("0101", 2)] = vertical
    table[int("0110", 2)] = 2 * diag
    table[int("0111", 2)] = diag
    table[int("1000", 2)] = diag
    table[int("1001", 2)] = 2 * diag
    table[int("1010", 2)] = vertical
    table[int("1011", 2)] = diag
    table[int("1100", 2)] = horizontal
    table[int("1101", 2)] = diag
    table[int("1110", 2)] = diag
    return table


def _compute_bounding_box(mask):
    """DeepMind metrics._compute_bounding_box (2D)."""
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)
    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return None, None
    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)
    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))
        max_over_axes.pop(axis)
        proj = np.amax(mask, axis=tuple(max_over_axes))
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)
    return bbox_min, bbox_max


def _crop_to_bounding_box(mask, bbox_min, bbox_max):
    """DeepMind metrics._crop_to_bounding_box (2D)."""
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                bbox_min[1]:bbox_max[1] + 1]
    return cropmask


def _sort_distances_surfels(distances, surfel_areas):
    sorted_surfels = np.array(sorted(zip(distances, surfel_areas)))
    return sorted_surfels[:, 0], sorted_surfels[:, 1]


def compute_surface_distances_deepmind(mask_gt, mask_pred, spacing_mm):
    """DeepMind metrics.compute_surface_distances (2D)."""
    from scipy import ndimage

    mask_gt = _bool(mask_gt)
    mask_pred = _bool(mask_pred)

    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    neighbour_code_to_surface_area = create_table_neighbour_code_to_contour_length(spacing_mm)
    kernel = _ENCODE_NEIGHBOURHOOD_2D_KERNEL
    full_true_neighbours = 0b1111

    neighbour_code_map_gt = ndimage.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    borders_gt = ((neighbour_code_map_gt != 0) &
                  (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    if borders_gt.any():
        distmap_gt = ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.inf * np.ones(borders_pred.shape)

    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    if distances_gt_to_pred.shape != (0,):
        distances_gt_to_pred, surfel_areas_gt = _sort_distances_surfels(
            distances_gt_to_pred, surfel_areas_gt)
    if distances_pred_to_gt.shape != (0,):
        distances_pred_to_gt, surfel_areas_pred = _sort_distances_surfels(
            distances_pred_to_gt, surfel_areas_pred)

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    """DeepMind metrics.compute_surface_dice_at_tolerance."""
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
        np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice


def nsd_score(pred, gt, spacing_mm=(1.0, 1.0), tolerance_mm=1.0):
    """DeepMind NSD (surface Dice at tolerance), pixel units."""
    try:
        sd = compute_surface_distances_deepmind(gt, pred, spacing_mm)
    except ImportError:
        return float("nan")
    total_area = float(np.sum(sd["surfel_areas_gt"]) + np.sum(sd["surfel_areas_pred"]))
    if total_area == 0:
        return float("nan")
    return float(compute_surface_dice_at_tolerance(sd, tolerance_mm))
