"""Tests for component metrics and optional component pruning."""

import numpy as np

from core.metrics.components import component_metrics
from core.postprocess.components import prune_small_components


def test_component_metrics_counts_and_ratios():
    gt = np.zeros((16, 16), dtype=bool)
    gt[1:6, 1:6] = True
    gt[9:14, 9:14] = True
    pred = np.zeros((16, 16), dtype=bool)
    pred[1:6, 1:6] = True
    pred[9:14, 9:14] = True
    pred[0, 15] = True  # isolated 1-pixel false positive
    m = component_metrics(pred, gt)
    assert m['components_gt'] == 2
    assert m['components_pred'] == 3
    assert m['component_count_error'] == 1
    assert m['small_components_pred'] == 1
    assert m['small_component_rate'] > 0
    assert m['fp_area_ratio'] > 0
    assert m['fn_area_ratio'] == 0
    assert 0 < m['largest_component_ratio'] < 1


def test_prune_small_components_keeps_large_and_removes_specks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:16] = True   # long component
    mask[15, 15] = True      # isolated speck
    pruned, stats = prune_small_components(
        mask, min_area=5, min_skeleton=5)
    assert not pruned[15, 15]
    assert pruned[2:6, 2:16].any()
    assert stats['removed_components'] == 1
    assert stats['removed_pixels'] == 1


def test_prune_small_components_never_erases_everything():
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    pruned, stats = prune_small_components(
        mask, min_area=100, min_skeleton=100)
    assert pruned.any()
    assert stats['removed_components'] == 0


def test_prune_empty_mask_is_noop():
    mask = np.zeros((8, 8), dtype=bool)
    pruned, stats = prune_small_components(
        mask, min_area=10, min_skeleton=10)
    assert not pruned.any()
    assert stats['components_before'] == 0
