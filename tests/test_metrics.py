"""Metric correctness tests: hand-computed values + official conventions."""

import numpy as np
import pytest

from core.metrics import (
    accuracy_score,
    asd_score,
    assd_score,
    betti_error,
    cldice_score,
    compute_metrics,
    dice_score,
    hausdorff_score,
    hd95_score,
    iou_score,
    mcc_score,
    nsd_score,
    precision_score,
    ravd_score,
    recall_score,
    sensitivity_score,
    skeleton_precision_score,
    skeleton_recall_score,
    specificity_score,
)


def test_pixel_metrics_hand_computed():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    # TP=1, FP=0, FN=1, TN=2
    # dice = 2*TP/(2*TP+FP+FN) = 2/(2+1) = 2/3 (medpy dc convention)
    assert dice_score(pred, gt) == pytest.approx(2 / 3)
    # iou = TP/(TP+FP+FN) = 1/2 (medpy jc convention)
    assert iou_score(pred, gt) == pytest.approx(1 / 2)
    assert precision_score(pred, gt) == pytest.approx(1.0)
    assert recall_score(pred, gt) == pytest.approx(0.5)
    assert sensitivity_score(pred, gt) == pytest.approx(0.5)
    assert specificity_score(pred, gt) == pytest.approx(1.0)
    assert accuracy_score(pred, gt) == pytest.approx(0.75)
    # mcc = (1*2 - 0*1) / sqrt(1*2*2*3) = 2/sqrt(12)
    assert mcc_score(pred, gt) == pytest.approx(2 / np.sqrt(12))


def test_medpy_empty_conventions():
    empty = np.zeros((4, 4), dtype=np.uint8)
    full = np.ones((4, 4), dtype=np.uint8)
    # both empty -> 1.0 (medpy ZeroDivisionError handling)
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    # degenerate divisions -> 0.0
    assert precision_score(empty, full) == 0.0
    assert recall_score(full, empty) == 0.0
    # distance metrics -> NaN on empty masks
    assert np.isnan(hd95_score(empty, full))
    assert np.isnan(assd_score(empty, full))
    assert np.isnan(ravd_score(full, empty))


def test_identical_masks():
    rng = np.random.default_rng(0)
    m = (rng.random((32, 32)) > 0.5).astype(np.uint8)
    assert dice_score(m, m) == pytest.approx(1.0)
    assert iou_score(m, m) == pytest.approx(1.0)
    assert hd95_score(m, m) == pytest.approx(0.0)
    assert hausdorff_score(m, m) == pytest.approx(0.0)
    assert assd_score(m, m) == pytest.approx(0.0)
    assert asd_score(m, m) == pytest.approx(0.0)
    assert nsd_score(m, m, tolerance_mm=1.0) == pytest.approx(1.0)
    assert cldice_score(m, m) == pytest.approx(1.0)


def test_cldice_empty_skeleton_is_nan():
    empty = np.zeros((8, 8), dtype=np.uint8)
    full = np.ones((8, 8), dtype=np.uint8)
    assert np.isnan(cldice_score(empty, full))


def test_skeleton_recall_and_precision():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[16, 4:28] = 1
    gt[4:28, 16] = 1
    assert skeleton_recall_score(gt, gt) == pytest.approx(1.0)
    assert skeleton_precision_score(gt, gt) == pytest.approx(1.0)

    # Only the horizontal branch: half of the GT skeleton is missed.
    pred = np.zeros_like(gt)
    pred[16, 4:28] = 1
    assert skeleton_recall_score(pred, gt) < 0.75
    # The predicted skeleton is entirely inside GT (no hallucinations).
    assert skeleton_precision_score(pred, gt) == pytest.approx(1.0)

    empty = np.zeros_like(gt)
    # Empty prediction covers none of the GT skeleton; its own skeleton is
    # undefined, so precision is NaN.
    assert skeleton_recall_score(empty, gt) == pytest.approx(0.0)
    assert np.isnan(skeleton_precision_score(empty, gt))


def test_betti_error_counts_components():
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    gt[10:13, 10:13] = 1
    pred = gt.copy()
    assert betti_error(pred, gt) == pytest.approx(0.0)
    # Cut one block in half -> exactly one extra foreground component
    pred2 = gt.copy()
    pred2[2:5, 3] = 0
    assert betti_error(pred2, gt) == pytest.approx(1.0)


def test_compute_metrics_keys_and_finiteness():
    rng = np.random.default_rng(1)
    gt = (rng.random((32, 32)) > 0.5).astype(np.float32)
    logits = (rng.normal(size=(32, 32)) * 2).astype(np.float32)
    m = compute_metrics(logits, gt)
    expected = {
        "dice", "iou", "precision", "recall", "sensitivity", "specificity",
        "accuracy", "mcc", "cldice", "skeleton_recall",
        "skeleton_precision", "hd", "hd95", "assd", "asd", "ravd",
        "nsd", "betti",
    }
    assert expected.issubset(m.keys())
    for k in ("dice", "iou", "precision", "recall", "accuracy", "mcc", "nsd"):
        assert np.isfinite(m[k]), k


def test_persistence_metrics_are_finite():
    """Regression: the optional persistence metrics must be computed, not NaN.

    A stale top-level ``losses.*`` import silently turned both metrics into
    NaN; this test uses a small line-like mask so both engines have work to do.
    """
    pytest.importorskip("networkx")
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[8:24, 8] = 1.0
    gt[16, 8:24] = 1.0
    logits = np.where(gt > 0, 2.0, -2.0).astype(np.float32)
    m = compute_metrics(
        logits, gt, compute_betti_matching=True, compute_topograph=True)
    assert "betti_matching" in m and np.isfinite(m["betti_matching"])
    assert "topograph_error" in m and np.isfinite(m["topograph_error"])
