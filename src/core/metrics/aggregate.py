"""Aggregate entry point: compute_metrics — the full metric suite for one
image. Prediction input convention: RAW LOGITS, binarized at threshold
0.0 (probability 0.5, SAM's mask_threshold). The original SACM
evaluation used 0.5 on logits (probability ~0.62), underestimating every
metric."""

import numpy as np

from .common import binarize, to_numpy
from .pixel import (
    accuracy_score,
    dice_score,
    iou_score,
    mcc_score,
    precision_score,
    recall_score,
    sensitivity_score,
    specificity_score,
)
from .distance import asd_score, assd_score, hausdorff_score, hd95_score, ravd_score
from .surface import nsd_score
from .topology import betti_error, cldice_score


def _auc_scores(probs, gt, n_thresholds=9):
    """Threshold-robustness curves: mean metric value over a uniform sweep
    of probability thresholds in [0.1, 0.9]. Defends the metrics against
    the binarization-threshold convention."""
    ts = np.linspace(0.1, 0.9, n_thresholds)
    dice_vals, cld_vals = [], []
    for t in ts:
        p = (probs > t).astype(np.uint8)
        dice_vals.append(dice_score(p, gt))
        c = cldice_score(p, gt)
        # At extreme thresholds the binarized prediction can lose its whole
        # skeleton (official clDice -> NaN); count it as zero contribution.
        cld_vals.append(c if not np.isnan(c) else 0.0)
    return float(np.mean(dice_vals)), float(np.mean(cld_vals))


def compute_metrics(
    pred_logits,
    gt,
    threshold=0.0,
    gt_threshold=0.5,
    nsd_tolerance=1.0,
    compute_auc=False,
    compute_betti_matching=False,
    compute_topograph=False,
    metric_names=None,
):
    """Compute the requested metric suite for one image.

    Args:
        pred_logits: RAW LOGITS from the mask decoder, shape [H, W],
            [1, H, W] or [1, 1, H, W]. Binarized at threshold=0.0.
        gt: ground-truth mask, same shapes, values in [0, 1] or {0, 1}.
        gt_threshold: binarization threshold for the GT (0.5; binary
            masks are unaffected).
        nsd_tolerance: tolerance (pixels) for the surface Dice / NSD.
        compute_auc: additionally compute dice_auc / cldice_auc
            (threshold-sweep means; ~9x skeletonization cost).
        compute_betti_matching: additionally compute betti_matching
            (persistence-based, official engine, at 256 resolution;
            slow — seconds per image).
        compute_topograph: additionally compute topograph_error
            (official engine, at 256 resolution; needs networkx/scipy).
        metric_names: optional iterable of names to compute.  ``None``
            (default) computes the full suite.  Requested metrics are
            skipped entirely, not computed and discarded, which matters
            for the expensive distance/topology metrics.  AUC and the
            optional engines are computed only when both their flag and
            their name are requested.
    Returns:
        dict with the requested subset of dice, iou, precision, recall,
        sensitivity, specificity, accuracy, mcc, cldice, hd, hd95, assd,
        asd, ravd, nsd, betti (+ dice_auc, cldice_auc when compute_auc,
        and betti_matching / topograph_error when their flags are set).
        Distance/topology metrics are NaN when a mask is empty or a
        dependency is missing; callers aggregate with NaN exclusion.
    """
    requested = None if metric_names is None else set(metric_names)

    def want(name):
        return requested is None or name in requested

    pred = binarize(to_numpy(pred_logits).squeeze(), threshold)
    ref = binarize(to_numpy(gt).squeeze(), gt_threshold)

    metrics = {}
    if want("dice"):
        metrics["dice"] = dice_score(pred, ref)
    if want("iou"):
        metrics["iou"] = iou_score(pred, ref)
    if want("precision"):
        metrics["precision"] = precision_score(pred, ref)
    if want("recall"):
        metrics["recall"] = recall_score(pred, ref)
    if want("sensitivity"):
        metrics["sensitivity"] = sensitivity_score(pred, ref)
    if want("specificity"):
        metrics["specificity"] = specificity_score(pred, ref)
    if want("accuracy"):
        metrics["accuracy"] = accuracy_score(pred, ref)
    if want("mcc"):
        metrics["mcc"] = mcc_score(pred, ref)
    if want("cldice"):
        metrics["cldice"] = cldice_score(pred, ref)
    if want("hd"):
        metrics["hd"] = hausdorff_score(pred, ref)
    if want("hd95"):
        metrics["hd95"] = hd95_score(pred, ref)
    if want("assd"):
        metrics["assd"] = assd_score(pred, ref)
    if want("asd"):
        metrics["asd"] = asd_score(pred, ref)
    if want("ravd"):
        metrics["ravd"] = ravd_score(pred, ref)
    if want("nsd"):
        metrics["nsd"] = nsd_score(pred, ref, tolerance_mm=nsd_tolerance)
    if want("betti"):
        metrics["betti"] = betti_error(pred, ref)
    if compute_auc and (requested is None or (
            "dice_auc" in requested or "cldice_auc" in requested)):
        probs = 1.0 / (1.0 + np.exp(-to_numpy(pred_logits).squeeze().astype(np.float64)))
        dice_auc, cldice_auc = _auc_scores(probs, ref)
        if want("dice_auc"):
            metrics["dice_auc"] = dice_auc
        if want("cldice_auc"):
            metrics["cldice_auc"] = cldice_auc
    if compute_betti_matching and want("betti_matching"):
        from .persistence import betti_matching_score
        metrics["betti_matching"] = betti_matching_score(pred_logits, gt)
    if compute_topograph and want("topograph_error"):
        from .persistence import topograph_error_score
        metrics["topograph_error"] = topograph_error_score(pred_logits, gt)
    return metrics
