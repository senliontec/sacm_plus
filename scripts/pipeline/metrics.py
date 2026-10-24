"""Shared metric dispatch for prediction evaluation.

Prediction/GT arrays are binary or 0-1; named metrics use a 0.5 threshold,
matching the canonical paper-metric path in ``core.metrics``.
"""

from __future__ import annotations

from core.metrics import (
    cldice_score,
    compute_metrics,
    dice_score,
    hd95_score,
    iou_score,
)

METRIC_FNS = {
    'dice': dice_score,
    'iou': iou_score,
    'cldice': cldice_score,
    'hd95': hd95_score,
}


def compute_selected(pred, gt, metric_names):
    if metric_names == 'all':
        return compute_metrics(pred, gt, threshold=0.5)
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    return {
        name: float(METRIC_FNS[name](pred_bin, gt_bin))
        for name in metric_names
    }
