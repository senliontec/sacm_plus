"""Shared metric dispatch for prediction evaluation.

Prediction/GT arrays are binary or 0-1; named metrics use a 0.5 threshold,
matching the canonical paper-metric path in ``core.metrics``.
"""

from __future__ import annotations

from core.metrics import binary_metrics, compute_metrics


def compute_selected(pred, gt, metric_names, empty_policy='penalty'):
    pred_bin = pred > 0.5
    gt_bin = gt > 0.5
    if metric_names == 'all':
        values = compute_metrics(pred, gt, threshold=0.5)
        if empty_policy != 'nan':
            fallback = binary_metrics(
                pred_bin, gt_bin, empty_policy=empty_policy)
            values['cldice'] = fallback['cldice']
            values['hd95'] = fallback['hd95']
        return values
    # Route the paper metrics through the shared empty-mask convention so
    # an empty prediction is scored as a failure instead of disappearing
    # from clDice/HD95 through NaN filtering.
    values = binary_metrics(
        pred_bin, gt_bin, empty_policy=empty_policy)
    return {name: float(values[name]) for name in metric_names}
