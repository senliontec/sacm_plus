"""Baseline-compatible per-image binary metrics.

This is the exact metric set used by the baseline runner
(``src/baseline_zoo/runtime.py::binary_metrics``): Dice/IoU/clDice/HD95
on the binarized prediction plus an empty-prediction flag.  The metric helpers use
this module and then macro-averages over the validation images, so its
W&B ``val/*`` keys and checkpoint-selection rule match the baselines.

Do not add the full 16-metric suite here; ``aggregate.compute_metrics``
remains the entry point for the rich table style.
"""

import numpy as np

from .distance import hd95_score
from .pixel import dice_score, iou_score
from .topology import cldice_score

BINARY_METRICS = ('dice', 'iou', 'cldice', 'hd95')
EMPTY_POLICIES = ('penalty', 'nan')


def _empty_pair_metrics(pred, gt, policy):
    """Define a finite convention for empty masks.

    The raw official implementations return NaN for clDice/HD95 on empty
    masks.  Averaging those values with ``nanmean`` silently removes the
    worst images/datasets from the reported metric, which can make a model
    that predicts empty masks look better than one that predicts a wrong
    nonempty mask.  The default ``penalty`` policy keeps every image in the
    denominator:

    * both empty: perfect segmentation (1/1/1, HD95=0);
    * exactly one empty: Dice/IoU/clDice=0, HD95=image diagonal.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())
    if not pred_empty and not gt_empty:
        return None
    if policy == 'nan':
        return {
            'dice': dice_score(pred, gt),
            'iou': iou_score(pred, gt),
            'cldice': cldice_score(pred, gt),
            'hd95': hd95_score(pred, gt),
        }
    if pred_empty and gt_empty:
        return {'dice': 1.0, 'iou': 1.0, 'cldice': 1.0, 'hd95': 0.0}
    height, width = pred.shape[-2:]
    diagonal = float(np.hypot(height, width))
    return {'dice': 0.0, 'iou': 0.0, 'cldice': 0.0, 'hd95': diagonal}


def binary_metrics(pred, gt, empty_policy='penalty'):
    """Return the baseline's per-image metrics for one mask pair.

    Args:
        pred: binary prediction at the original GT resolution.
        gt: binary ground truth at the same resolution.
        empty_policy: ``penalty`` (default) keeps empty masks in the
            denominator with a finite worst-case convention; ``nan`` keeps
            the raw official medpy/clDice semantics.
    """
    if empty_policy not in EMPTY_POLICIES:
        raise ValueError(
            f'empty_policy must be one of {EMPTY_POLICIES}, '
            f'got {empty_policy!r}')
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    values = _empty_pair_metrics(pred, gt, empty_policy)
    if values is None:
        values = {
            'dice': dice_score(pred, gt),
            'iou': iou_score(pred, gt),
            'cldice': cldice_score(pred, gt),
            'hd95': hd95_score(pred, gt),
        }
    # Non-empty but skeleton-disjoint masks yield 0/0 in cldice_score.
    # Under the penalty policy that is a worst-case topology failure, not a
    # reason to silently drop the image from the denominator.
    if empty_policy == 'penalty' and np.isnan(values['cldice']):
        values['cldice'] = 0.0
    values['empty'] = float(not bool(pred.any()))
    return values
