"""Baseline-compatible per-image binary metrics.

This is the exact metric set used by the baseline runner
(``src/baseline_zoo/runtime.py::binary_metrics``): Dice/IoU/clDice/HD95
on the binarized prediction plus an empty-prediction flag.  SACM-X calls
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


def binary_metrics(pred, gt):
    """Return the baseline's per-image metrics for one mask pair.

    Args:
        pred: binary prediction at the original GT resolution.
        gt: binary ground truth at the same resolution.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    return {
        'dice': dice_score(pred, gt),
        'iou': iou_score(pred, gt),
        'cldice': cldice_score(pred, gt),
        'hd95': hd95_score(pred, gt),
        'empty': float(pred.sum() == 0),
    }
