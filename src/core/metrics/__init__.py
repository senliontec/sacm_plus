"""Evaluation metrics package.

Strict-fidelity implementations, each function reproducing the behaviour
of its official source (verified line-by-line):

  medpy (metric/binary.py): dice, iou, precision, recall, sensitivity,
    specificity, hd, hd95, assd, asd, ravd — no smoothing, empty-empty
    -> 1.0, degenerate -> 0.0, 1-px surface borders.
  DeepMind surface-distance (metrics.py + lookup_tables.py, MICCAI 2020):
    nsd — surfel-area-weighted surface Dice at tolerance.
  clDice official repository (clDice_metric/clDice.py, CVPR 2021):
    cldice — hard skeletonization, no smoothing, NaN on empty skeletons.
  Official engines (via the losses package): betti_matching,
    topograph_error (256-resolution evaluation).

Project extensions (no external official implementation): accuracy, mcc,
betti (count-based), dice_auc / cldice_auc (threshold-sweep robustness).

Direct usage:

    from metrics import compute_metrics
    m = compute_metrics(logits, gt_mask, compute_auc=True)
"""

from .common import binarize, to_numpy, skeletonize
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
from .distance import (
    asd_score,
    assd_score,
    hausdorff_score,
    hd95_score,
    ravd_score,
)
from .surface import (
    compute_surface_dice_at_tolerance,
    compute_surface_distances_deepmind,
    nsd_score,
)
from .topology import betti_error, cldice_score
from .persistence import betti_matching_score, topograph_error_score
from .aggregate import compute_metrics
