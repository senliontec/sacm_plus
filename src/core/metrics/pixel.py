"""Pixel-level metrics — strict medpy conventions (verified against
medpy/metric/binary.py): no smoothing; dice/jaccard return 1.0 when BOTH
masks are empty; precision/recall/specificity return 0.0 on degenerate
divisions. accuracy/mcc are project extensions (no external official)."""

import numpy as np

from .common import _bool


def dice_score(pred, gt):
    """medpy `dc`: 2|A∩B|/(|A|+|B|), no smoothing; both empty -> 1.0."""
    pred = _bool(pred)
    gt = _bool(gt)
    intersection = np.count_nonzero(pred & gt)
    size_sum = np.count_nonzero(pred) + np.count_nonzero(gt)
    try:
        return float(2.0 * intersection / float(size_sum))
    except ZeroDivisionError:
        return 1.0


def iou_score(pred, gt):
    """medpy `jc`: |A∩B|/|A∪B|, no smoothing; both empty -> 1.0."""
    pred = _bool(pred)
    gt = _bool(gt)
    intersection = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    try:
        return float(intersection) / float(union)
    except ZeroDivisionError:
        return 1.0


def precision_score(pred, gt):
    """medpy `precision`: TP/(TP+FP); degenerate -> 0.0."""
    pred = _bool(pred)
    gt = _bool(gt)
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    try:
        return float(tp) / float(tp + fp)
    except ZeroDivisionError:
        return 0.0


def recall_score(pred, gt):
    """medpy `recall`: TP/(TP+FN); degenerate -> 0.0."""
    pred = _bool(pred)
    gt = _bool(gt)
    tp = np.count_nonzero(pred & gt)
    fn = np.count_nonzero(~pred & gt)
    try:
        return float(tp) / float(tp + fn)
    except ZeroDivisionError:
        return 0.0


def sensitivity_score(pred, gt):
    """medpy `sensitivity` == recall."""
    return recall_score(pred, gt)


def specificity_score(pred, gt):
    """medpy `specificity`: TN/(TN+FP); degenerate -> 0.0."""
    pred = _bool(pred)
    gt = _bool(gt)
    tn = np.count_nonzero(~pred & ~gt)
    fp = np.count_nonzero(pred & ~gt)
    try:
        return float(tn) / float(tn + fp)
    except ZeroDivisionError:
        return 0.0


def accuracy_score(pred, gt):
    """Project extension: (TP+TN)/total."""
    pred = _bool(pred)
    gt = _bool(gt)
    tp = np.count_nonzero(pred & gt)
    tn = np.count_nonzero(~pred & ~gt)
    total = pred.size
    return float((tp + tn) / float(total))


def mcc_score(pred, gt):
    """Project extension: Matthews correlation coefficient."""
    pred = _bool(pred)
    gt = _bool(gt)
    tp = np.count_nonzero(pred & gt)
    tn = np.count_nonzero(~pred & ~gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)
