"""Ported mask export path (synthetic calibration case)."""
import torch.nn.functional as F


def export_mask(logits, size):
    """Port: resize the probability map to the requested GT size."""
    return F.interpolate(logits, size=size, mode='nearest') > 0.5
