"""Official mask export path (synthetic calibration case)."""
import torch.nn.functional as F


def export_mask(logits, size):
    """Official: resize the probability map with bilinear interpolation."""
    return F.interpolate(
        logits, size=size, mode='bilinear', align_corners=False
    ) > 0.5
