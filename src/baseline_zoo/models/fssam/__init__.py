"""Self-contained FSSAM model package (SAM 2 + DINOv2)."""

from .model import FSSAM, Weighted_GAP, build_fssam, weighted_dice_loss

__all__ = ['FSSAM', 'Weighted_GAP', 'build_fssam', 'weighted_dice_loss']
