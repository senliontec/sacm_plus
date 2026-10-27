"""BLO-SAM: bilevel-optimized LoRA SAM for limited-data segmentation."""

from __future__ import annotations

from .lora import LoRA_Sam
from .losses import DiceLoss
from .segment_anything import sam_model_registry


def build_blo_sam(
    checkpoint: str,
    image_size: int = 256,
    num_classes: int = 1,
    rank: int = 4,
    pixel_mean=(0.0, 0.0, 0.0),
    pixel_std=(1.0, 1.0, 1.0),
) -> LoRA_Sam:
    """Build the upstream LoRA_Sam wrapper around a SAM ViT-B backbone."""
    sam, _ = sam_model_registry['vit_b'](
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=str(checkpoint),
        pixel_mean=list(pixel_mean),
        pixel_std=list(pixel_std),
    )
    return LoRA_Sam(sam, rank)


__all__ = ['LoRA_Sam', 'DiceLoss', 'build_blo_sam']
