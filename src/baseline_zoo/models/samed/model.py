"""Self-contained SAMed binary model.

SAMed (MICCAI 2023) keeps SAM's image encoder, injects rank-4 LoRA into the
q/v projections, makes the prompt encoder and mask decoder trainable, and
uses class-wise mask tokens for semantic segmentation.  The official forward
uses no prompts (``points=None, boxes=None, masks=None``).  This port uses a
binary head (background + foreground) and the official direct-512 input.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .lora import LoRA_Sam
from .segment_anything import (
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
)

BUILDERS = {
    'vit_b': build_sam_vit_b,
    'vit_l': build_sam_vit_l,
    'vit_h': build_sam_vit_h,
}


class SAMedNet(nn.Module):
    """SAMed LoRA model returning low-res logits and upscaled masks."""

    def __init__(self, sam: nn.Module, rank: int = 4):
        super().__init__()
        self.lora_sam = LoRA_Sam(sam, r=rank)

    def forward(self, x: torch.Tensor):
        out = self.lora_sam(x, False, x.shape[-1])
        return out['low_res_logits'], out['masks']


def build_samed(
    image_size: int = 512,
    num_classes: int = 1,
    sam_checkpoint: Optional[str] = None,
    rank: int = 4,
    sam_type: str = 'vit_b',
) -> SAMedNet:
    """Build the official SAMed architecture (binary by default).

    ``sam_checkpoint=None`` builds a random base SAM for prediction; the
    complete SAMed checkpoint is loaded by the runtime afterwards.
    """
    if sam_type not in BUILDERS:
        raise ValueError(f'unknown SAM type: {sam_type}')
    sam, _ = BUILDERS[sam_type](
        image_size=image_size,
        num_classes=num_classes,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
        checkpoint=sam_checkpoint,
    )
    return SAMedNet(sam, rank=rank)
