"""Self-contained SAMed binary model.

SAMed (arXiv 2023; often cited as MICCAI 2023) keeps SAM's image encoder,
injects rank-4 LoRA into the
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


class HighResRefinementHead(nn.Module):
    """Full-resolution residual refiner for thin structures.

    The SAMed mask decoder emits 128x128 logits at a 512 input, which is
    coarse for cracks, vessels and membranes.  This small head reads the
    input image at full resolution and adds a residual correction to the
    bilinearly upsampled SAMed logits.  The final projection is zero
    initialized, so the model starts exactly at SAMed's upsampled output
    and learns only the correction.  It is a task-specific adaptation, not
    part of the released SAMed architecture.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2,
                 channels: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, image: torch.Tensor,
                base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.body(image)


class SAMedNet(nn.Module):
    """SAMed LoRA model returning low-res and refined full-res logits."""

    def __init__(self, sam: nn.Module, rank: int = 4, num_classes: int = 1,
                 use_high_res_head: bool = True):
        super().__init__()
        self.lora_sam = LoRA_Sam(sam, r=rank)
        self.high_res_head = (
            HighResRefinementHead(
                in_channels=3, out_channels=num_classes + 1)
            if use_high_res_head else None)

    def forward(self, x: torch.Tensor):
        out = self.lora_sam(x, False, x.shape[-1])
        full_logits = out['masks']
        if self.high_res_head is not None:
            full_logits = self.high_res_head(x, full_logits)
        return out['low_res_logits'], full_logits


def build_samed(
    image_size: int = 512,
    num_classes: int = 1,
    sam_checkpoint: Optional[str] = None,
    rank: int = 4,
    sam_type: str = 'vit_b',
    use_high_res_head: bool = True,
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
    return SAMedNet(
        sam, rank=rank, num_classes=num_classes,
        use_high_res_head=use_high_res_head)
