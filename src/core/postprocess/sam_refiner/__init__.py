"""Self-contained SAMRefiner wrapper (ICLR 2025, MIT).

The upstream repository is vendored under this package so runtime code never
imports ``3rd/SAMRefiner``.  SAM weights are not bundled; pass a local SAM
checkpoint (ViT-H by default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .sam_refiner import sam_refiner
from .segment_anything import sam_model_registry


class SAMRefiner:
    """Model-agnostic binary-mask refinement with SAM prompts.

    Defaults follow the paper (single refinement pass, box margin 0.1,
    Gaussian amplitude 15, span 4).  The released code defaults to five
    passes, margin 0 and amplitude 30; pass those values explicitly to
    reproduce the code-default variant.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_type: str = 'vit_h',
        device: str | torch.device = 'cuda',
        iterations: int = 1,
        margin: float = 0.1,
        gamma: float = 4.0,
        strength: float = 15.0,
        use_point: bool = True,
        use_box: bool = True,
        use_mask: bool = True,
        add_neg: bool = True,
    ):
        if int(iterations) < 1:
            raise ValueError('SAMRefiner requires at least one pass')
        self.device = torch.device(device)
        self.model_type = model_type
        self.iterations = int(iterations)
        self.margin = float(margin)
        self.gamma = float(gamma)
        self.strength = float(strength)
        self.use_point = bool(use_point)
        self.use_box = bool(use_box)
        self.use_mask = bool(use_mask)
        self.add_neg = bool(add_neg)
        build = sam_model_registry[model_type]
        self.sam = build(checkpoint=str(checkpoint)).to(self.device)
        # Sam.device is a read-only property derived from pixel_mean.
        self.sam.eval()

    @torch.no_grad()
    def refine(self, image_path: str | Path,
               coarse_mask: np.ndarray) -> np.ndarray:
        """Refine one binary mask and return a boolean mask."""
        mask = np.asarray(coarse_mask)
        if mask.ndim != 2:
            raise ValueError(f'coarse_mask must be [H,W], got {mask.shape}')
        # Upstream's `coarse_masks[None: ,]` is a no-op for a 2-D mask; pass
        # the explicit [1, H, W] batch it expects.
        coarse = (mask > 0).astype(np.uint8)[None, ...]
        refined, _, _ = sam_refiner(
            str(image_path),
            coarse,
            self.sam,
            use_point=self.use_point,
            use_box=self.use_box,
            use_mask=self.use_mask,
            add_neg=self.add_neg,
            iters=self.iterations,
            margin=self.margin,
            gamma=self.gamma,
            strength=self.strength,
        )
        output = np.asarray(refined)
        if output.ndim == 3:
            output = output[0]
        return output.astype(bool)


__all__ = ['SAMRefiner']
