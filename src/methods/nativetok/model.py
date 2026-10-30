"""Conv-LoRA with native-resolution residual tokenization.

The Conv-LoRA backbone, prompt-free decoder, losses and optimizer recipe are
unchanged.  A forward hook on SAM's patch embedding adds a gated residual
computed from native-resolution image patches.  The hook is bypassed when
``native_images`` is None, so the wrapped model is exactly Conv-LoRA at
step 0 and remains a drop-in baseline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from baseline_zoo.models.convlora import ConvLoRASAM

from .tokenizer import NativeResidualTokenizer


class NativeTokenSAM(ConvLoRASAM):
    """Conv-LoRA plus native-resolution coarse-to-fine tokenization."""

    def __init__(self, base: ConvLoRASAM,
                 scales: Iterable[int] = (8, 4),
                 hidden: int = 32,
                 alpha_init: float = 0.0):
        super().__init__(base.sam, moe_weight=base.moe_weight)
        patch_embed = self.sam.image_encoder.patch_embed
        self.tokenizer = NativeResidualTokenizer(
            patch_embed.proj,
            in_dim=int(patch_embed.proj.out_channels),
            scales=scales,
            hidden=hidden,
            image_size=self.image_size,
            alpha_init=alpha_init,
        )
        self._native_images: Sequence[torch.Tensor] | torch.Tensor | None = None
        self._handle = patch_embed.register_forward_hook(self._inject)

    def _inject(self, module, inputs, output):
        del module, inputs
        if self._native_images is None:
            return output
        return self.tokenizer(output, self._native_images)

    def forward(self, images: torch.Tensor,
                native_images: Sequence[torch.Tensor] | torch.Tensor | None
                = None):
        self._native_images = native_images
        try:
            logits, moe_loss = super().forward(images)
        finally:
            self._native_images = None
        return logits, moe_loss


__all__ = ['NativeTokenSAM']
