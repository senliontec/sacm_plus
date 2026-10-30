"""CurvField model wrapper around an unchanged Conv-LoRA baseline.

The baseline ``ConvLoRASAM`` is not modified.  CurvField registers two
forward hooks on the frozen SAM encoder:

1. ``patch_embed`` -> predict the structure field from frozen patch tokens;
2. selected transformer blocks -> apply a zero-initialized
   structure-conditioned adapter to the block output.

The mask decoder and the Conv-LoRA qkv experts remain exactly the official
Conv-LoRA ones.  A zero-initialized field->logit bias can additionally
condition the decoder decision; both residual paths start at zero, so the
initial model is numerically the Conv-LoRA baseline.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from baseline_zoo.models.convlora import ConvLoRASAM

from .adapter import StructureConditionedAdapter
from .head import FieldDecoderBias, StructureFieldHead


def _block_indices(count: int, fractions: Iterable[float]) -> list[int]:
    indices = set()
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError('adapter fractions must be in [0, 1]')
        indices.add(min(count - 1, max(0, int(round(value * (count - 1))))))
    return sorted(indices)


class CurvFieldSAM(ConvLoRASAM):
    """Conv-LoRA plus a shared structure field and field-conditioned adapters."""

    def __init__(self, base: ConvLoRASAM, field_hidden: int = 128,
                 field_size: int = 128, adapter_hidden: int = 64,
                 adapter_fractions: Iterable[float] = (0.25, 0.5, 0.75),
                 film_scale: float = 0.5, decoder_bias: bool = True,
                 static_field: bool = False):
        super().__init__(base.sam, moe_weight=base.moe_weight)
        self.static_field = bool(static_field)
        self.field_size = int(field_size)
        self.decoder_bias = bool(decoder_bias)
        if self.field_size not in (64, 128):
            raise ValueError('field_size must be 64 or 128')
        patch_embed = self.sam.image_encoder.patch_embed
        in_dim = int(patch_embed.proj.out_channels)
        self.field_head = StructureFieldHead(
            in_dim, hidden=field_hidden, out_size=self.field_size)
        self.field_decoder = FieldDecoderBias(hidden=32)
        blocks = self.sam.image_encoder.blocks
        indices = _block_indices(len(blocks), adapter_fractions)
        self.adapter_indices = indices
        self.adapters = nn.ModuleList([
            StructureConditionedAdapter(
                in_dim, hidden=adapter_hidden, film_scale=film_scale)
            for _ in indices
        ])
        self._field: torch.Tensor | None = None
        self._handles = [
            patch_embed.register_forward_hook(self._capture_field)]
        for index, adapter in zip(indices, self.adapters):
            self._handles.append(
                blocks[index].register_forward_hook(
                    self._make_adapter_hook(adapter)))

    def _capture_field(self, module, inputs, output):
        del module, inputs
        if self.static_field:
            return None
        # SAM patch embedding returns [B,H,W,C].
        if output.ndim != 4:
            raise RuntimeError(
                f'unexpected patch embedding output: {tuple(output.shape)}')
        field = output.permute(0, 3, 1, 2)
        self._field = self.field_head(field)
        return None

    def _make_adapter_hook(self, adapter: StructureConditionedAdapter):
        def hook(module, inputs, output):
            del module, inputs
            if self._field is None:
                return output
            return adapter(output, self._field)
        return hook

    def forward(self, images: torch.Tensor):
        if self.static_field:
            batch = int(images.shape[0])
            field = torch.zeros(
                batch, 6, self.field_size, self.field_size,
                device=images.device, dtype=images.dtype)
            field[:, 0] = 1.0   # centerline
            field[:, 1] = 0.3   # normalized radius
            field[:, 2] = 1.0   # horizontal double-angle orientation
            field[:, 5] = 1.0   # coherence
            self._field = field
        else:
            self._field = None
        logits, moe_loss = super().forward(images)
        if self.decoder_bias and self._field is not None:
            bias = self.field_decoder(self._field)
            if bias.shape[-2:] != logits.shape[-2:]:
                bias = F.interpolate(
                    bias, size=logits.shape[-2:], mode='bilinear',
                    align_corners=False)
            logits = logits + bias
        return logits, moe_loss, self._field


__all__ = ['CurvFieldSAM']
