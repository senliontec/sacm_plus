"""Conv-LoRA + SACM-inspired global context and high-resolution decisions.

The Conv-LoRA qkv LoRA / multi-scale conv MoE / prompt-free decoder are kept.
A lightweight external-context residual aggregates a few encoder block
features; a second-stage refiner lifts the 256x256 decoder output to 512x512.
Both new paths are zero-initialized, so step 0 is numerically Conv-LoRA.

This is a new project model.  It is intentionally not the official SACM
baseline: SACM's buggy Stage-1/gating/head supervision are not reused.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from baseline_zoo.models.convlora import ConvLoRASAM


def _block_indices(count: int, fractions: Iterable[float]) -> list[int]:
    indices = {
        min(count - 1, max(0, int(round(float(f) * (count - 1)))))
        for f in fractions
    }
    return sorted(indices)


class ExternalContextAdapter(nn.Module):
    """Aggregate a few ViT block features into a global context residual."""

    def __init__(self, in_dims: Iterable[int], out_dim: int,
                 hidden: int = 64, pool_size: int = 16):
        super().__init__()
        in_dims = tuple(int(d) for d in in_dims)
        if not in_dims:
            raise ValueError('external context needs at least one block')
        self.pool_size = int(pool_size)
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, hidden, 1, bias=False) for dim in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * len(in_dims), hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, 1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, features: list[torch.Tensor],
                out_size: tuple[int, int]) -> torch.Tensor:
        pooled = []
        for projection, feature in zip(self.proj, features):
            pooled.append(F.adaptive_avg_pool2d(
                projection(feature), (self.pool_size, self.pool_size)))
        context = self.fuse(torch.cat(pooled, dim=1))
        if context.shape[-2:] != tuple(out_size):
            context = F.interpolate(
                context, size=out_size, mode='bilinear',
                align_corners=False)
        return context


class HighResRefiner(nn.Module):
    """Lift 256x256 coarse logits to 512x512 using a light image stem."""

    def __init__(self, out_size: int = 512, hidden: int = 16):
        super().__init__()
        if out_size != 512:
            raise ValueError('current ConvSACM refiner supports out_size=512')
        self.out_size = int(out_size)
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(1 + hidden, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, coarse_logits: torch.Tensor,
                images: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(
            coarse_logits, size=(self.out_size, self.out_size),
            mode='bilinear', align_corners=False)
        image_features = self.image_stem(images / 255.0)
        if image_features.shape[-2:] != coarse.shape[-2:]:
            image_features = F.interpolate(
                image_features, size=coarse.shape[-2:], mode='bilinear',
                align_corners=False)
        residual = self.fuse(torch.cat([coarse, image_features], dim=1))
        return coarse + residual


class ConvSACMSAM(ConvLoRASAM):
    """Conv-LoRA plus external context and a high-resolution second stage."""

    def __init__(self, base: ConvLoRASAM,
                 external_fractions: Iterable[float] = (0.25, 0.5, 0.75),
                 external_hidden: int = 64,
                 refiner_hidden: int = 16,
                 refiner_size: int = 512):
        super().__init__(base.sam, moe_weight=base.moe_weight)
        encoder = self.sam.image_encoder
        embed_dim = int(getattr(
            encoder, 'embed_dim', encoder.patch_embed.proj.out_channels))
        blocks = encoder.blocks
        self.external_indices = _block_indices(
            len(blocks), external_fractions)
        if not self.external_indices:
            raise ValueError('ConvSACM needs at least one external block')
        self.external = ExternalContextAdapter(
            [embed_dim] * len(self.external_indices),
            out_dim=int(encoder.neck[0].out_channels),
            hidden=external_hidden)
        self.refiner = HighResRefiner(
            out_size=refiner_size, hidden=refiner_hidden)
        self._block_features: list[torch.Tensor] = []
        self._external_handles = [
            blocks[index].register_forward_hook(self._capture_block)
            for index in self.external_indices
        ]

    def _capture_block(self, module, inputs, output):
        del module, inputs
        self._block_features.append(output)

    def _to_nchw(self, feature: torch.Tensor, embed_dim: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise RuntimeError(
                f'unexpected block output: {tuple(feature.shape)}')
        if feature.shape[-1] == embed_dim:
            return feature.permute(0, 3, 1, 2).contiguous()
        if feature.shape[1] == embed_dim:
            return feature
        raise RuntimeError(
            f'block channel mismatch: expected {embed_dim}, got '
            f'{tuple(feature.shape)}')

    def forward(self, images: torch.Tensor):
        self._block_features = []
        input_images = ((images / 255.0) - self.imagenet_mean) \
            / self.imagenet_std
        image_embeddings = self.sam.image_encoder(input_images)
        embed_dim = int(getattr(
            self.sam.image_encoder, 'embed_dim',
            self.sam.image_encoder.patch_embed.proj.out_channels))
        features = [
            self._to_nchw(feature, embed_dim)
            for feature in self._block_features
        ]
        if features:
            context = self.external(
                features, tuple(image_embeddings.shape[-2:]))
            image_embeddings = image_embeddings + context

        sparse, dense = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None)
        coarse, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        high_res = self.refiner(coarse, images)
        return coarse, high_res, self._moe_loss()


__all__ = ['ConvSACMSAM', 'ExternalContextAdapter', 'HighResRefiner']
