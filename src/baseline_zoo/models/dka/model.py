"""Self-contained DKA segmentation port (ICLR 2026).

The released DKA repository provides the dual-kernel adapter and
classification training code.  Its segmentation experiments use
Segmenter-B in MMSegmentation, which is not part of the release.  This module
keeps the official DKA module and reconstructs MMSegmentation's
Segmenter-B ViT-B/16 backbone in pure torch, including its exact state-dict
key layout, learned position embedding, 0.1 stochastic-depth schedule,
final LayerNorm and a linear (1x1) segmentation head.  The official
Segmenter-B checkpoint can therefore be loaded without key remapping.

DKA = down-projection to a narrow bottleneck, parallel depthwise 51x51
(large context) and 5x5 (local detail) convolutions, GELU, up-projection,
residual.  The paper initializes down/up projections to zero; the released
classification code has that initialization commented out, so the paper's
stated initialization is used here.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DKAModule(nn.Module):
    """Dual-Kernel Adapter (official DKA architecture)."""

    def __init__(self, input_dim: int, middle_dim: int = 192,
                 large_kernel: int = 51, small_kernel: int = 5):
        super().__init__()
        self.adapter_down = nn.Linear(input_dim, middle_dim, bias=False)
        self.adapter_up = nn.Linear(middle_dim, input_dim, bias=False)
        self.conv_large = nn.Conv2d(
            middle_dim, middle_dim, kernel_size=large_kernel, stride=1,
            padding=large_kernel // 2, groups=middle_dim, bias=False)
        self.conv_small = nn.Conv2d(
            middle_dim, middle_dim, kernel_size=small_kernel, stride=1,
            padding=small_kernel // 2, groups=middle_dim, bias=False)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        # Paper B.2: down-projection and up-projection initialize to zero.
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, x):
        batch, tokens, _ = x.shape
        reduced = self.act(self.adapter_down(x))
        side = int(round((tokens - 1) ** 0.5))
        if side * side != tokens - 1:
            raise ValueError(
                f'DKA expects a square patch grid, got {tokens} tokens')
        patches = reduced[:, 1:].reshape(
            batch, side, side, -1).permute(0, 3, 1, 2)
        patches = self.conv_large(patches) + self.conv_small(patches)
        patches = patches.permute(0, 2, 3, 1).reshape(
            batch, side * side, -1)
        reduced = torch.cat([reduced[:, :1], patches], dim=1)
        reduced = self.dropout(self.act(reduced))
        return x + self.adapter_up(reduced)


class _BlockWithDKA(nn.Module):
    """Official DKA insertion: apply the adapter after the transformer block."""

    def __init__(self, block: nn.Module, input_dim: int, middle_dim: int):
        super().__init__()
        self.block = block
        self.adapter = DKAModule(input_dim, middle_dim)

    def forward(self, x):
        return self.adapter(self.block(x))


class _DropPath(nn.Module):
    """Stochastic depth used by MMSeg's FFN (one rate per transformer layer)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class _PatchEmbed(nn.Module):
    """MMSeg PatchEmbed for the 512 input (16x16 patches, no padding needed)."""

    def __init__(self, in_channels: int = 3, embed_dims: int = 768,
                 patch_size: int = 16):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        size = (x.shape[-2], x.shape[-1])
        return x.flatten(2).transpose(1, 2), size


class _MMSegAttention(nn.Module):
    """MMSeg MultiheadAttention wrapper (state keys: ``attn.attn.*``)."""

    def __init__(self, embed_dims: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0.0, bias=True,
            batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class _MMSegFFN(nn.Module):
    """MMSeg FFN key layout (``ffn.layers.0.0`` and ``ffn.layers.1``)."""

    def __init__(self, embed_dims: int, feedforward_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.GELU(),
                nn.Dropout(0.0),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.layers(x)


class _MMSegBlock(nn.Module):
    """MMSeg TransformerEncoderLayer (attention has no DropPath; FFN does)."""

    def __init__(self, embed_dims: int, num_heads: int, mlp_ratio: float,
                 drop_path_rate: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.attn = _MMSegAttention(embed_dims, num_heads)
        self.ln2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.ffn = _MMSegFFN(
            embed_dims, int(embed_dims * mlp_ratio))
        self.drop_path = _DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


class SegmenterBackbone(nn.Module):
    """MMSegmentation Segmenter-B ViT-B/16 backbone (without the DKA wrapper)."""

    def __init__(self, input_size: int = 512, embed_dims: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1):
        super().__init__()
        side = int(input_size) // 16
        if side * 16 != int(input_size):
            raise ValueError('Segmenter-B input must be a multiple of 16')
        self.input_size = int(input_size)
        self.patch_embed = _PatchEmbed(3, embed_dims, 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, side * side + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(0.0)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList([
            _MMSegBlock(embed_dims, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)
        ])
        self.ln1 = nn.LayerNorm(embed_dims, eps=1e-6)

    def forward(self, image):
        batch = image.shape[0]
        tokens, (height, width) = self.patch_embed(image)
        cls_token = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_token, tokens], dim=1)
        x = self.drop_after_pos(x + self.pos_embed)
        for layer in self.layers:
            x = layer(x)
        x = self.ln1(x)
        x = x[:, 1:]
        return x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()


class SegmenterDKA(nn.Module):
    """Segmenter-B host with DKA after every transformer block."""

    def __init__(self, input_size: int = 512, middle_dim: int = 192,
                 num_classes: int = 2):
        super().__init__()
        self.input_size = int(input_size)
        self.middle_dim = int(middle_dim)
        self.num_classes = int(num_classes)
        self.embed_dims = 768
        self.vit = SegmenterBackbone(
            input_size=self.input_size, embed_dims=self.embed_dims,
            depth=12, num_heads=12, mlp_ratio=4.0, drop_path_rate=0.1)
        self.decoder = nn.Conv2d(self.embed_dims, self.num_classes, 1)
        # MMSeg BaseDecodeHead's Normal(std=0.01) override for conv_seg.
        nn.init.normal_(self.decoder.weight, std=0.01)
        nn.init.zeros_(self.decoder.bias)

    # ---- checkpoint ---------------------------------------------------
    @staticmethod
    def _unwrap_state_dict(payload):
        if not isinstance(payload, dict):
            raise RuntimeError('expected a Segmenter-B state dict')
        for key in ('state_dict', 'model'):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
        return {key.replace('module.', ''): value
                for key, value in payload.items()}

    def _resize_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        target = self.vit.pos_embed.shape[1]
        if pos_embed.shape[1] == target:
            return pos_embed
        source_side = int(math.sqrt(pos_embed.shape[1] - 1))
        target_side = int(math.sqrt(target - 1))
        cls_token = pos_embed[:, :1]
        patch = pos_embed[:, 1:].reshape(
            1, source_side, source_side, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(
            patch, size=(target_side, target_side), mode='bicubic',
            align_corners=False)
        patch = patch.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, patch], dim=1)

    def load_backbone(self, weights: str | Path) -> None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f'missing Segmenter-B checkpoint: {path}')
        state = self._unwrap_state_dict(torch.load(
            path, map_location='cpu', weights_only=False))
        if 'pos_embed' in state:
            state['pos_embed'] = self._resize_pos_embed(state['pos_embed'])
        missing, unexpected = self.vit.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f'{path}: Segmenter-B state dict mismatch '
                f'({len(missing)} missing, {len(unexpected)} unexpected)')

    # ---- DKA insertion / freezing -------------------------------------
    def insert_dka(self) -> None:
        if getattr(self.vit.layers[0], 'adapter', None) is not None:
            return
        self.vit.layers = nn.ModuleList([
            _BlockWithDKA(block, self.embed_dims, self.middle_dim)
            for block in self.vit.layers
        ])

    def freeze_backbone(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad_(False)
        for module in self.vit.layers:
            for parameter in module.adapter.parameters():
                parameter.requires_grad_(True)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The backbone parameters are frozen, but MMSeg's FFN DropPath must
        # still follow the training mode (it is stochastic depth, not state).
        # There is no BatchNorm in the ViT host, so leaving the frozen modules
        # in train mode is safe.  DKA and the head are the only trainable
        # parameters.
        for module in self.vit.layers:
            module.adapter.train(mode)
        self.decoder.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.vit(image)
        logits = self.decoder(features)
        return F.interpolate(
            logits, size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False)


def build_dka(vit_weights: str | Path | None,
              input_size: int = 512,
              middle_dim: int = 192) -> SegmenterDKA:
    """Build Segmenter-B + DKA, optionally from the official checkpoint."""
    model = SegmenterDKA(
        input_size=input_size, middle_dim=middle_dim, num_classes=2)
    if vit_weights is not None:
        model.load_backbone(vit_weights)
    model.insert_dka()
    model.freeze_backbone()
    return model


__all__ = ['DKAModule', 'SegmenterBackbone', 'SegmenterDKA', 'build_dka']
