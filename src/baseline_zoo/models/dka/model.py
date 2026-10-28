"""Self-contained DKA segmentation port (ICLR 2026).

The released DKA repository provides the dual-kernel adapter and
classification training code.  Its segmentation experiments use
Segmenter-B in MMSegmentation, which is not part of the release.  This port
applies the official DKA module to a frozen torchvision ViT-B/16 encoder
with Segmenter's linear decoder, matching the paper's segmentation recipe
(DKA in every transformer block, middle dimension 192, head LR 1e-4, DKA LR
1e-3, 300-epoch official budget, batch 1).

DKA = down-projection to a narrow bottleneck, parallel depthwise 51x51
(large context) and 5x5 (local detail) convolutions, GELU, up-projection,
residual.  The paper initializes down/up projections to zero; the released
classification code has that initialization commented out, so the paper's
stated initialization is used here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


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
    def __init__(self, block: nn.Module, input_dim: int, middle_dim: int):
        super().__init__()
        self.block = block
        self.adapter = DKAModule(input_dim, middle_dim)

    def forward(self, x):
        return self.adapter(self.block(x))


class SegmenterDKA(nn.Module):
    """Frozen ViT-B/16 + DKA in every block + Segmenter linear decoder."""

    def __init__(self, input_size: int = 512, middle_dim: int = 192,
                 num_classes: int = 2):
        super().__init__()
        self.input_size = int(input_size)
        self.middle_dim = int(middle_dim)
        self.num_classes = int(num_classes)
        self.vit = vit_b_16(weights=None)
        self.hidden_dim = self.vit.hidden_dim
        self.patch_size = self.vit.patch_size
        blocks = [
            _BlockWithDKA(block, self.hidden_dim, self.middle_dim)
            for block in self.vit.encoder.layers
        ]
        # torchvision's Encoder calls ``self.layers(input)`` directly, so
        # the adapter-wrapped blocks must be a callable Sequential.
        self.vit.encoder.layers = nn.Sequential(*blocks)
        self.decoder = nn.Conv2d(
            self.hidden_dim, self.num_classes, kernel_size=1)

    def load_backbone(self, weights: str | Path) -> None:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f'missing ViT-B/16 weights: {path}')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        state = {key.replace('module.', ''): value
                 for key, value in state.items()}
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing
                       if not key.startswith('heads.head')]
        bad_unexpected = [key for key in unexpected
                          if not key.startswith('heads.head')]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f'{path}: ViT-B/16 weights do not match '
                f'({len(bad_missing)} missing, {len(bad_unexpected)} '
                f'unexpected); first missing: {bad_missing[:5]}; '
                f'first unexpected: {bad_unexpected[:5]}')

    def resize_position_embedding(self) -> None:
        """Interpolate the 224-position embedding to ``input_size``."""
        if self.input_size == 224:
            self.vit.image_size = 224
            return
        side = self.input_size // self.patch_size
        pos = self.vit.encoder.pos_embedding
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:].reshape(
            1, 14, 14, self.hidden_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(side, side), mode='bicubic',
            align_corners=False)
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        self.vit.encoder.pos_embedding = nn.Parameter(
            torch.cat([cls_pos, patch_pos], dim=1))
        self.vit.image_size = self.input_size
        self.vit.seq_length = side * side + 1

    def freeze_backbone(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad_(False)
        for module in self.vit.encoder.layers:
            for parameter in module.adapter.parameters():
                parameter.requires_grad_(True)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen transformer trunk remains in eval mode.
        self.vit.encoder.eval()
        self.vit.conv_proj.eval()
        self.vit.class_token.requires_grad_(False)
        for module in self.vit.encoder.layers:
            module.adapter.train(mode)
        self.decoder.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        tokens = self.vit._process_input(image)
        class_token = self.vit.class_token.expand(batch, -1, -1)
        x = torch.cat([class_token, tokens], dim=1)
        x = self.vit.encoder(x)
        patch = x[:, 1:]
        side = int(round(patch.shape[1] ** 0.5))
        features = patch.reshape(
            batch, side, side, self.hidden_dim).permute(0, 3, 1, 2)
        logits = self.decoder(features)
        return F.interpolate(
            logits, size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False)


def build_dka(vit_weights: str | Path | None,
              input_size: int = 512,
              middle_dim: int = 192) -> SegmenterDKA:
    model = SegmenterDKA(
        input_size=input_size, middle_dim=middle_dim, num_classes=2)
    if vit_weights is not None:
        model.load_backbone(vit_weights)
    model.resize_position_embedding()
    model.freeze_backbone()
    return model


__all__ = ['DKAModule', 'SegmenterDKA', 'build_dka']
