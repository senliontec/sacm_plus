"""A2SAM prompt fusion.

The official release's ``PromptGenerator`` is a two-input Conv-BN-LeakyReLU
network:

* uncertainty branch: 1 -> 16 (3x3) -> 16 (1x1)
* coarse-map branch:  1 -> 16 (3x3) -> 32 (1x1)
* concatenate 48 -> 48 (3x3) -> 1 (3x3)

The released module expects single-image tensors and unsqueezes them.  This
port keeps the same parameter names, shapes and channel progression but is
batch-safe.  ``mode='release'`` reproduces the release inputs
``(1 - uncertainty, coarse_logit)``; ``mode='paper'`` uses
``(uncertainty, coarse_mask)`` from paper Eq. (6).  The release detaches the
output from the loss; the paper says the fusion blocks are learned, so the
default here is trainable and ``detach=True`` is available for parity tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptFusion(nn.Module):
    def __init__(self, in_channels: int = 1, out_conv_channels: int = 1):
        super().__init__()
        if in_channels != 1:
            raise ValueError('A2SAM prompt fusion expects 1-channel maps')
        entmap_channels = 16
        logits_channels = 16
        embedding_channels = entmap_channels + logits_channels * 2
        self.out_conv_channels = out_conv_channels
        self.entmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, entmap_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(entmap_channels, entmap_channels, 1, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logits_conv = nn.Sequential(
            nn.Conv2d(in_channels, logits_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(logits_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(logits_channels, logits_channels * 2, 1, bias=False),
            nn.BatchNorm2d(logits_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(embedding_channels, embedding_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(embedding_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embedding_channels, out_conv_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _as_map(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(
                f'{name} must be [B,1,H,W] or [B,H,W], got '
                f'{tuple(tensor.shape)}')
        return tensor

    def forward(
        self,
        uncertainty: torch.Tensor,
        coarse: torch.Tensor,
        *,
        mode: str = 'release',
        detach: bool = False,
    ) -> torch.Tensor:
        uncertainty = self._as_map(uncertainty, 'uncertainty')
        coarse = self._as_map(coarse, 'coarse')
        if uncertainty.shape != coarse.shape:
            raise ValueError(
                f'prompt maps must have the same shape, got '
                f'{tuple(uncertainty.shape)} and {tuple(coarse.shape)}')
        if mode == 'release':
            uncertainty_input = 1.0 - uncertainty
            coarse_input = coarse
        elif mode == 'paper':
            uncertainty_input = uncertainty
            coarse_input = coarse
        else:
            raise ValueError(f'unknown prompt-fusion mode: {mode!r}')
        uncertainty_info = self.entmap_conv(uncertainty_input)
        pred_info = self.logits_conv(coarse_input)
        fused = self.final_conv(
            torch.cat((uncertainty_info, pred_info), dim=1))
        return fused.detach() if detach else fused


# The official class name; useful for state-dict/parity tests.
PromptGenerator = PromptFusion
