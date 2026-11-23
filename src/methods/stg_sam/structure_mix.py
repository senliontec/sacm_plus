"""SCRWKV-inspired structure mixing for the SAM mask decoder.

Faithful port of the official SCRWKV modules (3rd/SCRWKV, ICML 2026):

* ``scrwkv_amcm.AMCM`` is copied verbatim from the official source.
* ``GBST`` is ported verbatim from ``models/SCIU.py`` (zero-filled
  directional channel-quarter shifts, outward/inward groups).
* ``DSCDGate`` keeps the official ``dynamic_w = sigmoid(conv(x))`` and
  ``(1 - dynamic_w)`` decay semantics, applied as a feature gate.

Deviation: the official bi-directional WKV CUDA recurrence is omitted.  It
requires an sm_86 custom kernel, while the local GPUs are sm_75, and the
SAM decoder already performs global token attention.  Everything else in
the block (conv256->64, AMCM, GBST, DSCD gate, conv64->256, residual)
follows the official SCIU block.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .scrwkv_amcm import AMCM


def GBST(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    """Official GBST (SCIU.py): zero-filled channel-quarter shifts."""
    B, N, C = input.shape
    input_reshaped = input.transpose(1, 2).reshape(
        B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input_reshaped.shape
    output = torch.zeros_like(input_reshaped)
    C_half = C // 2
    enlarge_total = C_half
    C_enlarge_quarter = C_half // 4
    enlarge_rem = enlarge_total - 4 * C_enlarge_quarter
    start, end = 0, C_enlarge_quarter
    output[:, start:end, :, shift_pixel:W] = input_reshaped[
        :, start:end, :, 0:W - shift_pixel]
    start, end = C_enlarge_quarter, C_enlarge_quarter * 2
    output[:, start:end, :, 0:W - shift_pixel] = input_reshaped[
        :, start:end, :, shift_pixel:W]
    start, end = C_enlarge_quarter * 2, C_enlarge_quarter * 3
    output[:, start:end, shift_pixel:H, :] = input_reshaped[
        :, start:end, 0:H - shift_pixel, :]
    start = C_enlarge_quarter * 3
    output[:, start:C_half, 0:H - shift_pixel, :] = input_reshaped[
        :, start:C_half, shift_pixel:H, :]
    if enlarge_rem > 0:
        start = C_enlarge_quarter * 4
        end = start + enlarge_rem
        output[:, start:end] = input_reshaped[:, start:end]
    shrink_channels_start_idx = C_half
    C_shrink = C - C_half
    C_shrink_quarter = C_shrink // 4
    shrink_rem = shrink_channels_start_idx - 4 * C_shrink_quarter
    start, end = shrink_channels_start_idx, shrink_channels_start_idx + C_shrink_quarter
    output[:, start:end, :, 0:W - shift_pixel] = input_reshaped[
        :, start:end, :, shift_pixel:W]
    start, end = shrink_channels_start_idx + C_shrink_quarter, shrink_channels_start_idx + C_shrink_quarter * 2
    output[:, start:end, :, shift_pixel:W] = input_reshaped[
        :, start:end, :, 0:W - shift_pixel]
    start, end = shrink_channels_start_idx + C_shrink_quarter * 2, shrink_channels_start_idx + C_shrink_quarter * 3
    output[:, start:end, 0:H - shift_pixel, :] = input_reshaped[
        :, start:end, shift_pixel:H, :]
    start = shrink_channels_start_idx + C_shrink_quarter * 3
    output[:, start:C, shift_pixel:H, :] = input_reshaped[
        :, start:C, 0:H - shift_pixel, :]
    if shrink_rem > 0:
        start = C_half + C_shrink_quarter * 4
        end = start + shrink_rem
        output[:, start:end] = input_reshaped[:, start:end]
    return output.flatten(2).transpose(1, 2)


class DSCDGate(nn.Module):
    """Official dynamic decay semantics as a feature gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.dynamic_w = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dynamic_w = torch.sigmoid(self.dynamic_w(x))
        return x * (1.0 - dynamic_w)


class StructureMix(nn.Module):
    def __init__(self, dim: int = 256, shift_pixel: int = 1):
        super().__init__()
        self.out_dim = dim
        self.hidden_dim = dim // 4
        self.shift_pixel = shift_pixel
        self.conv256to64 = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, 3, 1, 1,
                      groups=self.hidden_dim),
            nn.GroupNorm(num_groups=self.hidden_dim // 16,
                         num_channels=self.hidden_dim),
            nn.GELU(),
        )
        self.amcm = AMCM(self.hidden_dim)
        self.dscd = DSCDGate(self.hidden_dim)
        self.conv64to256 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, dim, 1),
            nn.GroupNorm(num_groups=dim // 32, num_channels=dim),
            nn.GELU(),
        )
        self.conlast256 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv256to64(x)
        h = self.amcm(h)
        B, C, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        tokens = GBST(tokens, self.shift_pixel, 1 / 4, (H, W))
        h = tokens.transpose(1, 2).reshape(B, C, H, W)
        h = self.dscd(h)
        h = self.conv64to256(h)
        return self.conlast256(h) + residual
