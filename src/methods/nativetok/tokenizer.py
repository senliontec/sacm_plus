"""Native-resolution residual tokenization for SAM.

SAM resizes every image to 1024x1024 and tokenizes it with 16x16 patches.
For thin curvilinear structures, a large part of the signal is already gone
after that resize: a 2-pixel vessel in a 3504x2336 image becomes sub-pixel
before the patch embedding sees it.  Adapters applied after tokenization
cannot recover information that was never tokenized.

This module adds a second, native-resolution path at the patch embedding:

1. the global 1024 image produces the usual 64x64 coarse tokens;
2. a tiny gate on those coarse tokens decides, per token, which finer
   scale (if any) is worth sampling;
3. fine patches are sampled from the *original-resolution* image at the
   token centres, resized to 16x16, and embedded with the same frozen SAM
   patch-embedding weights;
4. the fine embedding is added as a zero-initialized residual to the coarse
   token.

The coarse-to-fine decision is therefore moved from the decoder to the
tokenizer.  No extra ViT forward pass, no extra tokens, no new decoder and
no new loss are used.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

# Same normalization as Conv-LoRA: the released AutoGluon recipe uses
# ImageNet statistics, not SAM's pixel mean/std.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class NativeResidualTokenizer(nn.Module):
    """Add native-resolution fine patch tokens to SAM's coarse token grid.

    Args:
        patch_proj: the frozen SAM 16x16 patch-embedding convolution.  Its
            weight is copied into a buffer so this module cannot make the
            patch embedding trainable or duplicate it in the state dict.
        in_dim: token channel dimension.
        scales: fine patch side lengths measured in 1024-image pixels.  A
            value of 8 means the sampled region is half the width of a
            standard 16x16 patch; 4 means a quarter.  Values must be < 16.
        hidden: hidden width of the coarse-token gate.
        image_size: SAM input size (1024).
        alpha_init: initial residual scale.  Zero makes the model exactly
            the Conv-LoRA baseline at step 0.
    """

    def __init__(self, patch_proj: nn.Conv2d, in_dim: int,
                 scales: Iterable[int] = (8, 4), hidden: int = 32,
                 image_size: int = 1024, alpha_init: float = 0.0):
        super().__init__()
        if not isinstance(patch_proj, nn.Conv2d):
            raise TypeError('patch_proj must be a Conv2d')
        kernel = patch_proj.kernel_size
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if tuple(kernel) != (16, 16):
            raise ValueError(
                f'expected a 16x16 SAM patch embedding, got {kernel}')
        self.in_dim = int(in_dim)
        self.image_size = int(image_size)
        if self.in_dim != int(patch_proj.out_channels):
            raise ValueError('in_dim does not match patch_proj out_channels')
        self.scales = tuple(int(value) for value in scales)
        if not self.scales:
            raise ValueError('at least one fine scale is required')
        if any(value < 2 or value >= 16 for value in self.scales):
            raise ValueError('fine scales must be in [2, 15]')

        # Copy (do not register) the frozen patch-embedding weights.  This
        # avoids a duplicate submodule entry in the state dict while keeping
        # the exact SAM embedding used by the coarse path.
        self.register_buffer(
            'patch_weight', patch_proj.weight.detach().clone(),
            persistent=True)
        if patch_proj.bias is not None:
            self.register_buffer(
                'patch_bias', patch_proj.bias.detach().clone(),
                persistent=True)
        else:
            self.patch_bias = None
        self.register_buffer(
            'mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False)

        # Coarse tokens predict a distribution over {no-fine, scale_1, ...}.
        # The null class makes the residual sparse: a token can choose not
        # to be refined at all.  The null bias starts high so the model
        # begins close to the global baseline; alpha is zero as well.
        self.gate = nn.Sequential(
            nn.Conv2d(self.in_dim, int(hidden), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(int(hidden), len(self.scales) + 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.zero_()
            self.gate[-1].bias[0] = 2.0

        self.scale_embed = nn.Parameter(
            torch.zeros(len(self.scales), self.in_dim))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_gate: torch.Tensor | None = None

    def _native_patch(self, native: torch.Tensor, scale_px: int,
                      size: tuple[int, int]) -> torch.Tensor:
        """Sample one fine scale from a ``[3,H,W]`` native image.

        Returns ``[H_tokens, W_tokens, C]``.  The grid is built in native
        pixel coordinates and sampled bilinearly; no second image resize
        or encoder pass is performed.
        """
        if native.ndim != 3 or native.shape[0] != 3:
            raise ValueError(
                f'expected native image [3,H,W], got {tuple(native.shape)}')
        height, width = int(native.shape[-2]), int(native.shape[-1])
        tokens_h, tokens_w = size
        # Fine patch size in native pixels.  A scale of 8/1024 of the image
        # is half a standard SAM patch and therefore 2x finer in each axis.
        patch_w = max(2.0, scale_px / self.image_size * width)
        patch_h = max(2.0, scale_px / self.image_size * height)
        device = native.device
        dtype = native.dtype

        centers_y = (
            (torch.arange(tokens_h, device=device, dtype=dtype) + 0.5)
            / tokens_h * height)
        centers_x = (
            (torch.arange(tokens_w, device=device, dtype=dtype) + 0.5)
            / tokens_w * width)
        offsets = (
            torch.arange(16, device=device, dtype=dtype) - 7.5) / 16.0
        # [H_tokens, W_tokens, 16]
        ys = (
            centers_y[:, None, None] + offsets[None, None, :] * patch_h
        ).expand(tokens_h, tokens_w, 16)
        xs = (
            centers_x[None, :, None] + offsets[None, None, :] * patch_w
        ).expand(tokens_h, tokens_w, 16)
        # grid_sample uses normalized coordinates with align_corners=False:
        # pixel centre x maps to 2*(x+0.5)/W - 1.
        gy = 2.0 * (ys + 0.5) / height - 1.0
        gx = 2.0 * (xs + 0.5) / width - 1.0
        # [K,16,16] where rows are the 16 y samples of a token and columns
        # are its 16 x samples.
        gy = gy.reshape(tokens_h * tokens_w, 16)[:, :, None].expand(
            tokens_h * tokens_w, 16, 16)
        gx = gx.reshape(tokens_h * tokens_w, 16)[:, None, :].expand(
            tokens_h * tokens_w, 16, 16)
        grid = torch.stack([gy, gx], dim=-1).reshape(
            1, tokens_h * tokens_w * 16, 16, 2)
        patches = F.grid_sample(
            native.unsqueeze(0), grid, mode='bilinear',
            padding_mode='border', align_corners=False)
        patches = patches.reshape(tokens_h * tokens_w, 3, 16, 16)
        patches = (patches / 255.0 - self.mean) / self.std
        embed = F.conv2d(
            patches, self.patch_weight, self.patch_bias, stride=16)
        embed = embed.reshape(tokens_h, tokens_w, self.in_dim)
        return embed

    def forward(self, global_tokens: torch.Tensor,
                native_images: Sequence[torch.Tensor] | torch.Tensor | None
                ) -> torch.Tensor:
        """Return coarse tokens plus a gated native-resolution residual.

        ``global_tokens`` is ``[B,H,W,C]`` NHWC, the SAM patch-embedding
        output before positional embeddings.  ``native_images`` is a list
        of ``[3,H0,W0]`` tensors in 0-255 RGB, one per batch item.
        """
        if native_images is None:
            return global_tokens
        if global_tokens.ndim != 4:
            raise ValueError(
                f'expected global tokens [B,H,W,C], got '
                f'{tuple(global_tokens.shape)}')
        batch, tokens_h, tokens_w, channels = global_tokens.shape
        if channels != self.in_dim:
            raise ValueError('global token channel mismatch')
        if isinstance(native_images, torch.Tensor):
            if native_images.ndim != 4:
                raise ValueError(
                    'native_images tensor must be [B,3,H,W]')
            if native_images.shape[0] != batch:
                raise ValueError('native image batch mismatch')
            native_list = list(native_images)
        else:
            native_list = list(native_images)
            if len(native_list) != batch:
                raise ValueError(
                    f'expected {batch} native images, got '
                    f'{len(native_list)}')
        if any(image.ndim != 3 or image.shape[0] != 3
               for image in native_list):
            raise ValueError('each native image must be [3,H,W]')

        gate_logits = self.gate(
            global_tokens.permute(0, 3, 1, 2))
        gate = torch.softmax(gate_logits, dim=1)  # [B,S+1,H,W]
        self.last_gate = gate.detach()

        residuals = []
        for index, native in enumerate(native_list):
            fine = global_tokens.new_zeros(
                tokens_h, tokens_w, channels)
            for scale_index, scale_px in enumerate(self.scales):
                patch = self._native_patch(
                    native.to(device=global_tokens.device,
                              dtype=global_tokens.dtype),
                    scale_px, (tokens_h, tokens_w))
                patch = patch + self.scale_embed[scale_index]
                weight = gate[index, scale_index + 1][..., None]
                fine = fine + weight * patch
            residuals.append(self.alpha * fine)
        return global_tokens + torch.stack(residuals, dim=0)


__all__ = ['NativeResidualTokenizer']
