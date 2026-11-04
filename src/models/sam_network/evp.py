"""EVP / SAM-Adapter visual prompt generator.

Ported from the official SAM-Adapter ``PromptGenerator`` (EVP: high-frequency
FFT prompt) so SACM-X does not import the disposable baseline_zoo package.
The generator is a tiny trainable branch: FFT high-pass -> patch embedding ->
shared low-rank MLP -> one lightweight MLP per ViT block.  Prompts are added
to the block input before attention/MLP, leaving the frozen ViT untouched.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EVPPromptGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        img_size: int = 1024,
        patch_size: int = 16,
        scale_factor: int = 32,
        freq_nums: float = 0.25,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % scale_factor != 0:
            raise ValueError(
                f'embed_dim {embed_dim} is not divisible by '
                f'scale_factor {scale_factor}')
        if img_size % patch_size != 0:
            raise ValueError('img_size must be divisible by patch_size')
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.freq_nums = float(freq_nums)
        low_dim = embed_dim // scale_factor

        # 3x3 conv (same receptive field and parameter count as the official
        # non-overlapping PatchEmbed2 kernel).
        self.patch_embed = nn.Conv2d(
            3, low_dim, kernel_size=patch_size, stride=patch_size)
        self.shared_mlp = nn.Linear(low_dim, embed_dim)
        self.embedding_generator = nn.Linear(embed_dim, low_dim)
        self.lightweight_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(low_dim, low_dim), nn.GELU())
            for _ in range(self.depth)
        ])

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = (module.kernel_size[0] * module.kernel_size[1]
                           * module.out_channels) // module.groups
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()

        # For 3-shot fine-tuning, start from exactly the frozen-SAM feature
        # path instead of injecting random high-frequency prompts into every
        # block at step 0.  The shared MLP is the only output projection, so
        # zeroing it makes all prompts zero while still allowing gradients
        # to reach it on the first step.
        if zero_init:
            nn.init.zeros_(self.shared_mlp.weight)
            nn.init.zeros_(self.shared_mlp.bias)

    @staticmethod
    def _highpass(x: torch.Tensor, rate: float) -> torch.Tensor:
        """FFT high-pass with a central low-frequency square removed."""
        h, w = x.shape[-2:]
        line = max(1, int((w * h * rate) ** 0.5 // 2))
        mask = torch.zeros(1, 1, w, h, device=x.device, dtype=x.dtype)
        mask[:, :, w // 2 - line:w // 2 + line,
             h // 2 - line:h // 2 + line] = 1.0
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        fft = fft * (1.0 - mask)
        high = torch.fft.ifft2(
            torch.fft.ifftshift(fft), norm='forward').real
        return high.abs()

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return one prompt ``[B, H*W, embed_dim]`` per ViT block.

        Args:
            image: normalized input image ``[B, 3, H, W]``.
            tokens: patch tokens after patch_embed ``[B, H/p, W/p, C]``.
        """
        if image.shape[-2:] != (self.img_size, self.img_size):
            image = nn.functional.interpolate(
                image, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False)
        high = self._highpass(image, self.freq_nums)
        handcrafted = self.patch_embed(high)        # [B, low, h, w]
        handcrafted = handcrafted.flatten(2).transpose(1, 2)

        tokens_flat = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        embedding = self.embedding_generator(tokens_flat)

        prompts = []
        for mlp in self.lightweight_mlps:
            prompt = mlp(handcrafted + embedding)
            prompts.append(self.shared_mlp(prompt))
        return prompts
