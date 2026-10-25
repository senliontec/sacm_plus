"""Prompt-free SAM-Adapter model.

The image encoder is the official SAM-Adapter/EVP ViT: an FFT high-pass
prompt generator produces one prompt map per transformer block, and each
prompt is added to the block input.  The mask decoder receives an empty
sparse prompt and the learned ``no_mask_embed`` dense prompt, so inference
is prompt-free.

Vendored from ``3rd/SAM-Adapter-PyTorch`` (ICCV 2023).  The upstream
``models/sam.py`` class is reduced to construction and inference; the
official training loop lives in the baseline adapter.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer

# Official ``configs/cod-sam-vit-l.yaml`` encoder_mode.  The prompt-generator
# settings (highpass FFT, freq_nums=0.25, tuning_stage=1234) are hardcoded in
# the vendored ImageEncoderViT, exactly as upstream; the extra keys are kept
# for provenance.
DEFAULT_ENCODER_MODE = {
    'name': 'sam',
    'img_size': 1024,
    'mlp_ratio': 4,
    'patch_size': 16,
    'qkv_bias': True,
    'use_rel_pos': True,
    'window_size': 14,
    'out_chans': 256,
    'scale_factor': 32,
    'input_type': 'fft',
    'freq_nums': 0.25,
    'prompt_type': 'highpass',
    'prompt_embed_dim': 256,
    'tuning_stage': 1234,
    'handcrafted_tune': True,
    'embedding_tune': True,
    'adaptor': 'adaptor',
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'global_attn_indexes': [5, 11, 17, 23],
}


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies (upstream copy)."""

    def __init__(self, num_pos_feats: int = 64,
                 scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            'positional_encoding_gaussian_matrix',
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


class SAM(nn.Module):
    """SAM-Adapter with the official prompt-free decoder path."""

    def __init__(self, inp_size: int = 1024,
                 encoder_mode: Optional[dict] = None) -> None:
        super().__init__()
        mode = dict(DEFAULT_ENCODER_MODE if encoder_mode is None
                    else encoder_mode)
        self.embed_dim = mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=mode['patch_size'],
            in_chans=3,
            embed_dim=mode['embed_dim'],
            depth=mode['depth'],
            num_heads=mode['num_heads'],
            mlp_ratio=mode['mlp_ratio'],
            out_chans=mode['out_chans'],
            qkv_bias=mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=mode['window_size'],
            global_attn_indexes=tuple(mode['global_attn_indexes']),
        )

        self.prompt_embed_dim = mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.pe_layer = PositionEmbeddingRandom(
            self.prompt_embed_dim // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """Return foreground logits at ``inp_size`` resolution.

        The upstream ``SAM.forward``/``infer`` use a single prompt set and
        broadcast it over the image batch.  That is preserved here: the image
        encoder and decoder see the full batch, while the empty sparse prompt
        and ``no_mask_embed`` dense prompt have batch dimension one.  This is
        the official prompt-free behaviour, not a batched-prompt path.
        """
        prompt_batch = 1
        dtype = self.no_mask_embed.weight.dtype
        sparse_embeddings = torch.empty(
            (prompt_batch, 0, self.prompt_embed_dim),
            device=image.device, dtype=dtype)
        dense_embeddings = self.no_mask_embed.weight.reshape(
            1, -1, 1, 1).expand(
                prompt_batch, -1, self.image_embedding_size,
                self.image_embedding_size)

        features = self.image_encoder(image)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return self.postprocess_masks(
            low_res_masks, self.inp_size, self.inp_size)

    def postprocess_masks(self, masks: torch.Tensor,
                          input_size: int,
                          original_size: int) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode='bilinear', align_corners=False,
        )
        masks = masks[..., :input_size, :input_size]
        masks = F.interpolate(
            masks, (original_size, original_size),
            mode='bilinear', align_corners=False,
        )
        return masks


def build_sam_adapter(inp_size: int = 1024,
                      encoder_mode: Optional[dict] = None) -> SAM:
    """Build the official ViT-L SAM-Adapter (or a caller-supplied variant)."""
    return SAM(inp_size=inp_size, encoder_mode=encoder_mode)
