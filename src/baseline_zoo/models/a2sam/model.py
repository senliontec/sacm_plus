"""A2SAM model: SAM ViT-B + structural prompt generator + probabilistic prior.

The released A2SAM repository provides the SAM ViT-B modules and a minimal
prompt-fusion network; the paper's structural prompt generator and the
probabilistic U-Net prior are not part of the release.  This module assembles
the paper-level model: the prior produces a coarse likelihood and structural
uncertainty, the SPG turns them into a dense prompt, and SAM ViT-B decodes
the prompt.  The SAM modules are vendored verbatim from the release.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior import ProbabilisticUNet
from .sam import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
from .spg import StructuralPromptGenerator


class A2SAMModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 1024,
        prompt_size: int = 256,
        prior_base: int = 32,
        latent_dim: int = 16,
        spg_hidden: int = 32,
        spg_points: int = 512,
        spg_stride: int = 3,
        persistence: float = 5.0,
        relative_persistence: float = 0.1,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError('image_size must be divisible by 16')
        self.image_size = int(image_size)
        self.prior = ProbabilisticUNet(
            in_ch=3, base=prior_base, latent_dim=latent_dim)
        self.image_encoder = ImageEncoderViT(
            img_size=image_size,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            out_chans=256,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=(2, 5, 8, 11),
            tuning_mode='cps',
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_size // 16, image_size // 16),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.spg = StructuralPromptGenerator(
            prompt_size=prompt_size,
            hidden=spg_hidden,
            max_points=spg_points,
            stride=spg_stride,
            persistence=persistence,
            relative_persistence=relative_persistence,
        )

    # ------------------------------------------------------------------
    # SAM checkpoint
    # ------------------------------------------------------------------
    def load_sam_checkpoint(self, path: str | Path) -> dict[str, int]:
        """Load official SAM ViT-B weights strictly.

        The prior and SPG stay randomly initialized; only
        ``image_encoder``, ``prompt_encoder`` and ``mask_decoder`` come from
        the SAM checkpoint.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'missing SAM ViT-B checkpoint: {path}')
        payload = torch.load(str(path), map_location='cpu',
                             weights_only=False)
        state = payload.get('model', payload) if isinstance(payload, dict) \
            else payload
        if not isinstance(state, dict):
            raise RuntimeError(f'{path}: unsupported SAM checkpoint format')
        counts = {}
        for prefix in ('image_encoder.', 'prompt_encoder.', 'mask_decoder.'):
            module = getattr(self, prefix[:-1])
            subset = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not subset:
                raise RuntimeError(
                    f'{path}: no {prefix}* tensors for A2SAM')
            module.load_state_dict(subset, strict=True)
            counts[prefix[:-1]] = len(subset)
        return counts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def decode(self, image: torch.Tensor, prompt: torch.Tensor,
               original_size: tuple[int, int]):
        features = self.image_encoder(image)
        batch = image.shape[0]
        sparse = image.new_zeros(
            (batch, 0, self.prompt_encoder.embed_dim))
        _, dense = self.prompt_encoder(
            points=None, boxes=None, masks=prompt)
        low_res, iou = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        masks = F.interpolate(
            low_res, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False)
        masks = F.interpolate(
            masks, size=original_size, mode='bilinear',
            align_corners=False)
        return masks, low_res, iou

    def structural_prompt(
        self,
        prob: torch.Tensor,
        uncertainty: torch.Tensor,
        structural=None,
    ) -> torch.Tensor:
        if structural is None:
            return self.spg(prob, uncertainty)
        return self.spg.render(structural, prob, uncertainty)

    def forward(
        self,
        image: torch.Tensor,
        prob: torch.Tensor,
        uncertainty: torch.Tensor,
        original_size: tuple[int, int],
        *,
        structural=None,
        prompt: torch.Tensor | None = None,
    ):
        if prompt is None:
            prompt = self.structural_prompt(
                prob, uncertainty, structural=structural)
        return self.decode(image, prompt, original_size)


def build_a2sam(
    *,
    image_size: int = 1024,
    prompt_size: int = 256,
    prior_base: int = 32,
    latent_dim: int = 16,
    spg_hidden: int = 32,
    spg_points: int = 512,
    spg_stride: int = 3,
    persistence: float = 5.0,
    relative_persistence: float = 0.1,
) -> A2SAMModel:
    return A2SAMModel(
        image_size=image_size,
        prompt_size=prompt_size,
        prior_base=prior_base,
        latent_dim=latent_dim,
        spg_hidden=spg_hidden,
        spg_points=spg_points,
        spg_stride=spg_stride,
        persistence=persistence,
        relative_persistence=relative_persistence,
    )
