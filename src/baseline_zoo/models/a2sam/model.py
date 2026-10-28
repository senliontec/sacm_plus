"""A2SAM model: reconstructed SPG + vendored SAM ViT-B.

The official release only contains SAM, the two-branch prompt fusion, CPS and
the thickness loss.  This module assembles those official pieces with the
paper-level U-Net/prior reconstruction and the exact-persistence topology
backend.  It never imports from ``3rd/``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior import PersistencePrior
from .prompt import PromptFusion
from .sam import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
from .unet import CoarseUNet

SAM_CHECKPOINT_EXCLUDED = (
    'mask_tokens',
    'output_hypernetworks_mlps',
    'iou_prediction_head',
)


class A2SAMModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 1024,
        prompt_size: int = 256,
        unet_widths: tuple[int, int, int, int, int] = (36, 72, 144, 288, 288),
        prior_widths: tuple[int, int, int, int, int] = (24, 48, 96, 192, 384),
        prior_hidden: int = 256,
        prior_min_sigma: float = 0.05,
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError('image_size must be divisible by 16')
        self.image_size = int(image_size)
        self.prompt_size = int(prompt_size)
        self.unet = CoarseUNet(widths=unet_widths)
        self.prior = PersistencePrior(
            widths=prior_widths, hidden=prior_hidden,
            min_sigma=prior_min_sigma)
        self.prompt_fusion = PromptFusion()
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

    # ------------------------------------------------------------------
    # Official SAM checkpoint loading
    # ------------------------------------------------------------------
    def load_sam_checkpoint(self, path: str | Path) -> dict:
        """Load SAM components using the release's key allowlist.

        The release leaves ``mask_tokens``, ``output_hypernetworks_mlps`` and
        ``iou_prediction_head`` at their random initialization.  We report the
        exact loaded/random counts instead of silently using ``strict=False``.
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
        random_init = {}
        module_state = self.state_dict()
        for prefix in ('image_encoder.', 'prompt_encoder.', 'mask_decoder.'):
            module = getattr(self, prefix[:-1])
            subset = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
                and not any(excluded in key
                            for excluded in SAM_CHECKPOINT_EXCLUDED)
            }
            if not subset:
                raise RuntimeError(f'{path}: no {prefix}* tensors for A2SAM')
            missing, unexpected = module.load_state_dict(
                subset, strict=False)
            allowed_missing = [
                key for key in missing
                if any(excluded in key for excluded in SAM_CHECKPOINT_EXCLUDED)
            ]
            other_missing = [key for key in missing
                             if key not in allowed_missing]
            if other_missing or unexpected:
                raise RuntimeError(
                    f'{path}: {prefix} load mismatch; missing='
                    f'{other_missing[:5]}, unexpected={unexpected[:5]}')
            counts[prefix[:-1]] = len(subset)
            random_init[prefix[:-1]] = {
                'keys': allowed_missing,
                'parameters': sum(
                    module_state[prefix + key].numel()
                    for key in allowed_missing
                    if prefix + key in module_state),
            }
        return {'loaded': counts, 'random_init': random_init}

    # ------------------------------------------------------------------
    # Coarse U-Net / prior
    # ------------------------------------------------------------------
    def coarse(self, image: torch.Tensor):
        logits = self.unet(image)
        probability = torch.sigmoid(logits)
        return logits, probability, probability > 0.5

    def prior_stats(self, image: torch.Tensor,
                    probability: torch.Tensor):
        return self.prior(image, probability)

    # ------------------------------------------------------------------
    # Prompt / SAM
    # ------------------------------------------------------------------
    def make_prompt(
        self,
        uncertainty: torch.Tensor,
        coarse: torch.Tensor,
        *,
        mode: str = 'release',
        detach: bool = False,
    ) -> torch.Tensor:
        fused = self.prompt_fusion(
            uncertainty, coarse, mode=mode, detach=detach)
        return F.interpolate(
            fused, size=(self.prompt_size, self.prompt_size),
            mode='bilinear', align_corners=False)

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

    def forward(
        self,
        image: torch.Tensor,
        prompt: torch.Tensor,
        original_size: tuple[int, int],
    ):
        return self.decode(image, prompt, original_size)


def build_a2sam(
    *,
    image_size: int = 1024,
    prompt_size: int = 256,
    unet_widths: tuple[int, int, int, int, int] = (36, 72, 144, 288, 288),
    prior_widths: tuple[int, int, int, int, int] = (24, 48, 96, 192, 384),
    prior_hidden: int = 256,
    prior_min_sigma: float = 0.05,
) -> A2SAMModel:
    return A2SAMModel(
        image_size=image_size,
        prompt_size=prompt_size,
        unet_widths=unet_widths,
        prior_widths=prior_widths,
        prior_hidden=prior_hidden,
        prior_min_sigma=prior_min_sigma,
    )
