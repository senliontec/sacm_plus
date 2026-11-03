"""V3 structural decoder for prompt-free curvilinear segmentation.

The SAM mask decoder is a *prompt-candidate* decoder: four mask tokens,
gating over candidates, and a 32-channel linear mask head at 256.  That is
the wrong inductive bias for prompt-free dense segmentation of 1-px
structures.  V3 keeps the frozen SAM encoder (and its adapters) but
replaces the decoder with a dense structural path:

    64x64 SAM levels -> lightweight FPN
    64 -> 128 -> 256 -> 512 learned upsampling
    detail-stem fusion at 256 and 512, gated by a centerline prediction
    a 512 conv mask head
    training-only centerline / boundary / width heads
    image-guided 1024 refinement (logits + image + detail + centerline)

Only the mask head is used at inference.  There is no candidate selection,
no gating, no fixed head, and no TADT/ranker.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class _UpBlock(nn.Module):
    """2x bilinear upsample + depthwise/pointwise refinement."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False)
        self.norm = LayerNorm2d(in_dim)
        self.pw = nn.Conv2d(in_dim, out_dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear',
                          align_corners=False)
        x = self.act(self.norm(self.dw(x)))
        return self.act(self.pw(x))


class _ConvBlock(nn.Module):
    """3x3 conv + LayerNorm + GELU used by the 64x64 FPN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _DetailFuse(nn.Module):
    def __init__(self, feat_dim: int, detail_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            feat_dim + detail_dim, feat_dim, 1)
        self.norm = LayerNorm2d(feat_dim)
        self.act = nn.GELU()

    def forward(self, feat: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        if detail.shape[-2:] != feat.shape[-2:]:
            detail = F.interpolate(
                detail, size=feat.shape[-2:], mode='bilinear',
                align_corners=False)
        return self.act(self.norm(self.conv(
            torch.cat([feat, detail], dim=1))))


class StructuralDecoderV3(nn.Module):
    """Dense structural decoder (no prompt tokens, no candidate heads)."""

    def __init__(
        self,
        encoder_embed_dim: int = 1024,
        decoder_dim: int = 256,
        feature_dim: int = 128,
        mid_dim: int = 96,
        fine_dim: int = 64,
        detail_channels: int = 32,
        output_size: int = 512,
        refine_size: int = 1024,
        refine_channels: int = 16,
        use_centerline: bool = True,
        use_boundary: bool = True,
        use_width: bool = True,
        width_max: float = 16.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.refine_size = int(refine_size) if refine_size else None
        self.width_max = float(width_max)
        self.detail_channels = int(detail_channels)
        self.use_centerline = bool(use_centerline)
        self.use_boundary = bool(use_boundary)
        self.use_width = bool(use_width)
        self.num_mask_tokens = 1
        self.last_aux_outputs = None

        # Semantic pyramid at the ViT grid (64x64 for 1024 / patch 16).
        self.v3_lat0 = nn.Conv2d(encoder_embed_dim, feature_dim, 1)
        self.v3_lat1 = nn.Conv2d(encoder_embed_dim, feature_dim, 1)
        self.v3_lat2 = nn.Conv2d(encoder_embed_dim, feature_dim, 1)
        self.v3_lat3 = nn.Conv2d(encoder_embed_dim, feature_dim, 1)
        self.v3_lat_out = nn.Conv2d(decoder_dim, feature_dim, 1)
        self.v3_fpn3 = _ConvBlock(feature_dim)
        self.v3_fpn2 = _ConvBlock(feature_dim)
        self.v3_fpn1 = _ConvBlock(feature_dim)
        self.v3_fpn0 = _ConvBlock(feature_dim)

        # Learned dense upsampling: 64 -> 128 -> 256 -> 512.
        self.v3_up1 = _UpBlock(feature_dim, feature_dim)
        self.v3_up2 = _UpBlock(feature_dim, mid_dim)
        self.v3_up3 = _UpBlock(mid_dim, fine_dim)

        # Direct semantic shortcuts: the pretrained neck features reach the
        # mask/aux heads without passing through the randomly initialised
        # FPN stack.  This shortens the path from SAM features to logits.
        self.v3_skip256 = nn.Conv2d(decoder_dim, mid_dim, 1)
        self.v3_skip512 = nn.Conv2d(decoder_dim, fine_dim, 1)

        # Detail fusion, gated by the pre-fusion centerline logits.
        self.v3_fuse256 = _DetailFuse(mid_dim, detail_channels)
        self.v3_gate256 = nn.Conv2d(mid_dim, 1, 1)
        self.v3_fuse512 = _DetailFuse(fine_dim, detail_channels)
        self.v3_gate512 = nn.Conv2d(fine_dim, 1, 1)

        self.v3_coarse_out = nn.Conv2d(mid_dim, 1, 1)

        self.v3_mask_dw = nn.Conv2d(
            fine_dim, fine_dim, 3, padding=1, groups=fine_dim, bias=False)
        self.v3_mask_norm = LayerNorm2d(fine_dim)
        self.v3_mask_pw = nn.Conv2d(fine_dim, fine_dim, 1)
        self.v3_mask_out = nn.Conv2d(fine_dim, 1, 1)
        # Recall-first start: with the default zero bias the BCE/area terms
        # push a randomly initialised head below the 0.5 threshold for many
        # epochs on thin structures.  A positive bias starts in the
        # foreground and lets Dice/Tversky prune false positives.
        nn.init.constant_(self.v3_mask_out.bias, 1.0)

        self.v3_centerline_out = nn.Conv2d(fine_dim, 1, 1)
        self.v3_boundary_out = nn.Conv2d(fine_dim, 1, 1)
        self.v3_width_out = nn.Conv2d(fine_dim, 1, 1)

        self.v3_iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fine_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 1024 refinement: logits + image + detail + centerline.
        self.v3_refine = nn.Sequential(
            nn.Conv2d(1 + 3 + detail_channels + 1,
                      refine_channels, 3, padding=1),
            LayerNorm2d(refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, 3, padding=1),
            LayerNorm2d(refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, 1, 1),
        )
        # Near-identity start, but not exactly zero: a zero last conv would
        # block gradients to the first refine conv for one full step.
        nn.init.normal_(self.v3_refine[-1].weight, std=1e-3)
        nn.init.zeros_(self.v3_refine[-1].bias)

    @staticmethod
    def _to_bchw(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            b, n, c = feature.shape
            h = w = int(round(n ** 0.5))
            if h * w != n:
                raise ValueError(f'cannot reshape {n} tokens to a square grid')
            return feature.transpose(1, 2).reshape(b, c, h, w)
        if feature.ndim == 4:
            # Encoder intermediates are [B, H, W, C]; the neck output is
            # already [B, C, H, W].  H == W for both, so decide by the
            # channel dimension being much wider than the grid.
            if feature.shape[1] == feature.shape[2] and feature.shape[3] > feature.shape[1]:
                return feature.permute(0, 3, 1, 2)
            return feature
        raise ValueError(f'unsupported feature shape {tuple(feature.shape)}')

    @staticmethod
    def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(
            x, size=size, mode='bilinear', align_corners=False)

    def _semantic_pyramid(
        self,
        image_embeddings: torch.Tensor,
        intermediate_features: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        size = image_embeddings.shape[-2:]
        feats: List[torch.Tensor] = []
        if intermediate_features:
            for f in intermediate_features:
                feats.append(self._resize(self._to_bchw(f), size))
        while len(feats) < 4:
            feats.append(image_embeddings)

        p3 = self.v3_fpn3(self.v3_lat3(feats[3]))
        p2 = self.v3_fpn2(
            self.v3_lat2(feats[2]) + self._resize(p3, size))
        p1 = self.v3_fpn1(
            self.v3_lat1(feats[1]) + self._resize(p2, size))
        p0 = self.v3_fpn0(
            self.v3_lat_out(image_embeddings)
            + self.v3_lat0(feats[0])
            + self._resize(p1, size))
        return p0

    def _head(
        self,
        feat: torch.Tensor,
        detail: Optional[torch.Tensor],
        fuse: nn.Module,
        gate: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = gate(feat)
        if detail is not None:
            feat = feat + torch.sigmoid(gate_logits) * fuse(feat, detail)
        return feat, gate_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: Optional[torch.Tensor] = None,
        sparse_prompt_embeddings: Optional[torch.Tensor] = None,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        adapter_features: Optional[List[torch.Tensor]] = None,
        intermediate_features: Optional[List[torch.Tensor]] = None,
        detail_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image: Optional[torch.Tensor] = None,
        return_stage1: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        p0 = self._semantic_pyramid(image_embeddings, intermediate_features)
        u1 = self.v3_up1(p0)                       # 1/16 -> 1/8
        u2 = self.v3_up2(u1)                       # 1/8 -> 1/4 (256)
        u2 = u2 + self._resize(
            self.v3_skip256(image_embeddings), u2.shape[-2:])

        detail_256 = detail_features[0] if detail_features else None
        u2, centerline_256 = self._head(
            u2, detail_256, self.v3_fuse256, self.v3_gate256)
        coarse_logits = self.v3_coarse_out(u2)

        u3 = self.v3_up3(u2)                       # 1/4 -> 1/2 (512)
        u3 = u3 + self._resize(
            self.v3_skip512(image_embeddings), u3.shape[-2:])
        detail_512 = detail_features[1] if detail_features else None
        u3, _ = self._head(
            u3, detail_512, self.v3_fuse512, self.v3_gate512)

        mask_feat = F.gelu(self.v3_mask_pw(
            self.v3_mask_norm(self.v3_mask_dw(u3))))
        mask_logits = self.v3_mask_out(mask_feat)

        centerline = self.v3_centerline_out(u3)
        boundary = self.v3_boundary_out(u3) if self.use_boundary else None
        width = self.v3_width_out(u3) if self.use_width else None
        iou_pred = self.v3_iou_head(u3)

        if self.refine_size is not None and self.refine_size > self.output_size:
            size = (self.refine_size, self.refine_size)
            batch = mask_logits.shape[0]
            if detail_512 is not None:
                detail_up = self._resize(detail_512, size)
            else:
                detail_up = mask_logits.new_zeros(
                    (batch, self.detail_channels, *size))
            if image is not None:
                image_up = self._resize(image[:, :3], size)
            else:
                image_up = mask_logits.new_zeros((batch, 3, *size))
            parts = [
                self._resize(mask_logits, size),
                image_up,
                detail_up,
                self._resize(torch.sigmoid(centerline), size),
            ]
            mask_logits = parts[0] + self.v3_refine(torch.cat(parts, dim=1))

        self.last_aux_outputs = {
            'mask': mask_logits,
            'coarse': coarse_logits,
            'centerline_256': centerline_256,
            'centerline': centerline,
            'boundary': boundary,
            'width': width,
            'iou': iou_pred,
        }
        if return_stage1:
            return mask_logits, iou_pred, coarse_logits
        return mask_logits, iou_pred

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: Optional[torch.Tensor] = None,
        sparse_prompt_embeddings: Optional[torch.Tensor] = None,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            return_stage1=False,
        )
        return masks, iou
