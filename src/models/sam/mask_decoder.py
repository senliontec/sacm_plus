# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified: prompt-free curvilinear decoder with
#   - closed-loop coarse-to-fine refinement (CoarseToFineRefiner),
#   - Fusion v2 (cross-layer interaction weights + spatial gating),
#   - a multi-depth semantic path from encoder intermediate features.
# The gating_mlp dead module of the original SACM code was removed.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_adapter_skip: bool = True,
        adapter_embed_dim: int = 768,
        use_coarse_to_fine: bool = True,
        use_fusion_v2: bool = True,
        adapter_num_layers: int = 24,
        fusion_proj_dim: int = 32,
        fusion_hidden: int = 128,
        fusion_dropout: float = 0.1,
        use_multi_depth: bool = True,
        encoder_embed_dim: int = 1024,
        num_multidepth_levels: int = 4,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
          use_adapter_skip (bool): whether to use adapter skip connections
          adapter_embed_dim (int): dimension of adapter features
          use_coarse_to_fine (bool): whether to feed the Stage-1 coarse
            masks back into the Stage-2 feature map (closed-loop)
          use_fusion_v2 (bool): use Fusion v2 (concat+FFN cross-layer
            weights and spatial gating); otherwise the original SACM
            fusion (per-layer scoring) is used
          adapter_num_layers (int): number of external-adapter features
            consumed by the fusion module (window layers only when the
            geometric external adapter is enabled)
          fusion_proj_dim (int): projection dim of each layer descriptor
          fusion_hidden (int): hidden dim of the layer-weight FFN
          fusion_dropout (float): dropout on the layer-weight FFN
          use_multi_depth (bool): enable the multi-depth semantic path
            from encoder intermediate features
          encoder_embed_dim (int): channel dim of encoder intermediate
            features (1024 for ViT-L)
          num_multidepth_levels (int): number of intermediate features
            consumed by the multi-depth path
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.use_adapter_skip = use_adapter_skip
        self.use_fusion_v2 = use_fusion_v2
        self.use_multi_depth = use_multi_depth
        self.use_coarse_to_fine = use_coarse_to_fine

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        # Stage-1 hypernetworks (initial masks)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        # Stage-2 hypernetworks (refined masks)
        self.output_hypernetworks_mlps_stage2 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # IoU / quality prediction head (for each head)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        if self.use_adapter_skip:
            if use_fusion_v2:
                self.adapter_fusion_v2 = AdapterFusionV2(
                    embed_dim=adapter_embed_dim,
                    num_layers=adapter_num_layers,
                    proj_dim=fusion_proj_dim,
                    hidden=fusion_hidden,
                    dropout=fusion_dropout,
                    out_dim=transformer_dim // 8,
                )
            else:
                # Original SACM fusion modules (per-layer scoring)
                self.adapter_aggregator = nn.Sequential(
                    nn.LayerNorm(adapter_embed_dim),
                    nn.Linear(adapter_embed_dim, transformer_dim // 4),
                    activation(),
                    nn.Linear(transformer_dim // 4, transformer_dim // 8),
                )
                self.adapter_attention = nn.Sequential(
                    nn.LayerNorm(adapter_embed_dim),
                    nn.Linear(adapter_embed_dim, 1),
                    nn.Sigmoid()
                )

        if self.use_coarse_to_fine:
            self.coarse_refiner = CoarseToFineRefiner(
                feature_dim=transformer_dim // 8, num_heads=self.num_mask_tokens
            )

        if self.use_multi_depth:
            self.multi_depth_path = MultiDepthPath(
                embed_dim=encoder_embed_dim,
                mid_dim=256,
                out_dim=transformer_dim // 8,
                num_levels=num_multidepth_levels,
            )
            # Two-stream fuse, zero-initialized; used as a residual so the
            # main stream passes through unchanged at initialization.
            self.fuse_two = nn.Conv2d(
                2 * (transformer_dim // 8), transformer_dim // 8, kernel_size=1
            )
            nn.init.zeros_(self.fuse_two.weight)
            nn.init.zeros_(self.fuse_two.bias)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        adapter_features: List[torch.Tensor] = None,
        intermediate_features: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
          adapter_features (List[torch.Tensor]): adapter features from image encoder
          intermediate_features (List[torch.Tensor]): intermediate encoder
            features for the multi-depth path

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            adapter_features=adapter_features,
            intermediate_features=intermediate_features,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def _fusion_v1(self, adapter_features: List[torch.Tensor], target_size: Tuple[int, int]):
        """Original SACM fusion: per-layer scoring + softmax over layers +
        aggregator MLP + upsample."""
        adapter_weights = []
        for adapter_feat in adapter_features:
            pooled_feat = adapter_feat.mean(dim=[1, 2])
            weight = self.adapter_attention(pooled_feat)
            adapter_weights.append(weight)

        adapter_weights = torch.stack(adapter_weights, dim=0)
        adapter_weights = F.softmax(adapter_weights, dim=0)

        weighted_adapter_features = torch.zeros_like(adapter_features[0])
        for adapter_feat, weight in zip(adapter_features, adapter_weights):
            weighted_adapter_features = weighted_adapter_features + adapter_feat * weight.unsqueeze(-1).unsqueeze(-1)

        processed_adapter = self.adapter_aggregator(weighted_adapter_features)
        processed_adapter = processed_adapter.permute(0, 3, 1, 2)
        if processed_adapter.shape[2:] != target_size:
            processed_adapter = F.interpolate(
                processed_adapter,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        return processed_adapter

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_features: List[torch.Tensor] = None,
        intermediate_features: List[torch.Tensor] = None,
        return_stage1: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details.

        With return_stage1=True, also returns the Stage-1 coarse masks
        (reordered by the same gating) for deep supervision.
        """
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # ---------------- Adapter fusion injection ----------------
        if self.use_adapter_skip and adapter_features is not None:
            if self.use_fusion_v2:
                u_f = self.adapter_fusion_v2(adapter_features, upscaled_embedding.shape[2:])
            else:
                u_f = self._fusion_v1(adapter_features, upscaled_embedding.shape[2:])
            upscaled_embedding = upscaled_embedding + u_f

        # ---------------- Multi-depth semantic path ----------------
        if self.use_multi_depth and intermediate_features is not None:
            u_ms = self.multi_depth_path(intermediate_features, upscaled_embedding.shape[2:])
            upscaled_embedding = upscaled_embedding + self.fuse_two(
                torch.cat([upscaled_embedding, u_ms], dim=1)
            )

        # --------------------- Stage 1: initial masks ---------------------
        hyper_in_list_stage1: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list_stage1.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in_stage1 = torch.stack(hyper_in_list_stage1, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks_stage1 = (hyper_in_stage1 @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Compute gating scores to select best heads using Stage-1 spatial responses
        gating_scores = masks_stage1.view(b, self.num_mask_tokens, -1).max(dim=-1).values
        gating_weights = F.softmax(gating_scores, dim=1)

        # Order heads by gating score (descending), used in Stage-2 output ordering
        sorted_indices = torch.argsort(gating_weights, dim=1, descending=True)

        # ------------- Closed-loop coarse-to-fine refinement -------------
        # The coarse masks' spatial layout (where lines are vs background)
        # flows back into the feature map consumed by Stage-2. The refiner
        # is zero-initialized: at initialization U' == U.
        if self.use_coarse_to_fine:
            upscaled_embedding = self.coarse_refiner(upscaled_embedding, masks_stage1)

        # --------------------- Stage 2: refined masks ---------------------
        hyper_in_list_stage2: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list_stage2.append(self.output_hypernetworks_mlps_stage2[i](mask_tokens_out[:, i, :]))
        hyper_in_stage2 = torch.stack(hyper_in_list_stage2, dim=1)
        masks_stage2_all = (hyper_in_stage2 @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Reorder masks by gating scores so that index-0 is the best mask
        gather_indices = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        masks_stage2_ordered = torch.gather(masks_stage2_all, 1, gather_indices)

        # IoU prediction head and reorder by the same indices.
        # NOTE on training consistency: iou_pred's slot j is permanently
        # bound to mask token j. Reordering BOTH iou_pred and the IoU
        # targets (computed from the reordered masks) by the same
        # per-sample permutation is a no-op for the MSE — every slot j is
        # always trained against the quality of mask token j's output,
        # regardless of the gating order. Inference likewise selects the
        # reordered position with the best predicted score and takes the
        # mask at the same reordered position (same coordinate system).
        iou_pred = self.iou_prediction_head(iou_token_out)
        iou_pred_ordered = torch.gather(iou_pred, 1, sorted_indices)

        if return_stage1:
            masks_stage1_ordered = torch.gather(masks_stage1, 1, gather_indices)
            return masks_stage2_ordered, iou_pred_ordered, masks_stage1_ordered

        return masks_stage2_ordered, iou_pred_ordered


class CoarseToFineRefiner(nn.Module):
    """Closed-loop injection of Stage-1 coarse masks into Stage-2.

    U' = U + Conv3x3([U ; sigmoid(M1)])

    The 3x3 conv is zero-initialized, so at initialization U' == U and the
    module activates gradually. Concatenating ALL heads' coarse masks keeps
    every Stage-1 hypernetwork in the gradient path.

    Gradient note (exact): at step 0 the module is a DOUBLE no-op — the
    forward is identity AND d(U')/d(M1) = W^T·grad == 0 exactly (zero
    weights), so Stage-1 receives no gradient through this path until the
    first optimizer update makes the conv nonzero. Deep supervision covers
    Stage-1 during that one-step delay; without it (no_ds ablation), the
    Stage-1 hypernetworks get their first refiner-path gradient only from
    step 1 onward.
    """

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim + num_heads, feature_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, u: torch.Tensor, masks_stage1: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(masks_stage1)
        return u + self.conv(torch.cat([u, prob], dim=1))


class AdapterFusionV2(nn.Module):
    """Fusion with cross-layer interaction weights and spatial gating.

    alpha = Softmax(FFN(Concat(z_1, ..., z_L))),  z_l = W_l * AvgPool(E_l)
    F = sum(alpha_l * E_l)
    F = F * sigmoid(Conv1x1(F))          (per-location gate)
    out = UP(MLP(F))

    The aggregator output is zero-initialized so the fusion starts as a
    no-op and activates gradually. Dropout on the weight FFN regularizes
    the per-image weights under the few-shot regime.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        proj_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        out_dim: int = 32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_proj = nn.ModuleList(
            [nn.Linear(embed_dim, proj_dim) for _ in range(num_layers)]
        )
        self.weight_ffn = nn.Sequential(
            nn.Linear(num_layers * proj_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_layers),
        )
        self.spatial_gate = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.aggregator = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        # Zero-init aggregator output: fusion contributes nothing at start
        nn.init.zeros_(self.aggregator[-1].weight)
        nn.init.zeros_(self.aggregator[-1].bias)

    def forward(
        self, adapter_features: List[torch.Tensor], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        if len(adapter_features) != self.num_layers:
            raise ValueError(
                f"AdapterFusionV2 expects {self.num_layers} adapter features, "
                f"got {len(adapter_features)}. The encoder skips external "
                f"adapters on the global-attention layers only when "
                f"use_geo_e=True — check that the build flags match the "
                f"training configuration (e.g. a preset mismatch between "
                f"train_sam.py and test_sam.py)."
            )
        pooled = torch.stack(
            [f.mean(dim=(1, 2)) for f in adapter_features], dim=1
        )  # [B, L, C]
        z = torch.cat(
            [self.layer_proj[l](pooled[:, l]) for l in range(self.num_layers)], dim=-1
        )  # [B, L*P]
        alpha = F.softmax(self.weight_ffn(z), dim=-1)  # [B, L]

        fused = torch.zeros_like(adapter_features[0])
        for l in range(self.num_layers):
            fused = fused + adapter_features[l] * alpha[:, l].view(-1, 1, 1, 1)

        gate = torch.sigmoid(self.spatial_gate(fused.permute(0, 3, 1, 2)))  # [B, 1, H, W]
        fused = fused * gate.permute(0, 2, 3, 1)

        out = self.aggregator(fused).permute(0, 3, 1, 2)  # [B, out, H, W]
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class MultiDepthPath(nn.Module):
    """Multi-depth semantic path from encoder intermediate features.

    All ViT intermediate features share the token resolution (64x64 at
    1024 input), so this is a multi-DEPTH (semantic level) path, not a
    multi-scale one: deep-to-shallow progressive fusion, then a head
    projection and upsample to the decoder grid. It improves semantic
    continuity; it cannot recover sub-patch spatial detail (the 16x16
    patchification floor). Laterals/smooths/head are zero-initialized so
    the path activates gradually.
    """

    def __init__(self, embed_dim: int, mid_dim: int = 256, out_dim: int = 32, num_levels: int = 4):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(embed_dim, mid_dim, kernel_size=1) for _ in range(num_levels)]
        )
        self.smooths = nn.ModuleList(
            [
                nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim)
                for _ in range(num_levels - 1)
            ]
        )
        self.head = nn.Conv2d(mid_dim, out_dim, kernel_size=1)
        for m in list(self.laterals) + list(self.smooths) + [self.head]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self, features: List[torch.Tensor], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        # features: [B, H, W, C] tensors, shallow -> deep
        lat = [
            lateral(f.permute(0, 3, 1, 2).contiguous())
            for lateral, f in zip(self.laterals, features)
        ]
        p = lat[-1]  # deepest level
        for i in range(len(lat) - 2, -1, -1):
            p = self.smooths[i](lat[i] + p)
        out = self.head(p)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
