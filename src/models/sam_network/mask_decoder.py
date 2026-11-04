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
from .residual_moe import TopoResidualMoE
from .pepa import (
    CandidateRanker,
    SnakeUpsample2d,
    TargetAdaptiveThreshold,
)


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
        init_stage2_from_stage1: bool = False,
        use_typed_heads: bool = False,
        use_tcsu: bool = False,
        tcsu_scale: int = 4,
        tcsu_mid_channels: int = 32,
        tcsu_max_length: int = 5,
        use_tadt: bool = False,
        tadt_hidden: int = 128,
        tadt_t_min: float = -4.0,
        tadt_t_max: float = 4.0,
        use_ranker: bool = False,
        ranker_hidden: int = 128,
        ranker_pair_hidden: int = 128,
        use_detail_stem: bool = False,
        detail_channels: int = 32,
        high_res_output: bool = False,
        high_res_size: int = 512,
        high_res_refine: bool = False,
        high_res_refine_size: int = 1024,
        single_head: bool = False,
        use_structural_aux: bool = False,
        use_res_moe: bool = False,
        res_num_experts: int = 4,
        res_hidden: int = 24,
        res_gate_hidden: int = 32,
        res_gate_size: int = 512,
        res_gate_mode: str = 'learned',
        res_use_head_disagreement: bool = True,
        res_use_centerline: bool = True,
        res_gate_input_norm: bool = True,
        res_out_size: int = 512,
        res_head_index: int = 2,
        res_dropout: float = 0.0,
        res_disagreement_tau: float = 0.05,
        res_disagreement_scale: float = 0.10,
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
        self.use_typed_heads = bool(use_typed_heads)
        self.use_tcsu = bool(use_tcsu)
        self.use_tadt = bool(use_tadt)
        self.use_ranker = bool(use_ranker)
        self.use_detail_stem = bool(use_detail_stem)
        self.high_res_output = bool(high_res_output)
        self.high_res_size = int(high_res_size)
        self.high_res_refine = bool(high_res_refine)
        self.high_res_refine_size = int(high_res_refine_size)
        # Filled by predict_masks when TADT / the ranker is enabled.
        self.last_tadt_threshold = None
        self.last_rank_scores = None
        self.use_structural_aux = bool(use_structural_aux)
        self.last_aux_outputs = None
        self.use_res_moe = bool(use_res_moe)
        self.res_head_index = int(res_head_index)
        self.residual_moe = None

        self.single_head = bool(single_head)
        if self.single_head and self.use_typed_heads:
            raise ValueError('single_head and use_typed_heads are incompatible')
        self.num_multimask_outputs = 1 if self.single_head else num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 1 if self.single_head else num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        if self.use_typed_heads:
            # Each Stage-2 slot is a structural type.  Zero-initialized so
            # the typed preset starts from the exact SACM-X behaviour; the
            # type-specific losses (clDice/boundary/Tversky) differentiate
            # the slots during training.  Typed heads keep a stable
            # slot->type binding instead of following the gating order.
            self.type_embeddings = nn.Parameter(
                torch.zeros(self.num_mask_tokens, transformer_dim))

        if self.use_res_moe:
            if not 0 <= self.res_head_index < self.num_mask_tokens:
                raise ValueError(
                    f'res_head_index={self.res_head_index} outside '
                    f'[0, {self.num_mask_tokens})')
            self.residual_moe = TopoResidualMoE(
                num_experts=res_num_experts,
                hidden=res_hidden,
                gate_hidden=res_gate_hidden,
                gate_size=res_gate_size,
                gate_mode=res_gate_mode,
                use_head_disagreement=res_use_head_disagreement,
                use_centerline=res_use_centerline,
                gate_input_norm=res_gate_input_norm,
                out_size=res_out_size,
                head_index=self.res_head_index,
                dropout=res_dropout,
                disagreement_tau=res_disagreement_tau,
                disagreement_scale=res_disagreement_scale,
            )

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
        # NOTE: Stage-2 is intentionally NOT copied here.  The SAM
        # checkpoint is loaded by build_sam.py after construction, so a
        # copy in __init__ would only copy the random Stage-1 weights.
        # Call copy_stage1_to_stage2() after the base checkpoint is loaded.

        # IoU / quality prediction head (for each head)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # Optional PEPA-style post-encoder adapters.  Both are opt-in so the
        # default SACM-X graph and all existing checkpoints are unchanged.
        self.tcsu = None
        if self.use_tcsu:
            self.tcsu = SnakeUpsample2d(
                in_channels=transformer_dim,
                out_channels=transformer_dim // 8,
                scale=int(tcsu_scale),
                mid_channels=int(tcsu_mid_channels),
                max_length=int(tcsu_max_length),
                target_dim=transformer_dim,
            )
        self.tadt = None
        if self.use_tadt:
            self.tadt = TargetAdaptiveThreshold(
                dim=transformer_dim,
                hidden=int(tadt_hidden),
                t_min=float(tadt_t_min),
                t_max=float(tadt_t_max),
            )
        self.ranker = None
        if self.use_ranker:
            self.ranker = CandidateRanker(
                dim=transformer_dim,
                hidden=int(ranker_hidden),
                pair_hidden=int(ranker_pair_hidden),
            )
        self.detail_fuse_256 = None
        self.detail_fuse_512 = None
        if self.use_detail_stem:
            feature_dim = transformer_dim // 8
            self.detail_fuse_256 = nn.Conv2d(
                feature_dim + int(detail_channels), feature_dim, 1)
            self.detail_fuse_512 = nn.Conv2d(
                feature_dim + int(detail_channels), feature_dim, 1)
            nn.init.zeros_(self.detail_fuse_256.weight)
            nn.init.zeros_(self.detail_fuse_256.bias)
            nn.init.zeros_(self.detail_fuse_512.weight)
            nn.init.zeros_(self.detail_fuse_512.bias)
        self.final_refine = None
        if self.high_res_refine:
            self.final_refine = nn.Conv2d(
                self.num_mask_tokens + 3, self.num_mask_tokens, 3, padding=1)
            nn.init.zeros_(self.final_refine.weight)
            nn.init.zeros_(self.final_refine.bias)

        # Training-only structural heads for the prompt-decoder hybrid.
        self.structural_aux_conv = None
        if self.use_structural_aux:
            aux_dim = transformer_dim // 8
            self.structural_aux_conv = nn.Sequential(
                nn.Conv2d(aux_dim, aux_dim, 3, padding=1,
                          groups=aux_dim, bias=False),
                LayerNorm2d(aux_dim),
                nn.GELU(),
            )
            self.structural_aux_centerline = nn.Conv2d(aux_dim, 1, 1)
            self.structural_aux_boundary = nn.Conv2d(aux_dim, 1, 1)
            self.structural_aux_width = nn.Conv2d(aux_dim, 1, 1)

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

    def copy_stage1_to_stage2(self):
        """Warm-start Stage-2 heads from the *loaded* Stage-1 heads.

        Must be called after the base SAM checkpoint is loaded; doing it
        in __init__ only copies random Stage-1 parameters.
        """
        for stage1_mlp, stage2_mlp in zip(
                self.output_hypernetworks_mlps,
                self.output_hypernetworks_mlps_stage2):
            stage2_mlp.load_state_dict(stage1_mlp.state_dict())

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        adapter_features: List[torch.Tensor] = None,
        intermediate_features: List[torch.Tensor] = None,
        image: torch.Tensor = None,
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
          image (torch.Tensor): preprocessed input image, used by the
            optional residual-MoE refinement

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
            image=image,
        )

        if multimask_output:
            # SACM-X returns all heads in gating-ranked order; index 0 is
            # the best candidate.  Unlike vanilla SAM, token 0 is not a
            # separate single-mask token, so do not discard it.
            mask_slice = slice(0, self.num_multimask_outputs)
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

    def _refine_high_res(
        self,
        masks: torch.Tensor,
        image: torch.Tensor = None,
    ) -> torch.Tensor:
        """Image-guided full-resolution logit refinement.

        The 1/2-resolution candidates are upsampled and residual-refined
        with the preprocessed input image.  The refinement conv is
        zero-initialized, so at step 0 this is exactly an upsample.
        """
        if not self.high_res_refine or self.final_refine is None:
            return masks
        if image is None:
            return masks
        size = (self.high_res_refine_size, self.high_res_refine_size)
        if masks.shape[-2:] != size:
            masks = F.interpolate(
                masks, size=size, mode='bilinear', align_corners=False)
        if image.shape[-2:] != size:
            image = F.interpolate(
                image, size=size, mode='bilinear', align_corners=False)
        if image.shape[1] != 3:
            image = image[:, :3]
        return masks + self.final_refine(torch.cat([masks, image], dim=1))

    def _apply_res_moe(
        self,
        masks: torch.Tensor,
        image: torch.Tensor = None,
        all_head_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """Refine one Stage-2 head with the optional residual MoE."""
        if (self.residual_moe is None or image is None
                or masks.shape[1] <= self.res_head_index):
            return masks
        h = self.res_head_index
        corrected = self.residual_moe(
            image,
            masks[:, h:h + 1],
            all_head_logits=all_head_logits,
            aux=self.last_aux_outputs,
        )
        return torch.cat(
            [masks[:, :h], corrected, masks[:, h + 1:]], dim=1)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_features: List[torch.Tensor] = None,
        intermediate_features: List[torch.Tensor] = None,
        detail_features: Tuple[torch.Tensor, torch.Tensor] = None,
        image: torch.Tensor = None,
        return_stage1: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details.

        With return_stage1=True, also returns the Stage-1 coarse masks
        (reordered by the same gating) for deep supervision.
        """
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # ``tokens`` has one entry per image (or K prompts per image).
        # Repeat each image embedding by K, not by the total token batch,
        # otherwise batch_size > 1 becomes B*B.
        image_batch = image_embeddings.shape[0]
        repeats, remainder = divmod(tokens.shape[0], image_batch)
        if repeats < 1 or remainder:
            raise ValueError(
                f'token batch {tokens.shape[0]} is not a multiple of '
                f'image batch {image_batch}')
        src = image_embeddings.repeat_interleave(repeats, dim=0)
        pos_src = image_pe.repeat_interleave(repeats, dim=0)
        if dense_prompt_embeddings.shape[0] == image_batch and repeats > 1:
            dense_prompt_embeddings = dense_prompt_embeddings.repeat_interleave(
                repeats, dim=0)
        # Multi-prompt inference (SamPredictor / K prompts per image): the
        # encoder features are still per-image and must follow the same
        # repetition as the image embeddings.
        if adapter_features is not None and repeats > 1:
            adapter_features = [
                feat.repeat_interleave(repeats, dim=0)
                if feat.shape[0] == image_batch else feat
                for feat in adapter_features
            ]
        if intermediate_features is not None and repeats > 1:
            intermediate_features = [
                feat.repeat_interleave(repeats, dim=0)
                if feat.shape[0] == image_batch else feat
                for feat in intermediate_features
            ]
        src = src + dense_prompt_embeddings
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Target-conditioned snake upsampling (PEPA TCSU).  The branch is a
        # zero-initialized residual on the existing upsampling, so enabling it
        # starts from the exact baseline behaviour.
        if self.use_tcsu and self.tcsu is not None:
            target = mask_tokens_out.mean(dim=1)
            tcsu_out = self.tcsu(src, target)
            if tcsu_out.shape[-2:] != upscaled_embedding.shape[-2:]:
                tcsu_out = F.interpolate(
                    tcsu_out, size=upscaled_embedding.shape[-2:],
                    mode='bilinear', align_corners=False)
            upscaled_embedding = upscaled_embedding + tcsu_out

        # High-resolution detail fusion at the decoder grid (1/4 scale).
        if (self.use_detail_stem and detail_features is not None
                and self.detail_fuse_256 is not None):
            detail_256 = detail_features[0]
            if detail_256.shape[-2:] != upscaled_embedding.shape[-2:]:
                detail_256 = F.interpolate(
                    detail_256, size=upscaled_embedding.shape[-2:],
                    mode='bilinear', align_corners=False)
            upscaled_embedding = upscaled_embedding + self.detail_fuse_256(
                torch.cat([upscaled_embedding, detail_256], dim=1))

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
            token = mask_tokens_out[:, i, :]
            if self.use_typed_heads:
                token = token + self.type_embeddings[i]
            hyper_in_list_stage1.append(
                self.output_hypernetworks_mlps[i](token))
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

        # Optional high-resolution mask substrate (1/2 scale).  Built AFTER
        # the coarse-to-fine refiner so Stage-2 actually consumes U'.  Stage-1
        # and the refiner stay at the decoder grid.
        mask_features = upscaled_embedding
        if self.high_res_output:
            if (self.use_detail_stem and detail_features is not None
                    and self.detail_fuse_512 is not None):
                detail_512 = detail_features[1]
                target_size = detail_512.shape[-2:]
                mask_features = F.interpolate(
                    upscaled_embedding, size=target_size,
                    mode='bilinear', align_corners=False)
                mask_features = mask_features + self.detail_fuse_512(
                    torch.cat([mask_features, detail_512], dim=1))
            else:
                target_size = (self.high_res_size, self.high_res_size)
                mask_features = F.interpolate(
                    upscaled_embedding, size=target_size,
                    mode='bilinear', align_corners=False)

        # Training-only structural heads on the post-C2F high-res features.
        # They do not affect the mask logits; they add centerline/boundary/
        # width supervision to the shared features.
        if self.use_structural_aux:
            aux_feat = self.structural_aux_conv(mask_features)
            self.last_aux_outputs = {
                'centerline': self.structural_aux_centerline(aux_feat),
                'centerline_256': None,
                'boundary': self.structural_aux_boundary(aux_feat),
                'width': self.structural_aux_width(aux_feat),
            }
        else:
            self.last_aux_outputs = None

        # --------------------- Stage 2: refined masks ---------------------
        # Typed variant: keep a stable slot -> structural-type binding.
        # Unlike the gating-ranked SACM-X path, slot i always receives
        # mask token i plus its type embedding, so head i can specialize as
        # thin/blob/boundary/recall without chasing the gating permutation.
        if self.use_typed_heads:
            typed_tokens = mask_tokens_out + self.type_embeddings.unsqueeze(0)
            hyper_in_list_stage2: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list_stage2.append(
                    self.output_hypernetworks_mlps_stage2[i](
                        typed_tokens[:, i, :]))
            hyper_in_stage2 = torch.stack(hyper_in_list_stage2, dim=1)
            b2, c2, h2, w2 = mask_features.shape
            masks_stage2 = (
                hyper_in_stage2 @ mask_features.view(b2, c2, h2 * w2)
            ).view(b2, -1, h2, w2)
            head_iou = [
                self.iou_prediction_head(typed_tokens[:, i, :])[:, i]
                for i in range(self.num_mask_tokens)
            ]
            iou_pred = torch.stack(head_iou, dim=1)
            if self.use_ranker and self.ranker is not None:
                iou_pred = self.ranker(typed_tokens)
                self.last_rank_scores = iou_pred.detach()
            else:
                self.last_rank_scores = None
            if self.high_res_refine:
                masks_stage2 = self._refine_high_res(masks_stage2, image)
            if self.use_tadt and self.tadt is not None:
                masks_stage2, tadt_t = self.tadt(typed_tokens, masks_stage2)
                self.last_tadt_threshold = tadt_t.detach()
            else:
                self.last_tadt_threshold = None
            masks_stage2 = self._apply_res_moe(
                masks_stage2, image, all_head_logits=masks_stage2)
            if return_stage1:
                # Stage-1 and Stage-2 are both in stable type order.
                return masks_stage2, iou_pred, masks_stage1
            return masks_stage2, iou_pred

        # Eq.10: reorder the descriptors by the Stage-1 gating order, then
        # apply each Stage-2 MLP to its ranked slot.  The masks are produced
        # directly in ranked order, so they must NOT be gathered again.
        sorted_tokens = torch.gather(
            mask_tokens_out, 1,
            sorted_indices.unsqueeze(-1).expand(
                -1, -1, mask_tokens_out.shape[-1]))
        hyper_in_list_stage2: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list_stage2.append(
                self.output_hypernetworks_mlps_stage2[i](
                    sorted_tokens[:, i, :]))
        hyper_in_stage2 = torch.stack(hyper_in_list_stage2, dim=1)
        b2, c2, h2, w2 = mask_features.shape
        masks_stage2_ordered = (
            hyper_in_stage2 @ mask_features.view(b2, c2, h2 * w2)
        ).view(b2, -1, h2, w2)

        # Per-head IoU prediction: output slot j scores Stage-2 head j on
        # its ranked descriptor, so the slot-to-head binding survives the
        # gating permutation.
        head_iou = [
            self.iou_prediction_head(sorted_tokens[:, i, :])[:, i]
            for i in range(self.num_mask_tokens)
        ]
        iou_pred_ordered = torch.stack(head_iou, dim=1)

        if self.use_ranker and self.ranker is not None:
            iou_pred_ordered = self.ranker(sorted_tokens)
            self.last_rank_scores = iou_pred_ordered.detach()
        else:
            self.last_rank_scores = None

        if self.high_res_refine:
            masks_stage2_ordered = self._refine_high_res(
                masks_stage2_ordered, image)

        if self.use_tadt and self.tadt is not None:
            masks_stage2_ordered, tadt_t = self.tadt(
                sorted_tokens, masks_stage2_ordered)
            self.last_tadt_threshold = tadt_t.detach()
        else:
            self.last_tadt_threshold = None

        masks_stage2_ordered = self._apply_res_moe(
            masks_stage2_ordered, image,
            all_head_logits=masks_stage2_ordered)

        if return_stage1:
            gather_indices = sorted_indices.unsqueeze(-1).unsqueeze(
                -1).expand(-1, -1, h, w)
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
