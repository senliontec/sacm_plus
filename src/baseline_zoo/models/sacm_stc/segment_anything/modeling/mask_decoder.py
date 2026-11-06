# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        adapter_num_layers: int = 24,
        adapter_proj_dim: int = 32,
        adapter_hidden: int = 128,
        stable_stage2: bool = False,
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
          adapter_num_layers (int): number of Adapter-E features consumed by
            the Eq. 6 fusion (one per encoder block)
          adapter_proj_dim (int): per-layer low-rank projection used to build
            the concatenated layer descriptors
          adapter_hidden (int): hidden dim of the layer-weight FFN
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.use_adapter_skip = use_adapter_skip
        self.stable_stage2 = bool(stable_stage2)

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

        # IoU / quality prediction head.  Output slot j is permanently bound
        # to Stage-2 mask head j, so it is evaluated on the ranked Stage-2
        # descriptor of head j (not on the iou token as in the official
        # code) and the Eq. 10 descriptor reorder stays consistent.
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        if self.stable_stage2:
            # Stable task heads need a head-specific token offset; otherwise
            # the four MLPs see interchangeable SAM mask tokens.
            self.stage2_type_emb = nn.Parameter(
                torch.zeros(self.num_mask_tokens, transformer_dim))
            # Give Stage-1 a real inference role: its four coarse masks are
            # fused into the Stage-2 mask substrate instead of only providing
            # gating scores that stable heads no longer consume.
            self.stage1_prior = nn.Sequential(
                nn.Conv2d(self.num_mask_tokens, transformer_dim // 8, 3,
                          padding=1),
                LayerNorm2d(transformer_dim // 8),
                nn.GELU(),
                nn.Conv2d(transformer_dim // 8, transformer_dim // 8, 1),
            )
            nn.init.zeros_(self.stage1_prior[-1].weight)
            nn.init.zeros_(self.stage1_prior[-1].bias)
            # The width head regresses a bounded pixel width.  A random last
            # layer can emit saturated logits, and under the official
            # learning rate such a head never leaves saturation.  Start it
            # from a neutral zero logit instead.
            width_mlp = self.output_hypernetworks_mlps_stage2[-1]
            nn.init.zeros_(width_mlp.layers[-1].weight)
            nn.init.zeros_(width_mlp.layers[-1].bias)

        if self.use_adapter_skip:
            # Eq. 6: alpha = Softmax(FFN(Concat(z_1, ..., z_L))).
            # Each pooled layer descriptor is projected to a low-rank space
            # before concatenation; this is a tractable approximation of the
            # paper's L*D concatenation.
            self.adapter_num_layers = adapter_num_layers
            self.adapter_proj_dim = adapter_proj_dim
            self.adapter_weight_proj = nn.ModuleList([
                nn.Linear(adapter_embed_dim, adapter_proj_dim)
                for _ in range(adapter_num_layers)
            ])
            self.adapter_weight_ffn = nn.Sequential(
                nn.LayerNorm(adapter_num_layers * adapter_proj_dim),
                nn.Linear(adapter_num_layers * adapter_proj_dim,
                          adapter_hidden),
                activation(),
                nn.Linear(adapter_hidden, adapter_num_layers),
            )
            self.adapter_aggregator = nn.Sequential(
                nn.LayerNorm(adapter_embed_dim),
                nn.Linear(adapter_embed_dim, transformer_dim // 4),
                activation(),
                nn.Linear(transformer_dim // 4, transformer_dim // 8),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        adapter_features: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        This prompt-free SACM has ``num_mask_tokens`` (4) independent
        candidates instead of SAM's 3 disambiguating masks.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): if False, return the single candidate with
            the highest predicted IoU; if True, return all candidates in
            Stage-1 gating order
          adapter_features (List[torch.Tensor]): Adapter-E outputs from the
            image encoder, one feature map per transformer block

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
        )

        if multimask_output:
            return masks, iou_pred

        # The official code sliced [1:] here, which drops the gating
        # champion.  Prompt-free SACM has no "single-mask" semantic, so
        # select the candidate the trained IoU head considers best.
        best = iou_pred.argmax(dim=1)
        mask_indices = best[:, None, None, None].expand(
            -1, 1, *masks.shape[2:])
        masks = torch.gather(masks, 1, mask_indices)
        iou_pred = torch.gather(iou_pred, 1, best[:, None])
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_features: List[torch.Tensor] = None,
        return_stage1: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Predicts masks. See 'forward' for more details.

        With ``return_stage1=True`` the gating-ordered Stage-1 coarse masks
        are returned as a third value for deep supervision.
        """
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Number of prompt queries per image.  The official code repeated
        # each image tokens.shape[0] times, which becomes B*B for B > 1.
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
        src = src + dense_prompt_embeddings
        if adapter_features is not None and repeats > 1:
            adapter_features = [
                feat.repeat_interleave(repeats, dim=0)
                if feat.shape[0] == image_batch else feat
                for feat in adapter_features
            ]
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # ---------------- Adapter fusion (paper Eq. 5-7) ----------------
        if self.use_adapter_skip and adapter_features is not None:
            if len(adapter_features) != self.adapter_num_layers:
                raise ValueError(
                    f'fusion expects {self.adapter_num_layers} Adapter-E '
                    f'features, got {len(adapter_features)}; check that the '
                    'encoder emits one feature per transformer block')
            pooled = torch.stack(
                [feat.mean(dim=(1, 2)) for feat in adapter_features], dim=1
            )  # [B, L, C]
            descriptors = torch.cat(
                [self.adapter_weight_proj[l](pooled[:, l])
                 for l in range(self.adapter_num_layers)], dim=-1
            )  # [B, L*P]
            # Eq. 6: softmax over unbounded FFN logits.  The official code
            # applied Sigmoid first, compressing all 24 weights into about
            # [0.039, 0.106] and making the layer weighting useless.
            weights = F.softmax(self.adapter_weight_ffn(descriptors), dim=-1)

            fused = torch.zeros_like(adapter_features[0])
            for l in range(self.adapter_num_layers):
                fused = fused + adapter_features[l] * weights[
                    :, l].view(-1, 1, 1, 1)

            processed_adapter = self.adapter_aggregator(fused)
            processed_adapter = processed_adapter.permute(0, 3, 1, 2)
            if processed_adapter.shape[2:] != upscaled_embedding.shape[2:]:
                processed_adapter = F.interpolate(
                    processed_adapter,
                    size=upscaled_embedding.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            upscaled_embedding = upscaled_embedding + processed_adapter

        # --------------------- Stage 1: initial masks ---------------------
        hyper_in_list_stage1: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list_stage1.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in_stage1 = torch.stack(hyper_in_list_stage1, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks_stage1 = (hyper_in_stage1 @ upscaled_embedding.view(
            b, c, h * w)).view(b, -1, h, w)

        # Stage-1 confidence ranks the descriptors (paper Eq. 9).
        gating_scores = masks_stage1.view(
            b, self.num_mask_tokens, -1).max(dim=-1).values
        gating_weights = F.softmax(gating_scores, dim=1)
        if self.stable_stage2:
            sorted_indices = torch.arange(
                self.num_mask_tokens, device=gating_weights.device
            ).unsqueeze(0).expand(b, -1)
        else:
            sorted_indices = torch.argsort(
                gating_weights, dim=1, descending=True)

        # Stage-1 now contributes a coarse spatial prior to Stage-2.  The
        # final 1x1 conv is zero-initialised, so this is an identity start.
        if self.stable_stage2:
            upscaled_embedding = (
                upscaled_embedding + self.stage1_prior(masks_stage1))

        # --------------------- Stage 2: refined masks ---------------------
        # Eq. 10: MLP2 consumes the descriptors reordered by s.  The masks
        # are produced directly in ranked order, so the Stage-2 outputs must
        # NOT be gathered again.  (The official code reordered outputs
        # instead of inputs, pairing each descriptor with the wrong head.)
        sorted_tokens = torch.gather(
            mask_tokens_out, 1,
            sorted_indices.unsqueeze(-1).expand(
                -1, -1, mask_tokens_out.shape[-1]))
        if self.stable_stage2:
            sorted_tokens = sorted_tokens + self.stage2_type_emb.unsqueeze(0)
        hyper_in_list_stage2: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list_stage2.append(
                self.output_hypernetworks_mlps_stage2[i](
                    sorted_tokens[:, i, :]))
        hyper_in_stage2 = torch.stack(hyper_in_list_stage2, dim=1)
        masks_stage2 = (hyper_in_stage2 @ upscaled_embedding.view(
            b, c, h * w)).view(b, -1, h, w)

        # Per-head IoU prediction: output slot j scores Stage-2 head j on
        # its ranked descriptor, so the ranking permutation cannot break
        # the slot-to-head binding.
        head_iou = []
        for i in range(self.num_mask_tokens):
            head_iou.append(
                self.iou_prediction_head(sorted_tokens[:, i, :])[:, i])
        iou_pred = torch.stack(head_iou, dim=1)

        if return_stage1:
            stage1_indices = sorted_indices.unsqueeze(-1).unsqueeze(
                -1).expand(-1, -1, h, w)
            masks_stage1_ordered = torch.gather(
                masks_stage1, 1, stage1_indices)
            return masks_stage2, iou_pred, masks_stage1_ordered
        return masks_stage2, iou_pred


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
