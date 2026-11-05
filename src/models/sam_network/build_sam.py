# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified: builder arguments for the geometric adapters, the closed-loop
# coarse-to-fine decoder, Fusion v2 and the multi-depth path.

import logging

import torch

from functools import partial

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam
from .structural_decoder import StructuralDecoderV3
from .transformer import TwoWayTransformer


def load_compatible_state_dict(model, state_dict, *, allow_truncate=False):
    """Load a Skein checkpoint into a possibly different preset.

    Drops parameters that the target architecture does not have (for example
    the gate-only ``mask_decoder.residual_moe.*`` entries when loading the
    deployed checkpoint with the canonical ``skein`` preset), while still
    failing on any target key that receives no value.  This is deliberately
    stricter than ``strict=False``: an architecture mismatch must not
    silently leave a head randomly initialized.

    By default a checkpoint tensor whose first dimension is *larger* than the
    target is rejected as a shape mismatch.  Pass ``allow_truncate=True``
    only for an intentional head-count reduction; otherwise a stale head
    layout would be silently sliced.
    """
    target = model.state_dict()
    compatible = {}
    dropped = []
    shape_skipped = []
    for key, value in state_dict.items():
        key = key[len('module.'):] if key.startswith('module.') else key
        if key not in target:
            dropped.append(key)
            continue
        target_shape = target[key].shape
        if value.shape == target_shape:
            compatible[key] = value
        elif (allow_truncate
              and value.ndim == len(target_shape)
              and value.shape[1:] == tuple(target_shape[1:])
              and value.shape[0] > target_shape[0]):
            compatible[key] = value[:target_shape[0]]
        else:
            shape_skipped.append(key)
    missing = [key for key in target if key not in compatible]
    if missing or shape_skipped:
        raise RuntimeError(
            "checkpoint is incompatible with the target architecture: "
            f"{len(missing)} missing keys, "
            f"{len(shape_skipped)} shape-mismatched keys; "
            f"missing (first 10)={missing[:10]}, "
            f"shape-skipped (first 10)={shape_skipped[:10]}")
    model.load_state_dict(compatible, strict=True)
    return {
        'dropped': dropped,
        'shape_skipped': shape_skipped,
        'loaded': len(compatible),
        'target': len(target),
    }


def build_sam_vit_h(
    checkpoint=None,
    use_adapter=False,
    adapter_dim_ratio=0.1,
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    use_dynamic_strip=False,
    init_stage2_from_stage1=False,
    use_typed_heads=False,
    use_tcsu=False,
    tcsu_scale=4,
    tcsu_mid_channels=32,
    tcsu_max_length=5,
    use_tadt=False,
    tadt_hidden=128,
    tadt_t_min=-4.0,
    tadt_t_max=4.0,
    use_ranker=False,
    ranker_hidden=128,
    ranker_pair_hidden=128,
    use_detail_stem=False,
    detail_channels=32,
    detail_out_channels=32,
    high_res_output=False,
    high_res_size=512,
    high_res_refine=False,
    high_res_refine_size=1024,
    single_head=False,
    use_v3=False,
    use_structural_aux=False,
    v3_feature_dim=128,
    v3_mid_dim=96,
    v3_fine_dim=64,
    v3_refine_channels=16,
    v3_aux_centerline=True,
    v3_aux_boundary=True,
    v3_aux_width=True,
    use_internal_adapter=True,
    collect_adapter_features=True,
    use_convlora=False,
    convlora_rank=4,
    convlora_alpha=32.0,
    convlora_experts=8,
    use_evp=False,
    evp_freq_nums=0.25,
    evp_scale_factor=32,
    evp_zero_init=True,
    use_res_moe=False,
    res_num_experts=4,
    res_hidden=24,
    res_gate_hidden=32,
    res_gate_size=512,
    res_gate_mode='learned',
    res_use_head_disagreement=True,
    res_use_centerline=True,
    res_gate_input_norm=True,
    res_out_size=512,
    res_head_index=2,
    res_dropout=0.0,
    res_disagreement_tau=0.05,
    res_disagreement_scale=0.10,
):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        use_adapter=use_adapter,
        adapter_dim_ratio=adapter_dim_ratio,
        use_geo_i=use_geo_i,
        use_geo_e=use_geo_e,
        geo_e_layers=geo_e_layers,
        use_coarse_to_fine=use_coarse_to_fine,
        use_fusion_v2=use_fusion_v2,
        use_multi_depth=use_multi_depth,
        use_dynamic_strip=use_dynamic_strip,
        init_stage2_from_stage1=init_stage2_from_stage1,
        use_typed_heads=use_typed_heads,
        use_tcsu=use_tcsu,
        tcsu_scale=tcsu_scale,
        tcsu_mid_channels=tcsu_mid_channels,
        tcsu_max_length=tcsu_max_length,
        use_tadt=use_tadt,
        tadt_hidden=tadt_hidden,
        tadt_t_min=tadt_t_min,
        tadt_t_max=tadt_t_max,
        use_ranker=use_ranker,
        ranker_hidden=ranker_hidden,
        ranker_pair_hidden=ranker_pair_hidden,
        use_detail_stem=use_detail_stem,
        detail_channels=detail_channels,
        detail_out_channels=detail_out_channels,
        high_res_output=high_res_output,
        high_res_size=high_res_size,
        high_res_refine=high_res_refine,
        high_res_refine_size=high_res_refine_size,
        single_head=single_head,
        use_v3=use_v3,
        use_structural_aux=use_structural_aux,
        v3_feature_dim=v3_feature_dim,
        v3_mid_dim=v3_mid_dim,
        v3_fine_dim=v3_fine_dim,
        v3_refine_channels=v3_refine_channels,
        v3_aux_centerline=v3_aux_centerline,
        v3_aux_boundary=v3_aux_boundary,
        v3_aux_width=v3_aux_width,
        use_internal_adapter=use_internal_adapter,
        collect_adapter_features=collect_adapter_features,
        use_convlora=use_convlora,
        convlora_rank=convlora_rank,
        convlora_alpha=convlora_alpha,
        convlora_experts=convlora_experts,
        use_evp=use_evp,
        evp_freq_nums=evp_freq_nums,
        evp_scale_factor=evp_scale_factor,
        evp_zero_init=evp_zero_init,
        use_res_moe=use_res_moe,
        res_num_experts=res_num_experts,
        res_hidden=res_hidden,
        res_gate_hidden=res_gate_hidden,
        res_gate_size=res_gate_size,
        res_gate_mode=res_gate_mode,
        res_use_head_disagreement=res_use_head_disagreement,
        res_use_centerline=res_use_centerline,
        res_gate_input_norm=res_gate_input_norm,
        res_out_size=res_out_size,
        res_head_index=res_head_index,
        res_dropout=res_dropout,
        res_disagreement_tau=res_disagreement_tau,
        res_disagreement_scale=res_disagreement_scale,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(
    checkpoint=None,
    use_adapter=False,
    adapter_dim_ratio=0.1,
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    use_dynamic_strip=False,
    init_stage2_from_stage1=False,
    use_typed_heads=False,
    use_tcsu=False,
    tcsu_scale=4,
    tcsu_mid_channels=32,
    tcsu_max_length=5,
    use_tadt=False,
    tadt_hidden=128,
    tadt_t_min=-4.0,
    tadt_t_max=4.0,
    use_ranker=False,
    ranker_hidden=128,
    ranker_pair_hidden=128,
    use_detail_stem=False,
    detail_channels=32,
    detail_out_channels=32,
    high_res_output=False,
    high_res_size=512,
    high_res_refine=False,
    high_res_refine_size=1024,
    single_head=False,
    use_v3=False,
    use_structural_aux=False,
    v3_feature_dim=128,
    v3_mid_dim=96,
    v3_fine_dim=64,
    v3_refine_channels=16,
    v3_aux_centerline=True,
    v3_aux_boundary=True,
    v3_aux_width=True,
    use_internal_adapter=True,
    collect_adapter_features=True,
    use_convlora=False,
    convlora_rank=4,
    convlora_alpha=32.0,
    convlora_experts=8,
    use_evp=False,
    evp_freq_nums=0.25,
    evp_scale_factor=32,
    evp_zero_init=True,
    use_res_moe=False,
    res_num_experts=4,
    res_hidden=24,
    res_gate_hidden=32,
    res_gate_size=512,
    res_gate_mode='learned',
    res_use_head_disagreement=True,
    res_use_centerline=True,
    res_gate_input_norm=True,
    res_out_size=512,
    res_head_index=2,
    res_dropout=0.0,
    res_disagreement_tau=0.05,
    res_disagreement_scale=0.10,
):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        use_adapter=use_adapter,
        adapter_dim_ratio=adapter_dim_ratio,
        use_geo_i=use_geo_i,
        use_geo_e=use_geo_e,
        geo_e_layers=geo_e_layers,
        use_coarse_to_fine=use_coarse_to_fine,
        use_fusion_v2=use_fusion_v2,
        use_multi_depth=use_multi_depth,
        use_dynamic_strip=use_dynamic_strip,
        init_stage2_from_stage1=init_stage2_from_stage1,
        use_typed_heads=use_typed_heads,
        use_tcsu=use_tcsu,
        tcsu_scale=tcsu_scale,
        tcsu_mid_channels=tcsu_mid_channels,
        tcsu_max_length=tcsu_max_length,
        use_tadt=use_tadt,
        tadt_hidden=tadt_hidden,
        tadt_t_min=tadt_t_min,
        tadt_t_max=tadt_t_max,
        use_ranker=use_ranker,
        ranker_hidden=ranker_hidden,
        ranker_pair_hidden=ranker_pair_hidden,
        use_detail_stem=use_detail_stem,
        detail_channels=detail_channels,
        detail_out_channels=detail_out_channels,
        high_res_output=high_res_output,
        high_res_size=high_res_size,
        high_res_refine=high_res_refine,
        high_res_refine_size=high_res_refine_size,
        single_head=single_head,
        use_v3=use_v3,
        use_structural_aux=use_structural_aux,
        v3_feature_dim=v3_feature_dim,
        v3_mid_dim=v3_mid_dim,
        v3_fine_dim=v3_fine_dim,
        v3_refine_channels=v3_refine_channels,
        v3_aux_centerline=v3_aux_centerline,
        v3_aux_boundary=v3_aux_boundary,
        v3_aux_width=v3_aux_width,
        use_internal_adapter=use_internal_adapter,
        collect_adapter_features=collect_adapter_features,
        use_convlora=use_convlora,
        convlora_rank=convlora_rank,
        convlora_alpha=convlora_alpha,
        convlora_experts=convlora_experts,
        use_evp=use_evp,
        evp_freq_nums=evp_freq_nums,
        evp_scale_factor=evp_scale_factor,
        evp_zero_init=evp_zero_init,
        use_res_moe=use_res_moe,
        res_num_experts=res_num_experts,
        res_hidden=res_hidden,
        res_gate_hidden=res_gate_hidden,
        res_gate_size=res_gate_size,
        res_gate_mode=res_gate_mode,
        res_use_head_disagreement=res_use_head_disagreement,
        res_use_centerline=res_use_centerline,
        res_gate_input_norm=res_gate_input_norm,
        res_out_size=res_out_size,
        res_head_index=res_head_index,
        res_dropout=res_dropout,
        res_disagreement_tau=res_disagreement_tau,
        res_disagreement_scale=res_disagreement_scale,
    )


def build_sam_vit_b(
    checkpoint=None,
    use_adapter=False,
    adapter_dim_ratio=0.1,
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    use_dynamic_strip=False,
    init_stage2_from_stage1=False,
    use_typed_heads=False,
    use_tcsu=False,
    tcsu_scale=4,
    tcsu_mid_channels=32,
    tcsu_max_length=5,
    use_tadt=False,
    tadt_hidden=128,
    tadt_t_min=-4.0,
    tadt_t_max=4.0,
    use_ranker=False,
    ranker_hidden=128,
    ranker_pair_hidden=128,
    use_detail_stem=False,
    detail_channels=32,
    detail_out_channels=32,
    high_res_output=False,
    high_res_size=512,
    high_res_refine=False,
    high_res_refine_size=1024,
    single_head=False,
    use_v3=False,
    use_structural_aux=False,
    v3_feature_dim=128,
    v3_mid_dim=96,
    v3_fine_dim=64,
    v3_refine_channels=16,
    v3_aux_centerline=True,
    v3_aux_boundary=True,
    v3_aux_width=True,
    use_internal_adapter=True,
    collect_adapter_features=True,
    use_convlora=False,
    convlora_rank=4,
    convlora_alpha=32.0,
    convlora_experts=8,
    use_evp=False,
    evp_freq_nums=0.25,
    evp_scale_factor=32,
    evp_zero_init=True,
    use_res_moe=False,
    res_num_experts=4,
    res_hidden=24,
    res_gate_hidden=32,
    res_gate_size=512,
    res_gate_mode='learned',
    res_use_head_disagreement=True,
    res_use_centerline=True,
    res_gate_input_norm=True,
    res_out_size=512,
    res_head_index=2,
    res_dropout=0.0,
    res_disagreement_tau=0.05,
    res_disagreement_scale=0.10,
):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        use_adapter=use_adapter,
        adapter_dim_ratio=adapter_dim_ratio,
        use_geo_i=use_geo_i,
        use_geo_e=use_geo_e,
        geo_e_layers=geo_e_layers,
        use_coarse_to_fine=use_coarse_to_fine,
        use_fusion_v2=use_fusion_v2,
        use_multi_depth=use_multi_depth,
        use_dynamic_strip=use_dynamic_strip,
        init_stage2_from_stage1=init_stage2_from_stage1,
        use_typed_heads=use_typed_heads,
        use_tcsu=use_tcsu,
        tcsu_scale=tcsu_scale,
        tcsu_mid_channels=tcsu_mid_channels,
        tcsu_max_length=tcsu_max_length,
        use_tadt=use_tadt,
        tadt_hidden=tadt_hidden,
        tadt_t_min=tadt_t_min,
        tadt_t_max=tadt_t_max,
        use_ranker=use_ranker,
        ranker_hidden=ranker_hidden,
        ranker_pair_hidden=ranker_pair_hidden,
        use_detail_stem=use_detail_stem,
        detail_channels=detail_channels,
        detail_out_channels=detail_out_channels,
        high_res_output=high_res_output,
        high_res_size=high_res_size,
        high_res_refine=high_res_refine,
        high_res_refine_size=high_res_refine_size,
        single_head=single_head,
        use_v3=use_v3,
        use_structural_aux=use_structural_aux,
        v3_feature_dim=v3_feature_dim,
        v3_mid_dim=v3_mid_dim,
        v3_fine_dim=v3_fine_dim,
        v3_refine_channels=v3_refine_channels,
        v3_aux_centerline=v3_aux_centerline,
        v3_aux_boundary=v3_aux_boundary,
        v3_aux_width=v3_aux_width,
        use_internal_adapter=use_internal_adapter,
        collect_adapter_features=collect_adapter_features,
        use_convlora=use_convlora,
        convlora_rank=convlora_rank,
        convlora_alpha=convlora_alpha,
        convlora_experts=convlora_experts,
        use_evp=use_evp,
        evp_freq_nums=evp_freq_nums,
        evp_scale_factor=evp_scale_factor,
        evp_zero_init=evp_zero_init,
        use_res_moe=use_res_moe,
        res_num_experts=res_num_experts,
        res_hidden=res_hidden,
        res_gate_hidden=res_gate_hidden,
        res_gate_size=res_gate_size,
        res_gate_mode=res_gate_mode,
        res_use_head_disagreement=res_use_head_disagreement,
        res_use_centerline=res_use_centerline,
        res_gate_input_norm=res_gate_input_norm,
        res_out_size=res_out_size,
        res_head_index=res_head_index,
        res_dropout=res_dropout,
        res_disagreement_tau=res_disagreement_tau,
        res_disagreement_scale=res_disagreement_scale,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    use_adapter=False,
    adapter_dim_ratio=0.1,
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    use_dynamic_strip=False,
    init_stage2_from_stage1=False,
    use_typed_heads=False,
    use_tcsu=False,
    tcsu_scale=4,
    tcsu_mid_channels=32,
    tcsu_max_length=5,
    use_tadt=False,
    tadt_hidden=128,
    tadt_t_min=-4.0,
    tadt_t_max=4.0,
    use_ranker=False,
    ranker_hidden=128,
    ranker_pair_hidden=128,
    use_detail_stem=False,
    detail_channels=32,
    detail_out_channels=32,
    high_res_output=False,
    high_res_size=512,
    high_res_refine=False,
    high_res_refine_size=1024,
    single_head=False,
    use_v3=False,
    use_structural_aux=False,
    v3_feature_dim=128,
    v3_mid_dim=96,
    v3_fine_dim=64,
    v3_refine_channels=16,
    v3_aux_centerline=True,
    v3_aux_boundary=True,
    v3_aux_width=True,
    use_internal_adapter=True,
    collect_adapter_features=True,
    use_convlora=False,
    convlora_rank=4,
    convlora_alpha=32.0,
    convlora_experts=8,
    use_evp=False,
    evp_freq_nums=0.25,
    evp_scale_factor=32,
    evp_zero_init=True,
    use_res_moe=False,
    res_num_experts=4,
    res_hidden=24,
    res_gate_hidden=32,
    res_gate_size=512,
    res_gate_mode='learned',
    res_use_head_disagreement=True,
    res_use_centerline=True,
    res_gate_input_norm=True,
    res_out_size=512,
    res_head_index=2,
    res_dropout=0.0,
    res_disagreement_tau=0.05,
    res_disagreement_scale=0.10,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # External-adapter layer placement: 'all' = every window-attention
    # layer; 'shallow'/'deep' = the first/second HALF of the window
    # layers (for ViT-L this is exactly {0-4, 6-10} / {12-16, 18-22} —
    # the R2 层子集 of the design). The halves rule generalizes to any
    # depth (ViT-B/H) without ever producing an empty subset.
    window_layers = [i for i in range(encoder_depth)
                     if i not in set(encoder_global_attn_indexes)]
    half = len(window_layers) // 2
    if geo_e_layers == 'shallow':
        geo_e_indices = tuple(window_layers[:half])
    elif geo_e_layers == 'deep':
        geo_e_indices = tuple(window_layers[half:])
    else:
        geo_e_indices = tuple(window_layers)  # 'all'

    # Number of external-adapter feature layers consumed by the decoder
    # fusion module.
    adapter_num_layers = (
        len(geo_e_indices)
        if (use_adapter and use_geo_e)
        else encoder_depth
    )

    if use_v3:
        # Dense structural decoder: no prompt tokens, no candidate heads.
        mask_decoder = StructuralDecoderV3(
            encoder_embed_dim=encoder_embed_dim,
            decoder_dim=prompt_embed_dim,
            feature_dim=v3_feature_dim,
            mid_dim=v3_mid_dim,
            fine_dim=v3_fine_dim,
            detail_channels=detail_out_channels,
            output_size=high_res_size,
            refine_size=high_res_refine_size if high_res_refine else None,
            refine_channels=v3_refine_channels,
            use_centerline=v3_aux_centerline,
            use_boundary=v3_aux_boundary,
            use_width=v3_aux_width,
        )
    else:
        mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_adapter_skip=use_adapter,
            adapter_embed_dim=encoder_embed_dim,
            use_coarse_to_fine=use_coarse_to_fine,
            use_fusion_v2=use_fusion_v2,
            adapter_num_layers=adapter_num_layers,
            use_multi_depth=use_multi_depth,
            encoder_embed_dim=encoder_embed_dim,
            num_multidepth_levels=4,
            init_stage2_from_stage1=init_stage2_from_stage1,
            use_typed_heads=use_typed_heads,
            use_tcsu=use_tcsu,
            tcsu_scale=tcsu_scale,
            tcsu_mid_channels=tcsu_mid_channels,
            tcsu_max_length=tcsu_max_length,
            use_tadt=use_tadt,
            tadt_hidden=tadt_hidden,
            tadt_t_min=tadt_t_min,
            tadt_t_max=tadt_t_max,
            use_ranker=use_ranker,
            ranker_hidden=ranker_hidden,
            ranker_pair_hidden=ranker_pair_hidden,
            use_detail_stem=use_detail_stem,
            detail_channels=detail_out_channels,
            high_res_output=high_res_output,
            high_res_size=high_res_size,
            high_res_refine=high_res_refine,
            high_res_refine_size=high_res_refine_size,
            single_head=single_head,
            use_structural_aux=use_structural_aux,
            use_res_moe=use_res_moe,
            res_num_experts=res_num_experts,
            res_hidden=res_hidden,
            res_gate_hidden=res_gate_hidden,
            res_gate_size=res_gate_size,
            res_gate_mode=res_gate_mode,
            res_use_head_disagreement=res_use_head_disagreement,
            res_use_centerline=res_use_centerline,
            res_gate_input_norm=res_gate_input_norm,
            res_out_size=res_out_size,
            res_head_index=res_head_index,
            res_dropout=res_dropout,
            res_disagreement_tau=res_disagreement_tau,
            res_disagreement_scale=res_disagreement_scale,
        )

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_adapter=use_adapter,
            adapter_dim_ratio=adapter_dim_ratio,
            use_geo_i=use_geo_i,
            use_geo_e=use_geo_e,
            use_internal_adapter=use_internal_adapter,
            collect_adapter_features=(
                collect_adapter_features and not use_v3),
            geo_e_indices=geo_e_indices,
            use_dynamic_strip=use_dynamic_strip,
            use_detail_stem=use_detail_stem,
            detail_channels=detail_channels,
            detail_out_channels=detail_out_channels,
            use_convlora=use_convlora,
            convlora_rank=convlora_rank,
            convlora_alpha=convlora_alpha,
            convlora_experts=convlora_experts,
            use_evp=use_evp,
            evp_freq_nums=evp_freq_nums,
            evp_scale_factor=evp_scale_factor,
            evp_zero_init=evp_zero_init,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # Local, trusted SAM checkpoint (contains TorchVersion metadata).
            # PyTorch >= 2.6 defaults weights_only=True and rejects it.
            payload = torch.load(
                f, map_location='cpu', weights_only=False)
        # Accept both a raw state dict (SAM release) and a training
        # artifact ({'model_state_dict': ...}).  Without this unwrap a
        # warm-start from results/runs/... loads ZERO weights because every
        # top-level key ('model_state_dict', 'optimizer', ...) is unknown,
        # silently leaving the frozen encoder randomly initialised.
        if isinstance(payload, dict) and isinstance(
                payload.get('model_state_dict'), dict):
            state_dict = payload['model_state_dict']
            logging.info(
                'SAM checkpoint was a training artifact; unwrapped '
                'model_state_dict')
        elif isinstance(payload, dict) and isinstance(
                payload.get('state_dict'), dict):
            state_dict = payload['state_dict']
            logging.info(
                'SAM checkpoint was a training artifact; unwrapped state_dict')
        else:
            state_dict = payload
        state_dict = {
            (key[len('module.'):] if key.startswith('module.') else key): value
            for key, value in state_dict.items()
        }
        # ``strict=False`` does not ignore shape mismatches.  Single-head
        # models shrink mask_tokens / the IoU head from 4 slots to 1, and
        # v3 has no prompt decoder at all.  Slice a leading dimension when
        # possible, skip the rest, and report what was skipped.
        target = sam.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in target:
                continue  # unexpected for this architecture
            target_shape = target[key].shape
            if value.shape == target_shape:
                compatible[key] = value
            elif (value.ndim == len(target_shape)
                  and value.shape[1:] == tuple(target_shape[1:])
                  and value.shape[0] > target_shape[0]):
                compatible[key] = value[:target_shape[0]]
            else:
                skipped.append((key, tuple(value.shape),
                                tuple(target_shape)))
        missing, unexpected = sam.load_state_dict(compatible, strict=False)
        logging.info(
            "SAM checkpoint loaded: %d missing keys, %d unexpected keys, "
            "%d shape-skipped keys",
            len(missing), len(unexpected), len(skipped))
        if missing:
            logging.info("  missing (first 10): %s", missing[:10])
        if unexpected:
            logging.info("  unexpected (first 10): %s", unexpected[:10])
        if skipped:
            logging.info("  shape-skipped (first 10): %s", skipped[:10])

        # Stage-2 warm-start must happen AFTER the base checkpoint load.
        # A full Skein checkpoint already contains Stage-2 weights; never
        # overwrite those.  A base SAM checkpoint has no Stage-2 keys, so
        # copy the now-loaded Stage-1 heads into Stage-2.
        has_stage2 = any(
            key.startswith('mask_decoder.output_hypernetworks_mlps_stage2.')
            for key in state_dict)
        if (init_stage2_from_stage1 and not has_stage2
                and hasattr(sam.mask_decoder, 'copy_stage1_to_stage2')):
            sam.mask_decoder.copy_stage1_to_stage2()
            logging.info(
                "Initialized Stage-2 mask heads from the loaded Stage-1 heads")
    return sam
