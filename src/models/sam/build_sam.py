# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified: builder arguments for the geometric adapters, the closed-loop
# coarse-to-fine decoder, Fusion v2 and the multi-depth path.

import torch

from functools import partial

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam
from .transformer import TwoWayTransformer


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
            geo_e_indices=geo_e_indices,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
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
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)
    return sam
