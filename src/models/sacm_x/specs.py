"""SAM-family ModelSpecs (sam_l / sam_b / sam_h).

The build functions wrap segment_anything.build_sam_vit_* with the
architecture switches; the freeze protocol, parameter grouping and the
forward/prompt helpers are the single implementations previously
inlined in train_sam.py (moved here verbatim — behaviour unchanged).
"""

import logging

from models.sam_network import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from core.io import get_prompt_embeddings, predict_all
from core.registry import ModelSpec, register_model

# Decoder modules added by the new method (own learning-rate group).
NEW_DECODER_PREFIXES = (
    'mask_decoder.adapter_',
    'mask_decoder.coarse_refiner',
    'mask_decoder.multi_depth_path',
    'mask_decoder.fuse_two',
    # Stage-2 heads are newly added, not part of the pretrained decoder.
    'mask_decoder.output_hypernetworks_mlps_stage2',
    # Typed-head identity embeddings (full_typed only).
    'mask_decoder.type_embeddings',
    # PEPA adapters (full_pepa only).
    'mask_decoder.tcsu',
    'mask_decoder.tadt',
    'mask_decoder.ranker',
    'mask_decoder.detail_fuse_',
    'mask_decoder.final_refine',
    'mask_decoder.v3_',
    'mask_decoder.structural_aux',
)

# Architecture switches shared by the whole SAM family (defaults = the
# full model; presets in configs.py override them).
SAM_DEFAULTS = dict(
    use_adapter=True,
    adapter_dim_ratio=0.1,
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    use_dynamic_strip=False,
    # Default OFF: the typed-head prototype is opt-in through the
    # `full_typed` preset; the stable SACM-X path is unchanged.
    use_typed_heads=False,
    # PEPA-style post-encoder adapters (TCSU/TADT).  Opt-in through the
    # `full_pepa` preset; default behaviour is unchanged.
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
    # V3 dense structural decoder (opt-in via full_v3).
    use_v3=False,
    # Training-only structural heads on the prompt decoder.
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
    # Conv-LoRA qkv branch + EVP high-frequency prompts (opt-in, used as an
    # alternative to the SACM adapters).
    use_convlora=False,
    convlora_rank=4,
    convlora_alpha=32.0,
    convlora_experts=8,
    use_evp=False,
    evp_freq_nums=0.25,
    evp_scale_factor=32,
    evp_zero_init=True,
    # Default ON for SACM-X: the final masks come from the new Stage-2 heads,
    # so initialize them from SAM's pretrained Stage-1 heads instead of random.
    init_stage2_from_stage1=True,
)


def _build(checkpoint, builder, **overrides):
    cfg = {**SAM_DEFAULTS, **overrides}
    return builder(checkpoint=checkpoint, **cfg)


def freeze_sam(sam):
    """SAM freezing protocol (verbatim from train_sam.py)."""
    # Freeze image encoder parameters except for adapters
    for name, param in sam.image_encoder.named_parameters():
        # Train adapters, Conv-LoRA/EVP branches and the optional
        # high-resolution detail stem; leave the pretrained ViT frozen.
        if ('adapter' not in name and 'detail_stem' not in name
                and 'lora' not in name and 'evp' not in name):
            param.requires_grad = False

    # External adapters/norms on global-attention layers (and on layers
    # excluded by geo_e_layers) are never called by the encoder forward.
    # They are pure optimizer/checkpoint dead weight; keep them frozen.
    encoder = sam.image_encoder
    if (getattr(encoder, 'use_geo_e', False)
            and getattr(encoder, 'adapter_norms', None) is not None):
        used = getattr(encoder, 'geo_e_indices', None)
        for i in range(encoder.depth):
            skipped = (i in encoder.global_attn_indexes
                       or (used is not None and i not in used))
            if not skipped:
                continue
            for param in encoder.external_adapters[i].parameters():
                param.requires_grad = False
            for param in encoder.adapter_norms[i].parameters():
                param.requires_grad = False

    # Freeze prompt encoder parameters (no_mask_embed stays as pretrained anchor)
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    # Enable mask decoder parameters - explicitly set to trainable
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True


def sam_param_groups(sam, args):
    """Differential learning-rate groups (verbatim logic from
    train_sam.py): adapters 3e-4, new decoder modules 1e-4, original
    decoder 1e-5."""
    adapter_params = []
    new_decoder_params = []
    decoder_params = []
    for name, param in sam.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('image_encoder'):
            adapter_params.append(param)
        elif name.startswith(NEW_DECODER_PREFIXES):
            new_decoder_params.append(param)
        elif name.startswith('mask_decoder'):
            decoder_params.append(param)
        else:
            # Any other trainable parameter (none expected): treat as adapter
            adapter_params.append(param)

    logging.info(
        f"Parameter groups - adapters: {len(adapter_params)}, "
        f"new decoder modules: {len(new_decoder_params)}, original decoder: {len(decoder_params)}"
    )

    groups = [
        {'params': adapter_params, 'lr': args.adapter_lr, 'weight_decay': args.weight_decay},
        {'params': new_decoder_params, 'lr': args.new_module_lr, 'weight_decay': args.weight_decay},
        {'params': decoder_params, 'lr': args.decoder_lr, 'weight_decay': args.weight_decay},
    ]
    return [g for g in groups if g['params']]


def sam_preprocess(sam, images):
    """Apply SAM normalization on the scale it was pretrained with.

    Datasets use ``ToTensor`` -> [0, 1], while SAM's pixel_mean/pixel_std
    are defined for [0, 255].  Feeding [0, 1] directly subtracts a mean of
    ~123.7 and compresses the dynamic range by ~14x, which breaks the frozen
    encoder.  Scale raw images to [0, 255] first; already-normalized inputs
    (negative values) are left untouched.
    """
    lo = images.amin()
    hi = images.amax()
    if bool(lo >= 0.0) and bool(hi <= 1.0 + 1e-6):
        images = images * 255.0
    return sam.preprocess(images)


def _make_spec(name, builder, description):
    return ModelSpec(
        name=name,
        algorithm='sacm',
        build=lambda checkpoint, **kw: _build(checkpoint, builder, **kw),
        freeze=freeze_sam,
        param_groups=sam_param_groups,
        preprocess=sam_preprocess,
        prompt_builder=get_prompt_embeddings,
        forward=predict_all,
        input_size=1024,
        description=description,
    )


register_model('sam_l')(_make_spec(
    'sam_l', build_sam_vit_l, 'SAM ViT-L with the full SACM-X architecture (default)'))
register_model('sam_b')(_make_spec(
    'sam_b', build_sam_vit_b, 'SAM ViT-B with the same switches (faster, for debugging)'))
register_model('sam_h')(_make_spec(
    'sam_h', build_sam_vit_h, 'SAM ViT-H with the same switches'))
