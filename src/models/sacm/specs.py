"""SAM-family ModelSpecs (sam_l / sam_b / sam_h).

The build functions wrap segment_anything.build_sam_vit_* with the
architecture switches; the freeze protocol, parameter grouping and the
forward/prompt helpers are the single implementations previously
inlined in train_sam.py (moved here verbatim — behaviour unchanged).
"""

import logging

from models.sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from core.io import get_prompt_embeddings, predict_all
from core.registry import ModelSpec, register_model

# Decoder modules added by the new method (own learning-rate group).
NEW_DECODER_PREFIXES = (
    'mask_decoder.adapter_',
    'mask_decoder.coarse_refiner',
    'mask_decoder.multi_depth_path',
    'mask_decoder.fuse_two',
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
)


def _build(checkpoint, builder, **overrides):
    cfg = {**SAM_DEFAULTS, **overrides}
    return builder(checkpoint=checkpoint, **cfg)


def freeze_sam(sam):
    """SAM freezing protocol (verbatim from train_sam.py)."""
    # Freeze image encoder parameters except for adapters
    for name, param in sam.image_encoder.named_parameters():
        # Only train adapters and leave everything else frozen
        if 'adapter' not in name:
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
    'sam_l', build_sam_vit_l, 'SAM ViT-L with the full SACM-v2 architecture (default)'))
register_model('sam_b')(_make_spec(
    'sam_b', build_sam_vit_b, 'SAM ViT-B with the same switches (faster, for debugging)'))
register_model('sam_h')(_make_spec(
    'sam_h', build_sam_vit_h, 'SAM ViT-H with the same switches'))
