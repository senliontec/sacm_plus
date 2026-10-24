# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import partial
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from torch.nn import functional as F

def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter = args.encoder_adapter,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter = args.encoder_adapter,
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter = args.encoder_adapter,

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
    image_size,
    checkpoint,
    encoder_adapter,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
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
            use_rel_pos = True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train = encoder_adapter,
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
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    # sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        try:
            if 'model' in state_dict.keys():
                print(encoder_adapter)
                sam.load_state_dict(state_dict['model'], False)
            else:
                if image_size==1024 and encoder_adapter==True:
                    sam.load_state_dict(state_dict, False)
                else:
                    sam.load_state_dict(state_dict)
        except:
            print('*******interpolate')
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)   
            sam.load_state_dict(new_state_dict)
        print(f"*******load {checkpoint}")
        
    return sam


def load_from(sam, state_dicts, image_size, vit_patch_size):
    """Load a 1024-input SAM checkpoint into this (possibly 256) model.

    Official SAM-Med2D excludes three decoder groups
    (``mask_tokens``, ``output_hypernetworks_mlps``,
    ``iou_prediction_head``).  Those tensors are shape-compatible between
    vanilla SAM and this 256-input model, so excluding them would silently
    random-initialize part of the SAM decoder.  For the controlled
    "fine-tune from SAM" protocol we inherit every compatible tensor and
    only interpolate the image-size-dependent positional embeddings.
    """
    sam_dict = sam.state_dict()
    new_state_dict = {}
    for key, value in state_dicts.items():
        if key not in sam_dict:
            continue
        if sam_dict[key].shape == value.shape:
            new_state_dict[key] = value
        elif key == 'image_encoder.pos_embed' or 'rel_pos' in key:
            # Interpolated below when the spatial size differs.
            new_state_dict[key] = value

    pos_embed = new_state_dict.get('image_encoder.pos_embed')
    token_size = int(image_size // vit_patch_size)
    if pos_embed is not None and pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I
        # have no better idea (official comment)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(
            pos_embed, (token_size, token_size), mode='bilinear',
            align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed

    for key, value in list(new_state_dict.items()):
        if 'rel_pos' not in key:
            continue
        target = sam_dict[key]
        if value.shape == target.shape:
            continue
        # Official rel_pos tensors are [H, W]; interpolate to the target
        # spatial size without assuming a fixed global-attention index.
        rel_pos = value.unsqueeze(0).unsqueeze(0)
        rel_pos = F.interpolate(
            rel_pos, tuple(target.shape), mode='bilinear',
            align_corners=False)
        new_state_dict[key] = rel_pos[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict

