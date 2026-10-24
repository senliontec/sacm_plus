"""SegDINO: DINOv3 backbone + the official DPT decoder."""

from __future__ import annotations

import torch
import timm

from .dpt import DPT

_SIZE = {'s': 'small', 'small': 'small', 'b': 'base', 'base': 'base',
         'l': 'large', 'large': 'large'}


def build_segdino(dino_size: str = 'small', dino_ckpt: str | None = None,
                  random_init: bool = False):
    size = _SIZE.get(dino_size, dino_size)
    name = f'vit_{size}_patch16_dinov3'
    if random_init:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
    elif dino_ckpt:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
        payload = torch.load(dino_ckpt, map_location='cpu',
                             weights_only=False)
        state = payload.get('model', payload.get('state_dict', payload))
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f'loaded DINOv3 weights: {len(missing)} missing, '
              f'{len(unexpected)} unexpected')
        if missing:
            raise SystemExit(
                f'--dino-ckpt {dino_ckpt} did not provide '
                f'{len(missing)} backbone tensors (first: {missing[0]}); '
                'check that the checkpoint matches the selected --dino-size')
    else:
        try:
            backbone = timm.create_model(name, pretrained=True,
                                         num_classes=0)
        except Exception as exc:
            raise SystemExit(
                'DINOv3 weights are gated on Hugging Face; set HF_TOKEN or '
                'pass --dino-ckpt <checkpoint>, or use --random-init for a '
                f'code smoke only. Original error: {exc}') from exc
    # DPT selects different intermediate layers for small/base/large;
    # hard-coding 'base' silently used the wrong layers for ViT-L.
    return DPT(encoder_size=size, nclass=1, backbone=backbone)


__all__ = ['DPT', 'build_segdino']
