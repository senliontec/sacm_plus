"""SegDINO v2: DINOv3 backbone + the official TPA/SAD decoder.

The official v2 repository loads DINOv3 through a local `torch.hub` checkout.
This port uses the same timm DINOv3-S/16 backbone already shipped for the
SegDINO v1 baseline and the official `dpt.py` decoder, so no hub checkout or
network access is needed at runtime.

Source: https://github.com/script-Yang/segdino_v2 (Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch

from ..segdino import find_dinov3_checkpoint
from .dpt import DPT

_SIZE = {'s': 'small', 'small': 'small', 'b': 'base', 'base': 'base',
         'l': 'large', 'large': 'large'}


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SystemExit(
                f'reading {path} needs the safetensors package: '
                'pip install safetensors') from exc
        return load_file(str(path))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict):
        return payload.get('model', payload.get('state_dict', payload))
    return payload


def build_segdino_v2(dino_size: str = 'small',
                     dino_ckpt: str | None = None,
                     decoder_dim: int = 256,
                     random_init: bool = False) -> DPT:
    """Build the official v2 decoder on a local DINOv3 backbone."""
    size = _SIZE.get(dino_size, dino_size)
    if size not in ('small', 'base', 'large'):
        raise ValueError(f'unknown DINOv3 size: {dino_size!r}')
    name = f'vit_{size}_patch16_dinov3'
    if random_init:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
    else:
        ckpt = Path(dino_ckpt) if dino_ckpt else find_dinov3_checkpoint(size)
        if ckpt is None or not Path(ckpt).is_file():
            raise SystemExit(
                'SegDINO v2 needs a local DINOv3 checkpoint; place it under '
                'checkpoints/dinov3/ or pass --segdino-v2-dino-ckpt. '
                'The server is offline.')
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
        state = _load_state(ckpt)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            raise SystemExit(
                f'{ckpt} did not provide {len(missing)} DINOv3 tensors '
                f'(first: {missing[0]}); check the selected size')
        print(f'loaded DINOv3 ({size}) weights: {len(unexpected)} unexpected')
    return DPT(
        encoder_size=size,
        nclass=1,
        decoder_channels=int(decoder_dim),
        patch_size=16,
        use_bn=False,
        backbone=backbone,
    )


__all__ = ['DPT', 'build_segdino_v2', 'find_dinov3_checkpoint']
