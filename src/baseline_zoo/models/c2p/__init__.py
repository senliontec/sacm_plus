"""Self-contained Concept-to-Pixel model package."""

from __future__ import annotations

from pathlib import Path

import torch

from .network import C2P

# The released checkpoint was trained with auxiliary deep-supervision heads
# and an inverse projection that are not part of the inference network in
# the upstream release.  They are intentionally ignored at load time.
TRAINING_ONLY_PREFIXES = ('ds_head4.', 'ds_head5.', 'd2_proj_inv.')


def build_c2p(checkpoint: str | Path | None = None,
              random_init: bool = False) -> C2P:
    """Build C2P without network access.

    The official C2P checkpoint contains the ConvNeXtV2-B backbone, so the
    timm backbone is always constructed with ``pretrained=False`` and the
    checkpoint supplies every weight.
    """
    model = C2P(model='base', backbone_pretrained=False)
    if random_init:
        return model
    if checkpoint is None:
        raise FileNotFoundError('missing C2P checkpoint path')
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'missing C2P checkpoint: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict):
        for key in ('model', 'state_dict', 'model_state_dict', 'generator'):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    state = {key.replace('module.', ''): value
             for key, value in payload.items()}
    state = {
        key: value for key, value in state.items()
        if not key.startswith(TRAINING_ONLY_PREFIXES)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'{path}: C2P state dict does not match the architecture '
            f'({len(missing)} missing, {len(unexpected)} unexpected); '
            f'first missing: {missing[:5]}; first unexpected: '
            f'{unexpected[:5]}')
    return model


__all__ = ['C2P', 'build_c2p']
