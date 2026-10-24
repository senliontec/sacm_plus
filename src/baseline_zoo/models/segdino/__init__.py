"""SegDINO: DINOv3 backbone + the official DPT decoder."""

from __future__ import annotations

from pathlib import Path

import timm
import torch

from .dpt import DPT

_SIZE = {'s': 'small', 'small': 'small', 'b': 'base', 'base': 'base',
         'l': 'large', 'large': 'large'}

_SIZE_TAGS = {
    'small': ('small', 's16', 'vits16'),
    'base': ('base', 'b16', 'vitb16'),
    'large': ('large', 'l16', 'vitl16'),
}
_TORCH_SUFFIXES = {'.bin', '.pth', '.pt', '.ckpt'}
_SAFE_SUFFIXES = {'.safetensors'}
_GENERIC_NAMES = {
    'pytorch_model.bin', 'model.safetensors', 'model.pth', 'model.pt',
    'dinov3.bin', 'dinov3.pth', 'dinov3.pt',
}
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _checkpoint_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TORCH_SUFFIXES | _SAFE_SUFFIXES:
            continue
        # checkpoints/ also holds SAM/ResNet assets; outside the dedicated
        # dinov3/ folder only accept DINO-named files.
        if (path.parent.name.lower() != 'dinov3'
                and 'dino' not in path.name.lower()):
            continue
        files.append(path)
    return sorted(files)


def find_dinov3_checkpoint(dino_size: str = 'small',
                           root: str | Path | None = None) -> Path | None:
    """Return a local DINOv3 checkpoint matching ``dino_size``.

    ``root`` defaults to the repository's ``checkpoints/`` directory.  The
    search is recursive, so ``checkpoints/dinov3/pytorch_model.bin`` works
    out of the box, as do size-named files.  Ambiguous matches raise
    ``ValueError`` instead of silently training with the wrong backbone.
    """
    size = _SIZE.get(dino_size, dino_size)
    if size not in _SIZE_TAGS:
        raise ValueError(f'unknown DINOv3 size: {dino_size!r}')
    root = Path(root) if root is not None else _REPO_ROOT / 'checkpoints'
    files = _checkpoint_files(root)
    if not files:
        return None

    target_tags = _SIZE_TAGS[size]
    other_tags = tuple(
        tag for key, tags in _SIZE_TAGS.items() if key != size for tag in tags)
    scored: list[tuple[int, Path]] = []
    for path in files:
        name = path.name.lower()
        if (any(tag in name for tag in other_tags)
                and not any(tag in name for tag in target_tags)):
            continue
        if any(tag in name for tag in target_tags):
            score = 0
        elif name in _GENERIC_NAMES:
            score = 1
        else:
            score = 2
        scored.append((score, path))
    if not scored:
        return None

    best_score = min(score for score, _ in scored)
    candidates = [path for score, path in scored if score == best_score]
    if len(candidates) > 1:
        torch_candidates = [
            path for path in candidates
            if path.suffix.lower() in _TORCH_SUFFIXES]
        if len(torch_candidates) == 1:
            candidates = torch_candidates
        else:
            names = ', '.join(str(path) for path in candidates)
            raise ValueError(
                f'multiple DINOv3 checkpoints match size={size}: {names}; '
                'set DINO_CKPT or pass --dino-ckpt')
    return candidates[0]


def _load_dino_state(path: str | Path) -> dict[str, torch.Tensor]:
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


def build_segdino(dino_size: str = 'small', dino_ckpt: str | None = None,
                  random_init: bool = False):
    size = _SIZE.get(dino_size, dino_size)
    name = f'vit_{size}_patch16_dinov3'
    if random_init:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
    elif dino_ckpt:
        backbone = timm.create_model(name, pretrained=False, num_classes=0)
        state = _load_dino_state(dino_ckpt)
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


__all__ = ['DPT', 'build_segdino', 'find_dinov3_checkpoint']
