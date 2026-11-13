#!/usr/bin/env python3
"""Derive the inference-only fixed-slot-3 checkpoint from an SSH run.

The SSH base is trained with four role-specialized heads and a zero-initialised
linear fusion.  Deployment uses candidate 3 (the recall/Tversky expert)
directly:

    head_select     = fixed
    deployment_head = 3
    head_fusion     = 0

No weight is retrained.  The adapter config is changed and the
``head_fusion.*`` tensors are removed from the state dict so that the strict
checkpoint load in ``baseline_zoo`` succeeds.  Everything else (model weights,
split hash, best epoch, history) is preserved.

Usage:
    python scripts/pipeline/make_fixed3_checkpoint.py \
        --input results/runs/.../seed_0/best_model.pth \
        --output results/runs/.../seed_0/best_model_fixed3_nofusion.pth

For the strict native-control probe, pass ``--allow-native`` to accept a
checkpoint with ``specialized_heads=0`` and derive the same fixed candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

FIXED3_VERSION = 1
MAX_HEADS = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f'{path}: cannot load checkpoint: {exc}')
    if not isinstance(payload, dict):
        raise SystemExit(f'{path}: checkpoint must be a mapping')
    return payload


def _adapter_config(payload: dict) -> dict:
    config = payload.get('config')
    if not isinstance(config, dict):
        raise SystemExit('checkpoint has no config mapping')
    adapter = config.get('adapter_config')
    if not isinstance(adapter, dict):
        raise SystemExit('checkpoint config has no adapter_config mapping')
    return adapter


def _verify_source(payload: dict, path: Path,
                   allow_native: bool = False) -> dict:
    adapter = _adapter_config(payload)
    stored_adapter = payload.get('adapter')
    if stored_adapter and str(stored_adapter) != 'sacm_gcd':
        raise SystemExit(
            f'{path}: expected adapter=sacm_gcd, got {stored_adapter!r}')
    method = payload.get('config', {}).get('method')
    if method and str(method) != 'sacm_gcd':
        raise SystemExit(
            f'{path}: expected method=sacm_gcd, got {method!r}')
    specialized = int(adapter.get('specialized_heads', 0) or 0)
    if specialized != 1:
        if not (allow_native and specialized == 0):
            raise SystemExit(
                f'{path}: fixed3 requires an SSH checkpoint with '
                f'specialized_heads=1; got {specialized} '
                f'(use --allow-native for a native control checkpoint)')
    state = payload.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        raise SystemExit(f'{path}: checkpoint has no model_state_dict')
    return adapter


def _has_fusion_keys(payload: dict) -> bool:
    state = payload.get('model_state_dict') or {}
    return any(str(key).startswith('head_fusion.') for key in state)


def _already_derived(path: Path, source_sha: str, head: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_checkpoint(path)
        adapter = _adapter_config(payload)
    except SystemExit:
        return False
    except Exception:
        # A truncated/corrupt output must never satisfy the skip predicate.
        return False
    state = payload.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        return False
    return (
        payload.get('fixed3_source_sha256') == source_sha
        and int(payload.get('fixed3_deployment_head', -1)) == int(head)
        and str(adapter.get('head_select', '')) == 'fixed'
        and int(adapter.get('head_fusion', 1)) == 0
        and int(adapter.get('deployment_head', -1)) == int(head)
        and not _has_fusion_keys(payload)
    )


def _write_atomic(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f'.{output.name}.', suffix='.tmp', dir=str(output.parent))
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_fixed3(args: argparse.Namespace) -> dict:
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise SystemExit(f'input checkpoint does not exist: {source}')
    if source.resolve() == output.resolve():
        raise SystemExit('input and output must be different files')
    if output.exists() and output.is_dir():
        raise SystemExit(f'output is a directory: {output}')
    if not 0 <= int(args.deployment_head) < MAX_HEADS:
        raise SystemExit(
            f'deployment-head must be in [0, {MAX_HEADS - 1}]; '
            f'got {args.deployment_head}')

    source_sha = file_sha256(source)
    if not args.force and _already_derived(
            output, source_sha, int(args.deployment_head)):
        return {
            'version': FIXED3_VERSION,
            'skipped': True,
            'reason': 'output already derived from the same source',
            'input': str(source.resolve()),
            'output': str(output.resolve()),
            'source_sha256': source_sha,
            'deployment_head': int(args.deployment_head),
            'head_select': 'fixed',
            'head_fusion': 0,
        }

    payload = _load_checkpoint(source)
    adapter = _verify_source(
        payload, source, allow_native=bool(getattr(args, 'allow_native', False)))
    state = payload.get('model_state_dict')
    fusion_keys = sorted(
        str(key) for key in state if str(key).startswith('head_fusion.'))
    for key in fusion_keys:
        del state[key]

    adapter['head_select'] = 'fixed'
    adapter['deployment_head'] = int(args.deployment_head)
    adapter['head_fusion'] = 0

    payload['fixed3_version'] = FIXED3_VERSION
    payload['fixed3_source_checkpoint'] = str(source.resolve())
    payload['fixed3_source_sha256'] = source_sha
    payload['fixed3_created_at'] = datetime.now(
        timezone.utc).isoformat()
    payload['fixed3_head_select'] = 'fixed'
    payload['fixed3_deployment_head'] = int(args.deployment_head)
    payload['fixed3_head_fusion'] = 0
    payload['fixed3_source_best_epoch'] = payload.get('best_epoch')
    payload['fixed3_source_val_dice'] = payload.get('val_dice')

    _write_atomic(payload, output)
    return {
        'version': FIXED3_VERSION,
        'skipped': False,
        'input': str(source.resolve()),
        'output': str(output.resolve()),
        'source_sha256': source_sha,
        'removed_fusion_keys': len(fusion_keys),
        'deployment_head': int(args.deployment_head),
        'head_select': 'fixed',
        'head_fusion': 0,
        'best_epoch': payload.get('best_epoch'),
        'val_dice': payload.get('val_dice'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='SSH best_model.pth')
    parser.add_argument('--output', required=True,
                        help='derived fixed3 checkpoint to write')
    parser.add_argument('--deployment-head', type=int, default=3,
                        help='candidate to deploy (default: 3)')
    parser.add_argument('--allow-native', action='store_true',
                        help='allow specialized_heads=0 native control '
                             'checkpoint (for strict control experiments)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an already derived output')
    args = parser.parse_args(argv)
    summary = build_fixed3(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
