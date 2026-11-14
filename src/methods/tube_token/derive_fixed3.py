"""Derive the inference-only fixed candidate-3 checkpoint for TubeToken.

The baseline helper only accepts ``adapter=dcd``.  This method-local helper
performs the same configuration rewrite and removes the unused fusion keys,
while preserving every structure-control tensor.
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

MAX_HEADS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def derive(input_path: Path, output_path: Path, head: int = 3,
           force: bool = False) -> dict:
    source = Path(input_path)
    output = Path(output_path)
    if not source.is_file():
        raise SystemExit(f'input checkpoint does not exist: {source}')
    if source.resolve() == output.resolve():
        raise SystemExit('input and output must be different files')
    if not 0 <= int(head) < MAX_HEADS:
        raise SystemExit(f'deployment head must be in [0, {MAX_HEADS})')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit('checkpoint must be a mapping')
    if str(payload.get('adapter')) != 'tube_token':
        raise SystemExit(
            f'expected adapter=tube_token, got {payload.get("adapter")!r}')
    config = payload.get('config')
    if not isinstance(config, dict):
        raise SystemExit('checkpoint has no config mapping')
    adapter_config = config.get('adapter_config')
    if not isinstance(adapter_config, dict):
        raise SystemExit('checkpoint has no adapter_config mapping')
    state = payload.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        raise SystemExit('checkpoint has no model_state_dict')

    source_sha = _sha256(source)
    if output.is_file() and not force:
        old = torch.load(output, map_location='cpu', weights_only=False)
        old_adapter = (old.get('config') or {}).get('adapter_config', {})
        if (old.get('fixed3_source_sha256') == source_sha
                and str(old_adapter.get('head_select')) == 'fixed'
                and int(old_adapter.get('deployment_head', -1)) == int(head)
                and int(old_adapter.get('head_fusion', 1)) == 0
                and not any(str(k).startswith('head_fusion.')
                            for k in old.get('model_state_dict', {}))):
            return {'skipped': True, 'output': str(output.resolve())}

    removed = sorted(
        str(key) for key in list(state)
        if str(key).startswith('head_fusion.'))
    for key in removed:
        del state[key]
    adapter_config['head_select'] = 'fixed'
    adapter_config['deployment_head'] = int(head)
    adapter_config['head_fusion'] = 0
    payload['fixed3_version'] = 1
    payload['fixed3_source_checkpoint'] = str(source.resolve())
    payload['fixed3_source_sha256'] = source_sha
    payload['fixed3_created_at'] = datetime.now(timezone.utc).isoformat()
    payload['fixed3_head_select'] = 'fixed'
    payload['fixed3_deployment_head'] = int(head)
    payload['fixed3_head_fusion'] = 0
    payload['fixed3_source_best_epoch'] = payload.get('best_epoch')
    payload['fixed3_source_val_dice'] = payload.get('val_dice')

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
    return {
        'skipped': False,
        'output': str(output.resolve()),
        'removed_fusion_keys': len(removed),
        'deployment_head': int(head),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-head', type=int, default=3)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(
        derive(args.input, args.output, args.deployment_head, args.force),
        indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
