"""Prepare a CoSNA workdir from a trained BCTD checkpoint.

The copied checkpoint is written as ``<workdir>/best_model.pth`` and its
data-manifest hash is recomputed for the local absolute data root, so the
resume guard accepts it without touching the original checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from core.run_artifacts import build_data_manifest, data_manifest_fingerprint


def prepare(source: Path, workdir: Path, data_root: Path,
            epochs_add: int = 20, force: bool = False) -> int:
    """Copy ``source`` into ``workdir/best_model.pth`` and return the final epoch."""
    dest = workdir / 'best_model.pth'
    if dest.exists() and not force:
        raise SystemExit(f'{dest} exists; pass --force to overwrite')
    payload = torch.load(source, map_location='cpu',
                         weights_only=False)
    data_root = Path(data_root).resolve()
    protocol = data_root / 'protocol_manifest.csv'
    manifest = build_data_manifest(
        data_root,
        protocol=protocol if protocol.is_file() else None,
        hash_files=True)
    fingerprint = data_manifest_fingerprint(manifest)
    payload['data_manifest_sha256'] = fingerprint
    payload['adapter'] = 'cosna'
    config = payload.get('config') or {}
    config['data_manifest_sha256'] = fingerprint
    payload['config'] = config
    workdir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)
    epoch = int(payload.get('epoch', 0))
    meta = {
        'source': str(source),
        'destination': str(dest),
        'source_epoch': epoch,
        'epochs': epoch + int(epochs_add),
        'data_manifest_sha256': fingerprint,
    }
    (workdir / 'cosna_init.json').write_text(
        json.dumps(meta, indent=1), encoding='utf-8')
    return int(meta['epochs'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--epochs-add', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    print(prepare(args.source, args.workdir, args.data_root,
                  args.epochs_add, args.force))


if __name__ == '__main__':
    main()
