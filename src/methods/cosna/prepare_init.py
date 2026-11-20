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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--epochs-add', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    dest = args.workdir / 'best_model.pth'
    if dest.exists() and not args.force:
        raise SystemExit(f'{dest} exists; pass --force to overwrite')
    payload = torch.load(args.source, map_location='cpu',
                         weights_only=False)
    manifest = build_data_manifest(
        args.data_root.resolve(), hash_files=True)
    fingerprint = data_manifest_fingerprint(manifest)
    payload['data_manifest_sha256'] = fingerprint
    payload['adapter'] = 'cosna'
    config = payload.get('config') or {}
    config['data_manifest_sha256'] = fingerprint
    payload['config'] = config
    args.workdir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)
    epoch = int(payload.get('epoch', 0))
    meta = {
        'source': str(args.source),
        'destination': str(dest),
        'source_epoch': epoch,
        'epochs': epoch + int(args.epochs_add),
        'data_manifest_sha256': fingerprint,
    }
    (args.workdir / 'cosna_init.json').write_text(
        json.dumps(meta, indent=1), encoding='utf-8')
    print(meta['epochs'])


if __name__ == '__main__':
    main()
