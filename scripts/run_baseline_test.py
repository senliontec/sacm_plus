#!/usr/bin/env python3
"""One-click prediction + paper statistics + visualization for a baseline.

Defaults to the U-Net adapter and all 20 test datasets in the frozen protocol:

    python scripts/run_baseline_test.py

The script assumes the checkpoint already exists.  It derives ``NAME:DIR``
test specs from protocol_manifest.csv, so adding a dataset to the protocol
does not require editing this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_specs(protocol: Path, dataset_filter):
    roots = {}
    with open(protocol) as f:
        for row in csv.DictReader(f):
            if not row['split'].endswith('_test'):
                continue
            if dataset_filter and row['dataset'] not in dataset_filter:
                continue
            mask = Path(row['mask_path'])
            if not mask.is_absolute():
                mask = REPO_ROOT / mask
            root = mask.parent.parent
            previous = roots.setdefault(row['dataset'], root)
            if previous != root:
                raise SystemExit(
                    f'{row["dataset"]}: multiple test roots: {previous}, {root}')
    if not roots:
        raise SystemExit(f'no test rows in {protocol}')
    return [(name, roots[name]) for name in sorted(roots)]


def run(cmd):
    print('+', ' '.join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_split_hash(path: Path):
    try:
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=False)
        return payload.get('split_manifest_sha256')
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='unet')
    parser.add_argument('--protocol', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot' /
                        'protocol_manifest.csv')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPO_ROOT / 'workdirs' / 'baselines' /
                        'unet' / 'best_model.pth')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions')
    parser.add_argument('--paper-root', type=Path,
                        default=REPO_ROOT / 'results' / 'paper')
    parser.add_argument('--vis-root', type=Path,
                        default=REPO_ROOT / 'results' / 'visualizations')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--metrics', default='dice,iou,cldice,hd95')
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-predict', action='store_true')
    parser.add_argument('--skip-paper', action='store_true')
    parser.add_argument('--skip-visualize', action='store_true')
    parser.add_argument('--allow-stale-checkpoint', action='store_true',
                        help='allow a checkpoint older than the current split')
    args = parser.parse_args()

    specs = read_specs(args.protocol, set(args.datasets) if args.datasets else None)
    print(f'{args.method}: {len(specs)} test datasets')
    print('  ' + ' '.join(name for name, _ in specs))

    method_pred_root = args.pred_root / args.method
    if args.skip_predict:
        meta_path = method_pred_root / '.prediction_meta.json'
        if not meta_path.is_file():
            print('[warn] --skip-predict: no prediction metadata; '
                  'checkpoint provenance is not re-validated')
        else:
            meta = json.loads(meta_path.read_text())
            split = REPO_ROOT / 'data' / 'sacm_3shot' / 'split_manifest.csv'
            default_protocol = (REPO_ROOT / 'data' / 'sacm_3shot' /
                                'protocol_manifest.csv').resolve()
            custom_protocol = args.protocol.resolve() != default_protocol
            expected_split = (None if custom_protocol or not split.is_file()
                              else file_sha256(split))
            expected_ckpt = (file_sha256(args.checkpoint)
                             if args.checkpoint.is_file() else None)
            stale = (
                (expected_split is not None and
                 meta.get('split_manifest_sha256') != expected_split) or
                (expected_ckpt is not None and
                 meta.get('checkpoint_sha256') != expected_ckpt))
            if stale and not args.allow_stale_checkpoint:
                raise SystemExit(
                    'existing predictions do not match the current '
                    'checkpoint/split; regenerate them or pass '
                    '--allow-stale-checkpoint')
            if stale:
                print('[warn] --skip-predict: using stale predictions '
                      '(--allow-stale-checkpoint)')
    if not args.skip_predict:
        if args.method == 'unet':
            if not args.checkpoint.is_file():
                raise SystemExit(
                    f'missing checkpoint: {args.checkpoint}\n'
                    'train first: python baselines/adapters/unet.py '
                    '--stage train --epochs 200')
            split = REPO_ROOT / 'data' / 'sacm_3shot' / 'split_manifest.csv'
            if split.is_file() and not args.allow_stale_checkpoint:
                expected_hash = file_sha256(split)
                actual_hash = checkpoint_split_hash(args.checkpoint)
                if actual_hash is None:
                    raise SystemExit(
                        'checkpoint has no split_manifest_sha256 metadata; '
                        'retrain first or pass --allow-stale-checkpoint')
                if actual_hash != expected_hash:
                    raise SystemExit(
                        'checkpoint was trained on a different split; '
                        'retrain first or pass --allow-stale-checkpoint')
            method_pred_root = args.pred_root / args.method
            cmd = [sys.executable,
                   REPO_ROOT / 'baselines' / 'adapters' / 'unet.py',
                   '--stage', 'predict',
                   '--checkpoint', args.checkpoint,
                   '--pred-root', method_pred_root]
            if args.device:
                cmd += ['--device', args.device]
            cmd += ['--test-datasets']
            cmd += [f'{name}:{root}' for name, root in specs]
            run(cmd)
        else:
            print(f'[warn] no prediction adapter for {args.method}; '
                  'assuming predictions already exist')

    if not args.skip_paper:
        cmd = [sys.executable,
               REPO_ROOT / 'scripts' / 'collect_paper_metrics.py',
               '--methods', args.method,
               '--pred-root', args.pred_root,
               '--out-root', args.paper_root,
               '--metrics', args.metrics,
               '--protocol', args.protocol]
        if args.datasets:
            cmd += ['--datasets'] + list(args.datasets)
        run(cmd)

    if not args.skip_visualize:
        run([sys.executable,
             REPO_ROOT / 'scripts' / 'visualize_predictions.py',
             '--methods', args.method,
             '--pred-root', args.pred_root,
             '--out-root', args.vis_root,
             '--protocol', args.protocol,
             '--num-samples', str(args.num_samples)] +
            (['--datasets'] + list(args.datasets) if args.datasets else []))

    print('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
