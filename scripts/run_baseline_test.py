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
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.run_status import RunStatus  # noqa: E402
from core.wandb_utils import (  # noqa: E402
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

# Self-contained adapters.  A method without an entry is assumed to have
# predictions already on disk (reference-only or externally produced).
from baseline_zoo.adapters.registry import available_adapters  # noqa: E402

PORTED = set(available_adapters())
# nnWNet stores a model folder / KnowSAM a plain SGDL state dict; neither
# carries the unified split-hash field.
CUSTOM_CHECKPOINTS = {'nnwnet', 'knowsam'}
SAM_CHECKPOINT_METHODS = {
    'cwsam', 'sacm_official', 'sam_med2d', 'sam_octa', 'sam_vit_l_unet'}


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


def read_test_counts(protocol: Path, dataset_filter):
    counts = Counter()
    with open(protocol) as f:
        for row in csv.DictReader(f):
            if not row['split'].endswith('_test'):
                continue
            if dataset_filter and row['dataset'] not in dataset_filter:
                continue
            counts[row['dataset']] += 1
    return counts


def run(cmd, env=None):
    print('+', ' '.join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True,
                   env=env)


def repo_env():
    """Environment for child ``python -m baseline_zoo`` processes."""
    env = os.environ.copy()
    src = str(REPO_ROOT / 'src')
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = src + (os.pathsep + existing if existing else '')
    return env


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


def _as_bool(value):
    return str(value).strip().lower() not in ('0', 'false', 'no', 'off',
                                              'disabled', 'none')


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def log_paper_metrics(args, paper_root: Path):
    """Log the 20-dataset per-dataset and macro metrics to W&B.

    Non-fatal: a missing CSV, missing package, or unreachable server must not
    turn a successful evaluation into a failed run.
    """
    metrics_path = paper_root / 'paper_metrics.csv'
    macro_path = paper_root / 'paper_macro.csv'
    if not metrics_path.is_file():
        print(f'[warn] W&B: missing {metrics_path}; skip paper-metric logging')
        return

    run = setup_wandb(
        args,
        name=f'paper_metrics-{args.method}',
        config={
            'method': args.method,
            'protocol': str(args.protocol),
            'checkpoint': str(args.checkpoint),
            'pred_root': str(args.pred_root),
            'metrics': args.metrics,
            'datasets': args.datasets or 'all',
        },
    )
    if run is None:
        return
    try:
        import wandb

        with open(metrics_path, newline='') as f:
            rows = [row for row in csv.DictReader(f)
                    if row.get('method') == args.method]
        if not rows:
            print(f'[warn] W&B: no rows for method={args.method} in '
                  f'{metrics_path}; skip paper-metric logging')
            return

        table = wandb.Table(columns=[
            'method', 'dataset', 'role', 'dice_mean', 'iou_mean',
            'cldice_mean', 'hd95_mean', 'dice_n',
        ])
        for row in rows:
            table.add_data(
                row['method'], row['dataset'], row['role'],
                _float_or_none(row.get('dice_mean')),
                _float_or_none(row.get('iou_mean')),
                _float_or_none(row.get('cldice_mean')),
                _float_or_none(row.get('hd95_mean')),
                _float_or_none(row.get('dice_n')),
            )
        run.log({'paper/per_dataset': table})

        scalars = {}
        if macro_path.is_file():
            with open(macro_path, newline='') as f:
                macro_rows = [row for row in csv.DictReader(f)
                              if row.get('method') == args.method]
            for row in macro_rows:
                role = row.get('role', '').strip().lower()
                for metric in ('dice', 'iou', 'cldice', 'hd95'):
                    value = _float_or_none(row.get(f'{metric}_mean'))
                    if value is not None:
                        scalars[f'macro/{args.method}/{role}/{metric}'] = value
        if scalars:
            log_scalars(run, scalars)

        for path, artifact_name in (
            (metrics_path, 'paper_metrics'),
            (macro_path, 'paper_macro'),
        ):
            if path.is_file():
                log_artifact(run, path,
                             name=f'{artifact_name}-{args.method}',
                             artifact_type='metrics')
        print(f'W&B: logged paper metrics for {args.method} '
              f'({len(rows)} dataset rows)')
    except Exception as exc:
        print(f'[warn] W&B paper-metric logging failed: {exc}')
    finally:
        finish_run(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='unet')
    parser.add_argument('--protocol', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot' /
                        'protocol_manifest.csv')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='trained baseline checkpoint (default: '
                             'workdirs/baselines/<method>/best_model.pth)')
    parser.add_argument('--sam-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam_vit_l_0b3195.pth',
                        help='frozen SAM ViT-L checkpoint for '
                             'sam_vit_l_unet')
    parser.add_argument('--fob-sam-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam_vit_h_4b8939.pth',
                        help='frozen SAM ViT-H checkpoint for fob_sam')
    default_data_root = REPO_ROOT / 'data' / 'sacm_3shot'
    parser.add_argument('--data-root', type=Path,
                        default=default_data_root,
                        help='training split used for FoB support images')
    parser.add_argument('--sam2-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam2_hiera_large.pt',
                        help='frozen SAM2 Hiera checkpoint for sam2_unet')
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
    parser.add_argument('--workers', type=int, default=None,
                        help='parallel dataset workers for metric collection')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='images per metric worker task')
    parser.add_argument('--no-resume', action='store_true',
                        help='recompute per-dataset metrics instead of '
                             'reusing the cache')
    parser.add_argument('--allow-stale-checkpoint', action='store_true',
                        help='allow a checkpoint older than the current split')
    parser.add_argument('--use_wandb', nargs='?', const=True,
                        type=_as_bool, default=True,
                        help='log paper metrics to W&B (default: true; '
                             'pass false to disable)')
    parser.add_argument('--wandb_project',
                        default=os.environ.get('WANDB_PROJECT', 'sacm'))
    parser.add_argument('--wandb_entity',
                        default=os.environ.get('WANDB_ENTITY', 'buaazqk'))
    parser.add_argument('--wandb_host',
                        default=os.environ.get('WANDB_BASE_URL',
                                               'http://172.16.1.7:8080'))
    parser.add_argument('--wandb_api_key',
                        default=os.environ.get('WANDB_API_KEY', ''))
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_tags', default='paper_metrics,3shot')
    parser.add_argument('--wandb_mode', default=os.environ.get('WANDB_MODE'))
    parser.add_argument('--wandb_dir', default=os.environ.get('WANDB_DIR'))
    parser.add_argument('--wandb_group', default=None)
    parser.add_argument('--wandb_job_type', default='evaluation')
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = (REPO_ROOT / 'workdirs' / 'baselines' /
                           args.method / 'best_model.pth')
    # FoB needs labeled support images at test time.  If the caller did
    # not override --data-root, follow the protocol's own split root so a
    # seed-specific run never uses the frozen seed-42 support set.
    if (args.method == 'fob_sam'
            and args.data_root == default_data_root
            and args.protocol.parent != default_data_root):
        args.data_root = args.protocol.parent

    dataset_filter = set(args.datasets) if args.datasets else None
    specs = read_specs(args.protocol, dataset_filter)
    counts = read_test_counts(args.protocol, dataset_filter)
    total_images = sum(counts.values())
    print(f'{args.method}: {len(specs)} test datasets, {total_images} images')
    print('  ' + ' '.join(name for name, _ in specs))

    stage_names = []
    if not args.skip_predict and args.method in PORTED:
        stage_names.append('predict')
    if not args.skip_paper:
        stage_names.append('paper_metrics')
    if not args.skip_visualize:
        stage_names.append('visualize')

    status_path = (args.pred_root.parent / '.run_status' /
                   f'{args.method}.json')
    status = RunStatus(path=status_path, method=args.method, heartbeat=0)
    os.environ['SACM_RUN_STATUS'] = str(status_path)
    os.environ['SACM_RUN_METHOD'] = args.method
    stage_no = 0

    def begin_stage(stage):
        nonlocal stage_no
        stage_no += 1
        status.set_stage(
            stage, stage_index=stage_no, stage_total=len(stage_names),
            datasets_total=len(specs), overall_total=total_images)
        label = {
            'predict': 'predictions',
            'paper_metrics': 'paper metrics',
            'visualize': 'visualizations',
        }.get(stage, stage)
        print(f'== {label} ==', flush=True)

    try:
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
            if args.method in PORTED:
                if not args.checkpoint.is_file():
                    hint = (
                        'python -m baseline_zoo.run_one '
                        f'--model {args.method} --stage train --epochs 200')
                    raise SystemExit(
                        f'missing checkpoint: {args.checkpoint}\n'
                        f'train first: {hint}')
                # Use the split manifest belonging to the active protocol.
                # Seed-specific checkpoints must not be compared against the
                # frozen seed-42 split.
                split = args.protocol.parent / 'split_manifest.csv'
                if not split.is_file():
                    split = REPO_ROOT / 'data' / 'sacm_3shot' / 'split_manifest.csv'
                if (split.is_file() and not args.allow_stale_checkpoint
                        and args.method not in CUSTOM_CHECKPOINTS):
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
                cmd = [sys.executable, '-m', 'baseline_zoo.run_one',
                       '--model', args.method, '--stage', 'predict',
                       '--checkpoint', args.checkpoint,
                       '--pred-root', method_pred_root,
                       '--data-root', args.data_root]
                if args.method == 'fob_sam':
                    cmd += ['--sam-checkpoint', args.fob_sam_checkpoint]
                elif args.method in SAM_CHECKPOINT_METHODS:
                    cmd += ['--sam-checkpoint', args.sam_checkpoint]
                if args.device:
                    cmd += ['--device', args.device]
                cmd += ['--test-datasets']
                cmd += [f'{name}:{root}' for name, root in specs]
                begin_stage('predict')
                run(cmd, repo_env())
            else:
                print(f'[warn] no self-contained prediction adapter for '
                      f'{args.method}; assuming predictions already exist')

        if not args.skip_paper:
            begin_stage('paper_metrics')
            cmd = [sys.executable,
                   REPO_ROOT / 'scripts' / 'collect_paper_metrics.py',
                   '--methods', args.method,
                   '--pred-root', args.pred_root,
                   '--out-root', args.paper_root,
                   '--metrics', args.metrics,
                   '--protocol', args.protocol]
            if args.datasets:
                cmd += ['--datasets'] + list(args.datasets)
            if args.workers is not None:
                cmd += ['--workers', str(args.workers)]
            if args.chunk_size is not None:
                cmd += ['--chunk-size', str(args.chunk_size)]
            if args.no_resume:
                cmd += ['--no-resume']
            run(cmd, repo_env())
            if args.use_wandb:
                log_paper_metrics(args, args.paper_root)

        if not args.skip_visualize:
            begin_stage('visualize')
            run([sys.executable,
                 REPO_ROOT / 'scripts' / 'visualize_predictions.py',
                 '--methods', args.method,
                 '--pred-root', args.pred_root,
                 '--out-root', args.vis_root,
                 '--protocol', args.protocol,
                 '--num-samples', str(args.num_samples)] +
                (['--datasets'] + list(args.datasets) if args.datasets else []),
                repo_env())

        status.refresh()
        status.finish('done')
    except BaseException as exc:
        status.refresh()
        status.fail(exc)
        raise

    print('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
