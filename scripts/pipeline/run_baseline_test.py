#!/usr/bin/env python3
"""One-click prediction + paper statistics + visualization for a baseline.

Defaults to the U-Net adapter and all 20 test datasets in the frozen protocol:

    python scripts/pipeline/run_baseline_test.py

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
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from protocol_specs import dataset_specs  # noqa: E402

# Self-contained adapters.  A method without an entry is assumed to have
# predictions already on disk (reference-only or externally produced).
from baseline_zoo.adapters.registry import available_adapters  # noqa: E402
from baseline_zoo.registry import load_models  # noqa: E402
from core.run_layout import RUNS_ROOT, default_variant, run_root  # noqa: E402
from core.run_status import RunStatus  # noqa: E402
from core.wandb_utils import (  # noqa: E402
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

PORTED = set(available_adapters())
# nnWNet stores a model folder / KnowSAM a plain SGDL state dict; neither
# carries the unified split-hash field.
CUSTOM_CHECKPOINTS = {'nnunet2d', 'nnwnet', 'knowsam'}
SAM_CHECKPOINT_METHODS = {
    'cwsam', 'sacm', 'dcd', 'sam_adapter', 'sam_med2d',
    'sam_octa', 'sam_vit_l_unet', 'ucs'}


def read_specs(protocol: Path, dataset_filter):
    return [tuple(spec.split(':', 1)) for spec in dataset_specs(
        protocol, set(dataset_filter) if dataset_filter else None,
        REPO_ROOT)]


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
    # Deterministic child inference; these must be present before Python
    # starts in the prediction subprocess.
    env.setdefault('PYTHONHASHSEED', '0')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
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


def _resolve_checkpoint_pointer(path: Path) -> Path | None:
    """Resolve nnU-Net/KnowSAM ``*.txt`` model pointers."""
    candidate = Path(path)
    if candidate.suffix == '.txt':
        if not candidate.is_file():
            return None
        try:
            text = candidate.read_text().strip()
        except OSError:
            return None
        if not text:
            return None
        candidate = Path(text)
    return candidate


def checkpoint_fingerprint(path: Path) -> str | None:
    """Hash the checkpoint that prediction will actually deploy.

    Custom pipelines expose a model-folder pointer (nnU-Net) or a pointer to
    a state dict (KnowSAM).  Hashing the pointer file itself does not detect a
    retrained model, so resolve it and hash the deployed checkpoint file.
    """
    candidate = _resolve_checkpoint_pointer(path)
    if candidate is None:
        return None
    if candidate.is_file():
        try:
            return file_sha256(candidate)
        except OSError:
            return None
    if candidate.is_dir():
        names = (
            'checkpoint_benchmark_best.pth',
            'checkpoint_best.pth',
            'checkpoint_final.pth',
            'checkpoint_latest.pth',
            'checkpoint_benchmark_best.safetensors',
            'checkpoint_best.safetensors',
            'checkpoint_final.safetensors',
            'SGDL_best_model.pth',
            'best_model.pth',
        )
        for base in (candidate / 'fold_0', candidate):
            for name in names:
                target = base / name
                if target.is_file():
                    try:
                        return file_sha256(target)
                    except OSError:
                        return None
    return None


def resolve_variant(method: str, checkpoint: Path | None,
                    explicit: str | None) -> str:
    """Resolve the run variant without silently defaulting to ``default``.

    Priority: explicit CLI value -> checkpoint path under
    ``results/runs/<method>/<variant>/seed_<n>`` -> model config preset ->
    ``default`` (baselines).
    """
    if explicit:
        return str(explicit)
    if checkpoint is not None:
        try:
            rel = Path(checkpoint).resolve().relative_to(RUNS_ROOT.resolve())
        except (OSError, ValueError):
            rel = None
        if rel is not None and len(rel.parts) >= 3 and rel.parts[0] == method:
            return rel.parts[1]
    try:
        model = load_models().get(method)
        if model is not None:
            return default_variant(model.defaults)
    except Exception:
        pass
    return 'default'


def resolve_eval_root(paper_root: str | Path | None, canonical_run: Path,
                      train_root: Path) -> Path:
    """Map ``--paper-root`` to the eval output directory.

    Config templates pass the checkpoint's run root.  That must resolve to
    ``<run>/eval`` in the checkpoint's own variant, never to the method's
    default-variant eval directory (the old ``all_dense``/``default`` alias
    bug).  An explicit external path is preserved verbatim.
    """
    if paper_root is None:
        return Path(canonical_run) / 'eval'
    requested = Path(paper_root)
    if requested.resolve() in (Path(train_root).resolve(),
                               Path(canonical_run).resolve()):
        return requested / 'eval'
    return requested


def resolve_split_manifest(protocol: Path, seed: int) -> Path:
    """Require the split manifest belonging to the active protocol seed."""
    split = protocol.parent / 'split_manifest.csv'
    if split.is_file():
        return split
    if int(seed) == 42:
        legacy = REPO_ROOT / 'data' / 'sacm_3shot' / 'split_manifest.csv'
        if legacy.is_file():
            print('[warn] using legacy seed-42 split manifest for seed 42')
            return legacy
    raise SystemExit(
        f'missing split manifest for seed {seed}: {split}; '
        'pass a matching --protocol or regenerate the split')


def _as_bool(value):
    return str(value).strip().lower() not in ('0', 'false', 'no', 'off',
                                              'disabled', 'none')


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def log_paper_metrics(args, eval_root: Path):
    """Log the 20-dataset per-dataset and macro metrics to W&B.

    Non-fatal: a missing CSV, missing package, or unreachable server must not
    turn a successful evaluation into a failed run.
    """
    metrics_path = eval_root / 'tables' / 'all_metrics.csv'
    macro_path = eval_root / 'tables' / 'macro_metrics.csv'
    if not metrics_path.is_file():
        print(f'[warn] W&B: missing {metrics_path}; skip paper-metric logging')
        return

    run = setup_wandb(
        args,
        name=(f'paper_metrics-{args.method}-'
              f'{getattr(args, "variant", "default")}-seed{args.seed}'),
        config={
            'method': args.method,
            'variant': getattr(args, 'variant', 'default'),
            'seed': int(args.seed),
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
                    if row.get('method') in (None, args.method)]
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
                args.method, row['dataset'], row['role'],
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
                              if row.get('method') in (None, args.method)]
            for row in macro_rows:
                role = row.get('role', '').strip().lower()
                metric = row.get('metric', '').strip().lower()
                if metric not in ('dice', 'iou', 'cldice', 'hd95'):
                    continue
                value = _float_or_none(row.get('mean'))
                if value is not None:
                    scalars[f'macro/{args.method}/{role}/{metric}'] = value
        if scalars:
            log_scalars(run, scalars)

        for path, artifact_name in (
            (metrics_path, 'all_metrics'),
            (macro_path, 'macro_metrics'),
        ):
            if path.is_file():
                log_artifact(
                    run, path,
                    name=(f'{artifact_name}-{args.method}-'
                          f'seed{args.seed}'),
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
    parser.add_argument('--protocol', type=Path, default=None,
                        help='seed-specific protocol_manifest.csv (default: '
                             'data/sacm_3shot_seed0/protocol_manifest.csv)')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='trained baseline checkpoint (default: '
                             'results/runs/<method>/<variant>/seed_<seed>/'
                             'best_model.pth or the native checkpoint name)')
    parser.add_argument(
        '--variant', default=None,
        help='run variant (default: infer from the checkpoint run root, '
             'else the model config preset, else default)')
    parser.add_argument(
        '--allow-missing-prediction-meta', action='store_true',
        help='with --skip-predict, accept predictions without '
             '.prediction_meta.json (provenance is not validated)')
    parser.add_argument('--sam-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam_vit_l_0b3195.pth',
                        help='frozen SAM ViT-L checkpoint for '
                             'sam_vit_l_unet')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='training split used for support images '
                             '(default: the protocol manifest directory)')
    parser.add_argument('--sam2-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam2_hiera_large.pt',
                        help='frozen SAM2 Hiera checkpoint for sam2_unet')
    parser.add_argument('--pred-root', type=Path, default=None,
                        help='prediction root (default: '
                             '<run-root>/predictions)')
    parser.add_argument('--paper-root', type=Path, default=None,
                        help='canonical eval root (default: <run-root>/eval); '
                             'passing the run root is accepted for backward '
                             'compatibility')
    parser.add_argument('--vis-root', type=Path, default=None,
                        help='visualization root (default: '
                             '<run-root>/visualizations)')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--metrics', default='dice,iou,cldice,hd95')
    parser.add_argument(
        '--empty-metric-policy', default='penalty',
        choices=['penalty', 'nan'],
        help='empty prediction convention forwarded to the metric '
             'collector; penalty (default) keeps empty masks in clDice/HD95 '
             'with a finite worst-case value')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--invert-test-datasets', default='',
        help='comma-separated test dataset names whose images are inverted '
             'before prediction (engineering override; no retraining)')
    parser.add_argument(
        '--invert-test-threshold', type=float, default=0.5,
        help='binarisation threshold used for inverted datasets')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='protocol seed forwarded to prediction adapter (default: '
             'parsed from the protocol path, else 0)')
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
    parser.add_argument('--allow-partial-overwrite', action='store_true',
                        help='allow a dataset/metric subset to overwrite a '
                             'complete paper table')
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
    if args.protocol is None:
        legacy_protocol = (REPO_ROOT / 'data' / 'sacm_3shot' /
                           'protocol_manifest.csv')
        if args.seed is not None:
            candidate = (REPO_ROOT / 'data' /
                         f'sacm_3shot_seed{int(args.seed)}' /
                         'protocol_manifest.csv')
            if candidate.is_file():
                args.protocol = candidate
            elif int(args.seed) == 42 and legacy_protocol.is_file():
                args.protocol = legacy_protocol
            else:
                raise SystemExit(
                    f'missing seed-specific protocol for seed {args.seed}: '
                    f'{candidate}; pass --protocol explicitly')
        else:
            seed0 = (REPO_ROOT / 'data' / 'sacm_3shot_seed0' /
                     'protocol_manifest.csv')
            args.protocol = (seed0 if seed0.is_file() else legacy_protocol)
    if args.seed is None:
        seed = None
        summary_path = args.protocol.parent / 'protocol_summary.json'
        if summary_path.is_file():
            try:
                seed = json.loads(summary_path.read_text()).get('seed')
            except (OSError, json.JSONDecodeError):
                seed = None
        if seed is None:
            match = re.search(r'seed[_-]?(\d+)', str(args.protocol.parent))
            seed = int(match.group(1)) if match else 0
        args.seed = int(seed)
    if args.data_root is None:
        # The protocol and its support images live in the same seed split.
        args.data_root = args.protocol.parent
    args.variant = resolve_variant(
        args.method, args.checkpoint, args.variant)
    if args.method == 'dcd' and args.variant == 'default':
        raise SystemExit(
            'DCD has one canonical main variant: all_dense. '
            'The historical default/ tree was an eval alias and is archived; '
            'pass --variant all_dense or use the current run root.')
    canonical_run = run_root(args.method, args.variant, args.seed)
    if args.checkpoint is None:
        custom = {
            'nnunet2d': 'nnunet_model_dir.txt',
            'nnwnet': 'nnunet_model_dir.txt',
            'knowsam': 'SGDL_best_model.pth',
        }.get(args.method, 'best_model.pth')
        args.checkpoint = canonical_run / custom
    args.checkpoint = Path(args.checkpoint)
    train_root = args.checkpoint.parent
    if args.pred_root is None:
        args.pred_root = canonical_run / 'predictions'
    args.pred_root = Path(args.pred_root)
    args.paper_root = resolve_eval_root(
        args.paper_root, canonical_run, train_root)
    if args.vis_root is None:
        args.vis_root = canonical_run / 'visualizations'
    args.vis_root = Path(args.vis_root)
    # The checkpoint's own run root is the provenance source of truth.
    args.run_root = train_root
    args.train_root = train_root
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
                if not args.allow_missing_prediction_meta:
                    raise SystemExit(
                        '--skip-predict requires .prediction_meta.json so '
                        'the checkpoint/split provenance can be validated; '
                        'regenerate predictions or pass '
                        '--allow-missing-prediction-meta')
                print('[warn] --skip-predict: no prediction metadata; '
                      'checkpoint provenance is not re-validated '
                      '(--allow-missing-prediction-meta)')
            else:
                meta = json.loads(meta_path.read_text())
                split = resolve_split_manifest(args.protocol, args.seed)
                expected_split = file_sha256(split)
                custom_checkpoint = args.method in CUSTOM_CHECKPOINTS
                if custom_checkpoint:
                    expected_ckpt = checkpoint_fingerprint(args.checkpoint)
                    if (expected_ckpt is not None
                            and not meta.get('checkpoint_sha256')):
                        # Legacy custom metadata predates checkpoint hashes;
                        # warn instead of reporting a false stale mismatch.
                        print('[warn] --skip-predict: custom prediction '
                              'metadata has no checkpoint_sha256; cannot '
                              'prove it matches the current model')
                        expected_ckpt = None
                else:
                    expected_ckpt = (
                        file_sha256(args.checkpoint)
                        if args.checkpoint.is_file() else None)
                if custom_checkpoint and not meta.get(
                        'split_manifest_sha256'):
                    # Custom nnU-Net/KnowSAM metadata predates the unified
                    # hash fields; do not report a false stale mismatch.
                    expected_split = None
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
                    from baseline_zoo.registry import load_models
                    model = load_models().get(args.method)
                    epochs = (
                        model.defaults.get('epochs') if model is not None
                        else None)
                    epochs_arg = (
                        f' --epochs {epochs}' if epochs is not None else '')
                    hint = (
                        'python -m baseline_zoo.run_one '
                        f'--model {args.method} --stage train{epochs_arg}')
                    raise SystemExit(
                        f'missing checkpoint: {args.checkpoint}\n'
                        f'train first: {hint}')
                # Use the split manifest belonging to the active protocol.
                # Seed-specific checkpoints must not be compared against the
                # frozen seed-42 split.
                split = resolve_split_manifest(args.protocol, args.seed)
                if (not args.allow_stale_checkpoint
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
                       '--workdir', str(args.checkpoint.parent),
                       '--pred-root', method_pred_root,
                       '--data-root', args.data_root,
                       '--seed', str(args.seed)]
                if args.method == 'sam2_unet':
                    cmd += ['--sam2-checkpoint', args.sam2_checkpoint]
                elif args.method in SAM_CHECKPOINT_METHODS:
                    cmd += ['--sam-checkpoint', args.sam_checkpoint]
                if args.device:
                    cmd += ['--device', args.device]
                cmd += ['--test-datasets']
                cmd += [f'{name}:{root}' for name, root in specs]
                if args.invert_test_datasets:
                    cmd += [
                        '--invert-test-datasets', args.invert_test_datasets,
                        '--invert-test-threshold',
                        str(args.invert_test_threshold)]
                begin_stage('predict')
                run(cmd, repo_env())
            else:
                print(f'[warn] no self-contained prediction adapter for '
                      f'{args.method}; assuming predictions already exist')

        if not args.skip_paper:
            begin_stage('paper_metrics')
            cmd = [sys.executable,
                   REPO_ROOT / 'scripts' / 'pipeline' / 'collect_paper_metrics.py',
                   '--methods', args.method,
                   '--pred-root', args.pred_root,
                   '--out-root', args.paper_root,
                   '--metrics', args.metrics,
                   '--empty-metric-policy', args.empty_metric_policy,
                   '--protocol', args.protocol,
                   '--layout', 'canonical',
                   '--family', 'baseline',
                   '--variant', args.variant,
                   '--seed', str(args.seed)]
            if args.datasets:
                cmd += ['--datasets'] + list(args.datasets)
            if args.allow_partial_overwrite:
                cmd += ['--allow-partial-overwrite']
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
                 REPO_ROOT / 'scripts' / 'pipeline' / 'visualize_predictions.py',
                 '--methods', args.method,
                 '--pred-root', args.pred_root,
                 '--out-root', args.vis_root,
                 '--protocol', args.protocol,
                 '--num-samples', str(args.num_samples)] +
                (['--datasets'] + list(args.datasets) if args.datasets else []),
                repo_env())

        status.refresh()
        status.finish('done')
        requested_metrics = {item.strip() for item in args.metrics.split(',')
                             if item.strip()}
        full_metrics = (
            args.metrics.strip().lower() == 'all'
            or {'dice', 'iou', 'cldice', 'hd95'}.issubset(requested_metrics))
        collect_partial = False
        if not args.skip_paper:
            stats_path = (args.paper_root / 'tables' /
                          'collect_stats.json')
            if stats_path.is_file():
                try:
                    stats = json.loads(stats_path.read_text())
                except (OSError, json.JSONDecodeError):
                    stats = {}
                for item in stats.get('methods', {}).values():
                    if (int(item.get('missing', 0) or 0) > 0
                            or item.get('missing_datasets')):
                        collect_partial = True
                        break
        full_eval = (not args.datasets and full_metrics
                     and not args.skip_paper and not collect_partial)
        canonical_eval = (Path(args.paper_root).resolve() ==
                          (canonical_run / 'eval').resolve())
        finalize_allowed = canonical_eval and full_eval
        existing_manifest_path = args.paper_root / 'run_manifest.json'
        existing_status = None
        if existing_manifest_path.is_file():
            try:
                existing_status = json.loads(
                    existing_manifest_path.read_text()).get('status')
            except (OSError, json.JSONDecodeError):
                existing_status = None
        preserve_manifest = bool(
            args.skip_paper and existing_status == 'done')
        status_value = ('done' if preserve_manifest
                        else 'train_only' if args.skip_paper
                        else 'done' if full_eval else 'partial')
        checkpoint_hash = checkpoint_fingerprint(args.checkpoint)
        manifest = {
            'family': 'baseline',
            'method': args.method,
            'variant': args.variant,
            'seed': int(args.seed),
            'run_root': str(args.run_root),
            'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_sha256': checkpoint_hash,
            'protocol': str(args.protocol),
            'datasets': [name for name, _ in specs],
            'pred_root': str(method_pred_root),
            'eval_root': str(args.paper_root),
            'metrics_csv': str(
                args.paper_root / 'tables' / 'all_metrics.csv'),
            'macro_csv': str(
                args.paper_root / 'tables' / 'macro_metrics.csv'),
            'final_report': str(
                args.paper_root / 'tables' / 'final_report.md'),
            'per_dataset_metrics': str(
                args.paper_root / '<DATASET>' / 'metrics.csv'),
            'stages': stage_names,
            'status': status_value,
            'skip_predict': bool(args.skip_predict),
            'skip_paper': bool(args.skip_paper),
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        finalize = REPO_ROOT / 'scripts' / 'pipeline' / 'finalize_run.py'
        if preserve_manifest:
            print('[warn] --skip-paper: preserving existing done manifest')
        elif finalize_allowed:
            if not finalize.is_file():
                raise RuntimeError(
                    f'missing canonical finalizer: {finalize}')
            finalize_cmd = [
                sys.executable, str(finalize),
                '--family', 'baseline', '--method', args.method,
                '--variant', args.variant, '--seed', str(args.seed),
                '--source-train', str(train_root),
                '--source-eval', str(args.paper_root),
                '--out-root', str(RUNS_ROOT),
                '--checkpoint', str(args.checkpoint),
                '--protocol', str(args.protocol),
                '--pred-root', str(method_pred_root),
                '--status', status_value,
                '--cleanup']
            if checkpoint_hash:
                finalize_cmd += ['--checkpoint-sha256', checkpoint_hash]
            finalized = subprocess.run(finalize_cmd, cwd=REPO_ROOT,
                                       check=False)
            if finalized.returncode != 0:
                raise RuntimeError(
                    'finalize_run failed; the canonical results/runs tree '
                    'is incomplete')
        else:
            reason = ('non-canonical eval root' if not canonical_eval
                      else 'subset/partial run')
            print(f'[warn] not publishing to canonical results/runs '
                  f'({reason})')
            try:
                args.paper_root.mkdir(parents=True, exist_ok=True)
                (args.paper_root / 'run_manifest.json').write_text(
                    json.dumps(manifest, indent=2) + '\n')
            except OSError as exc:
                print(f'[warn] could not write run_manifest.json: {exc}')
        stale_failure = args.paper_root / 'run_manifest.failed.json'
        try:
            if stale_failure.is_file():
                stale_failure.unlink()
        except OSError:
            pass
    except BaseException as exc:
        status.refresh()
        status.fail(exc)
        try:
            args.paper_root.mkdir(parents=True, exist_ok=True)
            # The failed marker overrides any older success manifest in the
            # index, so a finalize failure cannot be reported as done.
            (args.paper_root / 'run_manifest.failed.json').write_text(
                json.dumps({
                    'family': 'baseline',
                    'method': args.method,
                    'seed': int(args.seed),
                    'checkpoint': str(args.checkpoint.resolve()),
                    'status': 'failed',
                    'error': str(exc),
                    'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }, indent=2) + '\n')
        except OSError:
            pass
        raise

    print('done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
