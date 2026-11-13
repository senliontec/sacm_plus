#!/usr/bin/env python3
"""One-click DCD 800-epoch main run + component ablation.

What this driver does
---------------------
1. Trains four arms (``ssh``, ``nofusion``, ``native_control``, ``notopo``)
   for the requested seeds at 800 epochs, using the existing config-driven
   ablation runner and ``baseline_zoo.run_one``.
2. Derives the inference-only fixed candidate-3 checkpoint for every
   completed arm/seed.  The native control is derived with ``--allow-native``.
3. Evaluates four deployment rows on the 20 protocol test datasets:
   every arm at fixed candidate 3, plus two eval-only deployment references
   (DCD gating+fusion and native gating) from the same checkpoints.
4. Writes a multi-arm macro / per-seed / per-dataset summary and, when the
   official SACM tables already exist, the standard fixed3-vs-SACM report.

The script is deliberately resumable.  The default output root is fresh, so
the first invocation trains all 16 arm/seed pairs; a later invocation skips
completed training and completed evaluation tables.  Use ``RESUME=0`` to
retrain from scratch.

This driver does not modify the model or the existing ablation runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / 'configs' / 'ablations' /
                  'dcd_full_800.yaml')
ARMS = ('ssh', 'nofusion', 'native_control', 'notopo')
MAIN_ARM = 'ssh'
FIXED3_CKPT = 'best_model_fixed3.pth'
METRICS = (
    ('dice', 'Dice', True),
    ('iou', 'IoU', True),
    ('cldice', 'clDice', True),
    ('hd95', 'HD95', False),
)
ROLES = ('All', 'Seen', 'Unseen', 'Novel')
TEST_SPLITS = ('seen_test', 'unseen_test', 'novel_test')

# These are the fields that must match when a completed run is reused.  Paths
# (SAM checkpoint, data root) are intentionally excluded: they can legitimately
# differ between the local machine and the server.
RECIPE_KEYS = (
    'adapter_dim_ratio',
    'bce_weight',
    'dice_weight',
    'bce_pos_weight',
    'binary_targets',
    'deploy_threshold',
    'deep_sup_weight',
    'specialized_heads',
    'specialized_head_weight',
    'specialized_topology_weight',
    'head_fusion',
    'fusion_loss_weight',
    'stage2_loss_scope',
    'topology_head',
    'cl_dice_weight',
    'skeleton_recall_weight',
    'topology_resolution',
    'topology_warmup_epochs',
    'head_select',
    'deployment_head',
)

ROW_LABELS = {
    'dcd_fixed3': 'DCD fixed3 (frozen canonical)',
    'nofusion_fixed3': 'DCD w/o fusion training',
    'native_fixed3': 'Native four-head, fixed candidate 3',
    'notopo_fixed3': 'DCD w/o topology supervision',
    'dcd_gating_fusion': 'DCD gating + fusion (same checkpoint)',
    'native_gating': 'Native four-head, gating',
    'sacm': 'SACM (official, reference)',
}
ABLATION_ROWS = ('nofusion_fixed3', 'native_fixed3', 'notopo_fixed3')
FIXED3_ROWS = ('dcd_fixed3', *ABLATION_ROWS)
REFERENCE_ROWS = ('dcd_gating_fusion', 'native_gating')
SUMMARY_ROWS = ('dcd_fixed3', *ABLATION_ROWS, *REFERENCE_ROWS, 'sacm')
ROW_ARMS = {
    'dcd_fixed3': 'ssh',
    'nofusion_fixed3': 'nofusion',
    'native_fixed3': 'native_control',
    'notopo_fixed3': 'notopo',
    'dcd_gating_fusion': 'ssh',
    'native_gating': 'native_control',
}


# ---------------------------------------------------------------------------
# Small parsing / path helpers
# ---------------------------------------------------------------------------
def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _parse_values(raw, default):
    values = raw if raw is not None else default
    if isinstance(values, str):
        values = [values]
    out = []
    for item in values:
        for token in str(item).replace(',', ' ').split():
            out.append(token)
    return out


def _parse_seeds(raw, default):
    tokens = _parse_values(raw, default)
    try:
        seeds = [int(token) for token in tokens]
    except ValueError as exc:
        raise SystemExit(f'invalid seed list: {tokens}') from exc
    if not seeds:
        raise SystemExit('seed list is empty')
    seeds = list(dict.fromkeys(seeds))
    if any(seed < 0 for seed in seeds):
        raise SystemExit(f'seeds must be non-negative: {seeds}')
    return seeds


def _parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in str(raw).split(',') if item.strip()]
    if not gpus:
        raise SystemExit('GPU list is empty')
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f'duplicate GPU ids: {gpus}')
    return gpus


def _resolve(path_text) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_text(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _file_sha256(path: Path) -> str | None:
    """Hash a deployed checkpoint, cached by path/size/mtime.

    ``run_baseline_test.py`` records the same SHA-256 in its manifest.  The
    summary uses it to reject an eval table that belongs to an older
    checkpoint with the same filename.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    cache_key = (str(path.resolve()), int(stat.st_size),
                 int(stat.st_mtime_ns))
    cached = _SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


# ---------------------------------------------------------------------------
# Configuration and recipe checks
# ---------------------------------------------------------------------------
def _load_config(path: Path) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f'cannot read config {path}: {exc}') from exc
    if not isinstance(config, dict):
        raise SystemExit(f'{path}: config must be a mapping')
    if int(config.get('schema_version', 0)) != 1:
        raise SystemExit(f'{path}: unsupported schema_version')
    if str(config.get('method')) != 'dcd':
        raise SystemExit(f'{path}: method must be dcd')
    if str(config.get('protocol')) != 'seeded5':
        raise SystemExit(f'{path}: protocol must be seeded5')
    arms = config.get('arms')
    if not isinstance(arms, dict) or not arms:
        raise SystemExit(f'{path}: arms must be a non-empty mapping')
    missing = [arm for arm in ARMS if arm not in arms]
    if missing:
        raise SystemExit(f'{path}: missing arms {missing}')
    for arm in ARMS:
        if not isinstance(arms[arm], dict):
            raise SystemExit(f'{path}: arm {arm!r} must be a mapping')
        if not isinstance(arms[arm].get('args'), dict):
            raise SystemExit(f'{path}: arm {arm!r} has no args mapping')
    common = config.get('common_train_args') or {}
    if not isinstance(common, dict):
        raise SystemExit(f'{path}: common_train_args must be a mapping')
    if str(common.get('head-select', 'gating')) != 'gating':
        raise SystemExit(
            f'{path}: training must use head-select=gating; fixed candidate 3 '
            'is an inference deployment, not a training selector')
    merged = _expected_recipe(config, MAIN_ARM)
    if int(merged.get('specialized_heads', 1)) != 1:
        raise SystemExit(f'{path}: ssh arm must set specialized-heads: 1')
    if int(merged.get('head_fusion', 1)) != 1:
        raise SystemExit(f'{path}: ssh arm must set head-fusion: 1')

    nofusion = _expected_recipe(config, 'nofusion')
    if int(nofusion.get('specialized_heads', 1)) != 1:
        raise SystemExit(
            f'{path}: nofusion arm must keep specialized-heads: 1')
    if int(nofusion.get('head_fusion', 1)) != 0:
        raise SystemExit(
            f'{path}: nofusion arm must set head-fusion: 0')

    native = _expected_recipe(config, 'native_control')
    if int(native.get('specialized_heads', 1)) != 0:
        raise SystemExit(
            f'{path}: native_control arm must set specialized-heads: 0')
    if int(native.get('head_fusion', 1)) != 0:
        raise SystemExit(
            f'{path}: native_control arm must set head-fusion: 0')

    notopo = _expected_recipe(config, 'notopo')
    if int(notopo.get('specialized_heads', 1)) != 1:
        raise SystemExit(
            f'{path}: notopo arm must keep specialized-heads: 1')
    if int(notopo.get('head_fusion', 1)) != 1:
        raise SystemExit(
            f'{path}: notopo arm must keep head-fusion: 1')
    for key in ('cl_dice_weight', 'skeleton_recall_weight',
                'specialized_topology_weight'):
        if abs(float(notopo.get(key, 1.0))) > 1e-12:
            raise SystemExit(
                f'{path}: notopo arm must set {key.replace("_", "-")}: 0')
    return config


def _expected_recipe(config: dict, arm: str) -> dict:
    merged = {}
    for mapping in (config.get('common_train_args') or {},
                    (config['arms'][arm].get('args') or {})):
        for key, value in mapping.items():
            merged[str(key).replace('-', '_')] = value
    return merged


def _recipe_mismatches(actual: dict, expected: dict) -> list[str]:
    mismatches = []
    for key in RECIPE_KEYS:
        if key not in expected:
            continue
        if key not in actual:
            mismatches.append(f'{key}: missing from train_config.json')
            continue
        got = actual[key]
        want = expected[key]
        numeric = (
            isinstance(want, (int, float))
            and not isinstance(want, bool)
            and isinstance(got, (int, float))
            and not isinstance(got, bool)
        )
        if numeric:
            if abs(float(got) - float(want)) > 1e-6:
                mismatches.append(f'{key}: {got!r} != {want!r}')
        elif str(got) != str(want):
            mismatches.append(f'{key}: {got!r} != {want!r}')
    return mismatches


def _train_config_args(run_root: Path):
    payload = _read_json(run_root / 'train_config.json')
    if not isinstance(payload, dict):
        return None
    args = payload.get('args')
    return args if isinstance(args, dict) else None


def _run_complete(run_root: Path, epochs: int) -> bool:
    info = _read_json(run_root / 'run_info.json')
    if not isinstance(info, dict):
        return False
    try:
        recorded = int(info.get('epochs', -1))
    except (TypeError, ValueError):
        return False
    checkpoint = run_root / 'best_model.pth'
    return (
        info.get('status') == 'done'
        and recorded == int(epochs)
        and checkpoint.is_file()
        and checkpoint.stat().st_size > 0
    )


def _check_existing_recipes(config: dict, run_root: Path, seeds: list[int],
                            epochs: int, resume: bool,
                            skip_train: bool) -> None:
    """Refuse to reuse a completed/resumable run trained with another recipe.

    The generic ablation runner's skip predicate checks status and epoch
    budget only.  Without this check a complete 800-epoch run from a
    different arm or an older recipe could silently satisfy a resume and
    poison the table.
    """
    if not resume and not skip_train:
        return
    errors = []
    for arm in ARMS:
        expected = _expected_recipe(config, arm)
        for seed in seeds:
            seed_root = run_root / arm / f'seed_{seed}'
            args = _train_config_args(seed_root)
            info = _read_json(seed_root / 'run_info.json')
            best_model = seed_root / 'best_model.pth'
            if isinstance(info, dict):
                try:
                    recorded = int(info.get('epochs'))
                except (TypeError, ValueError):
                    recorded = None
                try:
                    best_size = best_model.stat().st_size
                except OSError:
                    best_size = 0
                if (info.get('status') == 'done'
                        and recorded == int(epochs)
                        and best_size <= 0):
                    errors.append(
                        f'{arm}/seed{seed}: run_info is done but '
                        'best_model.pth is missing/empty; rerun with RESUME=0')
                    continue
            complete = _run_complete(seed_root, epochs)
            resumable = False
            if (seed_root / '.resume_state.pth').is_file():
                if args is not None:
                    try:
                        resumable = int(args.get('epochs')) == int(epochs)
                    except (TypeError, ValueError):
                        resumable = False
                else:
                    resumable = True
            if not complete and not resumable:
                continue
            if args is None:
                errors.append(
                    f'{arm}/seed{seed}: existing run has no '
                    'train_config.json; cannot verify its recipe')
                continue
            mismatches = _recipe_mismatches(args, expected)
            try:
                recorded_seed = int(args.get('seed'))
            except (TypeError, ValueError):
                recorded_seed = None
            if recorded_seed is not None and recorded_seed != seed:
                mismatches.append(
                    f'seed: {recorded_seed!r} != {seed!r}')
            try:
                recorded_epochs = int(args.get('epochs'))
            except (TypeError, ValueError):
                recorded_epochs = None
            if recorded_epochs is not None and recorded_epochs != int(epochs):
                mismatches.append(
                    f'epochs: {recorded_epochs!r} != {epochs!r}')
            data_root = str(args.get('data_root', ''))
            if data_root and Path(data_root).name != f'sacm_3shot_seed{seed}':
                mismatches.append(
                    f'data_root: {data_root!r} is not seed {seed}')
            if mismatches:
                errors.append(
                    f'{arm}/seed{seed}: ' + '; '.join(mismatches))
    if errors:
        detail = '\n  '.join(errors)
        raise SystemExit(
            'existing run recipe mismatch:\n  ' + detail + '\n'
            'Archive/move the affected arm/seed directories, or run with '
            'RESUME=0 (and FORCE_DERIVE=1 FORCE_EVAL=1) to retrain from '
            'this YAML recipe.')


# ---------------------------------------------------------------------------
# Preflight and command execution
# ---------------------------------------------------------------------------
def _check_gpus(gpus: list[str], max_used_mib: int,
                allow_busy: bool, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f'[warn] nvidia-smi unavailable; GPU-busy check skipped: {exc}',
              file=sys.stderr)
        return
    visible = {}
    for line in out.strip().splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            visible[parts[0]] = int(float(parts[1]))
        except ValueError:
            visible[parts[0]] = 0
    missing = [gpu for gpu in gpus if gpu not in visible]
    if missing:
        raise SystemExit(f'GPU(s) not visible to nvidia-smi: {missing}')
    busy = [(gpu, visible[gpu]) for gpu in gpus
            if visible[gpu] > max_used_mib]
    if busy and not allow_busy:
        detail = ', '.join(f'{gpu}={mib}MiB' for gpu, mib in busy)
        raise SystemExit(
            f'selected GPU(s) look busy: {detail}; free them or set '
            'ALLOW_BUSY=1 / MAX_USED_MIB=...')


def _protocol_counts(protocol: Path) -> tuple[dict[str, int],
                                               dict[str, int]]:
    datasets = {split: set() for split in TEST_SPLITS}
    images = {split: 0 for split in TEST_SPLITS}
    if not protocol.is_file():
        raise SystemExit(f'missing protocol: {protocol}')
    with protocol.open(newline='') as handle:
        for row in csv.DictReader(handle):
            split = str(row.get('split', '')).strip()
            dataset = str(row.get('dataset', '')).strip()
            if split in datasets:
                images[split] += 1
                if dataset:
                    datasets[split].add(dataset)
    return (
        {split: len(values) for split, values in datasets.items()},
        images,
    )


def _check_protocols(seeds: list[int], expected: int, dry_run: bool) -> None:
    for seed in seeds:
        root = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}'
        protocol = root / 'protocol_manifest.csv'
        split_manifest = root / 'split_manifest.csv'
        if not dry_run and not split_manifest.is_file():
            raise SystemExit(f'missing split manifest: {split_manifest}')
        if dry_run and not protocol.is_file():
            print(f'[dry-run] missing protocol would fail: {protocol}')
            continue
        dataset_counts, image_counts = _protocol_counts(protocol)
        total = sum(dataset_counts.values())
        if total < expected:
            raise SystemExit(
                f'{protocol}: {total} test datasets, expected at least '
                f'{expected}')
        print(
            f'protocol seed {seed}: '
            f'seen={dataset_counts["seen_test"]}/'
            f'{image_counts["seen_test"]} images, '
            f'unseen={dataset_counts["unseen_test"]}/'
            f'{image_counts["unseen_test"]} images, '
            f'novel={dataset_counts["novel_test"]}/'
            f'{image_counts["novel_test"]} images, '
            f'{total} datasets')


def _macro_complete(eval_root: Path, expected: int) -> bool:
    path = eval_root / 'tables' / 'macro_metrics.csv'
    if not path.is_file():
        return False
    metric_keys = {key for key, _, _ in METRICS}
    seen = set()
    all_n = None
    try:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                role = str(row.get('role', '')).strip()
                metric = str(row.get('metric', '')).strip().lower()
                if role not in ROLES or metric not in metric_keys:
                    continue
                try:
                    value = float(row.get('mean'))
                    n_datasets = int(float(row.get('n_datasets')))
                except (TypeError, ValueError):
                    continue
                if value != value:  # NaN
                    continue
                seen.add((role, metric))
                if role == 'All' and metric == 'dice':
                    all_n = n_datasets
    except (OSError, csv.Error):
        return False
    if all_n is None or all_n < int(expected):
        return False
    return all(
        (role, key) in seen
        for role in ROLES for key, _, _ in METRICS)


def _checkpoint_matches_manifest(eval_root: Path,
                                 checkpoint: Path) -> bool:
    """Require the eval manifest to describe the current checkpoint file."""
    if not checkpoint.is_file():
        return False
    manifest = _read_json(eval_root / 'run_manifest.json')
    if not isinstance(manifest, dict):
        return False
    if manifest.get('status') not in (None, 'done'):
        return False
    recorded = manifest.get('checkpoint')
    if not recorded or Path(str(recorded)).name != checkpoint.name:
        return False
    recorded_sha = manifest.get('checkpoint_sha256')
    current_sha = _file_sha256(checkpoint)
    return bool(recorded_sha and current_sha
                and str(recorded_sha) == current_sha)


def _eval_complete(eval_root: Path, expected: int,
                   checkpoint: Path | None = None,
                   require_macro: bool = False) -> bool:
    table = eval_root / 'tables' / 'all_metrics.csv'
    if not table.is_file():
        return False
    manifest = _read_json(eval_root / 'run_manifest.json')
    if not isinstance(manifest, dict):
        return False
    if manifest.get('status') not in (None, 'done'):
        return False
    if checkpoint is not None:
        if not _checkpoint_matches_manifest(eval_root, checkpoint):
            return False
    if require_macro and not _macro_complete(eval_root, expected):
        return False
    try:
        with table.open(newline='') as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row.get('dataset')]
    except (OSError, csv.Error):
        return False
    if len({row['dataset'] for row in rows}) < int(expected):
        return False
    for row in rows:
        for key, _, _ in METRICS:
            raw = row.get(f'{key}_mean')
            if raw in (None, ''):
                return False
            try:
                value = float(raw)
            except ValueError:
                return False
            if value != value:  # NaN
                return False
    return True


def _check_official(official_root: Path, seeds: list[int],
                    expected: int, require: bool) -> bool:
    missing = []
    for seed in seeds:
        eval_root = official_root / f'seed_{seed}' / 'eval'
        if not _eval_complete(
                eval_root, expected, checkpoint=None, require_macro=True):
            missing.append(eval_root)
    if missing:
        message = (
            'official SACM eval tables are incomplete: '
            + ', '.join(_repo_text(path) for path in missing))
        if require:
            raise SystemExit(message + '; complete SACM seeds first or set '
                             'REQUIRE_OFFICIAL=0')
        print(f'[warn] {message}; the summary will mark SACM as missing',
              file=sys.stderr)
        return False
    return True


def _official_budget_issues(official_root: Path, seeds: list[int],
                            epochs: int) -> list[str]:
    """Require SACM tables to come from the same epoch budget as DCD."""
    issues = []
    for seed in seeds:
        info = _read_json(
            official_root / f'seed_{seed}' / 'run_info.json')
        if not isinstance(info, dict):
            issues.append(f'sacm/seed{seed}: missing run_info.json')
            continue
        if info.get('status') not in (None, 'done'):
            issues.append(
                f'sacm/seed{seed}: status={info.get("status")!r}')
        try:
            recorded = int(info.get('epochs'))
        except (TypeError, ValueError):
            issues.append(f'sacm/seed{seed}: missing epochs')
            continue
        if recorded != int(epochs):
            issues.append(
                f'sacm/seed{seed}: epochs={recorded}, expected {epochs}')
    return issues


def _run_command_stream(command: list[str], log_path: Path,
                        dry_run: bool) -> int:
    printable = ' '.join(str(item) for item in command)
    print(f'+ {printable}', flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src = str(REPO_ROOT / 'src')
    env['PYTHONPATH'] = src + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with log_path.open('w') as log:
        proc = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        return proc.wait()


# ---------------------------------------------------------------------------
# Training / derivation / evaluation
# ---------------------------------------------------------------------------
def _train_command(config_path: Path, seeds: list[int],
                   train_gpus: list[str], epochs: int,
                   resume: bool, dry_run: bool) -> list[str]:
    command = [
        'bash', str(REPO_ROOT / 'scripts' / 'run_dcd_ablation.sh'),
        '--config', str(config_path),
        '--stage', 'train',
        '--arms', *ARMS,
        '--seeds', *[str(seed) for seed in seeds],
        '--gpus', ','.join(train_gpus),
        '--epochs', str(epochs),
    ]
    if not resume:
        command.append('--no-resume')
    if dry_run:
        command.append('--dry-run')
    return command


def _derive_command(source: Path, output: Path, arm: str,
                    force: bool) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' /
            'make_fixed3_checkpoint.py'),
        '--input', str(source),
        '--output', str(output),
        '--deployment-head', '3',
    ]
    if arm == 'native_control':
        command.append('--allow-native')
    if force:
        command.append('--force')
    return command


def _eval_command(job: 'EvalJob', gpu: str, config: dict,
                  no_resume: bool, eval_workers: int | None,
                  use_wandb: bool) -> list[str]:
    wandb = config.get('wandb') or {}
    protocol = (REPO_ROOT / 'data' /
                f'sacm_3shot_seed{job.seed}' / 'protocol_manifest.csv')
    command = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' / 'run_baseline_test.py'),
        '--method', 'dcd',
        '--protocol', str(protocol),
        '--seed', str(job.seed),
        '--variant', job.variant,
        '--checkpoint', str(job.checkpoint),
        '--pred-root', str(job.pred_root),
        '--paper-root', str(job.eval_root),
        '--device', f'cuda:{gpu}',
        '--use_wandb', 'true' if use_wandb else 'false',
        '--skip-visualize',
        '--metrics', 'dice,iou,cldice,hd95',
    ]
    if use_wandb:
        command += [
            '--wandb_project', str(wandb.get('project', 'sacm')),
            '--wandb_entity', str(wandb.get('entity', 'buaazqk')),
            '--wandb_group', 'dcd_full_800_eval',
            '--wandb_name', f'paper_metrics-dcd-{job.row}-seed{job.seed}',
            '--wandb_tags', 'dcd,ablation,800ep,fixed3,eval',
        ]
    if no_resume:
        command.append('--no-resume')
    if eval_workers is not None:
        command += ['--workers', str(eval_workers)]
    return command


@dataclass(frozen=True)
class EvalJob:
    row: str
    arm: str
    seed: int
    variant: str
    checkpoint: Path
    eval_root: Path
    pred_root: Path


def _main_fixed3_paths(run_root: Path, seed: int,
                       fixed3_root: Path) -> tuple[Path, Path, Path]:
    """Return ``(checkpoint, eval_root, pred_root)`` for the main fixed3 row."""
    checkpoint = (fixed3_root / 'checkpoints' / f'seed_{seed}' /
                  FIXED3_CKPT)
    eval_root = fixed3_root / 'eval' / f'seed_{seed}'
    pred_root = fixed3_root / 'predictions' / f'seed_{seed}'
    return checkpoint, eval_root, pred_root


def _arm_paths(run_root: Path, arm: str,
               seed: int) -> tuple[Path, Path, Path]:
    """Return ``(run_root, best_model, fixed3_checkpoint)`` for an arm."""
    root = run_root / arm / f'seed_{seed}'
    return root, root / 'best_model.pth', root / FIXED3_CKPT


def _build_eval_jobs(config: dict, run_root: Path, fixed3_root: Path,
                     seeds: list[int], epochs: int,
                     derive_ok: set[tuple[str, int]],
                     assume_complete: bool = False) -> tuple[list[EvalJob],
                                                              list[str]]:
    jobs = []
    problems = []
    for seed in seeds:
        main_derived, main_eval, main_pred = _main_fixed3_paths(
            run_root, seed, fixed3_root)
        main_root = run_root / MAIN_ARM / f'seed_{seed}'
        if not assume_complete and not _run_complete(main_root, epochs):
            problems.append(f'ssh/seed{seed}: training incomplete')
        elif (not assume_complete
              and (MAIN_ARM, seed) not in derive_ok):
            problems.append(f'ssh/seed{seed}: fixed3 derivation incomplete')
        elif not assume_complete and not main_derived.is_file():
            problems.append(f'ssh/seed{seed}: missing {main_derived.name}')
        else:
            jobs.append(EvalJob(
                row='dcd_fixed3', arm='ssh', seed=seed,
                variant='dcd_fixed3', checkpoint=main_derived,
                eval_root=main_eval, pred_root=main_pred))

        for arm, row, variant in (
                ('nofusion', 'nofusion_fixed3', 'nofusion_fixed3'),
                ('native_control', 'native_fixed3', 'native_fixed3'),
                ('notopo', 'notopo_fixed3', 'notopo_fixed3')):
            arm_root, _source, derived = _arm_paths(run_root, arm, seed)
            if not assume_complete and not _run_complete(arm_root, epochs):
                problems.append(f'{arm}/seed{seed}: training incomplete')
                continue
            if not assume_complete and (arm, seed) not in derive_ok:
                problems.append(
                    f'{arm}/seed{seed}: fixed3 derivation incomplete')
                continue
            if not assume_complete and not derived.is_file():
                problems.append(
                    f'{arm}/seed{seed}: missing {derived.name}')
                continue
            jobs.append(EvalJob(
                row=row, arm=arm, seed=seed, variant=variant,
                checkpoint=derived,
                eval_root=arm_root / 'eval_fixed3',
                pred_root=arm_root / 'predictions_fixed3'))

        ssh_root, ssh_source, _ = _arm_paths(run_root, 'ssh', seed)
        if assume_complete or _run_complete(ssh_root, epochs):
            jobs.append(EvalJob(
                row='dcd_gating_fusion', arm='ssh', seed=seed,
                variant='dcd_gating_fusion', checkpoint=ssh_source,
                eval_root=ssh_root / 'eval_gating_fusion',
                pred_root=ssh_root / 'predictions_gating_fusion'))

        native_root, native_source, _ = _arm_paths(
            run_root, 'native_control', seed)
        if assume_complete or _run_complete(native_root, epochs):
            jobs.append(EvalJob(
                row='native_gating', arm='native_control', seed=seed,
                variant='native_gating', checkpoint=native_source,
                eval_root=native_root / 'eval_gating',
                pred_root=native_root / 'predictions_gating'))

    priority = {
        'dcd_fixed3': 0,
        'nofusion_fixed3': 1,
        'native_fixed3': 2,
        'notopo_fixed3': 3,
        'dcd_gating_fusion': 4,
        'native_gating': 5,
    }
    jobs.sort(key=lambda job: (priority.get(job.row, 99), job.seed))
    return jobs, problems


def _run_eval_jobs(jobs: list[EvalJob], eval_gpus: list[str],
                   config: dict, log_dir: Path, no_resume: bool,
                   force_eval: bool, eval_workers: int | None,
                   use_wandb: bool, dry_run: bool,
                   expected: int) -> list[str]:
    failures = []
    pending = []
    if not jobs:
        print('  no evaluation jobs (training/derivation incomplete)',
              flush=True)
        return failures
    for job in jobs:
        if not force_eval and _eval_complete(
                job.eval_root, expected, job.checkpoint,
                require_macro=True):
            print(f'  skip complete: {job.row}/seed{job.seed}', flush=True)
            continue
        if not dry_run and not job.checkpoint.is_file():
            failures.append(
                f'{job.row}/seed{job.seed}: missing checkpoint '
                f'{job.checkpoint}')
            continue
        pending.append(job)
    if not pending:
        print('  all evaluation tables already complete', flush=True)
        return failures

    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in eval_gpus:
        gpu_queue.put(gpu)

    def evaluate(job: EvalJob):
        gpu = gpu_queue.get()
        try:
            log = (log_dir / 'eval' / job.row /
                   f'seed{job.seed}.log')
            command = _eval_command(
                job, gpu, config, no_resume, eval_workers, use_wandb)
            code = _run_command_stream(command, log, dry_run)
            if code != 0:
                return job, f'exit {code}; see {_repo_text(log)}'
            return job, None
        except Exception as exc:  # noqa: BLE001
            return job, str(exc)
        finally:
            gpu_queue.put(gpu)

    print(f'  {len(pending)} evaluation job(s), '
          f'workers={min(len(eval_gpus), len(pending))}', flush=True)
    if dry_run:
        for job in pending:
            evaluate(job)
        return failures

    with ThreadPoolExecutor(max_workers=len(eval_gpus)) as pool:
        futures = [pool.submit(evaluate, job) for job in pending]
        for future in as_completed(futures):
            job, error = future.result()
            if error is not None:
                failures.append(f'{job.row}/seed{job.seed}: {error}')
                print(f'[fail] {job.row}/seed{job.seed}: {error}',
                      file=sys.stderr, flush=True)
            else:
                print(f'[done] {job.row}/seed{job.seed}', flush=True)

    for job in jobs:
        if not _eval_complete(
                job.eval_root, expected, job.checkpoint,
                require_macro=True):
            message = (f'{job.row}/seed{job.seed}: '
                       'incomplete or stale eval table')
            if message not in failures:
                failures.append(message)
    return failures


# ---------------------------------------------------------------------------
# Evaluation table readers and summary
# ---------------------------------------------------------------------------
def _manifest_ok(eval_root: Path,
                 checkpoint: Path | None) -> str | None:
    manifest = _read_json(eval_root / 'run_manifest.json')
    if not isinstance(manifest, dict):
        return 'missing run_manifest.json'
    status = manifest.get('status')
    if status not in (None, 'done'):
        return f'evaluation status={status!r}'
    if checkpoint is not None:
        if not checkpoint.is_file():
            return f'missing checkpoint {checkpoint}'
        if not _checkpoint_matches_manifest(eval_root, checkpoint):
            return ('run_manifest.json does not match the current '
                    'checkpoint (stale eval)')
    return None


def _read_macro(eval_root: Path, expected: int,
                checkpoint: Path | None) -> dict:
    error = _manifest_ok(eval_root, checkpoint)
    if error:
        raise ValueError(error)
    path = eval_root / 'tables' / 'macro_metrics.csv'
    if not path.is_file():
        raise ValueError(f'missing {path}')
    metric_keys = {key for key, _, _ in METRICS}
    values = {}
    all_n = None
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            role = str(row.get('role', '')).strip()
            metric = str(row.get('metric', '')).strip().lower()
            if role not in ROLES or metric not in metric_keys:
                continue
            try:
                value = float(row.get('mean'))
                n_datasets = int(float(row.get('n_datasets')))
            except (TypeError, ValueError):
                continue
            if value != value:  # NaN
                continue
            if role == 'All' and metric == 'dice':
                all_n = n_datasets
            values[(role, metric)] = value
    if all_n is None or all_n < int(expected):
        raise ValueError(
            f'{path}: All role covers {all_n} datasets, '
            f'expected at least {expected}')
    missing = [
        (role, metric)
        for role in ROLES for metric in metric_keys
        if (role, metric) not in values
    ]
    if missing:
        raise ValueError(f'{path}: missing role/metric rows {missing}')
    return values


def _read_per_dataset(eval_root: Path, expected: int,
                      checkpoint: Path | None) -> dict:
    error = _manifest_ok(eval_root, checkpoint)
    if error:
        raise ValueError(error)
    path = eval_root / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        raise ValueError(f'missing {path}')
    datasets = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get('dataset', '')).strip()
            if not dataset:
                continue
            entry = {'role': str(row.get('role', '')).strip()}
            ok = True
            for key, _, _ in METRICS:
                try:
                    value = float(row.get(f'{key}_mean'))
                except (TypeError, ValueError):
                    ok = False
                    break
                if value != value:  # NaN
                    ok = False
                    break
                entry[key] = value
            if ok:
                datasets[dataset] = entry
    if len(datasets) < int(expected):
        raise ValueError(
            f'{path}: found {len(datasets)} datasets, '
            f'expected at least {expected}')
    return datasets


def _row_eval_root(row: str, seed: int, run_root: Path,
                   fixed3_root: Path, official_root: Path) -> Path:
    if row == 'dcd_fixed3':
        return fixed3_root / 'eval' / f'seed_{seed}'
    if row == 'nofusion_fixed3':
        return run_root / 'nofusion' / f'seed_{seed}' / 'eval_fixed3'
    if row == 'native_fixed3':
        return (run_root / 'native_control' / f'seed_{seed}' /
                'eval_fixed3')
    if row == 'notopo_fixed3':
        return run_root / 'notopo' / f'seed_{seed}' / 'eval_fixed3'
    if row == 'dcd_gating_fusion':
        return run_root / 'ssh' / f'seed_{seed}' / 'eval_gating_fusion'
    if row == 'native_gating':
        return (run_root / 'native_control' / f'seed_{seed}' /
                'eval_gating')
    if row == 'sacm':
        return official_root / f'seed_{seed}' / 'eval'
    raise ValueError(f'unknown summary row: {row}')


def _row_checkpoint_path(row: str, seed: int, run_root: Path,
                         fixed3_root: Path) -> Path | None:
    """Current checkpoint that an eval row must be scoring.

    ``sacm`` is an external reference table and has no local checkpoint.
    """
    if row == 'dcd_fixed3':
        return (fixed3_root / 'checkpoints' / f'seed_{seed}' /
                FIXED3_CKPT)
    if row == 'nofusion_fixed3':
        return run_root / 'nofusion' / f'seed_{seed}' / FIXED3_CKPT
    if row == 'native_fixed3':
        return (run_root / 'native_control' / f'seed_{seed}' /
                FIXED3_CKPT)
    if row == 'notopo_fixed3':
        return run_root / 'notopo' / f'seed_{seed}' / FIXED3_CKPT
    if row == 'dcd_gating_fusion':
        return run_root / 'ssh' / f'seed_{seed}' / 'best_model.pth'
    if row == 'native_gating':
        return (run_root / 'native_control' / f'seed_{seed}' /
                'best_model.pth')
    if row == 'sacm':
        return None
    raise ValueError(f'unknown summary row: {row}')


def _mean_std(values: list[float]):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def _fmt_value(metric: str, value: float | None) -> str:
    if value is None:
        return '—'
    if metric in ('dice', 'iou', 'cldice'):
        return f'{float(value) * 100.0:.2f}'
    return f'{float(value):.2f}'


def _fmt_delta(metric: str, value: float | None) -> str:
    if value is None:
        return '—'
    if metric in ('dice', 'iou', 'cldice'):
        return f'{float(value) * 100.0:+.2f}'
    return f'{float(value):+.2f}'


def _fmt_mean_std(metric: str, mean: float | None,
                  std: float | None) -> str:
    if mean is None:
        return '—'
    text = _fmt_value(metric, mean)
    if std is not None:
        text += ' ± ' + _fmt_value(metric, std)
    return text


def _build_summary_data(seeds: list[int], run_root: Path,
                        fixed3_root: Path, official_root: Path,
                        expected: int, epochs: int,
                        derive_ok: set[tuple[str, int]]) -> dict:
    per_seed = {}
    errors = {}
    for row in SUMMARY_ROWS:
        per_seed[row] = {}
        for seed in seeds:
            arm = ROW_ARMS.get(row)
            if arm is not None and not _run_complete(
                    run_root / arm / f'seed_{seed}', epochs):
                errors[(row, seed)] = 'training incomplete'
                continue
            if (row in FIXED3_ROWS
                    and (arm, seed) not in derive_ok):
                errors[(row, seed)] = 'fixed3 derivation incomplete'
                continue
            checkpoint = _row_checkpoint_path(
                row, seed, run_root, fixed3_root)
            if checkpoint is not None and not checkpoint.is_file():
                errors[(row, seed)] = f'missing checkpoint {checkpoint}'
                continue
            eval_root = _row_eval_root(
                row, seed, run_root, fixed3_root, official_root)
            try:
                macro = _read_macro(eval_root, expected, checkpoint)
                datasets = _read_per_dataset(
                    eval_root, expected, checkpoint)
            except (OSError, ValueError, csv.Error) as exc:
                errors[(row, seed)] = str(exc)
                continue
            per_seed[row][seed] = {
                'eval_root': str(eval_root),
                'macro': macro,
                'datasets': datasets,
            }

    aggregates = {}
    for row in SUMMARY_ROWS:
        aggregates[row] = {}
        for role in ROLES:
            for key, _, _ in METRICS:
                vals = [
                    item['macro'][(role, key)]
                    for item in per_seed[row].values()
                    if (role, key) in item['macro']
                ]
                mean, std = _mean_std(vals)
                aggregates[row][(role, key)] = (mean, std, len(vals))

    main = aggregates['dcd_fixed3']
    deltas = {}
    for row in SUMMARY_ROWS:
        deltas[row] = {}
        for role in ROLES:
            for key, _, _ in METRICS:
                value = aggregates[row][(role, key)][0]
                ref = main[(role, key)][0]
                deltas[row][(role, key)] = (
                    None if value is None or ref is None else value - ref)

    return {
        'per_seed': per_seed,
        'aggregates': aggregates,
        'deltas': deltas,
        'errors': errors,
    }


def _dataset_comparison(summary: dict, seeds: list[int]) -> list[dict]:
    """Return subject-vs-reference rows for every metric and dataset.

    For ablation rows the subject is the ablation and the reference is DCD
    fixed3.  For the ``sacm`` row the subject is DCD fixed3 and the reference
    is SACM, so ``better`` always means the subject is better.
    """
    per_seed = summary['per_seed']
    rows = []
    for row in SUMMARY_ROWS:
        if row == 'dcd_fixed3':
            continue
        for seed in seeds:
            main = per_seed['dcd_fixed3'].get(seed)
            other = per_seed[row].get(seed)
            if not main or not other:
                continue
            if row == 'sacm':
                subject = main
                reference = other
            else:
                subject = other
                reference = main
            for dataset, entry in subject['datasets'].items():
                ref = reference['datasets'].get(dataset)
                if not ref:
                    continue
                for key, _, higher_better in METRICS:
                    delta = entry[key] - ref[key]
                    if abs(delta) <= 1e-12:
                        outcome = 0
                    else:
                        subject_better = (
                            delta > 0 if higher_better else delta < 0)
                        outcome = 1 if subject_better else -1
                    rows.append({
                        'row': row,
                        'subject': (
                            'dcd_fixed3' if row == 'sacm' else row),
                        'reference': (
                            'sacm' if row == 'sacm' else 'dcd_fixed3'),
                        'seed': seed,
                        'dataset': dataset,
                        'role': entry['role'],
                        'metric': key,
                        'value': entry[key],
                        'reference_value': ref[key],
                        'delta': delta,
                        'outcome': outcome,
                    })
    return rows


def _dataset_aggregate(comparison: list[dict]) -> list[dict]:
    grouped = {}
    for item in comparison:
        key = (item['row'], item['dataset'], item['role'],
               item['metric'])
        grouped.setdefault(key, []).append(item)
    rows = []
    for (row, dataset, role, metric), items in grouped.items():
        deltas = [item['delta'] for item in items]
        values = [item['value'] for item in items]
        references = [item['reference_value'] for item in items]
        mean, std = _mean_std(deltas)
        rows.append({
            'row': row,
            'subject': items[0]['subject'],
            'reference': items[0]['reference'],
            'dataset': dataset,
            'role': role,
            'metric': metric,
            'n_seeds': len(items),
            'delta_mean': mean,
            'delta_std': std,
            'value_mean': statistics.fmean(values),
            'reference_mean': statistics.fmean(references),
            'wins': sum(1 for item in items if item['outcome'] > 0),
            'losses': sum(1 for item in items if item['outcome'] < 0),
            'ties': sum(1 for item in items if item['outcome'] == 0),
        })
    rows.sort(key=lambda item: (item['row'], item['metric'],
                                -abs(item['delta_mean'])))
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(config: dict, seeds: list[int], epochs: int,
                            expected: int, summary: dict,
                            aggregate: list[dict], output_root: Path,
                            official_ok: bool, missing: list[str]) -> None:
    aggregates = summary['aggregates']
    deltas = summary['deltas']
    lines = [
        '# DCD 800 — main run and component ablation',
        '',
        f'- Config: `{config.get("name")}`',
        f'- Seeds: `{seeds}`',
        f'- Epochs: {epochs}',
        f'- Evaluation: {expected} protocol test datasets; macro weights '
        'datasets equally.',
        '- Deployment: derived fixed candidate 3 '
        '(`head_select=fixed`, `deployment_head=3`, `head_fusion=0`) '
        'unless a row is marked as a deployment reference.',
        '- Dice / IoU / clDice are percentages; HD95 is pixels '
        '(lower is better).',
        '',
        '## 1. Fixed candidate-3 deployment (All role)',
        '',
        '| Method | Dice | IoU | clDice | HD95 | seeds | status |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in ('dcd_fixed3', *ABLATION_ROWS):
        cells = []
        for key, _, _ in METRICS:
            mean, std, _ = aggregates[row][('All', key)]
            cells.append(_fmt_mean_std(key, mean, std))
        n_seeds = max(
            [aggregates[row][('All', key)][2]
             for key, _, _ in METRICS] or [0])
        status = 'OK' if n_seeds == len(seeds) else (
            'PARTIAL' if n_seeds else 'MISSING')
        lines.append(
            f'| {ROW_LABELS[row]} | ' + ' | '.join(cells)
            + f' | {n_seeds}/{len(seeds)} | {status} |')

    lines += [
        '',
        '## 2. Component and deployment deltas against DCD fixed3 '
        '(All role)',
        '',
        '| Method | ΔDice | ΔIoU | ΔclDice | ΔHD95 |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in (*ABLATION_ROWS, *REFERENCE_ROWS):
        cells = [
            _fmt_delta(key, deltas[row][('All', key)])
            for key, _, _ in METRICS
        ]
        lines.append(f'| {ROW_LABELS[row]} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 2b. Component deltas by role',
        '',
        '| Role | Method | ΔDice | ΔIoU | ΔclDice | ΔHD95 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for role in ('Seen', 'Unseen', 'Novel'):
        for row in ABLATION_ROWS:
            cells = [
                _fmt_delta(key, deltas[row][(role, key)])
                for key, _, _ in METRICS
            ]
            lines.append(
                f'| {role} | {ROW_LABELS[row]} | '
                + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 3. Deployment references (same checkpoints, no retraining)',
        '',
        '| Row | Dice | IoU | clDice | HD95 | seeds | status |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in REFERENCE_ROWS:
        cells = []
        for key, _, _ in METRICS:
            mean, std, _ = aggregates[row][('All', key)]
            cells.append(_fmt_mean_std(key, mean, std))
        n_seeds = max(
            [aggregates[row][('All', key)][2]
             for key, _, _ in METRICS] or [0])
        status = 'OK' if n_seeds == len(seeds) else (
            'PARTIAL' if n_seeds else 'MISSING')
        lines.append(
            f'| {ROW_LABELS[row]} | ' + ' | '.join(cells)
            + f' | {n_seeds}/{len(seeds)} | {status} |')

    lines += [
        '',
        '## 4. DCD fixed3 versus SACM official',
        '',
        '| Role | Dice | ΔDice | IoU | ΔIoU | clDice | ΔclDice | '
        'HD95 | ΔHD95 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for role in ROLES:
        cells = []
        for key, _, _ in METRICS:
            value = aggregates['dcd_fixed3'][(role, key)][0]
            delta = deltas['sacm'][(role, key)]
            cells += [_fmt_value(key, value), _fmt_delta(key, delta)]
        lines.append(f'| {role} | ' + ' | '.join(cells) + ' |')
    if not official_ok:
        lines += [
            '',
            'SACM rows are missing or incomplete; the delta columns are '
            'therefore not a complete official comparison.',
        ]

    lines += [
        '',
        '## 5. Per-seed All-role macro: DCD fixed3 and SACM',
        '',
        '| Seed | DCD Dice | DCD IoU | DCD clDice | DCD HD95 | '
        'SACM Dice | SACM IoU | SACM clDice | SACM HD95 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for seed in seeds:
        dcd = summary['per_seed']['dcd_fixed3'].get(seed)
        sacm = summary['per_seed']['sacm'].get(seed)
        cells = []
        for data in (dcd, sacm):
            if not data:
                cells += ['—', '—', '—', '—']
            else:
                cells += [
                    _fmt_value('dice', data['macro'][('All', 'dice')]),
                    _fmt_value('iou', data['macro'][('All', 'iou')]),
                    _fmt_value('cldice', data['macro'][('All', 'cldice')]),
                    _fmt_value('hd95', data['macro'][('All', 'hd95')]),
                ]
        lines.append(f'| {seed} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 6. Dataset-level wins/losses (subject vs reference)',
        '',
        '| Comparison | Metric | wins | losses | ties |',
        '|---|---|---:|---:|---:|',
    ]
    for row, label in (
            ('sacm', 'DCD fixed3 vs SACM'),
            ('nofusion_fixed3', 'No-fusion vs DCD'),
            ('native_fixed3', 'Native fixed3 vs DCD'),
            ('notopo_fixed3', 'No-topology vs DCD')):
        for key, _, _ in METRICS:
            items = [item for item in aggregate
                     if item['row'] == row and item['metric'] == key]
            wins = sum(item['wins'] for item in items)
            losses = sum(item['losses'] for item in items)
            ties = sum(item['ties'] for item in items)
            if items:
                lines.append(
                    f'| {label} | {key} | {wins} | {losses} | {ties} |')

    lines += [
        '',
        '## 7. Largest dataset changes: DCD fixed3 vs SACM',
        '',
        '| Metric | Direction | Dataset | Role | Δ mean | wins/losses |',
        '|---|---|---|---|---:|---:|',
    ]
    for key, _, higher_better in METRICS:
        items = [item for item in aggregate
                 if item['row'] == 'sacm' and item['metric'] == key]
        if key == 'hd95':
            better_items = [
                item for item in items
                if item['delta_mean'] is not None
                and item['delta_mean'] < 0]
            worse_items = [
                item for item in items
                if item['delta_mean'] is not None
                and item['delta_mean'] > 0]
        else:
            better_items = [
                item for item in items
                if item['delta_mean'] is not None
                and item['delta_mean'] > 0]
            worse_items = [
                item for item in items
                if item['delta_mean'] is not None
                and item['delta_mean'] < 0]
        better = sorted(
            better_items, key=lambda item: item['delta_mean'],
            reverse=higher_better)[:5]
        worse = sorted(
            worse_items, key=lambda item: item['delta_mean'],
            reverse=not higher_better)[:5]
        for direction, group in (('better', better), ('worse', worse)):
            for item in group:
                lines.append(
                    f'| {key} | {direction} | {item["dataset"]} | '
                    f'{item["role"]} | {_fmt_delta(key, item["delta_mean"])} '
                    f'| {item["wins"]}/{item["losses"]} |')

    if missing:
        lines += [
            '',
            '## Incomplete jobs',
            '',
            *[f'- {item}' for item in missing],
        ]

    errors = summary['errors']
    if errors:
        lines += [
            '',
            '## Missing or invalid eval tables',
            '',
            *[f'- {row}/seed{seed}: {message}'
              for (row, seed), message in sorted(errors.items())],
        ]

    lines += [
        '',
        '## Notes and limitations',
        '',
        '- Every arm is trained with `head-select=gating`; fixed candidate 3 '
        'is an inference-only deployment derived from the same weights.',
        '- The native control has no specialized role losses and no fusion; '
        'its fixed3 checkpoint is derived with `--allow-native`.',
        '- Gating / gating+fusion rows reuse the trained checkpoints without '
        'retraining and are deployment references, not additional arms.',
        '- For the `sacm` comparison row, the CSV subject is DCD fixed3 and '
        'the reference is SACM; for ablation rows, the subject is the '
        'ablation and the reference is DCD fixed3.',
        '- The 300-epoch dense/structure scope factorial is a separate '
        'budget study and is not part of this 800-epoch table.',
        '- HD95 is lower-better. The summary does not run a paired '
        'significance test; use the standard fixed3-vs-SACM report for its '
        'Wilcoxon table.',
        '',
    ]
    (output_root / 'summary.md').write_text(
        '\n'.join(lines), encoding='utf-8')


def _write_summary_html(seeds: list[int], epochs: int,
                        expected: int, summary: dict,
                        output_root: Path) -> None:
    aggregates = summary['aggregates']
    deltas = summary['deltas']
    table_rows = []
    for row in ('dcd_fixed3', *ABLATION_ROWS, *REFERENCE_ROWS, 'sacm'):
        cells = [f'<td>{html.escape(ROW_LABELS[row])}</td>']
        for key, _, _ in METRICS:
            mean, std, _n = aggregates[row][('All', key)]
            cells.append(
                f'<td>{html.escape(_fmt_mean_std(key, mean, std))}</td>')
        n_seeds = max(
            [aggregates[row][('All', key)][2]
             for key, _, _ in METRICS] or [0])
        cells.append(f'<td>{n_seeds}/{len(seeds)}</td>')
        table_rows.append('<tr>' + ''.join(cells) + '</tr>')

    delta_rows = []
    for row in (*ABLATION_ROWS, *REFERENCE_ROWS):
        cells = [f'<td>{html.escape(ROW_LABELS[row])}</td>']
        for key, _, _ in METRICS:
            text = _fmt_delta(key, deltas[row][('All', key)])
            cells.append(f'<td>{html.escape(text)}</td>')
        delta_rows.append('<tr>' + ''.join(cells) + '</tr>')

    body = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>DCD 800 full ablation</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:32px;color:#111}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px}}
table{{border-collapse:collapse;margin-top:12px}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f2f2ef}}
.note{{color:#555;max-width:980px}}
</style></head><body>
<h1>DCD 800: main run and component ablation</h1>
<p class="note">Seeds {seeds}, {epochs} epochs, {expected} test datasets.
Dice / IoU / clDice are percentages; HD95 is pixels (lower is better).
All DCD rows except the two deployment references use the derived fixed
candidate 3. The native control is derived with <code>--allow-native</code>.
The SACM row is the official reference.</p>
<h2>Fixed candidate-3 deployment and references</h2>
<table><thead><tr><th>Method</th><th>Dice</th><th>IoU</th><th>clDice</th>
<th>HD95</th><th>seeds</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<h2>Deltas against DCD fixed3, All role</h2>
<table><thead><tr><th>Method</th><th>ΔDice</th><th>ΔIoU</th>
<th>ΔclDice</th><th>ΔHD95</th></tr></thead>
<tbody>{''.join(delta_rows)}</tbody></table>
<p class="note">See <code>summary.md</code>, <code>summary_macro.csv</code>,
<code>summary_per_seed.csv</code>, and <code>dataset_comparison.csv</code>
for the role-wise and dataset-wise results.</p>
</body></html>'''
    (output_root / 'summary.html').write_text(body, encoding='utf-8')


def _write_summary(config: dict, seeds: list[int], epochs: int,
                   expected: int, summary: dict, comparison: list[dict],
                   aggregate: list[dict], output_root: Path,
                   official_ok: bool, missing: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'config_snapshot.yaml').write_text(
        yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    macro_rows = []
    for row in SUMMARY_ROWS:
        for role in ROLES:
            for key, _, _ in METRICS:
                mean, std, n = summary['aggregates'][row][(role, key)]
                macro_rows.append({
                    'row': row,
                    'label': ROW_LABELS[row],
                    'role': role,
                    'metric': key,
                    'mean': mean,
                    'std': std,
                    'n_seeds': n,
                    'n_requested': len(seeds),
                    'status': ('OK' if n == len(seeds) else
                               ('PARTIAL' if n else 'MISSING')),
                })
    _write_csv(
        output_root / 'summary_macro.csv', macro_rows,
        ['row', 'label', 'role', 'metric', 'mean', 'std', 'n_seeds',
         'n_requested', 'status'])

    per_seed_rows = []
    for row in SUMMARY_ROWS:
        for seed in seeds:
            item = summary['per_seed'][row].get(seed)
            for role in ROLES:
                for key, _, _ in METRICS:
                    value = (item['macro'][(role, key)]
                             if item else None)
                    per_seed_rows.append({
                        'row': row,
                        'seed': seed,
                        'role': role,
                        'metric': key,
                        'value': value,
                    })
    _write_csv(
        output_root / 'summary_per_seed.csv', per_seed_rows,
        ['row', 'seed', 'role', 'metric', 'value'])

    delta_rows = []
    for row in SUMMARY_ROWS:
        for role in ROLES:
            for key, _, _ in METRICS:
                delta_rows.append({
                    'row': row,
                    'role': role,
                    'metric': key,
                    'delta_vs_dcd_fixed3': summary['deltas'][row][
                        (role, key)],
                })
    _write_csv(
        output_root / 'summary_deltas.csv', delta_rows,
        ['row', 'role', 'metric', 'delta_vs_dcd_fixed3'])

    _write_csv(
        output_root / 'dataset_comparison.csv', comparison,
        ['row', 'subject', 'reference', 'seed', 'dataset', 'role',
         'metric', 'value', 'reference_value', 'delta', 'outcome'])
    _write_csv(
        output_root / 'dataset_aggregate.csv', aggregate,
        ['row', 'subject', 'reference', 'dataset', 'role', 'metric',
         'n_seeds', 'delta_mean', 'delta_std', 'value_mean',
         'reference_mean', 'wins', 'losses', 'ties'])

    _write_summary_markdown(
        config, seeds, epochs, expected, summary, aggregate,
        output_root, official_ok, missing)
    _write_summary_html(
        seeds, epochs, expected, summary, output_root)

    payload = {
        'name': config.get('name'),
        'seeds': seeds,
        'epochs': epochs,
        'expected_datasets': expected,
        'official_ok': official_ok,
        'missing': missing,
        'errors': {
            f'{row}/seed{seed}': message
            for (row, seed), message in sorted(
                summary['errors'].items())
        },
        'rows': {
            row: {
                role: {
                    key: {
                        'mean': summary['aggregates'][row][(role, key)][0],
                        'std': summary['aggregates'][row][(role, key)][1],
                        'n_seeds': summary['aggregates'][row][(role, key)][2],
                    }
                    for key, _, _ in METRICS
                }
                for role in ROLES
            }
            for row in SUMMARY_ROWS
        },
    }
    (output_root / 'summary.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _write_standard_skip_notice(output_root: Path, reason: str) -> None:
    """Move old report files aside and leave a clear skip marker.

    A skipped/failed standard summary must not leave a previous
    ``summary.md``/``summary.json``/CSV set at the documented path, because a
    morning reader could mistake it for the current run.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    previous = output_root / 'previous'
    for path in list(output_root.iterdir()):
        if path.name in ('SKIPPED.txt', 'previous') or path.is_dir():
            continue
        previous.mkdir(parents=True, exist_ok=True)
        path.replace(previous / path.name)
    (output_root / 'SKIPPED.txt').write_text(
        reason.strip() + '\n', encoding='utf-8')


def _run_standard_summary(fixed3_root: Path, official_root: Path,
                          seeds: list[int], epochs: int, expected: int,
                          output_root: Path, log_dir: Path,
                          dry_run: bool) -> bool:
    command = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' /
            'summarize_fixed3_seeds.py'),
        '--fixed3-root', str(fixed3_root),
        '--official-root', str(official_root),
        '--seeds', *[str(seed) for seed in seeds],
        '--epochs', str(epochs),
        '--expected-datasets', str(expected),
        '--fixed3-checkpoint-name', FIXED3_CKPT,
        '--output-root', str(output_root),
    ]
    try:
        code = _run_command_stream(
            command, log_dir / 'fixed3_vs_sacm.log', dry_run)
    except OSError as exc:
        print(f'[warn] cannot run standard summary: {exc}',
              file=sys.stderr)
        return False
    return code == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(env.get('CONFIG', str(DEFAULT_CONFIG))))
    parser.add_argument('--seeds', nargs='*', default=None)
    parser.add_argument(
        '--epochs', type=int,
        default=int(env.get('EPOCHS', '800')))
    parser.add_argument(
        '--train-gpus',
        default=env.get('TRAIN_GPUS', env.get('GPU_LIST',
                                               '0,1,2,3,4,5,6,7')))
    parser.add_argument(
        '--eval-gpus',
        default=env.get('EVAL_GPUS', env.get('GPU_LIST',
                                             '0,1,2,3,4,5,6,7')))
    parser.add_argument('--skip-train', action='store_true',
                        default=_as_bool(env.get('SKIP_TRAIN', '0')))
    parser.add_argument('--skip-eval', action='store_true',
                        default=_as_bool(env.get('SKIP_EVAL', '0')))
    parser.add_argument('--skip-standard-summary', action='store_true',
                        default=_as_bool(
                            env.get('SKIP_STANDARD_SUMMARY', '0')))
    parser.add_argument('--dry-run', action='store_true',
                        default=_as_bool(env.get('DRY_RUN', '0')))
    parser.add_argument('--no-resume', action='store_true',
                        default=not _as_bool(env.get('RESUME', '1')))
    parser.add_argument('--force-derive', action='store_true',
                        default=_as_bool(env.get('FORCE_DERIVE', '0')))
    parser.add_argument('--force-eval', action='store_true',
                        default=_as_bool(env.get('FORCE_EVAL', '0')))
    parser.add_argument('--force', action='store_true',
                        help='force derivation and evaluation; combine with '
                             'RESUME=0 to retrain')
    parser.add_argument('--allow-busy', action='store_true',
                        default=_as_bool(env.get('ALLOW_BUSY', '0')))
    parser.add_argument(
        '--max-used-mib', type=int,
        default=int(env.get('MAX_USED_MIB', '2000')))
    parser.add_argument(
        '--min-free-gb', type=float,
        default=float(env.get('MIN_FREE_GB', '80')))
    parser.add_argument(
        '--allow-low-disk', action='store_true',
        default=_as_bool(env.get('ALLOW_LOW_DISK', '0')))
    parser.add_argument(
        '--require-official', action='store_true',
        default=_as_bool(env.get('REQUIRE_OFFICIAL', '1')))
    parser.add_argument(
        '--eval-workers', type=int,
        default=(int(env['EVAL_WORKERS'])
                 if env.get('EVAL_WORKERS') else None))
    parser.add_argument(
        '--official-root',
        default=env.get('OFFICIAL_ROOT', 'results/runs/sacm/default'))
    parser.add_argument(
        '--run-root', default=env.get('RUN_ROOT'))
    parser.add_argument(
        '--summary-root',
        default=env.get('SUMMARY_ROOT',
                        'results/ablations/dcd_full_800'))
    parser.add_argument('--log-dir', default=env.get('LOG_DIR'))
    args = parser.parse_args(argv)

    force_derive = bool(args.force or args.force_derive)
    force_eval = bool(args.force or args.force_eval or force_derive)
    # RESUME=0 means the checkpoints are about to change.  Never reuse an
    # eval table or a derived checkpoint whose provenance may predate them.
    if args.no_resume:
        force_derive = True
        force_eval = True

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_config(config_path)

    seeds = _parse_seeds(
        args.seeds if args.seeds is not None else env.get('SEEDS'),
        config.get('seeds', [0, 1, 2, 3]))
    train_gpus = _parse_gpus(args.train_gpus)
    eval_gpus = _parse_gpus(args.eval_gpus)
    epochs = int(args.epochs)
    if epochs <= 0:
        raise SystemExit('epochs must be positive')
    expected = int(env.get(
        'EXPECTED_DATASETS', config.get('expected_datasets', 20)))
    if expected <= 0:
        raise SystemExit('expected_datasets must be positive')

    config_run_root = _resolve(config.get('output_root'))
    if args.run_root:
        run_root = _resolve(args.run_root)
        if run_root != config_run_root:
            raise SystemExit(
                'RUN_ROOT must match config output_root: '
                f'{run_root} != {config_run_root}')
    else:
        run_root = config_run_root
    fixed3_root = run_root / 'fixed3'
    official_root = _resolve(args.official_root)
    summary_root = _resolve(args.summary_root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = (_resolve(args.log_dir) if args.log_dir else
               _resolve(f'results/logs/dcd_full_800_{stamp}'))
    use_wandb = _as_bool((config.get('wandb') or {}).get('use_wandb', True))

    print('=' * 68)
    print(' DCD full 800-epoch run + component ablation')
    print(f' config:       {config_path}')
    print(f' seeds:        {seeds}')
    print(f' epochs:       {epochs}')
    print(f' train GPUs:   {train_gpus}')
    print(f' eval GPUs:    {eval_gpus}')
    print(f' run root:     {run_root}')
    print(f' fixed3 root:  {fixed3_root}')
    print(f' official:     {official_root} (read-only)')
    print(f' summary:      {summary_root}')
    print(f' log dir:      {log_dir}')
    print(f' arms:         {list(ARMS)}')
    print(' deployment:   head_select=fixed, deployment_head=3, '
          'head_fusion=0')
    if args.dry_run:
        print(' mode:         DRY RUN')
    print('=' * 68)

    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    _check_protocols(seeds, expected, args.dry_run)
    sam = REPO_ROOT / str(
        (config.get('common_train_args') or {}).get(
            'sam-checkpoint', 'checkpoints/sam_vit_l_0b3195.pth'))
    if not sam.is_file() and not args.dry_run:
        raise SystemExit(f'missing SAM checkpoint: {sam}')
    _check_official(
        official_root, seeds, expected,
        bool(args.require_official) and not args.dry_run)
    official_budget_issues = _official_budget_issues(
        official_root, seeds, epochs)
    if official_budget_issues:
        message = ('official SACM budget/provenance mismatch: '
                   + '; '.join(official_budget_issues))
        if bool(args.require_official) and not args.dry_run:
            raise SystemExit(
                message + '; complete SACM at the same epoch budget or set '
                          'REQUIRE_OFFICIAL=0')
        print(f'[warn] {message}', file=sys.stderr)
    if not args.dry_run:
        free_gb = shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            message = (f'disk free {free_gb:.1f} GiB < '
                       f'MIN_FREE_GB={args.min_free_gb:g}')
            if args.allow_low_disk:
                print(f'[warn] {message}', file=sys.stderr)
            else:
                raise SystemExit(
                    f'{message}; free space or set ALLOW_LOW_DISK=1')
        print(f'disk free: {free_gb:.1f} GiB')
        _check_existing_recipes(
            config, run_root, seeds, epochs, not args.no_resume,
            args.skip_train)
        _check_gpus(
            list(dict.fromkeys([*train_gpus, *eval_gpus])),
            args.max_used_mib, args.allow_busy, args.dry_run)

    status = 0

    # 1. Train all arms.
    if args.skip_train:
        print('[1/5] training skipped (SKIP_TRAIN=1)')
    else:
        print(f'[1/5] training {len(ARMS)} arms x {len(seeds)} seeds '
              f'at {epochs} epochs')
        command = _train_command(
            config_path, seeds, train_gpus, epochs,
            not args.no_resume, args.dry_run)
        code = _run_command_stream(
            command, log_dir / 'train.log', args.dry_run)
        if code != 0:
            print(f'[warn] training driver exited {code}; continuing with '
                  'completed runs', file=sys.stderr)
            status = 1

    # 2. Derive fixed3 checkpoints.
    print('[2/5] deriving fixed candidate-3 checkpoints')
    derive_failures = []
    derive_ok: set[tuple[str, int]] = set()
    for arm in ARMS:
        for seed in seeds:
            arm_root, source, output = _arm_paths(run_root, arm, seed)
            if not args.dry_run and not _run_complete(arm_root, epochs):
                derive_failures.append(
                    f'{arm}/seed{seed}: training incomplete')
                continue
            if not args.dry_run and not source.is_file():
                derive_failures.append(
                    f'{arm}/seed{seed}: missing {source}')
                continue
            if arm == 'ssh':
                output = _main_fixed3_paths(
                    run_root, seed, fixed3_root)[0]
            command = _derive_command(
                source, output, arm, force_derive)
            code = _run_command_stream(
                command, log_dir / 'derive' / f'{arm}_seed{seed}.log',
                args.dry_run)
            if code != 0:
                derive_failures.append(
                    f'{arm}/seed{seed}: derivation exited {code}')
                continue
            if not args.dry_run and (
                    not output.is_file()
                    or output.stat().st_size <= 0):
                derive_failures.append(
                    f'{arm}/seed{seed}: derivation produced no checkpoint')
                continue
            derive_ok.add((arm, seed))
    if derive_failures:
        status = 1
        print('[warn] derivation incomplete:', file=sys.stderr)
        for item in derive_failures:
            print(f'  {item}', file=sys.stderr)

    # 3. Evaluate the fixed3 arms and deployment references.
    jobs, job_problems = _build_eval_jobs(
        config, run_root, fixed3_root, seeds, epochs, derive_ok,
        assume_complete=args.dry_run)
    if job_problems:
        status = 1
    missing = list(dict.fromkeys(
        list(job_problems) + list(derive_failures)
        + list(official_budget_issues)))
    if args.skip_eval:
        print('[3/5] evaluation skipped (SKIP_EVAL=1)')
        eval_failures = []
    else:
        print(f'[3/5] evaluating {len(jobs)} deployment row(s) on '
              f'{expected} datasets')
        eval_failures = _run_eval_jobs(
            jobs, eval_gpus, config, log_dir, args.no_resume,
            force_eval, args.eval_workers, use_wandb, args.dry_run,
            expected)
        if eval_failures:
            status = 1
            print('[warn] evaluation incomplete:', file=sys.stderr)
            for item in eval_failures:
                print(f'  {item}', file=sys.stderr)
        missing += eval_failures

    if args.dry_run:
        print('[4/5] dry-run: summary not written')
        print('[5/5] dry-run: standard fixed3-vs-SACM report not written')
        return 0

    # 4. Multi-arm summary.
    print('[4/5] writing multi-arm summary')
    summary = _build_summary_data(
        seeds, run_root, fixed3_root, official_root, expected,
        epochs, derive_ok)
    comparison = _dataset_comparison(summary, seeds)
    aggregate = _dataset_aggregate(comparison)
    official_ok = (
        not official_budget_issues
        and all(seed in summary['per_seed']['sacm'] for seed in seeds))
    if summary['errors']:
        status = 1
        print('[warn] summary is incomplete:', file=sys.stderr)
        for (row, seed), message in sorted(summary['errors'].items()):
            print(f'  {row}/seed{seed}: {message}', file=sys.stderr)
    if missing:
        status = 1
    if not official_ok:
        status = 1
        print('[warn] official SACM rows are incomplete', file=sys.stderr)
    _write_summary(
        config, seeds, epochs, expected, summary, comparison, aggregate,
        summary_root, official_ok, missing)
    print(f'  {summary_root}/summary.md')
    print(f'  {summary_root}/summary_macro.csv')
    print(f'  {summary_root}/summary_per_seed.csv')
    print(f'  {summary_root}/dataset_comparison.csv')

    # 5. Standard fixed3-vs-SACM report (uses the same eval tables).
    standard_root = summary_root / 'fixed3_vs_sacm'
    main_fixed3_errors = [
        seed for seed in seeds if ('dcd_fixed3', seed) in summary['errors']
    ]
    if args.skip_standard_summary:
        print('[5/5] standard fixed3-vs-SACM report skipped')
    elif main_fixed3_errors:
        print('[5/5] standard fixed3-vs-SACM report skipped: '
              'main fixed3 eval is incomplete/stale')
        _write_standard_skip_notice(
            standard_root,
            'skipped: main fixed3 eval is incomplete or stale for seed(s) '
            + ', '.join(str(seed) for seed in main_fixed3_errors))
        status = 1
    elif not official_ok:
        print('[5/5] standard fixed3-vs-SACM report skipped: '
              'SACM tables incomplete')
        _write_standard_skip_notice(
            standard_root,
            'skipped: official SACM eval tables are incomplete')
        status = 1
    else:
        print('[5/5] writing standard fixed3-vs-SACM report')
        ok = _run_standard_summary(
            fixed3_root, official_root, seeds, epochs, expected,
            standard_root, log_dir, args.dry_run)
        if ok:
            try:
                (standard_root / 'SKIPPED.txt').unlink()
            except FileNotFoundError:
                pass
            print(f'  {standard_root}/summary.md')
        else:
            print('[warn] standard fixed3-vs-SACM report failed; see '
                  f'{log_dir}/fixed3_vs_sacm.log', file=sys.stderr)
            _write_standard_skip_notice(
                standard_root,
                'failed: see fixed3_vs_sacm.log in the driver log directory')
            status = 1

    print('=' * 68)
    if status == 0:
        print(' DONE')
    else:
        print(f' FINISHED WITH FAILURES (exit {status})')
    print(f' log dir:    {log_dir}')
    print(f' training:   {run_root}/<arm>/seed_*/')
    print(f' fixed3:     {fixed3_root}/')
    print(f' summary:    {summary_root}/summary.md')
    print('=' * 68)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
