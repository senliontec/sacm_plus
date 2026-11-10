#!/usr/bin/env python3
"""Config-driven runner for the SACM-STC scope/structure ablation.

The experiment matrix lives in
``configs/ablations/sacm_stc_scope_structure.yaml``.  This runner only:

* validates the YAML and protocol seeds;
* builds ``baseline_zoo.run_one``, ``run_baseline_test.py`` and
  ``diagnose_sacm_stc_candidates.py`` commands from the arm definitions;
* distributes jobs round-robin over the requested GPUs;
* skips completed stages unless ``--no-resume`` is passed;
* invokes the paired-analysis script after an ``all``/``analyze`` stage.

It intentionally contains no arm names or model hyperparameters.  Add or
change an arm in YAML, not here.

Stages:
    audit     candidate audit of the existing full checkpoints (no training)
    train     train every selected arm/seed
    eval      evaluate every selected arm/seed
    diagnose  candidate diagnostics for every selected arm/seed
    all       train -> eval -> diagnose -> analyze
    analyze   paired contrast tables only

Example (server):
    bash scripts/run_sacm_stc_ablation.sh --stage audit --gpus 4,5,6,7
    bash scripts/run_sacm_stc_ablation.sh --stage all --gpus 4,5,6,7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / 'configs' / 'ablations' /
                  'sacm_stc_scope_structure.yaml')
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.protocols import load_protocol  # noqa: E402


def _resolve(path_text) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_text(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f'{path}: config must be a mapping')
    if int(config.get('schema_version', 0)) != 1:
        raise SystemExit(f'{path}: unsupported schema_version')
    for key in ('name', 'protocol', 'method', 'seeds', 'arms'):
        if key not in config:
            raise SystemExit(f'{path}: missing required field {key!r}')
    if str(config['method']) != 'sacm_stc':
        raise SystemExit(
            'this runner currently supports method=sacm_stc; '
            f'got {config["method"]!r}')
    if not isinstance(config['arms'], dict) or not config['arms']:
        raise SystemExit(f'{path}: arms must be a non-empty mapping')
    common = config.get('common_train_args') or {}
    if not isinstance(common, dict):
        raise SystemExit(f'{path}: common_train_args must be a mapping')
    for arm, spec in config['arms'].items():
        if not isinstance(spec, dict):
            raise SystemExit(f'{path}: arm {arm!r} must be a mapping')
        if not isinstance(spec.get('args'), dict):
            raise SystemExit(f'{path}: arm {arm!r}.args must be a mapping')
    for contrast in config.get('contrasts') or []:
        for key in ('id', 'left', 'right'):
            if key not in contrast:
                raise SystemExit(
                    f'{path}: contrast missing {key!r}: {contrast}')
        for key in ('left', 'right'):
            if contrast[key] not in config['arms']:
                raise SystemExit(
                    f'{path}: contrast {contrast["id"]} references '
                    f'unknown arm {contrast[key]!r}')
    return config


def _parse_seeds(values, protocol, default):
    raw = values if values is not None else default
    if raw is None:
        raw = list(protocol.seeds)
    # argparse nargs='*' may yield one quoted string ("0 1 2 3") or one
    # comma-separated item ("0,1,2,3").  Accept both forms.
    seeds = []
    for value in raw:
        for item in str(value).replace(',', ' ').split():
            seeds.append(int(item))
    unknown = sorted(set(seeds) - set(protocol.seeds))
    if unknown:
        raise SystemExit(
            f'seed(s) {unknown} not in protocol {protocol.name} '
            f'{list(protocol.seeds)}')
    return list(dict.fromkeys(seeds))


def _parse_arms(values, config):
    if values:
        arms = [item for value in values
                for item in str(value).replace(',', ' ').split()]
    else:
        arms = list(config['arms'])
    unknown = [arm for arm in arms if arm not in config['arms']]
    if unknown:
        raise SystemExit(
            f'unknown arm(s) {unknown}; configured: {list(config["arms"])}')
    return list(dict.fromkeys(arms))


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in str(value).split(',') if item.strip()]
    if not gpus:
        raise SystemExit('--gpus cannot be empty')
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f'duplicate GPU ids: {gpus}')
    return gpus


def _bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _cli_args(mapping: dict) -> list[str]:
    """Flatten a YAML mapping into CLI flags without hardcoding arm names."""
    args = []
    for key, value in mapping.items():
        flag = f'--{str(key).replace("_", "-")}'
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                args.extend([flag, '0'])
            continue
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def _train_command(config, arm: str, seed: int, gpu: str,
                   split_root: Path, run_root: Path, epochs: int,
                   resume: bool) -> list[str]:
    arm_args = dict(config.get('common_train_args') or {})
    arm_args.update(config['arms'][arm]['args'])
    wandb = config.get('wandb') or {}
    command = [
        sys.executable, '-m', 'baseline_zoo.run_one',
        '--model', 'sacm_stc', '--stage', 'train',
        '--data-root', str(split_root),
        '--workdir', str(run_root),
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--device', f'cuda:{gpu}',
    ]
    command += _cli_args(arm_args)
    # Runtime/Core args use underscores, while adapter args use hyphens.
    # Keep the W&B options explicit instead of running them through the
    # adapter-flag flattener.
    if _bool_arg(wandb.get('use_wandb', True)):
        command += [
            '--use_wandb',
            '--wandb_project', str(wandb.get('project', 'sacm')),
            '--wandb_entity', str(wandb.get('entity', 'buaazqk')),
            '--wandb_group', str(wandb.get('group', config['name'])),
            '--wandb_name', f'{config["name"]}-{arm}-seed{seed}',
            '--wandb_tags', str(wandb.get('tags', 'sacm_stc,ablation')),
            '--wandb_job_type', 'train',
        ]
    else:
        command += ['--use_wandb', '0']
    if resume:
        command.append('--resume')
    return command


def _eval_checkpoint(config) -> str:
    """Checkpoint evaluated after the fixed training budget.

    The ablation default is the final epoch checkpoint, so "300 epochs then
    test" is literal; set ``eval_checkpoint: best_model.pth`` to use the
    val-Dice-selected checkpoint instead.
    """
    return str(config.get('eval_checkpoint', 'best_model.pth'))


def _eval_command(config, arm: str, seed: int, gpu: str,
                  protocol_path: Path, run_root: Path) -> list[str]:
    wandb = config.get('wandb') or {}
    use_wandb = _bool_arg(wandb.get('use_wandb', True))
    command = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' / 'run_baseline_test.py'),
        '--method', 'sacm_stc',
        '--protocol', str(protocol_path),
        '--seed', str(seed),
        '--variant', arm,
        '--checkpoint', str(run_root / _eval_checkpoint(config)),
        '--pred-root', str(run_root / 'predictions'),
        '--paper-root', str(run_root),
        '--device', f'cuda:{gpu}',
        '--use_wandb', 'true' if use_wandb else 'false',
        '--skip-visualize',
    ]
    if use_wandb:
        command += [
            '--wandb_project', str(wandb.get('project', 'sacm')),
            '--wandb_entity', str(wandb.get('entity', 'buaazqk')),
            '--wandb_group', str(wandb.get('group', config['name'])),
            '--wandb_name', f'paper_metrics-{config["name"]}-{arm}-seed{seed}',
        ]
    return command


def _diagnose_command(config, checkpoint: Path, protocol_path: Path,
                      seed: int, gpu: str, output_dir: Path,
                      splits: list[str], max_per_dataset: int,
                      empty_policy: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'pipeline' /
            'diagnose_sacm_stc_candidates.py'),
        '--checkpoint', str(checkpoint),
        '--protocol', str(protocol_path),
        '--seed', str(seed),
        '--data-root', str(protocol_path.parent),
        '--splits', *splits,
        '--max-per-dataset', str(max_per_dataset),
        '--empty-metric-policy', str(empty_policy),
        '--device', f'cuda:{gpu}',
        '--output-dir', str(output_dir),
        '--name', str(config['name']),
    ]


def _run_info_done(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        return payload.get('status') == 'done'
    except (OSError, ValueError, TypeError):
        return False


def _train_complete(path: Path, epochs: int) -> bool:
    """Treat a run as complete only if it used the requested epoch budget.

    A previous 200-epoch run must not satisfy a 300-epoch ablation: its
    ``run_info.json`` says done, but its checkpoint is not the requested
    budget.  Requiring the recorded epochs makes the resume safe when the
    YAML budget changes.
    """
    try:
        payload = json.loads(path.read_text())
        recorded = int(payload.get('epochs'))
    except (OSError, ValueError, TypeError):
        return False
    return payload.get('status') == 'done' and recorded == int(epochs)


def _training_incomplete_reason(run_root: Path, epochs: int) -> str | None:
    """Explain why a run is not safe to evaluate/analyse yet.

    Separate ``train`` and ``eval`` stages are useful for scheduling, but they
    lose the per-job failure propagation of the ``all`` stage.  Without this
    guard a run that crashed at epoch 150 still has ``best_model.pth`` and
    would be evaluated as if it had completed the full budget.
    """
    path = run_root / 'run_info.json'
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return 'missing or invalid run_info.json (training not complete)'
    status = payload.get('status')
    if status != 'done':
        return f'training status={status!r}, expected done'
    try:
        recorded = int(payload.get('epochs'))
    except (TypeError, ValueError):
        return 'run_info.json has no valid epochs'
    if recorded != int(epochs):
        return f'training epochs={recorded}, expected {int(epochs)}'
    return None


def _resume_epochs_match(run_root: Path, epochs: int) -> bool:
    """Only resume a partial run started with the same epoch budget.

    The optimizer/scheduler checkpoint stores a cosine schedule built for the
    old budget.  Resuming a 200-epoch partial run under ``--epochs 300``
    would silently keep the old schedule, so a budget change starts fresh.
    """
    try:
        payload = json.loads(
            (run_root / 'train_config.json').read_text())
        recorded = int((payload.get('args') or {}).get('epochs'))
    except (OSError, ValueError, TypeError):
        return False
    return recorded == int(epochs)


def _artifact_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _eval_complete(run_root: Path, expected_datasets: int,
                   checkpoint_name: str | None = None) -> bool:
    """Require a completed 20-dataset table, not merely a non-empty file.

    A crashed eval can leave a header/partial table.  Checking only file size
    would make a later resume skip the missing datasets and poison the
    analysis.  The canonical table has one row per test dataset.

    When ``checkpoint_name`` is supplied, the eval manifest must record that
    checkpoint.  This prevents a stale best_model evaluation from satisfying
    a new last_model (epoch-300) evaluation.
    """
    table = run_root / 'eval' / 'tables' / 'all_metrics.csv'
    if not _artifact_nonempty(table):
        return False
    manifest = run_root / 'eval' / 'run_manifest.json'
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text())
            status = payload.get('status')
            if status not in (None, 'done'):
                return False
            if checkpoint_name is not None:
                recorded = payload.get('checkpoint')
                if (not recorded
                        or Path(recorded).name != checkpoint_name):
                    return False
        except (OSError, ValueError, TypeError):
            return False
    elif checkpoint_name is not None:
        # Without a manifest we cannot prove which checkpoint was scored.
        return False
    try:
        with open(table, newline='') as handle:
            datasets = {
                row.get('dataset') for row in csv.DictReader(handle)
                if row.get('dataset')}
    except (OSError, csv.Error):
        return False
    return len(datasets) >= int(expected_datasets)


def _diagnostics_complete(path: Path, checkpoint_name: str) -> bool:
    """Require diagnostics for the same checkpoint as the requested eval."""
    try:
        payload = json.loads(path.read_text())
        recorded = payload.get('checkpoint')
    except (OSError, ValueError, TypeError):
        return False
    return bool(recorded) and Path(recorded).name == checkpoint_name


def _job_label(stage: str, job_id: str) -> str:
    return f'{stage}:{job_id}'


def _run_command(command: list[str], log_path: Path,
                 label: str) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{label}] $ {" ".join(command)}', flush=True)
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '')
    src = str(REPO_ROOT / 'src')
    env['PYTHONPATH'] = src + (
        os.pathsep + env['PYTHONPATH'] if env['PYTHONPATH'] else '')
    try:
        with open(log_path, 'w') as log:
            result = subprocess.run(
                command, cwd=REPO_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True)
    except OSError as exc:
        print(f'[{label}] failed to start: {exc}', file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f'[{label}] FAILED (exit {result.returncode}); '
              f'log: {_repo_text(log_path)}', file=sys.stderr)
        return False
    print(f'[{label}] ok; log: {_repo_text(log_path)}', flush=True)
    return True


def _run_gpu_jobs(gpu: str, jobs: list[dict], log_root: Path,
                  dry_run: bool, stamp: str) -> bool:
    del gpu
    ok = True
    for job in jobs:
        if job.get('error'):
            print(f'[{job["id"]}] ERROR: {job["error"]}', file=sys.stderr)
            ok = False
            continue
        for command_spec in job['commands']:
            label = _job_label(command_spec['stage'], job['id'])
            if command_spec.get('skip'):
                print(f'[{label}] skip: {command_spec["skip"]}', flush=True)
                continue
            if dry_run:
                print(f'[{label}] dry-run: '
                      f'{" ".join(command_spec["command"])}', flush=True)
                continue
            log_path = (log_root / command_spec['stage'] /
                        f'{job["id"]}_{stamp}.log')
            if not _run_command(command_spec['command'], log_path, label):
                ok = False
                # Skip the remaining stages of this arm/seed, but continue
                # with the next independent job on this GPU.  An overnight
                # run should maximize completed pairs, not stop an entire
                # GPU queue because one run failed.
                break
    return ok


def _dispatch(jobs: list[dict], gpus: list[str], log_root: Path,
              dry_run: bool, stamp: str) -> int:
    by_gpu = {gpu: [] for gpu in gpus}
    for index, job in enumerate(jobs):
        gpu = job.get('gpu') or gpus[index % len(gpus)]
        job['gpu'] = gpu
        by_gpu[gpu].append(job)
    if not jobs:
        print('no jobs to run', flush=True)
        return 0
    for gpu in gpus:
        if by_gpu[gpu]:
            print(f'[gpu {gpu}] {len(by_gpu[gpu])} job(s)', flush=True)
    if dry_run or len(gpus) == 1:
        ok = True
        for gpu in gpus:
            ok = _run_gpu_jobs(gpu, by_gpu[gpu], log_root,
                               dry_run, stamp) and ok
        return 0 if ok else 1
    ok = True
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(_run_gpu_jobs, gpu, by_gpu[gpu], log_root,
                        dry_run, stamp)
            for gpu in gpus if by_gpu[gpu]
        ]
        for future in as_completed(futures):
            ok = future.result() and ok
    return 0 if ok else 1


def _run_root(config, arm: str, seed: int) -> Path:
    return _resolve(config['output_root']) / arm / f'seed_{seed}'


def build_jobs(config: dict, protocol, stage: str, arms: list[str],
               seeds: list[int], epochs: int, resume: bool,
               allow_untrained_eval: bool = False) -> list[dict]:
    jobs = []
    diagnostics = config.get('diagnostics') or {}
    diag_splits = list(diagnostics.get('splits', ['val', 'test']))
    diag_limit = int(diagnostics.get('max_per_dataset', 20))
    diag_policy = str(diagnostics.get('empty_metric_policy', 'penalty'))
    diag_root = _resolve(diagnostics.get(
        'output_root', 'results/ablations/sacm_stc/candidate_diagnostics'))

    if stage == 'audit':
        audit = config.get('audit') or {}
        variant = str(audit.get('variant', 'all_dense'))
        checkpoint_name = str(audit.get('checkpoint_name', 'best_model.pth'))
        output_root = _resolve(audit.get(
            'output_root', 'results/ablations/sacm_stc/candidate_audit'))
        audit_splits = list(audit.get('splits', diag_splits))
        audit_limit = int(audit.get('max_per_dataset', diag_limit))
        audit_policy = str(audit.get(
            'empty_metric_policy', diag_policy))
        for seed in seeds:
            run_root = _resolve(config['output_root']) / variant / f'seed_{seed}'
            checkpoint = run_root / checkpoint_name
            output_dir = output_root / f'seed_{seed}'
            done = output_dir / 'summary.json'
            jobs.append({
                'id': f'audit_seed{seed}',
                'commands': [{
                    'stage': 'audit',
                    'command': _diagnose_command(
                        config, checkpoint,
                        protocol.split_for(seed).test_manifest,
                        seed, '{gpu}', output_dir,
                        audit_splits, audit_limit, audit_policy),
                    'skip': (f'already exists ({_repo_text(done)})'
                             if resume and done.is_file() else None),
                }],
            })
        return jobs

    for arm in arms:
        for seed in seeds:
            run_root = _run_root(config, arm, seed)
            training_error = None
            if stage in ('eval', 'diagnose') and not allow_untrained_eval:
                training_error = _training_incomplete_reason(run_root, epochs)
            if training_error:
                jobs.append({
                    'id': f'{arm}_seed{seed}',
                    'arm': arm,
                    'seed': seed,
                    'commands': [],
                    'error': training_error,
                })
                continue
            split_root = protocol.split_for(seed).split_root
            protocol_path = protocol.split_for(seed).test_manifest
            commands = []
            train_skip = False
            if stage in ('train', 'all'):
                done = run_root / 'run_info.json'
                train_skip = bool(
                    resume and _train_complete(done, epochs))
                train_resume = bool(
                    resume and not train_skip
                    and (run_root / 'last_model.pth').is_file()
                    and _resume_epochs_match(run_root, epochs))
                commands.append({
                    'stage': 'train',
                    'command': _train_command(
                        config, arm, seed, '{gpu}', split_root, run_root,
                        epochs, train_resume),
                    'skip': (f'training complete ({_repo_text(done)})'
                             if train_skip else None),
                })
            eval_skip = False
            expected = int(config.get('expected_datasets', 20))
            checkpoint_name = _eval_checkpoint(config)
            if stage in ('eval', 'all'):
                done = run_root / 'eval' / 'tables' / 'all_metrics.csv'
                if stage == 'all':
                    # If training is about to run, any old eval table belongs
                    # to the old checkpoint and must not be reused.
                    eval_skip = bool(
                        train_skip and _eval_complete(
                            run_root, expected, checkpoint_name))
                else:
                    eval_skip = bool(
                        resume and _eval_complete(
                            run_root, expected, checkpoint_name))
                commands.append({
                    'stage': 'eval',
                    'command': _eval_command(
                        config, arm, seed, '{gpu}', protocol_path, run_root),
                    'skip': (f'evaluation complete ({_repo_text(done)})'
                             if eval_skip else None),
                })
            if stage in ('diagnose', 'all'):
                done = diag_root / arm / f'seed_{seed}' / 'summary.json'
                diag_done = _diagnostics_complete(done, checkpoint_name)
                if stage == 'all':
                    diag_skip = bool(
                        train_skip and eval_skip and diag_done)
                else:
                    diag_skip = bool(resume and diag_done)
                commands.append({
                    'stage': 'diagnose',
                    'command': _diagnose_command(
                        config, run_root / _eval_checkpoint(config),
                        protocol_path, seed, '{gpu}',
                        diag_root / arm / f'seed_{seed}',
                        diag_splits, diag_limit, diag_policy),
                    'skip': (f'diagnostics complete ({_repo_text(done)})'
                             if diag_skip else None),
                })
            jobs.append({
                'id': f'{arm}_seed{seed}',
                'arm': arm,
                'seed': seed,
                'commands': commands,
            })
    return jobs


def _fill_gpu(command: list[str], gpu: str) -> list[str]:
    return [item.replace('{gpu}', gpu) for item in command]


def _dispatch_with_gpu_placeholders(jobs: list[dict], gpus: list[str],
                                    log_root: Path, dry_run: bool,
                                    stamp: str) -> int:
    for index, job in enumerate(jobs):
        job['gpu'] = gpus[index % len(gpus)]
        for spec in job['commands']:
            spec['command'] = _fill_gpu(spec['command'], job['gpu'])
    return _dispatch(jobs, gpus, log_root, dry_run, stamp)


def _stage_label(stage: str) -> str:
    return {
        'audit': 'candidate audit',
        'train': 'training',
        'eval': '20-dataset evaluation',
        'diagnose': 'candidate diagnostics',
        'all': 'train + eval + diagnostics',
        'analyze': 'paired analysis',
    }.get(stage, stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        '--stage', choices=['audit', 'train', 'eval', 'diagnose',
                            'all', 'analyze'],
        default='audit',
        help='audit is the safe default; use all for the full matrix')
    parser.add_argument('--arms', nargs='*', default=None,
                        help='arm ids or comma-separated list; default all')
    parser.add_argument('--seeds', nargs='*', default=None,
                        help='seed ids or comma-separated list')
    parser.add_argument('--gpus',
                        default=os.environ.get('GPU_LIST', '4,5,6,7'),
                        help='comma-separated GPU ids (default: the '
                             'server ablation cards 4,5,6,7)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override config epochs (default: config)')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='skip completed stages (default: true)')
    parser.add_argument(
        '--allow-untrained-eval', action='store_true',
        help='allow eval/diagnose when run_info.json is missing or does not '
             'record a completed matching-budget training run; intended only '
             'for externally produced checkpoints')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    protocol = load_protocol(str(config['protocol']))
    if args.stage == 'audit' and args.seeds is None:
        raw_seeds = config.get('audit', {}).get('seeds', config['seeds'])
    else:
        raw_seeds = args.seeds if args.seeds is not None else config.get('seeds')
    seeds = _parse_seeds(raw_seeds, protocol, protocol.seeds)
    arms = _parse_arms(args.arms, config)
    gpus = _parse_gpus(args.gpus)
    epochs = int(args.epochs if args.epochs is not None
                 else config.get('epochs', 200))
    if epochs <= 0:
        raise SystemExit('epochs must be positive')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = _resolve('results/logs/sacm_stc_ablation') / config['name']

    print(f'ablation: {config["name"]}')
    print(f'  config: {config_path}')
    print(f'  stage: {args.stage} ({_stage_label(args.stage)})')
    print(f'  protocol: {protocol.name} seeds={seeds}')
    if args.stage not in ('audit', 'analyze'):
        print(f'  arms: {arms}')
        print(f'  epochs: {epochs}')
    print(f'  gpus: {gpus}')
    if args.dry_run:
        print('  dry-run: no commands will execute')

    if args.stage == 'analyze':
        command = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'pipeline' /
                'analyze_sacm_stc_ablation.py'),
            '--config', str(config_path),
            '--arms', *arms,
            '--seeds', *[str(seed) for seed in seeds],
        ]
        if args.dry_run:
            print('analyze dry-run:', ' '.join(command))
            return 0
        log_path = log_root / 'analyze' / f'{stamp}.log'
        return 0 if _run_command(command, log_path, 'analysis') else 1

    jobs = build_jobs(
        config, protocol, args.stage, arms, seeds, epochs, bool(args.resume),
        bool(args.allow_untrained_eval))
    if not jobs:
        print('no jobs to run')
        return 0
    status = _dispatch_with_gpu_placeholders(
        jobs, gpus, log_root, args.dry_run, stamp)
    # Always attempt the analysis after `all`, even if some jobs failed.  It
    # reports missing pairs explicitly, so a morning operator still gets the
    # completed subset instead of only a failed driver log.
    if args.stage == 'all' and not args.dry_run:
        command = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'pipeline' /
                'analyze_sacm_stc_ablation.py'),
            '--config', str(config_path),
            '--arms', *arms,
            '--seeds', *[str(seed) for seed in seeds],
        ]
        analysis_log = log_root / 'analyze' / f'{stamp}.log'
        if not _run_command(command, analysis_log, 'analysis'):
            status = 1
    return status


if __name__ == '__main__':
    raise SystemExit(main())
