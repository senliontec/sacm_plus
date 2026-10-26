#!/usr/bin/env python3
"""Optuna hyperparameter search for the paper baselines.

Each trial runs the normal ``baseline_zoo.run_one`` trainer in a unique
workdir and scores the trial by the maximum 6-image validation score from
``metrics_history.json`` (Dice by default; ``--selection`` can change it).
The test split is never used.

Examples:
  python scripts/tools/tune_baselines.py --models dscnet --trials 8 --gpus 0
  python scripts/tools/tune_baselines.py --models unet,samed --trials 12 \
      --gpus 0,1,2,3 --epochs 50
  python scripts/tools/tune_baselines.py --models all --seeds 0 1 2 3 4 \
      --trials 8 --gpus 0,1,2,3,4,5,6
  python scripts/tools/tune_baselines.py --models all --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.__main__ import (  # noqa: E402
    _format_template,
    _job_context,
    _pythonize_command,
)
from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.protocols import load_protocol  # noqa: E402
from baseline_zoo.registry import load_models  # noqa: E402

BASELINE_MODELS = (
    'unet', 'cs2net', 'bcunet', 'maskvsc', 'nnwnet', 'jtfn', 'dscnet',
    'segdino', 'samed', 'topolora', 'sam_adapter', 'sam_octa', 'cwsam',
    'sacm_official', 'fob_sam', 'persam', 'knowsam', 'sam2_unet',
    'sam_med2d', 'ucs', 'matcher', 'sam_vit_l_unet',
)


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(',') if item.strip()]
    if not gpus:
        raise SystemExit('--gpus cannot be empty')
    if any(not item.isdigit() for item in gpus):
        raise SystemExit(f'--gpus must be comma-separated GPU ids: {gpus}')
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f'--gpus contains duplicates: {gpus}')
    return gpus


def _parse_models(value: str) -> list[str]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        raise SystemExit('--models cannot be empty')
    return items


def _run_tag(args, default_epochs: int | None = None) -> str:
    tag = args.selection
    epochs = args.epochs if args.epochs is not None else default_epochs
    if epochs is not None:
        tag += f'_ep{epochs}'
    return tag


def _parse_seeds(values: list[str] | None, legacy_seed: int | None,
                 protocol) -> list[int]:
    if values is not None:
        if not values:
            raise SystemExit('--seeds requires at least one value')
        if len(values) == 1 and values[0].lower() == 'all':
            requested = list(protocol.seeds)
        else:
            tokens = [
                token
                for value in values
                for token in value.replace(',', ' ').split()
            ]
            if not tokens:
                raise SystemExit('--seeds requires at least one value')
            try:
                requested = [int(token) for token in tokens]
            except ValueError:
                raise SystemExit(
                    f'--seeds must be integers or "all", got {values}'
                ) from None
    elif legacy_seed is not None:
        requested = [legacy_seed]
    else:
        requested = [protocol.seeds[0]]
    try:
        return protocol.select_seeds(requested)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def _load_spaces(path: Path) -> tuple[dict, dict]:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    defaults = data.get('defaults') or {}
    models = data.get('models') or {}
    if not isinstance(defaults, dict) or not isinstance(models, dict):
        raise SystemExit(f'invalid tuning spaces file: {path}')
    return defaults, models


def _merged_spec(defaults: dict, overrides: dict, key: str) -> dict:
    spec = dict(defaults.get(key, {}) or {})
    spec.update(overrides.get(key, {}) or {})
    return spec


def _absolute_range(spec: dict, official: float) -> tuple[float, float]:
    low = spec.get('low')
    high = spec.get('high')
    if low is None and 'low_factor' in spec:
        low = official * float(spec['low_factor'])
    if high is None and 'high_factor' in spec:
        high = official * float(spec['high_factor'])
    if low is None or high is None:
        raise SystemExit(f'spec needs low/high or factors: {spec}')
    return float(low), float(high)


def _suggest(trial, name: str, spec: dict, official: float | None):
    kind = spec.get('type', 'choices' if 'choices' in spec else None)
    if kind == 'choices':
        choices = spec.get('choices')
        if not choices:
            raise SystemExit(f'{name}: empty choices')
        return trial.suggest_categorical(name, choices)
    if official is not None and official <= 0:
        # Relative factors are meaningless for a zero official value.
        low = float(spec.get('low', 1e-6))
        high = float(spec.get('high', 1e-2))
    else:
        low, high = _absolute_range(spec, float(official or 0.0))
    if low <= 0 or high <= 0 or high < low:
        raise SystemExit(f'{name}: invalid range [{low}, {high}]')
    if kind == 'loguniform':
        return trial.suggest_float(name, low, high, log=True)
    if kind == 'uniform':
        return trial.suggest_float(name, low, high)
    raise SystemExit(f'{name}: unknown parameter type {kind!r}')


def _trial_command(*, model, model_id: str, protocol, seed: int, gpu: str,
                   trial_dir: Path, params: dict, epochs: int | None,
                   selection: str, train_entrypoint: str) -> str:
    context = _job_context(
        model, protocol, seed, gpu, str(trial_dir))
    if epochs is not None:
        context['epochs'] = str(epochs)
    command = _pythonize_command(
        _format_template(model, train_entrypoint, context))
    command += (
        f' --lr {params["lr"]} --weight-decay {params["weight_decay"]}'
        f' --scheduler {params["scheduler"]} --selection {selection}'
    )
    if params.get('warmup_epochs') is not None:
        command += f' --warmup-epochs {params["warmup_epochs"]}'
    for key, value in params.items():
        if key.startswith('_') or key in (
                'lr', 'weight_decay', 'scheduler', 'warmup_epochs'):
            continue
        command += f' --{key.replace("_", "-")} {value}'
    command += (
        f' --wandb_group tuning --wandb_name'
        f' {model_id}-seed{seed}-trial{params.get("_trial_number", "x")}'
    )
    return command


def _kill_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def _remove_trial_checkpoints(trial_dir: Path) -> None:
    """Keep logs/metrics, but drop per-trial weights unless requested.

    A full HPO batch can otherwise write hundreds of gigabytes of
    checkpoints that are never used: only the best config is retrained at
    full budget after the search.
    """
    for pattern in (
            'best_model.pth',
            'best_model.before_*.pth',
            'best_model*.safetensors'):
        for path in trial_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _read_score(train_dir: Path, selection: str) -> float:
    history = train_dir / 'metrics_history.json'
    if not history.is_file():
        raise RuntimeError(f'missing {history}')
    payload = json.loads(history.read_text())
    rows = payload.get('history', [])
    values = []
    for row in rows:
        overall = row.get('overall', {}) or {}
        value = overall.get(selection)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if not values:
        raise RuntimeError(
            f'no finite {selection} values in {history}')
    return max(values)


def _run_trial(trial_number: int, model_id: str, model, protocol, args,
               params: dict, gpu_queue: queue.Queue) -> float:
    gpu = gpu_queue.get()
    try:
        trial_dir = (
            REPO_ROOT / 'results' / 'tuning' / 'trials' / model_id /
            f'seed_{args.seed}' / _run_tag(
                args, model.defaults.get('epochs')) /
            f'trial_{trial_number:04d}')
        trial_dir.mkdir(parents=True, exist_ok=True)
        params['_trial_number'] = trial_number
        command = _trial_command(
            model=model, model_id=model_id, protocol=protocol,
            seed=args.seed, gpu=gpu, trial_dir=trial_dir,
            params=params, epochs=args.epochs, selection=args.selection,
            train_entrypoint=model.train_entrypoint)
        log_path = trial_dir / 'tune_train.log'
        env = os.environ.copy()
        env['PYTHONPATH'] = (
            str(REPO_ROOT / 'src')
            + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
        timeout = args.trial_timeout or None
        with open(log_path, 'w') as log:
            log.write(command + '\n\n')
            log.flush()
            # Run in a new session so a timeout kills the trainer and all of
            # its DataLoader workers.  Killing only the shell leaves children
            # holding GPU memory, which would poison the next trial.
            proc = subprocess.Popen(
                command, shell=True, cwd=REPO_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # SIGTERM may be ignored by a trainer; make sure workers do
                # not keep the GPU after the shell has exited.
                _kill_process_group(proc.pid, signal.SIGKILL)
                if proc.poll() is None:
                    proc.wait()
                raise RuntimeError(
                    f'trial {trial_number} timed out after {timeout}s; '
                    f'see {log_path}') from None
        if proc.returncode != 0:
            _kill_process_group(proc.pid, signal.SIGKILL)
            raise RuntimeError(
                f'trial {trial_number} failed (exit {proc.returncode}); '
                f'see {log_path}')
        value = _read_score(trial_dir, args.selection)
        if not args.keep_trial_checkpoints:
            try:
                _remove_trial_checkpoints(trial_dir)
            except OSError as exc:
                print(
                    f'[tune] warning: could not remove trial checkpoints '
                    f'in {trial_dir}: {exc}', file=sys.stderr, flush=True)
        print(
            f'[tune] {model_id} trial {trial_number}: '
            f'{args.selection}={value:.6f}', flush=True)
        return value
    finally:
        gpu_queue.put(gpu)


def _sample_params(trial, defaults: dict, overrides: dict,
                   adapter) -> dict:
    lr_spec = _merged_spec(defaults, overrides, 'lr')
    wd_spec = _merged_spec(defaults, overrides, 'weight_decay')
    sched_spec = _merged_spec(defaults, overrides, 'scheduler')
    warm_spec = _merged_spec(defaults, overrides, 'warmup_epochs')

    lr = _suggest(trial, 'lr', lr_spec, adapter.official_lr)
    wd = _suggest(
        trial, 'weight_decay', wd_spec,
        adapter.official_weight_decay)
    scheduler = _suggest(trial, 'scheduler', sched_spec, None)
    warmup = None
    if scheduler != 'official' and warm_spec.get('choices'):
        warmup = trial.suggest_categorical(
            'warmup_epochs', warm_spec['choices'])

    params = {
        'lr': float(lr),
        'weight_decay': float(wd),
        'scheduler': scheduler,
        'warmup_epochs': warmup,
    }
    extra = _merged_spec(defaults, overrides, 'extra')
    for name, spec in extra.items():
        params[name] = _suggest(trial, name, spec, None)
    return params


def _run_model(model_id: str, model, adapter, protocol, args,
               defaults: dict, overrides: dict) -> bool:
    import optuna

    storage = args.storage
    storage_path = Path(storage.removeprefix('sqlite:///'))
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    tag = _run_tag(args, model.defaults.get('epochs'))
    study_name = (
        f'{args.study_prefix}_{model_id}_{args.protocol}'
        f'_seed{args.seed}_{tag}')
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    gpu_queue: queue.Queue = queue.Queue()
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    def objective(trial):
        params = _sample_params(trial, defaults, overrides, adapter)
        try:
            return _run_trial(
                trial.number, model_id, model, protocol, args,
                params, gpu_queue)
        except Exception as exc:
            print(f'[tune] {model_id} trial {trial.number} pruned: {exc}',
                  file=sys.stderr, flush=True)
            raise optuna.TrialPruned() from exc

    existing = len(study.trials)
    remaining = max(0, args.trials - existing)
    print(
        f'[tune] study={study_name} storage={storage} '
        f'existing={existing} new={remaining}', flush=True)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=min(len(args.gpus), remaining),
        )

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print(f'[tune] {model_id}: no completed trials', file=sys.stderr)
        return False
    best = study.best_trial
    out = (
        REPO_ROOT / 'results' / 'tuning' / model_id /
        f'best_{args.protocol}_seed{args.seed}_{tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model_id,
        'study': study_name,
        'protocol': args.protocol,
        'seed': args.seed,
        'run_tag': tag,
        'trial_number': best.number,
        'value': best.value,
        'selection': args.selection,
        'params': best.params,
        'official_lr': adapter.official_lr,
        'official_weight_decay': adapter.official_weight_decay,
        'epochs_override': args.epochs,
        'trials': len(study.trials),
        'storage': storage,
    }
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(
        f'[tune] best {model_id}: {best.value:.6f} -> {out}\n'
        f'[tune] params: {json.dumps(best.params, sort_keys=True)}',
        flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='all',
                        help='comma-separated model ids or all')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--protocol', default='seeded5')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='legacy single-seed option; use --seeds for a batch')
    parser.add_argument(
        '--seeds', nargs='+', default=None,
        help='one or more seeds, or "all" for every protocol seed')
    parser.add_argument('--selection', default='dice',
                        choices=['dice', 'iou', 'cldice'])
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='optional search-budget override; final runs should use the '
             'model config default')
    parser.add_argument('--trial-timeout', type=float, default=None,
                        help='seconds before a trial is pruned')
    parser.add_argument(
        '--keep-trial-checkpoints', action='store_true',
        help='keep per-trial best_model.pth files (default: delete them; '
             'only metrics/logs are needed for HPO scoring)')
    parser.add_argument(
        '--spaces', type=Path,
        default=REPO_ROOT / 'configs' / 'tuning' / 'spaces.yaml')
    parser.add_argument(
        '--storage', default='sqlite:///results/tuning/optuna.db')
    parser.add_argument('--study-prefix', default='tuning')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    if args.trials <= 0:
        raise SystemExit('--trials must be positive')
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit('--epochs must be positive')
    if args.trial_timeout is not None and args.trial_timeout <= 0:
        raise SystemExit('--trial-timeout must be positive')
    if args.seeds and args.seed is not None:
        raise SystemExit('use either --seed or --seeds, not both')
    args.gpus = _parse_gpus(args.gpus)
    defaults, model_spaces = _load_spaces(args.spaces)
    models = load_models()
    protocol = load_protocol(args.protocol)
    seeds = _parse_seeds(args.seeds, args.seed, protocol)
    if args.models.strip() == 'all':
        requested = list(BASELINE_MODELS)
    else:
        requested = _parse_models(args.models)
    unknown = sorted(set(requested) - set(models))
    if unknown:
        raise SystemExit(f'unknown model(s): {", ".join(unknown)}')

    if args.dry_run:
        print(
            f'[tune] models={len(requested)} seeds={seeds} '
            f'gpus={args.gpus} trials={args.trials}', flush=True)
        for model_id in requested:
            overrides = model_spaces.get(model_id, {}) or {}
            tunable = bool(overrides.get(
                'tunable', defaults.get('tunable', True)))
            if not tunable:
                print(f'{model_id}: tunable=false (skipped)')
                continue
            try:
                adapter = load_adapter(model_id)
            except Exception as exc:
                print(f'{model_id}: unavailable: {exc}')
                continue
            print(
                f'{model_id}: lr~{adapter.official_lr:g}, '
                f'wd~{adapter.official_weight_decay:g}, '
                f'epochs={models[model_id].defaults.get("epochs")}, '
                f'spaces={args.spaces}')
        return 0

    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'optuna is not installed; run: pip install optuna') from exc

    failures = []
    for seed in seeds:
        args.seed = seed
        print(f'\n[tune] ===== seed {seed} =====', flush=True)
        for model_id in requested:
            overrides = model_spaces.get(model_id, {}) or {}
            tunable = bool(overrides.get(
                'tunable', defaults.get('tunable', True)))
            if not tunable:
                print(f'[tune] {model_id}: tunable=false; skipped',
                      flush=True)
                continue
            try:
                adapter = load_adapter(model_id)
            except Exception as exc:
                print(
                    f'[tune] {model_id} seed {seed} unavailable: {exc}',
                    file=sys.stderr, flush=True)
                failures.append(f'{model_id}/seed{seed}')
                continue
            try:
                completed = _run_model(
                    model_id, models[model_id], adapter, protocol, args,
                    defaults, overrides)
            except Exception as exc:
                print(
                    f'[tune] {model_id} seed {seed} failed: {exc}',
                    file=sys.stderr, flush=True)
                completed = False
            if not completed:
                failures.append(f'{model_id}/seed{seed}')
    if failures:
        print(
            f'[tune] FAILED studies ({len(failures)}): '
            f'{", ".join(failures)}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
