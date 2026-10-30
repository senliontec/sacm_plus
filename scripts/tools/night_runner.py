#!/usr/bin/env python3
"""Generic multi-arm screening campaign helper.

The caller supplies ``MODELS`` and ``WANDB_GROUP``; this module runs one
task per GPU, reuses finished checkpoints/metrics, and writes a summary.
Training and evaluation use the same W&B/protocol/artifacts as the shared
baseline entrypoints.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DEFAULT_ROOT = REPO_ROOT / 'results' / 'night_campaign'
SAM_B = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'
WANDB_GROUP = 'night_campaign'

# Set by the caller before invoking _summary/_worker.
MODELS: tuple = ()


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT / 'src') + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    # Same self-hosted W&B defaults as scripts/tools/wandb_env.sh.  The API
    # key is never set here; it comes from the environment or ~/.netrc.
    env.setdefault('WANDB_BASE_URL', 'http://172.16.1.7:8080')
    env.setdefault('WANDB_ENTITY', 'buaazqk')
    env.setdefault('WANDB_PROJECT', 'sacm')
    env.setdefault('WANDB_MODE', 'online')
    env.setdefault('WANDB_FINISH_TIMEOUT', '60')
    return env


def _wandb_args(tag: str, seed: int, job_type: str,
                extra_tags: tuple[str, ...] = ()) -> list[str]:
    tags = (tag, f'seed{seed}', *extra_tags, WANDB_GROUP)
    return [
        '--use_wandb', 'true',
        '--wandb_name', f'{WANDB_GROUP}-{tag}-seed{seed}-{job_type}',
        '--wandb_group', WANDB_GROUP,
        '--wandb_job_type', job_type,
        '--wandb_tags', ','.join(tags),
    ]


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print('+', ' '.join(str(part) for part in cmd), flush=True)
    with log.open('a') as handle:
        handle.write('\n+ ' + ' '.join(str(part) for part in cmd) + '\n')
        handle.flush()
        subprocess.run(
            [str(part) for part in cmd], cwd=REPO_ROOT,
            env=_env(), stdout=handle, stderr=subprocess.STDOUT,
            check=True)


def _best_val(log: Path) -> float | None:
    if not log.is_file():
        return None
    text = log.read_text(errors='ignore')
    matches = re.findall(r'Best\s+E\d+\s+dice=([0-9.]+)', text)
    return float(matches[-1]) if matches else None


def _read_macro(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _task_dirs(root: Path, seed: int, tag: str) -> tuple[Path, Path]:
    workdir = root / f'seed{seed}' / tag
    return workdir, workdir / 'paper' / 'paper_macro.csv'


def _train(root: Path, seed: int, tag: str, method: str,
           extra: tuple[str, ...], gpu: int, epochs: int) -> None:
    workdir, _ = _task_dirs(root, seed, tag)
    checkpoint = workdir / 'best_model.pth'
    log = root / 'logs' / f'{tag}_seed{seed}_train.log'
    if checkpoint.is_file():
        print(f'skip train {tag} seed{seed}: checkpoint exists', flush=True)
        return
    data_root = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}'
    cmd = [
        PYTHON, '-m', 'baseline_zoo.run_one',
        '--model', method, '--stage', 'train',
        '--data-root', str(data_root),
        '--workdir', str(workdir),
        '--epochs', str(epochs), '--batch-size', '1',
        '--seed', str(seed), '--device', f'cuda:{gpu}',
        *_wandb_args(tag, seed, 'train', ('3shot',)),
        '--convlora-model-type', 'vit_b',
        '--convlora-checkpoint', str(SAM_B),
        *extra,
    ]
    _run(cmd, log)


def _evaluate(root: Path, seed: int, tag: str, method: str,
              gpu: int) -> None:
    workdir, macro = _task_dirs(root, seed, tag)
    checkpoint = workdir / 'best_model.pth'
    log = root / 'logs' / f'{tag}_seed{seed}_eval.log'
    if macro.is_file():
        print(f'skip eval {tag} seed{seed}: metrics exist', flush=True)
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f'missing checkpoint: {checkpoint}')
    protocol = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}' / \
        'protocol_manifest.csv'
    cmd = [
        PYTHON, str(REPO_ROOT / 'scripts' / 'pipeline' /
                    'run_baseline_test.py'),
        '--method', method,
        '--protocol', str(protocol),
        '--seed', str(seed),
        '--checkpoint', str(checkpoint),
        '--pred-root', str(workdir / 'predictions'),
        '--paper-root', str(workdir / 'paper'),
        '--device', f'cuda:{gpu}',
        *_wandb_args(tag, seed, 'eval', ('3shot', 'paper_metrics')),
        '--skip-visualize',
    ]
    _run(cmd, log)


def _worker(root: Path, gpu: int,
            tasks: list[tuple[int, str, str, tuple[str, ...]]],
            epochs: int) -> None:
    for seed, tag, method, extra in tasks:
        try:
            _train(root, seed, tag, method, extra, gpu, epochs)
            _evaluate(root, seed, tag, method, gpu)
        except Exception as exc:  # keep the queue alive, report later
            log = root / 'logs' / f'{tag}_seed{seed}_error.txt'
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f'{type(exc).__name__}: {exc}\n')
            print(f'FAILED {tag} seed{seed}: {exc}', flush=True)


def _summary(root: Path, seeds: list[int]) -> None:
    rows = []
    for seed in seeds:
        for tag, method, _ in MODELS:
            _, macro_path = _task_dirs(root, seed, tag)
            for row in _read_macro(macro_path):
                rows.append({
                    'tag': tag,
                    'method': method,
                    'seed': seed,
                    'role': row.get('role', ''),
                    'dice': row.get('dice_mean', ''),
                    'iou': row.get('iou_mean', ''),
                    'cldice': row.get('cldice_mean', ''),
                    'hd95': row.get('hd95_mean', ''),
                    'n': row.get('dice_n_datasets', ''),
                    'val_best': _best_val(
                        root / 'logs' / f'{tag}_seed{seed}_train.log'),
                })
    out = root / 'summary.csv'
    if rows:
        with out.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = ['# Overnight screening', '',
             'Matrix: seeds 0/1/2, 100 epochs, ViT-B, full 20-dataset test.', '']
    lines.append('## All / Novel by model and seed')
    lines.append('')
    lines.append('| model | seed | All Dice | All clDice | All HD95 | Novel Dice | Novel clDice | val best |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for seed in seeds:
        for tag, _, _ in MODELS:
            _, macro_path = _task_dirs(root, seed, tag)
            macro = {r['role']: r for r in _read_macro(macro_path)}
            all_row = macro.get('All', {})
            novel = macro.get('Novel', {})
            val = _best_val(root / 'logs' / f'{tag}_seed{seed}_train.log')

            def f(row, key):
                try:
                    return f'{float(row[key]):.3f}'
                except (KeyError, TypeError, ValueError):
                    return 'NA'
            lines.append(
                f"| {tag} | {seed} | {f(all_row, 'dice_mean')} | "
                f"{f(all_row, 'cldice_mean')} | {f(all_row, 'hd95_mean')} | "
                f"{f(novel, 'dice_mean')} | {f(novel, 'cldice_mean')} | "
                f"{val if val is not None else 'NA'} |")

    lines += ['', '## Aggregated across seeds', '',
              '| model | All Dice | All clDice | Novel Dice | Novel clDice |',
              '|---|---:|---:|---:|---:|']
    for tag, _, _ in MODELS:
        collected = {key: [] for key in
                     ('dice', 'cldice', 'novel_dice', 'novel_cldice')}
        for seed in seeds:
            _, macro_path = _task_dirs(root, seed, tag)
            macro = {r['role']: r for r in _read_macro(macro_path)}
            for key, role, field in (
                ('dice', 'All', 'dice_mean'),
                ('cldice', 'All', 'cldice_mean'),
                ('novel_dice', 'Novel', 'dice_mean'),
                ('novel_cldice', 'Novel', 'cldice_mean'),
            ):
                try:
                    collected[key].append(float(macro[role][field]))
                except (KeyError, TypeError, ValueError):
                    pass

        def mean_std(values):
            if not values:
                return 'NA'
            import statistics
            if len(values) > 1:
                return (f'{statistics.mean(values):.3f} ± '
                        f'{statistics.stdev(values):.3f}')
            return f'{values[0]:.3f}'
        lines.append(
            f"| {tag} | {mean_std(collected['dice'])} | "
            f"{mean_std(collected['cldice'])} | "
            f"{mean_std(collected['novel_dice'])} | "
            f"{mean_std(collected['novel_cldice'])} |")

    lines += ['', '## Failures', '']
    errors = sorted(root.glob('logs/*_error.txt'))
    lines += [f'- {path.name}: {path.read_text().strip()}'
              for path in errors] or ['- none']
    log_root = root.relative_to(REPO_ROOT) / 'logs'
    lines += ['', f'Per-run metrics: `{out}`', '',
              f'Raw training/eval logs: `{log_root}`.', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if len(args.gpus) < 2:
        raise SystemExit('two GPUs are required for the overnight queue')
    missing = [
        str(REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}')
        for seed in args.seeds
        if not (REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}').is_dir()
    ]
    if missing:
        raise SystemExit('missing seed roots: ' + ', '.join(missing))
    if not SAM_B.is_file():
        raise SystemExit(f'missing SAM ViT-B checkpoint: {SAM_B}')

    tasks = [(seed, tag, method, extra)
             for seed in args.seeds
             for tag, method, extra in MODELS]
    if args.dry_run:
        for seed, tag, method, extra in tasks:
            print(f'{tag} seed{seed}: {method} {" ".join(extra)}')
        return 0

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'logs').mkdir(exist_ok=True)
    queues = [tasks[index::len(args.gpus)]
              for index in range(len(args.gpus))]
    threads = [
        threading.Thread(
            target=_worker, args=(args.root, gpu, queue, args.epochs),
            daemon=False)
        for gpu, queue in zip(args.gpus, queues)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _summary(args.root, args.seeds)
    print(f'wrote {args.root / "SUMMARY.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
