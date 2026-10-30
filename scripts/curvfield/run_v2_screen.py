#!/usr/bin/env python3
"""One-click seed0 screen for CurvField v2.

Trains Conv-LoRA and CurvField v2 on seed0, runs the oracle-field causal gate
on the CurvField checkpoint, and only then evaluates the full 20-dataset test
set.  The two tasks run on two GPUs and log to the shared W&B server.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_night as night  # noqa: E402

REPO_ROOT = night.REPO_ROOT
DEFAULT_ROOT = REPO_ROOT / 'results' / 'curvfield_v2_screen'
SAM_B = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'
WANDB_GROUP = 'curvfield_v2_screen'
MODELS = (
    ('convlora', 'convlora', ()),
    ('curvfield_v2', 'curvfield',
     ('--curvfield-field-size', '128',
      '--curvfield-film-scale', '0.5',
      '--curvfield-decoder-bias', 'true',
      '--curvfield-field-weight', '0.2')),
)


def _field_gate(root: Path, seed: int, workdir: Path, gpu: int,
                diag_n: int, min_mean: float, min_max: float) -> None:
    log = root / 'logs' / f'curvfield_v2_seed{seed}_field_diag.log'
    protocol = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}' / \
        'protocol_manifest.csv'
    cmd = [
        night.PYTHON, str(REPO_ROOT / 'scripts' / 'curvfield' /
                          'diagnose_field.py'),
        '--checkpoint', str(workdir / 'best_model.pth'),
        '--protocol', str(protocol),
        '--device', f'cuda:{gpu}',
        '--n-per', str(diag_n),
        '--min-mean-delta', str(min_mean),
        '--min-max-delta', str(min_max),
        '--json-out', str(workdir / 'field_diagnosis.json'),
    ]
    night._run(cmd, log)


def _worker(root: Path, gpu: int, tasks, epochs: int, diag_n: int,
            min_mean: float, min_max: float) -> None:
    for seed, tag, method, extra in tasks:
        workdir, _ = night._task_dirs(root, seed, tag)
        try:
            night._train(root, seed, tag, method, extra, gpu, epochs)
            if method == 'curvfield':
                _field_gate(root, seed, workdir, gpu, diag_n,
                            min_mean, min_max)
            night._evaluate(root, seed, tag, method, gpu)
        except Exception as exc:  # keep the other GPU alive, report later
            log = root / 'logs' / f'{tag}_seed{seed}_error.txt'
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f'{type(exc).__name__}: {exc}\n')
            print(f'FAILED {tag} seed{seed}: {exc}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--diag-n', type=int, default=12)
    parser.add_argument('--min-mean-delta', type=float, default=0.2)
    parser.add_argument('--min-max-delta', type=float, default=0.5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if len(args.gpus) != 2:
        raise SystemExit('v2 screen requires exactly two GPUs')
    data_root = REPO_ROOT / 'data' / f'sacm_3shot_seed{args.seed}'
    if not data_root.is_dir():
        raise SystemExit(f'missing seed root: {data_root}')
    if not SAM_B.is_file():
        raise SystemExit(f'missing SAM ViT-B checkpoint: {SAM_B}')

    night.WANDB_GROUP = WANDB_GROUP
    night.MODELS = MODELS
    tasks = [(args.seed, tag, method, extra)
             for tag, method, extra in MODELS]
    if args.dry_run:
        for seed, tag, method, extra in tasks:
            print(f'{tag} seed{seed}: {method} {" ".join(extra)}')
        return 0

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'logs').mkdir(exist_ok=True)
    threads = [
        threading.Thread(
            target=_worker,
            args=(args.root, gpu, [task], args.epochs, args.diag_n,
                  args.min_mean_delta, args.min_max_delta),
            daemon=False)
        for gpu, task in zip(args.gpus, tasks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    night._summary(args.root, [args.seed])
    summary = args.root / 'SUMMARY.md'
    if summary.is_file():
        text = summary.read_text()
        text = text.replace(
            'Matrix: seeds 0/1/2, 100 epochs, ViT-B, full 20-dataset test.',
            f'Matrix: seed {args.seed}, 100 epochs, ViT-B, full 20-dataset test.')
        text = text.replace(
            'Raw training/eval logs: `results/curvfield_night/logs/`.',
            f'Raw training/eval logs: `{args.root.relative_to(REPO_ROOT)}/logs/`.')
        gate = args.root / 'logs' / f'curvfield_v2_seed{args.seed}_field_diag.log'
        if gate.is_file():
            text += (f'\nOracle-field gate log: '
                     f'`{gate.relative_to(REPO_ROOT)}`.\n')
        summary.write_text(text)
    print(f'wrote {args.root / "SUMMARY.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
