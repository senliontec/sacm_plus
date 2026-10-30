#!/usr/bin/env python3
"""One-click seed0 screen: SACM official vs NativeToken.

The main comparison is the official SACM baseline (SAM ViT-L, official
adapters) against NativeToken (Conv-LoRA backbone + native-resolution
residual tokenization, SAM ViT-B).  Conv-LoRA is the absorbed base model,
not the main comparison.  Training and evaluation reuse the shared
CurvField campaign helpers so W&B, protocol and artifacts match the
baselines.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault('SACM_ENABLE_PROJECT_MODELS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import night_runner as night  # noqa: E402

REPO_ROOT = night.REPO_ROOT
DEFAULT_ROOT = REPO_ROOT / 'results' / 'nativetok_vs_sacm'
SAM_B = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'
SAM_L = REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth'
WANDB_GROUP = 'nativetok_vs_sacm'
MODELS = (
    ('sacm_official', 'sacm_official', ()),
    ('nativetok', 'nativetok',
     ('--nativetok-scales', '8', '4',
      '--nativetok-hidden', '32',
      '--nativetok-alpha-init', '0',
      '--nativetok-new-lr-mult', '10',
      '--nativetok-native-max-side', '0')),
)


def _train(root: Path, seed: int, tag: str, method: str,
           extra: tuple[str, ...], gpu: int, epochs: int) -> None:
    """Train one arm with the arguments that its adapter actually owns."""
    workdir, _ = night._task_dirs(root, seed, tag)
    checkpoint = workdir / 'best_model.pth'
    log = root / 'logs' / f'{tag}_seed{seed}_train.log'
    if checkpoint.is_file():
        print(f'skip train {tag} seed{seed}: checkpoint exists', flush=True)
        return
    data_root = REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}'
    cmd = [
        night.PYTHON, '-m', 'baseline_zoo.run_one',
        '--model', method, '--stage', 'train',
        '--data-root', str(data_root),
        '--workdir', str(workdir),
        '--epochs', str(epochs), '--batch-size', '1',
        '--seed', str(seed), '--device', f'cuda:{gpu}',
        *night._wandb_args(tag, seed, 'train', ('3shot',)),
    ]
    if method == 'sacm_official':
        cmd += ['--sam-checkpoint', str(SAM_L)]
    else:
        cmd += [
            '--convlora-model-type', 'vit_b',
            '--convlora-checkpoint', str(SAM_B),
        ]
    cmd += list(extra)
    night._run(cmd, log)


def _worker(root: Path, gpu: int, tasks, epochs: int) -> None:
    for seed, tag, method, extra in tasks:
        try:
            _train(root, seed, tag, method, extra, gpu, epochs)
            night._evaluate(root, seed, tag, method, gpu)
        except Exception as exc:
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
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if len(args.gpus) != 2:
        raise SystemExit('NativeToken screen requires exactly two GPUs')
    data_root = REPO_ROOT / 'data' / f'sacm_3shot_seed{args.seed}'
    if not data_root.is_dir():
        raise SystemExit(f'missing seed root: {data_root}')
    if not SAM_B.is_file():
        raise SystemExit(f'missing SAM ViT-B checkpoint: {SAM_B}')
    if not SAM_L.is_file():
        raise SystemExit(f'missing SAM ViT-L checkpoint: {SAM_L}')

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
            target=_worker, args=(args.root, gpu, [task], args.epochs),
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
            '# Overnight screening',
            '# NativeToken vs SACM official seed0 screen')
        text = text.replace(
            'Matrix: seeds 0/1/2, 100 epochs, ViT-B, full 20-dataset test.',
            f'Matrix: seed {args.seed}, 100 epochs, SACM ViT-L vs '
            'NativeToken ViT-B, full 20-dataset test.')
        summary.write_text(text)
    print(f'wrote {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
