#!/usr/bin/env python3
"""One-click seed0 screen: Conv-LoRA vs Conv-LoRA + SDSConv.

Both models run 100 epochs on the same 3-shot split and are evaluated on all
20 test datasets.  W&B uses the shared server/group convention.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'curvfield'))
import run_night as night  # noqa: E402

REPO_ROOT = night.REPO_ROOT
DEFAULT_ROOT = REPO_ROOT / 'results' / 'sdsconv_screen'
SAM_B = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'
WANDB_GROUP = 'sdsconv_screen'
MODELS = (
    ('convlora', 'convlora', ()),
    ('sdsconv', 'sdsconv',
     ('--sdsconv-kernel-size', '7',
      '--sdsconv-hidden', '64',
      '--sdsconv-new-lr-mult', '10')),
)


def _worker(root: Path, gpu: int, tasks, epochs: int) -> None:
    for seed, tag, method, extra in tasks:
        try:
            night._train(root, seed, tag, method, extra, gpu, epochs)
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
        raise SystemExit('SDSConv screen requires exactly two GPUs')
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
            'Matrix: seeds 0/1/2, 100 epochs, ViT-B, full 20-dataset test.',
            f'Matrix: seed {args.seed}, 100 epochs, ViT-B, full 20-dataset test.')
        text = text.replace(
            'Raw training/eval logs: `results/curvfield_night/logs/`.',
            f'Raw training/eval logs: `{args.root.relative_to(REPO_ROOT)}/logs/`.')
        summary.write_text(text)
    print(f'wrote {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
