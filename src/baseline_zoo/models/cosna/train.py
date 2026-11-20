"""Train CoSNA from a trained four-candidate BCTD-scaffold checkpoint.

CoSNA keeps the scaffold frozen and fits only the 18k-parameter TCSU
residual, so a random initialization is not a valid starting point.  This
wrapper copies the seed-specific BCTD champion checkpoint into the CoSNA
workdir (``prepare_init``), computes the absolute final epoch, and forwards
all remaining flags to the shared baseline runtime.

    python -m baseline_zoo.models.cosna.train \
      --data-root data/sacm_3shot_seed0 --workdir results/runs/cosna/seed_0 \
      --epochs-add 20 --seed 0 --device cuda:0 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from baseline_zoo.models.cosna.adapter import CoSNAAdapter
from baseline_zoo.models.cosna.prepare_init import prepare
from baseline_zoo.runtime import REPO_ROOT, main_for

DEFAULT_SOURCES = {
    0: REPO_ROOT / 'results/runs/topostrand/pilot200_bnd005_h2/seed_0/'
                    'best_model_fixed3.pth',
}
DEFAULT_SOURCE_TEMPLATE = (
    REPO_ROOT / 'results/runs/topostrand/bctd200/seed_{seed}/'
                 'best_model_fixed3.pth')


def default_source(seed: int) -> Path:
    return DEFAULT_SOURCES.get(seed, Path(str(DEFAULT_SOURCE_TEMPLATE).format(
        seed=int(seed))))


def _strip_option(argv: list[str], name: str) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            index += 2
            continue
        if token.startswith(name + '='):
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-checkpoint', type=Path, default=None)
    parser.add_argument('--epochs-add', type=int, default=20)
    parser.add_argument('--force-init', action='store_true')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    known, rest = parser.parse_known_args(argv)

    workdir = known.workdir
    checkpoint = workdir / 'best_model.pth'
    if checkpoint.is_file() and not known.force_init:
        payload = torch.load(checkpoint, map_location='cpu',
                             weights_only=False)
        source_epoch = int(payload.get('epoch', 0))
        final_epoch = source_epoch + int(known.epochs_add)
    else:
        source = known.source_checkpoint or default_source(known.seed)
        if not source.is_file():
            raise SystemExit(
                f'missing initialization checkpoint: {source}; pass '
                '--source-checkpoint or pre-populate the workdir')
        final_epoch = prepare(
            source, workdir, known.data_root,
            epochs_add=known.epochs_add, force=known.force_init)

    forwarded = _strip_option(rest, '--epochs')
    forwarded += [
        '--epochs', str(final_epoch),
        '--resume', '--resume-checkpoint', 'best',
    ]
    return main_for(CoSNAAdapter(), forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
