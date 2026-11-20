"""Train CoSNA from scratch or from a CoSNA initialization checkpoint.

``--init-mode source`` keeps the canonical two-stage recipe: copy a trained
CoSNA/BCTD-scaffold checkpoint, freeze the scaffold and fit the TCSU
residual.  ``--init-mode scratch`` does not read any initialization
checkpoint; it trains the scaffold adapters/decoder and the TCSU residual
jointly from the SAM ViT-L checkpoint, so the run is fully independent of
BCTD files.

    python -m baseline_zoo.models.cosna.train \
      --init-mode scratch \
      --data-root data/sacm_3shot_seed0 --workdir results/runs/cosna/seed_0 \
      --epochs-total 800 --seed 0 --device cuda:0 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from baseline_zoo.models.cosna.adapter import CoSNAAdapter
from baseline_zoo.models.cosna.prepare_init import prepare
from baseline_zoo.runtime import main_for


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
    parser.add_argument('--init-mode', choices=['source', 'scratch'],
                        default='source',
                        help='source=copy a trained CoSNA init checkpoint; '
                             'scratch=train from SAM ViT-L only')
    parser.add_argument('--source-checkpoint', type=Path, default=None)
    parser.add_argument('--epochs-add', type=int, default=20,
                        help='additional TCSU epochs (canonical CoSNA uses 20)')
    parser.add_argument('--epochs-total', type=int, default=0,
                        help='absolute final epoch; overrides --epochs-add')
    parser.add_argument('--force-init', action='store_true')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    known, rest = parser.parse_known_args(argv)
    if known.epochs_add < 0 or known.epochs_total < 0:
        raise SystemExit('epochs-add/epochs-total must be non-negative')

    workdir = known.workdir
    data_root = Path(known.data_root).resolve()
    checkpoint = workdir / 'best_model.pth'
    if known.init_mode == 'scratch':
        if checkpoint.is_file():
            print(f'[cosna] scratch mode: ignoring existing {checkpoint}',
                  flush=True)
        source_epoch = 0
        final_epoch = (int(known.epochs_total)
                       if int(known.epochs_total) > 0
                       else int(known.epochs_add))
        if final_epoch <= 0:
            raise SystemExit('scratch mode needs --epochs-total or '
                             '--epochs-add > 0')
        resume_args = ['--cosna-train-scope', 'full']
        print(f'[cosna] seed={known.seed} init=scratch '
              f'final_epoch={final_epoch}', flush=True)
    else:
        if checkpoint.is_file() and not known.force_init:
            payload = torch.load(checkpoint, map_location='cpu',
                                 weights_only=False)
            source_epoch = int(payload.get('epoch', 0))
        else:
            source = known.source_checkpoint
            if source is None or not source.is_file():
                raise SystemExit(
                    'source mode requires an explicit --source-checkpoint '
                    '(a CoSNA init checkpoint); use --init-mode scratch '
                    'for the BCTD-free recipe, or pre-populate the workdir')
            # Establish the CoSNA-labelled source checkpoint first, then
            # decide the final epoch.  This keeps --epochs-total independent
            # of the source checkpoint's epoch.
            prepare(source, workdir, data_root, epochs_add=0,
                    force=known.force_init)
            payload = torch.load(checkpoint, map_location='cpu',
                                 weights_only=False)
            source_epoch = int(payload.get('epoch', 0))

        if int(known.epochs_total) > 0:
            final_epoch = int(known.epochs_total)
        else:
            final_epoch = source_epoch + int(known.epochs_add)
        if final_epoch <= source_epoch:
            raise SystemExit(
                f'final epoch {final_epoch} must be greater than the '
                f'resume epoch {source_epoch}')
        resume_args = ['--resume', '--resume-checkpoint', 'best']
        print(f'[cosna] seed={known.seed} resume_epoch={source_epoch} '
              f'final_epoch={final_epoch}', flush=True)

    forwarded = _strip_option(rest, '--epochs')
    forwarded += [
        '--stage', 'train',
        '--workdir', str(known.workdir),
        '--data-root', str(data_root),
        '--seed', str(known.seed),
        '--epochs', str(final_epoch),
        *resume_args,
    ]
    return main_for(CoSNAAdapter(), forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
