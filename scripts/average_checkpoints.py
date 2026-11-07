#!/usr/bin/env python
"""Equal-weight checkpoint averaging (SWA) for baseline_zoo checkpoints.

The unified checkpoints store ``model_state_dict`` plus the run config, so an
averaged checkpoint remains directly loadable by ``run_one --stage predict``
and by the paper evaluator.

Usage:
  PYTHONPATH=src python scripts/average_checkpoints.py \
    --checkpoint results/diagnostics/x/checkpoints/epoch_0020.pth \
                 results/diagnostics/x/checkpoints/epoch_0030.pth \
                 results/diagnostics/x/checkpoints/epoch_0040.pth \
    --output results/diagnostics/x/swa_model.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    paths = [Path(p) for p in args.checkpoint]
    payloads = [torch.load(p, map_location='cpu', weights_only=False)
                for p in paths]
    adapters = {p.get('adapter') for p in payloads}
    if len(adapters) > 1:
        raise SystemExit(f'mixed adapters: {sorted(adapters)}')

    states = [p.get('model_state_dict') or {} for p in payloads]
    if any(set(s) != set(states[0]) for s in states[1:]):
        raise SystemExit('checkpoint state_dict keys do not match')

    averaged = {}
    for key, first in states[0].items():
        if not torch.is_floating_point(first):
            for state in states[1:]:
                if not torch.equal(state[key], first):
                    raise SystemExit(
                        f'non-floating buffer {key!r} differs across '
                        'checkpoints; cannot average')
            averaged[key] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float64)
        for state in states:
            value = state[key]
            if value.shape != first.shape:
                raise SystemExit(f'shape mismatch for {key!r}')
            acc += value.to(torch.float64)
        averaged[key] = (acc / len(states)).to(first.dtype)

    base = dict(payloads[0])
    base['model_state_dict'] = averaged
    epochs = [int(p.get('epoch', 0)) for p in payloads]
    base['epoch'] = max(epochs)
    base['epoch_completed'] = max(epochs)
    base['swa'] = {
        'checkpoints': [str(p) for p in paths],
        'epochs': epochs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(f'averaged {len(paths)} checkpoints (epochs {epochs}) -> {out}')


if __name__ == '__main__':
    main()
