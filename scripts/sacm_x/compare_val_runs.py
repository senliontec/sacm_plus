#!/usr/bin/env python3
"""Print a compact validation comparison for two SACM-X runs.

Usage:
    python scripts/sacm_x/compare_val_runs.py \
        full_v2_skel=results/.../metrics_history.json \
        full_v3=results/.../metrics_history.json

Only stdlib is used.  The best epoch is the history entry with the highest
``val/dice`` (the checkpoint-selection metric).
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple


def _load(spec: str) -> Tuple[str, Dict]:
    if '=' not in spec:
        raise SystemExit(f'expected NAME=PATH, got {spec!r}')
    name, path = spec.split('=', 1)
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return name, payload


def _best_entry(history: List[Dict]) -> Dict:
    if not history:
        return {}
    return max(history, key=lambda item: float(item.get('val/dice', -1.0)))


def _datasets(entry: Dict) -> List[str]:
    names = set()
    for key in entry:
        if (key.startswith('val/') and key.endswith('/dice')
                and key != 'val/dice'):
            name = key[len('val/'):-len('/dice')]
            if name:
                names.add(name)
    return sorted(names)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    runs = [_load(spec) for spec in argv]

    print()
    print('Validation comparison (best epoch by val Dice)')
    print('=' * 78)
    print(f"{'preset':18s} {'best_ep':>7s} {'Dice':>8s} {'IoU':>8s} "
          f"{'clDice':>8s} {'HD95':>8s} {'last_ep':>7s} {'last_Dice':>9s}")
    for name, payload in runs:
        history = payload.get('history', [])
        best = _best_entry(history)
        last = history[-1] if history else {}
        print(f"{name:18s} {int(best.get('epoch', -1)):7d} "
              f"{float(best.get('val/dice', 0)):8.4f} "
              f"{float(best.get('val/iou', 0)):8.4f} "
              f"{float(best.get('val/cldice', 0)):8.4f} "
              f"{float(best.get('val/hd95', 0)):8.2f} "
              f"{int(last.get('epoch', -1)):7d} "
              f"{float(last.get('val/dice', 0)):9.4f}")

    baseline_name, baseline_payload = runs[0]
    candidate_name, candidate_payload = runs[1]
    baseline_best = _best_entry(baseline_payload.get('history', []))
    candidate_best = _best_entry(candidate_payload.get('history', []))
    baseline_sets = _datasets(baseline_best)
    candidate_sets = _datasets(candidate_best)
    shared = sorted(set(baseline_sets) & set(candidate_sets))
    if shared:
        print()
        print(f"Per-dataset val Dice: {candidate_name} - {baseline_name}")
        print('=' * 78)
        wins = 0
        for dataset in shared:
            key = f'val/{dataset}/dice'
            delta = (float(candidate_best.get(key, 0.0))
                     - float(baseline_best.get(key, 0.0)))
            wins += int(delta > 0)
            print(f'{dataset:24s} {delta:+8.4f}')
        print('-' * 78)
        print(f'{candidate_name} wins {wins}/{len(shared)} datasets')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
