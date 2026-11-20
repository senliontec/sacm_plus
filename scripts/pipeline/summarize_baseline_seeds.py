#!/usr/bin/env python3
"""Summarize per-seed baseline tables into a 4-seed dataset-equal macro.

The canonical baseline pipeline writes one
``<run_root>/seed_<n>/eval/tables/all_metrics.csv`` per seed.  This script
validates that every expected test dataset is present, computes the
dataset-equal macro for each seed, then averages the four seeds.  It does not
recompute metrics or touch predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

METRICS = ('dice', 'iou', 'cldice', 'hd95')


def _display(metric: str, value: float) -> float:
    """Dice/IoU/clDice as percent; HD95 stays in pixels."""
    return value * 100.0 if metric != 'hd95' else value


def _expected_datasets(protocol: Path) -> list[str]:
    names: list[str] = []
    with protocol.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if str(row.get('split', '')).endswith('_test'):
                name = str(row.get('dataset', '')).strip()
                if name and name not in names:
                    names.append(name)
    if not names:
        raise SystemExit(f'no test datasets found in {protocol}')
    return names


def _read_seed(table: Path, expected: list[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with table.open(newline='') as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get('dataset', '')).strip()
            if dataset not in expected:
                continue
            if dataset in rows:
                raise SystemExit(f'{table}: duplicate dataset row {dataset}')
            values = {}
            for metric in METRICS:
                raw = row.get(f'{metric}_mean')
                if raw in (None, '', 'nan', 'NaN'):
                    raise SystemExit(f'{table}: missing {metric}_mean for {dataset}')
                values[metric] = float(raw)
            rows[dataset] = values
    missing = [name for name in expected if name not in rows]
    if missing:
        raise SystemExit(f'{table}: missing datasets: {missing}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--protocol', type=Path,
                        default=Path('data/sacm_3shot_seed0/protocol_manifest.csv'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('results/reports/cosna_server_4seed'))
    args = parser.parse_args()

    expected = _expected_datasets(args.protocol)
    per_seed = []
    for seed in args.seeds:
        table = args.run_root / f'seed_{seed}' / 'eval' / 'tables' / 'all_metrics.csv'
        if not table.is_file():
            raise SystemExit(f'missing evaluation table: {table}')
        rows = _read_seed(table, expected)
        macro = {
            metric: sum(rows[name][metric] for name in expected) / len(expected)
            for metric in METRICS
        }
        per_seed.append({
            'seed': int(seed),
            'table': str(table),
            'macro': {k: round(v, 8) for k, v in macro.items()},
            'macro_pct': {k: round(_display(k, v), 4)
                          for k, v in macro.items()},
            'datasets': {name: {k: round(v, 8) for k, v in rows[name].items()}
                         for name in expected},
        })

    macro = {
        metric: sum(item['macro'][metric] for item in per_seed) / len(per_seed)
        for metric in METRICS
    }
    payload = {
        'model': 'cosna',
        'run_root': str(args.run_root),
        'seeds': [int(s) for s in args.seeds],
        'datasets': expected,
        'protocol': str(args.protocol),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'metric_units': ('fraction for raw macro; macro_pct uses percent for '
                         'Dice/IoU/clDice and pixels for HD95'),
        'per_seed': per_seed,
        'macro': {k: round(v, 8) for k, v in macro.items()},
        'macro_pct': {k: round(_display(k, v), 4)
                      for k, v in macro.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'summary.json').write_text(
        json.dumps(payload, indent=1), encoding='utf-8')
    with (args.out_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', *METRICS])
        for item in per_seed:
            writer.writerow([item['seed'], *[
                f"{item['macro_pct'][metric]:.4f}" for metric in METRICS]])
        writer.writerow(['macro', *[
            f"{payload['macro_pct'][metric]:.4f}" for metric in METRICS]])
    lines = [
        '# CoSNA 4-seed summary',
        '',
        f'Run root: `{args.run_root}`',
        f'Protocol: `{args.protocol}`',
        f'Datasets: {len(expected)}',
        '',
        '| seed | Dice | IoU | clDice | HD95 |',
        '|---|---:|---:|---:|---:|',
    ]
    for item in per_seed:
        lines.append(
            f"| {item['seed']} | "
            + ' | '.join(f"{item['macro_pct'][m]:.4f}" for m in METRICS)
            + ' |')
    lines.append(
        '| **macro** | '
        + ' | '.join(f"**{payload['macro_pct'][m]:.4f}**" for m in METRICS)
        + ' |')
    (args.out_dir / 'summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload['macro_pct'], indent=1))


if __name__ == '__main__':
    main()
