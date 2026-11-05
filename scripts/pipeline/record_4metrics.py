#!/usr/bin/env python3
"""Record per-dataset Dice/IoU/clDice/HD95 for one model/variant/seeds.

Input (written by the canonical evaluators):
  results/runs/<model>/<variant>/seed_<s>/eval/tables/all_metrics.csv

Outputs:
  results/runs/<model>/<variant>/seed_<s>/eval/tables/per_dataset_4metrics.csv
  results/reports/<model>_<variant>_seeds<a>_<b>_per_dataset.csv
  results/reports/<model>_<variant>_seeds<a>_<b>_mean_std.csv
  results/reports/<model>_<variant>_seeds<a>_<b>_mean_std.md

The helper is deliberately model-agnostic: the same all_metrics.csv schema is
used by the Skein evaluator and by the official-baseline evaluator.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2}


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return number if math.isfinite(number) else float('nan')


def _metric(row, name):
    for key in (f'{name}_mean', name, f'mean_{name}'):
        if key in row and row[key] not in (None, ''):
            value = _float(row[key])
            if math.isfinite(value):
                return value
    return float('nan')


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_seed(path: Path, expected: int):
    with path.open(newline='') as handle:
        raw = [row for row in csv.DictReader(handle) if row.get('dataset')]
    if expected and len(raw) != expected:
        raise SystemExit(
            f'{path}: found {len(raw)} dataset rows, expected {expected}')
    rows = []
    for row in raw:
        rows.append({
            'dataset': row['dataset'].strip(),
            'role': (row.get('role') or '').strip(),
            **{name: _metric(row, name) for name in METRICS},
        })
    rows.sort(key=lambda item: (ROLE_ORDER.get(item['role'], 9),
                                item['dataset'].lower()))
    return rows


def _fmt(value: float) -> str:
    return '—' if not math.isfinite(value) else f'{value:.4f}'


def _print_table(rows, title):
    print('')
    print(title)
    print(f"{'dataset':28s} {'role':7s} " +
          ' '.join(f'{name:>9}' for name in METRICS))
    for row in rows:
        print(f"{row['dataset']:28s} {row['role']:7s} " +
              ' '.join(f'{_fmt(row[name]):>9}' for name in METRICS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=None)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--expected-datasets', type=int, default=20)
    args = parser.parse_args()

    repo = (args.repo_root.resolve() if args.repo_root is not None
            else Path(__file__).resolve().parents[2])
    seeds = sorted(dict.fromkeys(args.seeds))
    all_rows = []
    per_seed = {}

    for seed in seeds:
        metrics_path = (repo / 'results' / 'runs' / args.model / args.variant /
                        f'seed_{seed}' / 'eval' / 'tables' / 'all_metrics.csv')
        if not metrics_path.is_file():
            raise SystemExit(f'missing metrics file: {metrics_path}')
        rows = _read_seed(metrics_path, args.expected_datasets)
        per_seed[seed] = rows
        _write_csv(
            metrics_path.parent / 'per_dataset_4metrics.csv',
            ['dataset', 'role', *METRICS], rows)
        for row in rows:
            all_rows.append({
                'model': args.model,
                'variant': args.variant,
                'seed': seed,
                **row,
            })
        _print_table(rows, f'=== {args.model}/{args.variant} seed {seed} '
                           f'({len(rows)} datasets) ===')

    report_dir = repo / 'results' / 'reports'
    tag = (f'{args.model}_{args.variant}_seeds'
           f'{seeds[0]}_{seeds[-1]}'
           if len(seeds) > 1 else
           f'{args.model}_{args.variant}_seed{seeds[0]}')

    long_path = report_dir / f'{tag}_per_dataset.csv'
    _write_csv(
        long_path,
        ['model', 'variant', 'seed', 'dataset', 'role', *METRICS],
        all_rows)

    grouped = {}
    for row in all_rows:
        grouped.setdefault((row['dataset'], row['role']), []).append(row)

    summary_rows = []
    for (dataset, role), rows in sorted(
            grouped.items(),
            key=lambda item: (ROLE_ORDER.get(item[0][1], 9),
                              item[0][0].lower())):
        summary = {
            'dataset': dataset,
            'role': role,
            'n_seeds': len(rows),
        }
        for name in METRICS:
            values = [row[name] for row in rows if math.isfinite(row[name])]
            summary[f'{name}_mean'] = (
                statistics.fmean(values) if values else float('nan'))
            summary[f'{name}_std'] = (
                statistics.stdev(values) if len(values) > 1 else 0.0)
        summary_rows.append(summary)

    summary_path = report_dir / f'{tag}_mean_std.csv'
    _write_csv(
        summary_path,
        ['dataset', 'role', 'n_seeds',
         *[f'{name}_{suffix}' for name in METRICS
           for suffix in ('mean', 'std')]],
        summary_rows)

    md = [
        f'# {args.model} / {args.variant}: per-dataset test results',
        '',
        f'Seeds: {seeds}. Expected datasets per seed: '
        f'{args.expected_datasets}.',
        '',
        '| Dataset | Role | Dice (mean±std) | IoU (mean±std) | '
        'clDice (mean±std) | HD95 (mean±std) |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        cells = []
        for name in METRICS:
            mean = row[f'{name}_mean']
            std = row[f'{name}_std']
            if not math.isfinite(mean):
                cells.append('—')
            elif row['n_seeds'] > 1:
                cells.append(f'{mean:.4f} ± {std:.4f}')
            else:
                cells.append(f'{mean:.4f}')
        md.append(f"| {row['dataset']} | {row['role']} | " +
                  ' | '.join(cells) + ' |')
    md_path = report_dir / f'{tag}_mean_std.md'
    md_path.write_text('\n'.join(md) + '\n', encoding='utf-8')

    print('')
    print(f'per-seed records: '
          f'results/runs/{args.model}/{args.variant}/seed_*/'
          f'eval/tables/per_dataset_4metrics.csv')
    print(f'long table:       {long_path}')
    print(f'mean/std table:   {summary_path}')
    print(f'markdown table:   {md_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
