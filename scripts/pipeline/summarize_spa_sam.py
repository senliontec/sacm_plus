#!/usr/bin/env python3
"""Summarize SPA-SAM seed runs and optional DCD/SACM reference tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLES = ('Seen', 'Unseen', 'Novel', 'All')
DISPLAY = {'dice': 'Dice', 'iou': 'IoU', 'cldice': 'clDice', 'hd95': 'HD95'}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return number if math.isfinite(number) else float('nan')


def _finite(values):
    numbers = []
    for value in values:
        if value is None:
            continue
        number = _number(value)
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def mean(values):
    numbers = _finite(values)
    return statistics.mean(numbers) if numbers else float('nan')


def std(values):
    numbers = _finite(values)
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def read_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def role_macro(rows, role):
    selected = rows if role == 'All' else [
        row for row in rows if row.get('role') == role]
    return {m: mean(row.get(f'{m}_mean') for row in selected)
            for m in METRICS}


def dataset_macro(rows):
    return {row['dataset']: {m: _number(row.get(f'{m}_mean'))
                             for m in METRICS} for row in rows}


def find_tables(root):
    patterns = ('seed_*/eval/tables/all_metrics.csv',
                'eval/seed_*/tables/all_metrics.csv',
                'seed_*/tables/all_metrics.csv')
    found = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            for part in path.parts:
                if part.startswith('seed_'):
                    try:
                        found.setdefault(int(part.split('_', 1)[1]), path)
                    except ValueError:
                        pass
                    break
    return found


def load(root):
    return {seed: read_rows(path) for seed, path in find_tables(root).items()}


def summarize(seed_rows):
    seeds = sorted(seed_rows)
    per_seed = {s: role_macro(seed_rows[s], 'All') for s in seeds}
    macro = {m: mean(per_seed[s][m] for s in seeds) for m in METRICS}
    macro_std = {m: std(per_seed[s][m] for s in seeds) for m in METRICS}
    roles = {role: {m: mean(role_macro(seed_rows[s], role)[m]
                           for s in seeds) for m in METRICS}
             for role in ROLES}
    datasets = {}
    names = set()
    for rows in seed_rows.values():
        names.update(dataset_macro(rows))
    for name in sorted(names):
        datasets[name] = {
            m: mean(dataset_macro(seed_rows[s]).get(name, {}).get(m)
                    for s in seeds) for m in METRICS}
    return seeds, per_seed, macro, macro_std, roles, datasets


def epochs(root):
    values = []
    for path in sorted(root.glob('seed_*/train_config.json')):
        try:
            value = json.loads(path.read_text()).get('args', {}).get('epochs')
            if value is not None:
                values.append(int(value))
        except (OSError, ValueError, TypeError):
            pass
    return values[0] if values and len(set(values)) == 1 else None


def fmt(metric, value):
    """Dice/IoU/clDice are shown as percentages; HD95 stays in pixels."""
    number = _number(value)
    if not math.isfinite(number):
        return 'nan'
    return f'{number:.2f}' if metric == 'hd95' else f'{100.0 * number:.2f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dcd-root', type=Path, default=None)
    parser.add_argument('--sacm-root', type=Path, default=None)
    parser.add_argument('--name', default='SPA-SAM')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    seed_rows = load(args.root)
    if not seed_rows:
        raise SystemExit(f'no seed tables under {args.root}')
    seeds, per_seed, macro, macro_std, roles, datasets = summarize(seed_rows)

    with open(args.out / 'macro.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['method', 'seed', *METRICS])
        writer.writeheader()
        for seed in seeds:
            writer.writerow({'method': args.name, 'seed': seed,
                             **per_seed[seed]})
        writer.writerow({'method': args.name, 'seed': 'mean', **macro})
        writer.writerow({'method': args.name, 'seed': 'std', **macro_std})

    baseline = {}
    dcd_datasets = None
    if args.dcd_root and args.dcd_root.is_dir():
        dcd_rows = load(args.dcd_root)
        if dcd_rows:
            _, _, dcd_macro, _, _, dcd_data = summarize(dcd_rows)
            baseline['DCD'] = dcd_macro
            dcd_datasets = {}
            for name in datasets:
                if name in dcd_data:
                    dcd_datasets[name] = {
                        'ours': datasets[name], 'dcd': dcd_data[name],
                        'delta': {m: datasets[name][m] - dcd_data[name][m]
                                  for m in METRICS}}
            with open(args.out / 'per_dataset_vs_dcd.csv', 'w',
                      newline='') as handle:
                fields = ['dataset']
                for m in METRICS:
                    fields += [f'ours_{m}', f'dcd_{m}', f'delta_{m}']
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for name, values in sorted(dcd_datasets.items()):
                    row = {'dataset': name}
                    for m in METRICS:
                        row[f'ours_{m}'] = values['ours'][m]
                        row[f'dcd_{m}'] = values['dcd'][m]
                        row[f'delta_{m}'] = values['delta'][m]
                    writer.writerow(row)
    if args.sacm_root and args.sacm_root.is_dir():
        sacm_rows = load(args.sacm_root)
        if sacm_rows:
            baseline['SACM'] = summarize(sacm_rows)[2]

    lines = [f'# {args.name} summary', '']
    budget = epochs(args.root)
    if budget is not None:
        lines += [f'Training budget: {budget} epochs per seed.', '']
    lines += ['## Macro (20 datasets, mean ± std across seeds)', '',
              '| Method | Dice | IoU | clDice | HD95 |',
              '|---|---:|---:|---:|---:|']
    lines.append(f"| {args.name} | " + ' | '.join(
        f'{fmt(m, macro[m])} ± {fmt(m, macro_std[m])}'
        for m in METRICS) + ' |')
    for name, values in baseline.items():
        lines.append(f'| {name} | ' + ' | '.join(
            fmt(m, values[m]) for m in METRICS) + ' |')
    lines += ['', '## Per seed', '',
              '| Seed | Dice | IoU | clDice | HD95 |',
              '|---|---:|---:|---:|---:|']
    for seed in seeds:
        lines.append(f'| {seed} | ' + ' | '.join(
            fmt(m, per_seed[seed][m]) for m in METRICS) + ' |')
    lines += ['', '## Per role (seed mean)', '',
              '| Role | Dice | IoU | clDice | HD95 |',
              '|---|---:|---:|---:|---:|']
    for role in ROLES:
        lines.append(f'| {role} | ' + ' | '.join(
            fmt(m, roles[role][m]) for m in METRICS) + ' |')
    if dcd_datasets is not None:
        lines += ['', '## Per-dataset wins vs DCD (seed mean)', '']
        for m in METRICS:
            if m == 'hd95':
                wins = sum(1 for v in dcd_datasets.values()
                           if v['delta'][m] < 0)
                losses = sum(1 for v in dcd_datasets.values()
                             if v['delta'][m] > 0)
                delta_text = f'{mean(v["delta"][m] for v in dcd_datasets.values()):.2f}'
            else:
                wins = sum(1 for v in dcd_datasets.values()
                           if v['delta'][m] > 0)
                losses = sum(1 for v in dcd_datasets.values()
                             if v['delta'][m] < 0)
                delta_text = (
                    f'{100.0 * mean(v["delta"][m] for v in dcd_datasets.values()):.2f} pp')
            lines.append(f'- {DISPLAY[m]}: {wins} wins / {losses} losses, '
                         f'mean delta {delta_text}')
    lines += ['', 'Dice/IoU/clDice are percentages; HD95 is in pixels. '
              'macro.csv stores raw fractions.',
              'Seeds use distinct official 3-shot splits; evaluation uses no '
              'target labels or support images.', '']
    (args.out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {args.out}/summary.md')
    print(f'wrote {args.out}/macro.csv')


if __name__ == '__main__':
    main()
