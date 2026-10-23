#!/usr/bin/env python3
"""Build a readable four-metric SACM-X report from aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROLE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2, 'All': 3}
METRICS = [('dice', 'Dice'), ('iou', 'IoU'),
           ('cldice', 'clDice'), ('hd95', 'HD95')]
PERCENT = {'dice', 'iou', 'cldice'}


def read_csv(path: Path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fmt(value, metric):
    if value in (None, '', 'nan', 'NaN', 'None'):
        return '—'
    try:
        number = float(value)
    except ValueError:
        return '—'
    if not math.isfinite(number):
        return '—'
    if metric in PERCENT:
        number *= 100.0
    return f'{number:.2f}'


def role_key(role):
    return ROLE_ORDER.get(role, 9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', type=Path, required=True)
    parser.add_argument('--preset', default='full')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    tables = args.tables_dir
    out_dir = args.out_dir or tables
    all_path = tables / 'all_metrics.csv'
    macro_path = tables / 'macro_metrics.csv'
    if not all_path.is_file():
        raise SystemExit(f'missing {all_path}; run evaluation + aggregate first')
    if not macro_path.is_file():
        raise SystemExit(f'missing {macro_path}; run aggregate first')

    rows = [r for r in read_csv(all_path) if r.get('preset') == args.preset]
    if not rows:
        raise SystemExit(f'no rows for preset={args.preset!r} in {all_path}')
    rows.sort(key=lambda r: (role_key(r.get('role', '')),
                             r.get('dataset', '')))

    macro_rows = [r for r in read_csv(macro_path)
                  if r.get('preset') == args.preset]
    macro = {}
    for row in macro_rows:
        metric = row.get('metric')
        if metric in dict(METRICS):
            macro.setdefault(row.get('role', ''), {})[metric] = row

    lines = ['# SACM-X final report', '']
    lines.append(f'- Preset: `{args.preset}`')
    if args.checkpoint:
        lines.append(f'- Checkpoint: `{args.checkpoint}`')
    lines.append(f'- Per-dataset source: `{all_path}`')
    lines.append(f'- Macro source: `{macro_path}`')
    lines.append('')
    lines.append('## Macro (equal-dataset average)')
    lines.append('')
    lines.append('| Role | Dice (%) | IoU (%) | clDice (%) | HD95 | n |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for role in sorted(macro, key=role_key):
        vals = macro[role]
        cells = []
        n = '—'
        for metric, _ in METRICS:
            row = vals.get(metric)
            cells.append(fmt(row.get('mean') if row else None, metric))
            if row and row.get('n_datasets'):
                n = row['n_datasets']
        lines.append(f"| {role} | " + ' | '.join(cells) + f' | {n} |')

    lines.append('')
    lines.append('## Per dataset')
    lines.append('')
    lines.append('| Dataset | Role | Dice (%) | IoU (%) | clDice (%) | HD95 |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for row in rows:
        cells = [fmt(row.get(f'{metric}_mean'), metric) for metric, _ in METRICS]
        lines.append(f"| {row.get('dataset', '')} | {row.get('role', '')} | " +
                     ' | '.join(cells) + ' |')

    markdown = '\n'.join(lines) + '\n'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'final_report.md'
    report_path.write_text(markdown)
    print(markdown)
    print(f'report: {report_path}')

    csv_path = out_dir / 'final_report.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'role', 'dice', 'iou', 'cldice', 'hd95'])
        for row in rows:
            writer.writerow([row.get('dataset', ''), row.get('role', '')] +
                            [row.get(f'{metric}_mean', '') for metric, _ in METRICS])
    print(f'csv:    {csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
