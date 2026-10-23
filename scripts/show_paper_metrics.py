#!/usr/bin/env python3
"""Print the four paper metrics per dataset from results/paper/*.csv.

Examples:
    python scripts/show_paper_metrics.py
    python scripts/show_paper_metrics.py --method unet
    python scripts/show_paper_metrics.py --raw   # keep Dice/IoU/clDice in 0-1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2}
METRICS = [('dice', 'Dice'), ('iou', 'IoU'),
           ('cldice', 'clDice'), ('hd95', 'HD95')]
PERCENT_METRICS = {'dice', 'iou', 'cldice'}


def read_csv(path: Path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fmt(value, metric, percent=True):
    if value in (None, '', 'nan', 'NaN', 'None'):
        return 'NaN'
    try:
        number = float(value)
    except ValueError:
        return 'NaN'
    if percent and metric in PERCENT_METRICS:
        number *= 100
    return f'{number:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        default=REPO_ROOT / 'results' / 'paper' /
                        'paper_metrics.csv')
    parser.add_argument('--macro-csv', type=Path,
                        default=REPO_ROOT / 'results' / 'paper' /
                        'paper_macro.csv')
    parser.add_argument('--method', default=None)
    parser.add_argument('--raw', action='store_true',
                        help='show Dice/IoU/clDice in 0-1 instead of percent')
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(
            f'missing {args.csv}; run scripts/run_baseline_test.py first')

    rows = read_csv(args.csv)
    if args.method:
        rows = [r for r in rows if r.get('method') == args.method]
    if not rows:
        raise SystemExit(f'no rows for method={args.method!r} in {args.csv}')

    methods = sorted({r['method'] for r in rows})
    macro_rows = []
    if args.macro_csv.is_file():
        macro_rows = read_csv(args.macro_csv)
        if args.method:
            macro_rows = [r for r in macro_rows
                          if r.get('method') == args.method]

    percent = not args.raw
    for method in methods:
        print(f'\n[{method}] per-dataset (mean)')
        print('  ' + f'{"dataset":<14} {"role":<7} ' +
              ' '.join(f'{label:>8}' for _, label in METRICS))
        method_rows = sorted(
            (r for r in rows if r['method'] == method),
            key=lambda r: (ROLE_ORDER.get(r.get('role', ''), 9),
                           r.get('dataset', '')))
        for row in method_rows:
            cells = [fmt(row.get(f'{key}_mean'), key, percent)
                     for key, _ in METRICS]
            print('  ' + f'{row.get("dataset", ""):<14} '
                  f'{row.get("role", ""):<7} ' +
                  ' '.join(f'{cell:>8}' for cell in cells))

        macro_method = [r for r in macro_rows if r['method'] == method]
        if macro_method:
            print(f'[{method}] macro')
            order = {role: index for index, role in
                     enumerate(['Seen', 'Unseen', 'Novel', 'All'])}
            macro_method.sort(key=lambda r: order.get(r.get('role', ''), 9))
            for row in macro_method:
                cells = [fmt(row.get(f'{key}_mean'), key, percent)
                         for key, _ in METRICS]
                print('  ' + f'{row.get("role", ""):<7}          ' +
                      ' '.join(f'{cell:>8}' for cell in cells))
    print(f'\nsource: {args.csv}')
    if macro_rows:
        print(f'macro:  {args.macro_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
