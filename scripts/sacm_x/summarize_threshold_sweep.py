#!/usr/bin/env python3
"""Summarize ``threshold_audit.csv`` from one SACM-X evaluation.

The evaluator writes one row per (image, threshold) with quick binary metrics
and component statistics.  This script aggregates by threshold and reports
the trade-off between overlap, recall and fragmentation.  It is diagnostic:
do not choose a test-set threshold for the paper; calibrate on validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ('dice', 'iou', 'precision', 'recall')
COMPONENT_METRICS = (
    'components_pred',
    'components_gt',
    'component_count_error',
    'small_component_rate',
    'fp_area_ratio',
    'fn_area_ratio',
)


def _float(value, default=math.nan):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def _fmt(value, digits=4):
    return '—' if not math.isfinite(value) else f'{value:.{digits}f}'


def _load(path: Path):
    groups = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            threshold = _float(row.get('threshold'))
            if not math.isfinite(threshold):
                continue
            groups[threshold].append(row)
    if not groups:
        raise SystemExit(f'no threshold rows found in {path}')
    return groups


def _summarize(rows):
    item = {'n': len(rows)}
    for key in METRICS + COMPONENT_METRICS:
        item[key] = _mean([_float(row.get(key)) for row in rows])
    ratios = []
    for row in rows:
        pred = _float(row.get('components_pred'), 0.0)
        gt = max(_float(row.get('components_gt'), 0.0), 1.0)
        ratios.append(pred / gt)
    item['components_ratio'] = _mean(ratios)
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--baseline', type=float, default=0.0,
        help='reference logit threshold for deltas (default: 0.0)')
    args = parser.parse_args()

    groups = _load(args.input)
    summary = {
        threshold: _summarize(rows)
        for threshold, rows in sorted(groups.items())
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fields = ['threshold', 'n', *METRICS, 'components_ratio',
              'components_pred', 'components_gt', 'component_count_error',
              'small_component_rate', 'fp_area_ratio', 'fn_area_ratio']
    csv_path = args.out_dir / 'threshold_summary.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold, item in summary.items():
            writer.writerow({'threshold': threshold, **item})

    baseline_threshold = min(summary, key=lambda value: abs(value - args.baseline))
    baseline = summary[baseline_threshold]
    best_threshold = max(
        summary, key=lambda value: (
            summary[value]['dice'] if math.isfinite(summary[value]['dice'])
            else -math.inf,
            value,
        ))
    best = summary[best_threshold]

    lines = [
        '# OCTA threshold sweep',
        '',
        f'- Input: `{args.input}`',
        f'- Images: {baseline["n"]}',
        f'- Reference threshold: {baseline_threshold:g}',
        '',
        '> Diagnostic only. Do not select a test-set threshold for the paper; '
        'calibrate on validation.',
        '',
        '| threshold | n | Dice | IoU | Precision | Recall | comp ratio | '
        'small comp | FP area | FN area |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for threshold in sorted(summary):
        item = summary[threshold]
        lines.append(
            f"| {threshold:g} | {item['n']} | {_fmt(item['dice'])} | "
            f"{_fmt(item['iou'])} | {_fmt(item['precision'])} | "
            f"{_fmt(item['recall'])} | {_fmt(item['components_ratio'], 2)} | "
            f"{_fmt(item['small_component_rate'], 3)} | "
            f"{_fmt(item['fp_area_ratio'], 3)} | "
            f"{_fmt(item['fn_area_ratio'], 3)} |"
        )
    lines += [
        '',
        '## Extremes',
        '',
        f'- Best Dice threshold: {best_threshold:g} '
        f"(Dice={_fmt(best['dice'])}, recall={_fmt(best['recall'])})",
        f"- Reference Dice: {_fmt(baseline['dice'])} "
        f"(recall={_fmt(baseline['recall'])})",
        f"- Best Dice delta vs reference: "
        f"{_fmt(best['dice'] - baseline['dice'], 4)}",
        f"- Best threshold components ratio: "
        f"{_fmt(best['components_ratio'], 2)} "
        f"(reference {_fmt(baseline['components_ratio'], 2)})",
        '',
    ]
    md_path = args.out_dir / 'threshold_summary.md'
    md_path.write_text('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {md_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
