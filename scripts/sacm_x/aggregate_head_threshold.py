#!/usr/bin/env python3
"""Aggregate ``head_threshold_audit.csv`` from a calibration split.

The evaluator writes one row per (image, head, threshold).  This tool maps
each image back to its dataset through the protocol manifest and prints
per-dataset and equal-dataset macro Dice.  It is deliberately read-only:
the only output is an optional CSV summary.

This is for val/support calibration only.  Do not use a test split to pick
the deployment head or threshold.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd


def protocol_map(path: Path, split: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('split') != split:
                continue
            image = Path(row['image_path']).name
            mapping[image] = row['dataset']
            mapping[Path(image).stem] = row['dataset']
    if not mapping:
        raise SystemExit(f'no {split!r} rows in {path}')
    return mapping


def attach_dataset(rows: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    def lookup(value: str) -> str:
        base = os.path.basename(str(value))
        stem = os.path.splitext(base)[0]
        return mapping.get(base) or mapping.get(stem) or 'unknown'

    rows = rows.copy()
    rows['dataset'] = rows['image'].map(lookup)
    unknown = sorted(rows.loc[rows['dataset'] == 'unknown', 'image'].unique())
    if unknown:
        print(f'[warn] {len(unknown)} images not in manifest: {unknown[:5]}')
    return rows


def fmt(value: float) -> str:
    return 'nan' if pd.isna(value) else f'{value:.3f}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True,
                        help='head_threshold_audit.csv from evaluate.py')
    parser.add_argument('--protocol', type=Path, required=True,
                        help='protocol_manifest.csv (image -> dataset)')
    parser.add_argument('--split', default='val', choices=['val', 'train'],
                        help='protocol split used for calibration')
    parser.add_argument('--out', type=Path, default=None,
                        help='optional per_dataset_head_threshold.csv')
    parser.add_argument('--sort-metric', default='dice',
                        help='metric used to rank policies (default: dice)')
    args = parser.parse_args()

    if not args.audit.is_file():
        raise SystemExit(f'missing audit CSV: {args.audit}')
    if not args.protocol.is_file():
        raise SystemExit(f'missing protocol: {args.protocol}')

    rows = pd.read_csv(args.audit)
    required = {'image', 'head', 'threshold', args.sort_metric}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f'{args.audit}: missing columns {sorted(missing)}')
    rows = attach_dataset(rows, protocol_map(args.protocol, args.split))

    metric_cols = [c for c in (
        'dice', 'iou', 'precision', 'recall', 'components_pred',
        'components_gt', 'small_component_rate', 'fp_area_ratio',
        'fn_area_ratio') if c in rows.columns]
    grouped = rows.groupby(
        ['dataset', 'head', 'threshold'], dropna=False).agg(
        n=('image', 'nunique'),
        **{c: (c, 'mean') for c in metric_cols},
    ).reset_index()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(args.out, index=False)
        print(f'CSV: {args.out}')

    print(f'\nCalibration split: {args.split}')
    print(f'Images: {rows["image"].nunique()}; datasets: {rows["dataset"].nunique()}')
    print(f'Rows: {len(rows)}; heads: {sorted(rows["head"].unique())}; '
          f'thresholds: {sorted(rows["threshold"].unique())}')

    print('\n== per-dataset best (rank by mean ' + args.sort_metric + ') ==')
    print('dataset,n,head,threshold,' + ','.join(metric_cols))
    for dataset, group in grouped.groupby('dataset', sort=True):
        best = group.sort_values(
            [args.sort_metric, 'dice'], ascending=False).iloc[0]
        vals = ','.join(fmt(best[c]) for c in metric_cols)
        print(f'{dataset},{int(best["n"])},{int(best["head"])},'
              f'{float(best["threshold"]):.1f},{vals}')

    print('\n== equal-dataset macro (rank by mean ' + args.sort_metric + ') ==')
    macro = grouped.groupby(['head', 'threshold'], dropna=False).agg(
        n_datasets=('dataset', 'nunique'),
        **{c: (c, 'mean') for c in metric_cols},
    ).reset_index().sort_values(
        [args.sort_metric, 'dice'], ascending=False)
    print('head,threshold,n_datasets,' + ','.join(metric_cols))
    for _, row in macro.iterrows():
        vals = ','.join(fmt(row[c]) for c in metric_cols)
        print(f'{int(row["head"])},{float(row["threshold"]):.1f},'
              f'{int(row["n_datasets"])},{vals}')

    reference = grouped[(grouped['head'] == 2) & (grouped['threshold'] == 0.0)]
    if not reference.empty:
        print('\n== fixed head2 / threshold0 reference ==')
        print('dataset,n,dice,iou,precision,recall')
        for _, row in reference.sort_values('dataset').iterrows():
            print(f'{row["dataset"]},{int(row["n"])},{fmt(row["dice"])},'
                  f'{fmt(row["iou"])},{fmt(row["precision"])},'
                  f'{fmt(row["recall"])}')
        print(f'macro_dice,{fmt(reference["dice"].mean())}')

    small = grouped[grouped['n'] < 10]['dataset'].nunique()
    if small:
        print(f'\n[warn] {small} dataset(s) have <10 calibration images; '
              'per-dataset winners are noisy. Prefer a global policy or '
              'collect more labeled support.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
