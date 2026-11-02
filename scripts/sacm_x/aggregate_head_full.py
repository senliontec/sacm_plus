#!/usr/bin/env python3
"""Aggregate the full per-head audit (Dice/clDice/HD95/ASSD/NSD)."""

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
                        help='head_full_audit.csv from evaluate.py')
    parser.add_argument('--protocol', type=Path, required=True,
                        help='protocol_manifest.csv (image -> dataset)')
    parser.add_argument('--split', default='val', choices=['val', 'train'])
    parser.add_argument('--out', type=Path, default=None,
                        help='optional per_dataset_head_full.csv')
    args = parser.parse_args()

    if not args.audit.is_file():
        raise SystemExit(f'missing audit CSV: {args.audit}')
    if not args.protocol.is_file():
        raise SystemExit(f'missing protocol: {args.protocol}')

    rows = pd.read_csv(args.audit)
    required = {'image', 'head', 'dice', 'cldice', 'hd95'}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f'{args.audit}: missing columns {sorted(missing)}')
    rows = attach_dataset(rows, protocol_map(args.protocol, args.split))

    metric_cols = [c for c in (
        'dice', 'iou', 'precision', 'recall', 'cldice', 'hd95', 'assd',
        'nsd', 'components_pred', 'components_gt', 'small_component_rate',
        'fp_area_ratio', 'fn_area_ratio') if c in rows.columns]
    grouped = rows.groupby(
        ['dataset', 'head'], dropna=False).agg(
        n=('image', 'nunique'),
        **{c: (c, 'mean') for c in metric_cols},
    ).reset_index()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(args.out, index=False)
        print(f'CSV: {args.out}')

    print(f'\nSplit: {args.split}; images: {rows["image"].nunique()}; '
          f'datasets: {rows["dataset"].nunique()}')

    print('\n== per-dataset per-head ==')
    print('dataset,head,n,dice,cldice,hd95,assd,nsd,components_pred')
    for _, row in grouped.sort_values(['dataset', 'head']).iterrows():
        print(f'{row["dataset"]},{int(row["head"])},{int(row["n"])},'
              f'{fmt(row["dice"])},{fmt(row["cldice"])},'
              f'{fmt(row["hd95"])},{fmt(row["assd"])},{fmt(row["nsd"])},'
              f'{fmt(row["components_pred"])}')

    print('\n== equal-dataset macro per head ==')
    macro = grouped.groupby('head', dropna=False).agg(
        n_datasets=('dataset', 'nunique'),
        **{c: (c, 'mean') for c in metric_cols},
    ).reset_index().sort_values('head')
    print('head,n_datasets,dice,cldice,hd95,assd,nsd,components_pred')
    for _, row in macro.iterrows():
        print(f'{int(row["head"])},{int(row["n_datasets"])},'
              f'{fmt(row["dice"])},{fmt(row["cldice"])},'
              f'{fmt(row["hd95"])},{fmt(row["assd"])},{fmt(row["nsd"])},'
              f'{fmt(row["components_pred"])}')

    best = {}
    for metric, ascending in (('dice', False), ('cldice', False),
                              ('hd95', True)):
        if metric in macro.columns:
            key = macro.sort_values(metric, ascending=ascending).iloc[0]
            best[metric] = int(key['head'])
    print('\n== macro winners ==')
    for metric, head in best.items():
        print(f'{metric}: head {head}')

    small = grouped[grouped['n'] < 10]['dataset'].nunique()
    if small:
        print(f'\n[warn] {small} dataset(s) have <10 calibration images; '
              'per-dataset winners are noisy. Treat this as a diagnostic.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
