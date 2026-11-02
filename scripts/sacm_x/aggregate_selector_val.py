#!/usr/bin/env python3
"""Compare selector variants from per-config metrics.csv files."""

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
    return rows


def fmt(value: float) -> str:
    return 'nan' if pd.isna(value) else f'{value:.3f}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='directory containing one subdir per selector')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['val', 'train'])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    mapping = protocol_map(args.protocol, args.split)
    metric_cols = ['dice', 'iou', 'precision', 'recall', 'cldice',
                   'hd95', 'assd', 'nsd', 'components_pred',
                   'small_component_rate']
    rows = []
    head_counts = {}
    for config_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        metrics_path = config_dir / 'metrics.csv'
        if not metrics_path.is_file():
            print(f'[skip] {config_dir.name}: no metrics.csv')
            continue
        df = pd.read_csv(metrics_path)
        missing = set(metric_cols) - set(df.columns)
        if missing:
            print(f'[skip] {config_dir.name}: missing {sorted(missing)}')
            continue
        df = attach_dataset(df, mapping)
        per_ds = df.groupby('dataset')[metric_cols].mean()
        row = {'config': config_dir.name, 'n_images': len(df),
               'n_datasets': df['dataset'].nunique()}
        for c in metric_cols:
            row[c] = float(per_ds[c].mean())
        rows.append(row)
        if 'selected_head' in df.columns:
            counts = df['selected_head'].dropna().astype(int).value_counts()
            head_counts[config_dir.name] = counts.sort_index().to_dict()

    if not rows:
        raise SystemExit('no complete selector metrics found')
    summary = pd.DataFrame(rows).sort_values('dice', ascending=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f'CSV: {args.out}')

    print(f'\nSplit: {args.split}')
    print('config,n_images,n_datasets,dice,cldice,hd95,assd,nsd,'
          'components_pred,small_component_rate')
    for _, row in summary.iterrows():
        print(f'{row["config"]},{int(row["n_images"])},'
              f'{int(row["n_datasets"])},{fmt(row["dice"])},'
              f'{fmt(row["cldice"])},{fmt(row["hd95"])},'
              f'{fmt(row["assd"])},{fmt(row["nsd"])},'
              f'{fmt(row["components_pred"])},'
              f'{fmt(row["small_component_rate"])}')

    if head_counts:
        print('\nselected_head distribution (where recorded):')
        for config, counts in head_counts.items():
            print(f'  {config}: {counts}')

    base = summary[summary['config'] == 'iou']
    if not base.empty:
        b = base.iloc[0]
        print('\ndelta vs iou (same val split):')
        for _, row in summary.iterrows():
            if row['config'] == 'iou':
                continue
            print(f'  {row["config"]}: dice {row["dice"] - b["dice"]:+.3f}, '
                  f'cldice {row["cldice"] - b["cldice"]:+.3f}, '
                  f'hd95 {row["hd95"] - b["hd95"]:+.3f}, '
                  f'assd {row["assd"] - b["assd"]:+.3f}, '
                  f'nsd {row["nsd"] - b["nsd"]:+.3f}')

    small = summary[summary['n_images'] < 10]
    if not small.empty:
        print('\n[warn] fewer than 10 images; differences are diagnostic only.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
