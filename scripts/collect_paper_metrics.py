#!/usr/bin/env python3
"""Collect paper metrics for every baseline from the prediction contract.

Reads ``data/sacm_3shot/protocol_manifest.csv`` and expects
``results/predictions/<method>/<dataset>/<stem>.png`` for every test image.
Writes per-image metrics, per-dataset mean/std, role macro averages, and
LaTeX/Markdown tables.

Examples:
    # One method, four main metrics
    python scripts/collect_paper_metrics.py --methods unet

    # All methods found under results/predictions
    python scripts/collect_paper_metrics.py --metrics dice,iou,cldice,hd95

    # Fast smoke test
    python scripts/collect_paper_metrics.py --methods unet --datasets DRIVE
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from evaluate_predictions import compute_selected  # noqa: E402

MAIN_METRICS = ['dice', 'iou', 'cldice', 'hd95']
METHOD_META = {
    'unet': ('U-Net', '✗'),
    'cs2net': ('CS²Net', '✗'),
    'bcunet': ('BCUNet', '✗'),
    'maskvsc': ('MaskVSC', '✗'),
    'nnwnet': ('nnWNet', '✗'),
    'sam-med2d': ('SAM-Med2D', '✓'),
    'sam-octa': ('SAM-OCTA', '✓'),
    'cwsam': ('CWSAM', '✓'),
    'knowsam': ('KnowSAM', '✓'),
    'segdino': ('SegDINO', '✓'),
    'sacm': ('SACM', '✗'),
}
ROLE_ORDER = ['Seen', 'Unseen', 'Novel']


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_rows(protocol_path: Path, dataset_filter):
    rows = []
    with open(protocol_path) as f:
        for row in csv.DictReader(f):
            if not row['split'].endswith('_test'):
                continue
            if dataset_filter and row['dataset'] not in dataset_filter:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f'no test rows found in {protocol_path}')
    return rows


def discover_methods(pred_root: Path):
    methods = []
    for child in sorted(pred_root.iterdir()):
        if child.is_dir() and any(p.is_dir() for p in child.iterdir()):
            methods.append(child.name)
    return methods


def collect_method(method, rows, pred_root, metric_names, max_images=None):
    per_dataset = defaultdict(list)
    missing = []
    for row in rows:
        dataset = row['dataset']
        if max_images is not None and len(per_dataset[dataset]) >= max_images:
            continue
        stem = Path(row['mask_path']).stem
        pred_path = pred_root / method / dataset / f'{stem}.png'
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue
        gt_path = resolve(row['mask_path'])
        pred = np.array(Image.open(pred_path).convert('L'),
                        dtype=np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert('L'),
                      dtype=np.float32) / 255.0
        if pred.shape != gt.shape:
            raise SystemExit(
                f'{pred_path}: prediction shape {pred.shape} != GT {gt.shape}')
        metrics = compute_selected(pred, gt, metric_names)
        per_dataset[dataset].append({
            'image': stem,
            'role': row['role'],
            **metrics,
        })
    return per_dataset, missing


def summarize(per_dataset, metric_names):
    summary = {}
    for dataset, rows in per_dataset.items():
        if not rows:
            continue
        summary[dataset] = {}
        for metric in metric_names:
            values = np.asarray([r[metric] for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                summary[dataset][metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'n': int(values.size),
                }
            else:
                summary[dataset][metric] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'n': 0,
                }
    return summary


def role_of(dataset, rows):
    roles = {row['role'] for row in rows if row['dataset'] == dataset}
    if len(roles) != 1:
        raise SystemExit(f'{dataset}: expected one role, got {roles}')
    return roles.pop().capitalize()


def macro(summary, role_map, metric):
    groups = defaultdict(list)
    for dataset, values in summary.items():
        item = values.get(metric)
        if item and math.isfinite(item['mean']):
            groups[role_map[dataset]].append(item['mean'])
    out = {}
    for role in ROLE_ORDER:
        vals = groups.get(role, [])
        out[role] = {
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'n': len(vals),
        }
    all_vals = [v for vals in groups.values() for v in vals]
    out['All'] = {
        'mean': float(np.mean(all_vals)) if all_vals else float('nan'),
        'std': float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0,
        'n': len(all_vals),
    }
    return out


PERCENT_METRICS = {
    'dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity',
    'accuracy', 'mcc', 'cldice', 'nsd', 'dice_auc', 'cldice_auc',
}


def fmt(value, metric, percent=True):
    if not math.isfinite(value):
        return 'NaN'
    if percent and metric in PERCENT_METRICS:
        value *= 100.0
    return f'{value:.2f}'


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, methods, summary, macros, metric_names):
    lines = ['# Paper metrics', '']
    for metric in metric_names:
        lines += [f'## {metric}', '', '| Method | Prompt | ' +
                  ' | '.join(ROLE_ORDER + ['All']) + ' |',
                  '|---|---|' + '---:|' * (len(ROLE_ORDER) + 1)]
        for method in methods:
            display, prompt = METHOD_META.get(method, (method, '?'))
            vals = [fmt(macros[method][metric][role]['mean'], metric)
                    for role in ROLE_ORDER + ['All']]
            lines.append(f'| {display} | {prompt} | ' + ' | '.join(vals) + ' |')
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')


def write_latex(out_dir, methods, summary, macros, metric_names):
    # ``summary`` is method -> dataset -> metrics.  Build the union of
    # datasets once; looking up summary[method] by dataset is required.
    datasets = sorted({dataset for method in methods for dataset in summary[method]})
    for metric in metric_names:
        lines = [
            '\\begin{tabular}{l' + 'c' * len(methods) + '}',
            '\\toprule',
            'Dataset & ' + ' & '.join(
                METHOD_META.get(m, (m, '?'))[0] for m in methods) + ' \\\\',
            '\\midrule',
        ]
        for dataset in datasets:
            vals = []
            for method in methods:
                item = summary[method].get(dataset, {}).get(metric)
                vals.append('--' if item is None else
                            f"{fmt(item['mean'], metric)} $\\pm$ "
                            f"{fmt(item['std'], metric)}")
            lines.append(f'{dataset} & ' + ' & '.join(vals) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '']
        (out_dir / f'table_dataset_{metric}.tex').write_text(
            '\n'.join(lines))
        lines = [
            '\\begin{tabular}{l' + 'c' * (len(ROLE_ORDER) + 1) + '}',
            '\\toprule',
            'Method & ' + ' & '.join(ROLE_ORDER + ['All']) + ' \\\\',
            '\\midrule',
        ]
        for method in methods:
            vals = [fmt(macros[method][metric][role]['mean'], metric)
                    for role in ROLE_ORDER + ['All']]
            lines.append(f'{METHOD_META.get(method, (method, "?"))[0]} & ' +
                         ' & '.join(vals) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '']
        (out_dir / f'table_macro_{metric}.tex').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot' /
                        'protocol_manifest.csv')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions')
    parser.add_argument('--out-root', type=Path,
                        default=REPO_ROOT / 'results' / 'paper')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='method directory names (default: discover all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='optional dataset-name filter')
    parser.add_argument('--metrics', default='dice,iou,cldice,hd95',
                        help="comma-separated metrics or 'all'")
    parser.add_argument('--max-images', type=int, default=None,
                        help='per-dataset cap for smoke tests')
    parser.add_argument('--allow-missing', action='store_true',
                        help='write partial results instead of failing')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit('--max-images must be positive (omit it to use all)')

    collect_metrics = 'all' if args.metrics == 'all' else [
        m.strip() for m in args.metrics.split(',') if m.strip()]
    if collect_metrics != 'all':
        unknown = [m for m in collect_metrics if m not in MAIN_METRICS]
        if unknown:
            raise SystemExit(f'unknown metrics: {unknown}')

    rows = read_rows(args.protocol, set(args.datasets) if args.datasets else None)
    methods = args.methods or discover_methods(args.pred_root)
    if not methods:
        raise SystemExit(f'no methods under {args.pred_root}')

    all_summaries = {}
    all_macros = {}
    stats = {'methods': {}, 'missing': {}}
    metric_names = None
    for method in methods:
        per_dataset, missing = collect_method(
            method, rows, args.pred_root, collect_metrics, args.max_images)
        if missing and not args.allow_missing:
            raise SystemExit(
                f'{method}: {len(missing)} missing predictions; '
                f'first={missing[0]} (use --allow-missing to continue)')
        if metric_names is None:
            first_rows = next(iter(per_dataset.values()), [])
            if not first_rows:
                raise SystemExit(
                    f'{method}: no predictions were collected; '
                    'check --pred-root and --allow-missing')
            if collect_metrics == 'all':
                metric_names = [k for k in first_rows[0]
                                if k not in ('image', 'role')]
            else:
                metric_names = collect_metrics
        summary = summarize(per_dataset, metric_names)
        role_map = {dataset: role_of(dataset, rows) for dataset in summary}
        macros = {m: macro(summary, role_map, m) for m in metric_names}
        expected = {row['dataset'] for row in rows}
        missing_datasets = sorted(expected - set(summary))
        if missing_datasets:
            print(f'[warn] {method}: missing datasets {missing_datasets}; '
                  'macro denominators are reduced')
        all_summaries[method] = summary
        all_macros[method] = macros
        stats['methods'][method] = {
            'datasets': len(summary),
            'images': sum(len(v) for v in per_dataset.values()),
            'missing': len(missing),
            'missing_datasets': missing_datasets,
        }
        stats['missing'][method] = missing[:100]
        print(f'{method}: {stats["methods"][method]["images"]} images, '
              f'{stats["methods"][method]["datasets"]} datasets, '
              f'{len(missing)} missing')

        method_dir = args.out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for dataset, rows_ds in per_dataset.items():
            fields = ['image', 'role'] + [
                m for m in metric_names if rows_ds and m in rows_ds[0]]
            write_csv(method_dir / dataset / 'metrics.csv', fields, rows_ds)

    if metric_names is None:
        raise SystemExit('no metrics collected')
    args.out_root.mkdir(parents=True, exist_ok=True)
    metric_fields = []
    for m in metric_names:
        metric_fields += [f'{m}_mean', f'{m}_std', f'{m}_n']
    dataset_rows = []
    for method in methods:
        for dataset, vals in sorted(all_summaries[method].items()):
            row = {'method': method, 'dataset': dataset,
                   'role': role_of(dataset, rows)}
            for m in metric_names:
                item = vals.get(m, {})
                row[f'{m}_mean'] = item.get('mean')
                row[f'{m}_std'] = item.get('std')
                row[f'{m}_n'] = item.get('n', 0)
            dataset_rows.append(row)
    write_csv(args.out_root / 'paper_metrics.csv',
              ['method', 'dataset', 'role'] + metric_fields, dataset_rows)

    macro_rows = []
    for method in methods:
        for role in ROLE_ORDER + ['All']:
            row = {'method': method, 'role': role}
            for m in metric_names:
                item = all_macros[method][m][role]
                row[f'{m}_mean'] = item['mean']
                row[f'{m}_std'] = item['std']
                row[f'{m}_n_datasets'] = item['n']
            macro_rows.append(row)
    macro_fields = []
    for m in metric_names:
        macro_fields += [f'{m}_mean', f'{m}_std', f'{m}_n_datasets']
    write_csv(args.out_root / 'paper_macro.csv',
              ['method', 'role'] + macro_fields, macro_rows)

    write_markdown(args.out_root / 'paper_tables.md',
                   methods, all_summaries, all_macros, metric_names)
    write_latex(args.out_root, methods, all_summaries, all_macros,
                metric_names)
    stats.update({
        'protocol': str(args.protocol),
        'pred_root': str(args.pred_root),
        'metrics': metric_names,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'test_rows': len(rows),
        'macro': {
            method: {
                m: {role: all_macros[method][m][role]
                    for role in ROLE_ORDER + ['All']}
                for m in metric_names
            }
            for method in methods
        },
    })
    with open(args.out_root / 'paper_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
        f.write('\n')
    print(f'wrote {args.out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
