#!/usr/bin/env python3
"""Build SACM Table-2-style per-dataset and macro tables.

Inputs:
  * canonical baselines: <baseline-root>/<model>/default/seed_*/eval/tables/
    all_metrics.csv (repeat --baseline-root for multiple result roots)
  * final SACM-X: <final-root>/component_eval_all20_*/seed_*/per_dataset.csv
    (or an explicitly supplied --component-eval directory)

The script never loads a model and never modifies run artifacts.  It writes:
  per_dataset_long.csv
  per_dataset_<metric>.md    (wide tables, one per metric)
  macro.csv
  macro.md

Metrics are dataset-level seed means (Dice/IoU/clDice/HD95), with std across
seeds.  Macro values are unweighted means over datasets within each role and
All, matching the canonical evaluator's macro convention.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

ROLE_ORDER = ('Seen', 'Unseen', 'Novel')
METRICS = ('dice', 'iou', 'cldice', 'hd95')
CANONICAL_METRICS = {metric: f'{metric}_mean' for metric in METRICS}
FINAL_METRICS = {
    'dice': 'test_prune_dice',
    'iou': 'test_prune_iou',
    'cldice': 'test_prune_cldice',
    'hd95': 'test_prune_hd95',
}
MODEL_LABELS = {
    'sacm_official': 'SACM (official)',
    'sacm_plus': 'SACM+ (fixed)',
    'sacm_x': 'SACM-X (ours)',
}
MODEL_ORDER = {
    'sacm_official': 0,
    'sacm_plus': 1,
    'sacm_x': 2,
}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def seed_number(path: Path):
    name = path.name
    if not name.startswith('seed_'):
        return None
    try:
        return int(name.replace('seed_', ''))
    except ValueError:
        return None


def add_observation(store, model, dataset, role, seed, metric, value):
    value = finite(value)
    if value is None:
        return
    item = store.setdefault(model, {}).setdefault(dataset, {
        'role': role, 'values': {metric: {} for metric in METRICS}})
    item['values'][metric][seed] = value


def load_canonical(root: Path, models_filter=None):
    store = {}
    if not root.is_dir():
        return store
    for path in sorted(root.glob('*/default/seed_*/eval/tables/all_metrics.csv')):
        method = path.parents[4].name
        if method == 'sacm_x':
            continue
        if models_filter and method not in models_filter:
            continue
        seed = seed_number(path.parents[2])
        if seed is None:
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                dataset = row.get('dataset', '').strip()
                if not dataset:
                    continue
                role = (row.get('role', '') or 'Other').strip()
                for metric, field in CANONICAL_METRICS.items():
                    add_observation(
                        store, method, dataset, role, seed, metric,
                        row.get(field))
    return store


def component_dir(final_root: Path, explicit: Path | None):
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    all20 = sorted(final_root.glob('component_eval_all20_*'))
    if all20:
        return all20[-1]
    legacy = final_root / 'component_eval'
    return legacy if legacy.is_dir() else None


def load_final(final_root: Path, explicit_component: Path | None):
    directory = component_dir(final_root, explicit_component)
    if directory is None:
        return {}, None
    store = {}
    for path in sorted(directory.glob('seed_*/per_dataset.csv')):
        seed = seed_number(path.parent)
        if seed is None:
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                dataset = row.get('dataset', '').strip()
                if not dataset:
                    continue
                role = (row.get('role', '') or 'Other').strip()
                for metric, field in FINAL_METRICS.items():
                    add_observation(
                        store, 'sacm_x', dataset, role, seed, metric,
                        row.get(field))
    return store, directory


def merged_models(*stores):
    output = {}
    for store in stores:
        for model, datasets in store.items():
            for dataset, item in datasets.items():
                prior = output.setdefault(model, {}).setdefault(dataset, {
                    'role': item['role'],
                    'values': {metric: {} for metric in METRICS}})
                for metric, values in item['values'].items():
                    prior['values'][metric].update(values)
    return output


def summarize(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def label(model):
    return MODEL_LABELS.get(model, model)


def sort_models(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.get(m, 99), m))


def fmt(value, std=None, digits=4):
    if value is None:
        return '—'
    if std is None or not math.isfinite(std) or std == 0:
        return f'{value:.{digits}f}'
    return f'{value:.{digits}f}±{std:.{digits}f}'


def write_csv(path: Path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(args):
    baseline_roots = [Path(root) for root in
                      (args.baseline_root or ['results/runs'])]
    final_root = Path(args.final_root) if args.final_root else None
    models_filter = set(args.models.split(',')) if args.models else None

    canonical = {}
    for root in baseline_roots:
        canonical = merged_models(
            canonical, load_canonical(root, models_filter))
    if canonical:
        print('canonical models: ' + ', '.join(sort_models(canonical)))
    final_store, component_used = load_final(
        final_root, Path(args.component_eval) if args.component_eval else None
    ) if final_root is not None else ({}, None)
    if final_store and (models_filter is None or 'sacm_x' in models_filter):
        models = merged_models(canonical, final_store)
    else:
        models = merged_models(canonical)

    if not models:
        raise SystemExit('no result tables found')

    # Refuse to silently publish a partial final 20-dataset table.
    if final_store:
        final_datasets = sorted(final_store.get('sacm_x', {}))
        if (args.expected_datasets
                and len(final_datasets) != args.expected_datasets
                and not args.allow_partial):
            raise SystemExit(
                f'final component evaluation has {len(final_datasets)} '
                f'datasets, expected {args.expected_datasets}; use '
                '--allow-partial only for a diagnostic table')
        print(f'final component evaluation: {component_used} '
              f'({len(final_datasets)} datasets)')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long_rows = []
    macro_rows = []
    by_dataset = {}

    for model in sort_models(models):
        model_store = models[model]
        for dataset, item in sorted(model_store.items(), key=lambda kv: kv[0]):
            role = item['role']
            row = {
                'model': label(model),
                'model_id': model,
                'dataset': dataset,
                'role': role,
            }
            for metric in METRICS:
                mean, std, n = summarize(item['values'][metric].values())
                row[f'{metric}_mean'] = '' if mean is None else mean
                row[f'{metric}_std'] = '' if std is None else std
                if metric == 'dice':
                    row['n_seeds'] = n
            long_rows.append(row)
            by_dataset.setdefault(dataset, {'role': role, 'models': {}})
            by_dataset[dataset]['models'][model] = row

        # Macro: unweighted over datasets, matching canonical macro tables.
        for role in ROLE_ORDER + ('All',):
            selected = [
                (dataset, item) for dataset, item in model_store.items()
                if role == 'All' or item['role'] == role
            ]
            if not selected:
                continue
            row = {
                'model': label(model),
                'model_id': model,
                'role': role,
                'n_datasets': len(selected),
            }
            for metric in METRICS:
                means = []
                for _, item in selected:
                    mean, _, _ = summarize(item['values'][metric].values())
                    if mean is not None:
                        means.append(mean)
                row[f'{metric}_mean'] = (
                    statistics.fmean(means) if means else '')
                row[f'{metric}_std'] = (
                    statistics.stdev(means) if len(means) > 1 else 0.0
                )
            macro_rows.append(row)

    long_fields = ['model', 'model_id', 'dataset', 'role', 'n_seeds']
    for metric in METRICS:
        long_fields += [f'{metric}_mean', f'{metric}_std']
    write_csv(out / 'per_dataset_long.csv', long_rows, long_fields)

    macro_fields = ['model', 'model_id', 'role', 'n_datasets']
    for metric in METRICS:
        macro_fields += [f'{metric}_mean', f'{metric}_std']
    write_csv(out / 'macro.csv', macro_rows, macro_fields)

    model_order = sort_models(models)
    dataset_order = sorted(
        by_dataset,
        key=lambda d: (ROLE_ORDER.index(by_dataset[d]['role'])
                       if by_dataset[d]['role'] in ROLE_ORDER else 99, d))
    for metric in METRICS:
        lines = [
            f'# {metric.upper()} by dataset',
            '',
            'Mean±std across seeds. HD95 is in pixels (lower is better).',
            '',
            '| Dataset | Role | ' + ' | '.join(
                label(model) for model in model_order) + ' |',
            '|---|---|' + '---:|' * len(model_order),
        ]
        for dataset in dataset_order:
            role = by_dataset[dataset]['role']
            cells = []
            for model in model_order:
                row = by_dataset[dataset]['models'].get(model)
                if row is None:
                    cells.append('—')
                else:
                    cells.append(fmt(
                        finite(row[f'{metric}_mean']),
                        finite(row[f'{metric}_std']),
                        digits=4 if metric != 'hd95' else 2))
            lines.append(f'| {dataset} | {role} | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Macro (unweighted over datasets):', '',
                  '| Role | ' + ' | '.join(
                      label(model) for model in model_order) + ' |',
                  '|---|' + '---:|' * len(model_order)]
        for role in ROLE_ORDER + ('All',):
            cells = []
            for model in model_order:
                row = next(
                    (r for r in macro_rows
                     if r['model_id'] == model and r['role'] == role),
                    None)
                if row is None:
                    cells.append('—')
                else:
                    cells.append(fmt(
                        finite(row[f'{metric}_mean']),
                        finite(row[f'{metric}_std']),
                        digits=4 if metric != 'hd95' else 2))
            lines.append(f'| {role} | ' + ' | '.join(cells) + ' |')
        (out / f'per_dataset_{metric}.md').write_text(
            '\n'.join(lines) + '\n')

    macro_lines = [
        '# Macro metrics by model',
        '',
        'Unweighted over datasets; mean±std across datasets within role.',
        '',
        '| Model | Role | Dice | IoU | clDice | HD95 | n |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in macro_rows:
        cells = []
        for metric in METRICS:
            cells.append(fmt(finite(row[f'{metric}_mean']),
                             finite(row[f'{metric}_std']),
                             digits=4 if metric != 'hd95' else 2))
        macro_lines.append(
            f"| {row['model']} | {row['role']} | " + ' | '.join(cells)
            + f" | {row['n_datasets']} |")
    (out / 'macro.md').write_text('\n'.join(macro_lines) + '\n')

    print(f'wrote {out}')
    for name in ('per_dataset_long.csv', 'macro.csv', 'macro.md'):
        print(f'  {out / name}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--baseline-root', action='append', default=None,
        help='root containing <method>/default/seed_*/eval/tables/'
             'all_metrics.csv; repeat for multiple roots (default: '
             'results/runs)')
    parser.add_argument(
        '--final-root', default=None,
        help='final run root containing component_eval_all20_*; default: '
             'latest results/runs/final_gpu0_4_*')
    parser.add_argument(
        '--component-eval', default=None,
        help='explicit final component_eval directory (overrides auto-detect)')
    parser.add_argument(
        '--models', default=None,
        help='comma-separated canonical model ids; default: discover all '
             'under baseline-root')
    parser.add_argument(
        '--out-dir', default='results/tables_paper',
        help='output directory')
    parser.add_argument(
        '--expected-datasets', type=int, default=20,
        help='require this many final datasets; 0 disables')
    parser.add_argument(
        '--allow-partial', action='store_true',
        help='allow a diagnostic table when final is not complete')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.final_root is None:
        candidates = sorted(Path('results/runs').glob('final_gpu0_4_*'))
        args.final_root = str(candidates[-1]) if candidates else ''
    build(args)


if __name__ == '__main__':
    main()
