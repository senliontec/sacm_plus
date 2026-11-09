#!/usr/bin/env python3
"""Build a SACM-Table-2-style per-dataset paper table.

Reads the evaluator's canonical per-seed table first:
  results/runs/<model>/<variant>/seed_<s>/eval/tables/all_metrics.csv
with a fallback to the recorder's derived copy ``per_dataset_4metrics.csv``.
Preferring the canonical file avoids stale derived data after a re-evaluation.

Outputs (default ``results/reports/paper_table2``):
  table2.csv          long table: dataset, method, metric, mean, std, unit
  table2_macro.csv    role-level macro rows
  table2.md           Markdown, mean ± std
  table2_means.md     Markdown, means only, best bold / second underlined
  table2.tex          LaTeX, means only, best bold / second underlined
  table2_with_std.tex LaTeX, mean ± std

Display follows the SACM paper's Table 2: Dice/IoU/clDice in percent,
HD95 in pixels, best in bold and second best underlined.  CSV files keep
the raw fractions for the overlap metrics.

The table groups datasets by Seen / Unseen / Novel and appends
Seen / Unseen / Novel / All macro rows.  Macro rows average the per-dataset
means *within each seed* first, then report mean/std across seeds.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

METRICS = ('dice', 'iou', 'cldice', 'hd95')
METRIC_LABEL = {'dice': 'Dice', 'iou': 'IoU',
                'cldice': 'clDice', 'hd95': 'HD95'}
LOWER_BETTER = {'hd95'}
ROLE_ORDER = {'Seen': 0, 'Unseen': 1, 'Novel': 2}
LABELS = {
    ('sacm_official', 'default'): 'SACM (official)',
    ('sacm_stc', 'default'): 'SACM-STC',
    ('sacm_stc', 'all_dense'): 'SACM-STC',
}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(row, name):
    for key in (f'{name}_mean', name, f'mean_{name}'):
        if key in row and row[key] not in (None, ''):
            value = _finite(row[key])
            if value is not None:
                return value
    return float('nan')


def _model_label(model, variant):
    return LABELS.get((model, variant), f'{model}/{variant}')


def _parse_models(values):
    parsed = []
    for value in values:
        if ':' in value:
            model, variant = value.split(':', 1)
        else:
            model = value
            # Most rows use variant=default; non-default method variants
            # (for example SACM-STC all_dense) should be passed explicitly.
            variant = 'all_dense' if model == 'sacm_stc' else 'default'
        parsed.append((model, variant))
    return parsed


def _read_seed(path: Path, expected: int, variant: str | None = None):
    with path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('dataset')]
    if expected and len(rows) != expected:
        raise SystemExit(
            f'{path}: found {len(rows)} dataset rows, expected {expected}')
    out = {}
    for row in rows:
        row_variant = (row.get('variant') or '').strip()
        if variant and row_variant and row_variant != variant:
            raise SystemExit(
                f'{path}: row variant={row_variant!r} but requested '
                f'{variant!r}')
        dataset = row['dataset'].strip()
        if dataset in out:
            raise SystemExit(f'{path}: duplicate dataset row {dataset}')
        role = (row.get('role') or '').strip()
        role = {'seen': 'Seen', 'unseen': 'Unseen', 'novel': 'Novel'}.get(
            role.lower(), role)
        out[dataset] = {
            'role': role,
            **{name: _metric(row, name) for name in METRICS},
        }
    return out


def _load_method(repo, model, variant, seeds, expected):
    data = {}
    for seed in seeds:
        tables = (repo / 'results' / 'runs' / model / variant /
                  f'seed_{seed}' / 'eval' / 'tables')
        path = tables / 'all_metrics.csv'
        if not path.is_file():
            path = tables / 'per_dataset_4metrics.csv'
        if not path.is_file():
            raise SystemExit(f'missing metrics for {model}/{variant} '
                             f'seed {seed}: tried {tables}')
        data[seed] = _read_seed(path, expected, variant)
    datasets = set(data[seeds[0]])
    for seed in seeds[1:]:
        if set(data[seed]) != datasets:
            raise SystemExit(
                f'{model}/{variant}: dataset set differs for seed {seed}')
        for dataset in datasets:
            if data[seed][dataset]['role'] != data[seeds[0]][dataset]['role']:
                raise SystemExit(
                    f'{model}/{variant}: role for {dataset} differs '
                    f'between seed {seeds[0]} and seed {seed}')
    return data


def _protocol_roles(path: Path):
    """Return {test_dataset: role} from a protocol_manifest.csv."""
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    roles = {}
    for row in rows:
        if not (row.get('split') or '').strip().endswith('_test'):
            continue
        dataset = (row.get('dataset') or '').strip()
        role = (row.get('role') or '').strip().capitalize()
        if dataset and role:
            roles[dataset] = role
    return roles


def _mean_std(values):
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float('nan'), float('nan')
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _aggregate(method_data, datasets, seeds, roles):
    per = {}
    for dataset in datasets:
        for metric in METRICS:
            per[(dataset, metric)] = _mean_std(
                [method_data[seed][dataset][metric] for seed in seeds])

    macro = {}
    role_groups = {role: [d for d in datasets if roles[d] == role]
                   for role in ROLE_ORDER}
    role_groups['All'] = list(datasets)
    for role, group in role_groups.items():
        if not group:
            continue
        for metric in METRICS:
            per_seed = []
            for seed in seeds:
                values = [method_data[seed][dataset][metric]
                          for dataset in group]
                values = [value for value in values if math.isfinite(value)]
                if values:
                    per_seed.append(statistics.fmean(values))
            macro[(role, metric)] = _mean_std(per_seed)
    return per, macro


def _fmt(mean, std, decimals, with_std, latex=False, scale=1.0):
    if not math.isfinite(mean):
        return '--'
    mean = mean * scale
    std = std * scale
    if not with_std or not math.isfinite(std):
        return f'{mean:.{decimals}f}'
    if latex:
        return f'{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}'
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


def _best_ranks(datasets, macro_roles, methods, per, macro, decimals,
                percent):
    """Rank each cell: 0 = best, 1 = second best, None = rest.

    Ranks compare the displayed (scaled and rounded) means so a bold mark
    never lands on a number that looks worse than an unmarked one.
    """
    ranks = {}
    for row_key in list(datasets) + list(macro_roles):
        source = per if row_key in datasets else macro
        for metric in METRICS:
            scale = 100.0 if percent and metric not in LOWER_BETTER else 1.0
            shown = {}
            for label in methods:
                mean = source[(label, row_key, metric)][0]
                if math.isfinite(mean):
                    shown[label] = round(mean * scale, decimals)
            ordered = sorted(set(shown.values()),
                             reverse=metric not in LOWER_BETTER)
            rank_of = {value: index for index, value in enumerate(ordered)}
            ranks[(row_key, metric)] = {
                label: rank_of[value] for label, value in shown.items()}
    return ranks


def _decorate(text, rank, latex=False):
    if rank == 0:
        return f'\\textbf{{{text}}}' if latex else f'**{text}**'
    if rank == 1:
        return f'\\underline{{{text}}}' if latex else f'<u>{text}</u>'
    return text


def _legend(percent, markdown=False, bold=True):
    if percent:
        unit = ('Dice/IoU/clDice: %; HD95: px' if markdown
                else 'Dice/IoU/clDice: \\%; HD95: px')
    else:
        unit = 'Dice/IoU/clDice: fraction in [0,1]; HD95: px'
    marks = ('best in bold, second best underlined.' if bold
             else 'no ranking marks.')
    if markdown:
        return f'_{unit}. ↑/↓ mark the better direction; {marks}_'
    return (f'{unit}. $\\uparrow$/$\\downarrow$ mark the better direction; '
            f'{marks}')


def _markdown(datasets, macro_roles, methods, roles, per, macro,
              decimals, with_std, bold_best, percent):
    ranks = (_best_ranks(datasets, macro_roles, methods, per, macro,
                         decimals, percent) if bold_best else {})

    header = ['Dataset', 'Role']
    for label in methods:
        for metric in METRICS:
            arrow = '↓' if metric in LOWER_BETTER else '↑'
            header.append(f'{label} {METRIC_LABEL[metric]}{arrow}')
    lines = [_legend(percent, markdown=True, bold=bold_best), '',
             '| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']

    def emit(row_name, role, row_key, source):
        cells = [row_name, role]
        for label in methods:
            for metric in METRICS:
                mean, std = source[(label, row_key, metric)]
                scale = (100.0 if percent and metric not in LOWER_BETTER
                         else 1.0)
                text = _fmt(mean, std, decimals, with_std, scale=scale)
                rank = ranks.get((row_key, metric), {}).get(label)
                cells.append(_decorate(text, rank))
        lines.append('| ' + ' | '.join(cells) + ' |')

    for role in ROLE_ORDER:
        group = [d for d in datasets if roles[d] == role]
        if not group:
            continue
        for dataset in group:
            emit(dataset, role, dataset, per)
        lines.append('')
        emit(f'{role} macro', role, role, macro)
        lines.append('')
    if 'All' in macro_roles:
        emit('All macro', 'All', 'All', macro)
    return '\n'.join(lines) + '\n'


def _latex(datasets, macro_roles, methods, roles, per, macro,
           decimals, with_std, bold_best, percent):
    ranks = (_best_ranks(datasets, macro_roles, methods, per, macro,
                         decimals, percent) if bold_best else {})

    lines = [
        '% Requires: \\usepackage{booktabs,multirow,graphicx}',
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        '\\setlength{\\tabcolsep}{3pt}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{ll' + 'cccc' + '|cccc' * (len(methods) - 1) + '}',
        '\\toprule',
    ]
    first = ['\\multirow{2}{*}{Dataset}', '\\multirow{2}{*}{Role}']
    for index, label in enumerate(methods):
        rule = '|c' if index > 0 else 'c'
        first.append(f'\\multicolumn{{4}}{{{rule}}}{{{label}}}')
    lines.append(' & '.join(first) + ' \\\\')
    second = ['', '']
    arrows = {metric: ('$\\downarrow$' if metric in LOWER_BETTER
                       else '$\\uparrow$') for metric in METRICS}
    for _ in methods:
        second.extend([f'{METRIC_LABEL[metric]} {arrows[metric]}'
                       for metric in METRICS])
    lines.append(' & '.join(second) + ' \\\\')
    lines.append('\\midrule')

    def emit(row_name, role, row_key, source):
        cells = [row_name, role]
        for label in methods:
            for metric in METRICS:
                mean, std = source[(label, row_key, metric)]
                scale = (100.0 if percent and metric not in LOWER_BETTER
                         else 1.0)
                text = _fmt(mean, std, decimals, with_std, latex=True,
                            scale=scale)
                rank = ranks.get((row_key, metric), {}).get(label)
                cells.append(_decorate(text, rank, latex=True))
        lines.append(' & '.join(cells) + ' \\\\')

    for role in ROLE_ORDER:
        group = [d for d in datasets if roles[d] == role]
        if not group:
            continue
        for dataset in group:
            emit(dataset, role, dataset, per)
        lines.append('\\midrule')
        emit(f'{role} macro', role, role, macro)
        lines.append('\\midrule')
    if 'All' in macro_roles:
        emit('All macro', 'All', 'All', macro)
    lines.extend([
        '\\bottomrule',
        '\\end{tabular}}',
        f'\\caption{{Per-dataset 3-shot segmentation results on the '
        f'{len(datasets)}-dataset protocol.  '
        + _legend(percent, bold=bold_best) + '}',
        '\\label{tab:per-dataset}',
        '\\end{table*}',
    ])
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=None)
    parser.add_argument('--models', nargs='+', required=True,
                        help='model:variant specs, e.g. sacm_stc:all_dense '
                             'sacm_official:default')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--expected-datasets', type=int, default=20)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('results/reports/paper_table2'))
    parser.add_argument('--decimals', type=int, default=None,
                        help='display decimals (default: 2 for percent, '
                             '4 for fractions)')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--percent', dest='percent', action='store_true',
                         default=True,
                         help='display Dice/IoU/clDice in percent (default)')
    display.add_argument('--no-percent', dest='percent',
                         action='store_false',
                         help='display Dice/IoU/clDice as [0,1] fractions')
    parser.add_argument('--no-bold', action='store_true')
    args = parser.parse_args()

    repo = (args.repo_root.resolve() if args.repo_root is not None
            else Path(__file__).resolve().parents[2])
    out_dir = (args.out_dir if args.out_dir.is_absolute()
               else repo / args.out_dir)
    seeds = sorted(dict.fromkeys(args.seeds))
    decimals = (args.decimals if args.decimals is not None
                else (2 if args.percent else 4))
    specs = _parse_models(args.models)
    labels = [_model_label(model, variant) for model, variant in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f'duplicate method labels: {labels}; '
                         'pass each model:variant once')

    method_data = {}
    roles = {}
    datasets = None
    for (model, variant), label in zip(specs, labels):
        data = _load_method(repo, model, variant, seeds,
                            args.expected_datasets)
        method_data[label] = data
        current = set(data[seeds[0]])
        if datasets is None:
            datasets = current
            roles = {dataset: data[seeds[0]][dataset]['role']
                     for dataset in datasets}
        elif current != datasets:
            raise SystemExit(
                f'{label}: dataset set differs from {labels[0]}; '
                f'missing={sorted(datasets - current)}, '
                f'extra={sorted(current - datasets)}')
        else:
            for dataset in datasets:
                if data[seeds[0]][dataset]['role'] != roles[dataset]:
                    raise SystemExit(
                        f'{label}: role for {dataset} differs from '
                        f'{labels[0]}')

    unknown_roles = sorted({role for role in roles.values()
                            if role not in ROLE_ORDER})
    if unknown_roles:
        raise SystemExit(
            f'unknown or missing dataset roles {unknown_roles}; expected '
            f'{sorted(ROLE_ORDER)}')

    protocol = (repo / 'data' / f'sacm_3shot_seed{seeds[0]}' /
                'protocol_manifest.csv')
    if protocol.is_file():
        expected_roles = _protocol_roles(protocol)
        if expected_roles:
            if set(expected_roles) != set(datasets):
                raise SystemExit(
                    f'{protocol}: test dataset set differs; '
                    f'missing={sorted(set(expected_roles) - set(datasets))}, '
                    f'extra={sorted(set(datasets) - set(expected_roles))}')
            for dataset in datasets:
                if expected_roles[dataset] != roles[dataset]:
                    raise SystemExit(
                        f'{protocol}: role for {dataset} is '
                        f'{roles[dataset]!r}, protocol says '
                        f'{expected_roles[dataset]!r}')
        else:
            print(f'[warn] {protocol}: no *_test rows; skipping name check')
    else:
        print(f'[warn] protocol manifest not found: {protocol}; '
              'skipping dataset-name check')

    datasets = sorted(
        datasets,
        key=lambda dataset: (ROLE_ORDER.get(roles[dataset], 9),
                             dataset.lower()))
    per = {}
    macro = {}
    for label in labels:
        label_per, label_macro = _aggregate(
            method_data[label], datasets, seeds, roles)
        for dataset, metric in label_per:
            per[(label, dataset, metric)] = label_per[(dataset, metric)]
        for role, metric in label_macro:
            macro[(label, role, metric)] = label_macro[(role, metric)]

    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for dataset in datasets:
        for label in labels:
            for metric in METRICS:
                mean, std = per[(label, dataset, metric)]
                long_rows.append({
                    'dataset': dataset, 'role': roles[dataset],
                    'method': label, 'metric': metric,
                    'mean': mean, 'std': std,
                    'unit': 'fraction' if metric != 'hd95' else 'px',
                    'n_seeds': len(seeds),
                })
    with (out_dir / 'table2.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'role', 'method', 'metric',
                                'mean', 'std', 'unit', 'n_seeds'])
        writer.writeheader()
        writer.writerows(long_rows)

    macro_roles = [role for role in ROLE_ORDER
                   if any(roles[d] == role for d in datasets)] + ['All']
    macro_rows = []
    for role in macro_roles:
        for label in labels:
            for metric in METRICS:
                mean, std = macro[(label, role, metric)]
                macro_rows.append({
                    'role': role, 'method': label, 'metric': metric,
                    'mean': mean, 'std': std,
                    'unit': 'fraction' if metric != 'hd95' else 'px',
                    'n_seeds': len(seeds),
                })
    with (out_dir / 'table2_macro.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['role', 'method', 'metric',
                                'mean', 'std', 'unit', 'n_seeds'])
        writer.writeheader()
        writer.writerows(macro_rows)

    bold = not args.no_bold
    (out_dir / 'table2.md').write_text(
        _markdown(datasets, macro_roles, labels, roles,
                  per, macro, decimals, True, bold, args.percent),
        encoding='utf-8')
    (out_dir / 'table2_means.md').write_text(
        _markdown(datasets, macro_roles, labels, roles,
                  per, macro, decimals, False, bold, args.percent),
        encoding='utf-8')
    (out_dir / 'table2.tex').write_text(
        _latex(datasets, macro_roles, labels, roles,
               per, macro, decimals, False, bold, args.percent),
        encoding='utf-8')
    (out_dir / 'table2_with_std.tex').write_text(
        _latex(datasets, macro_roles, labels, roles,
               per, macro, decimals, True, bold, args.percent),
        encoding='utf-8')

    print(f'methods: {", ".join(labels)}')
    print(f'datasets: {len(datasets)}; seeds: {seeds}')
    print(f'display: {"percent" if args.percent else "fraction"}; '
          f'decimals={decimals}')
    print(f'wrote: {out_dir}/table2.tex')
    print(f'       {out_dir}/table2_with_std.tex')
    print(f'       {out_dir}/table2.md')
    print(f'       {out_dir}/table2_means.md')
    print(f'       {out_dir}/table2.csv')
    print(f'       {out_dir}/table2_macro.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
