#!/usr/bin/env python3
"""Summarize fixed3 versus the already-evaluated official SACM baseline.

The paper comparison is:

    fixed3 (SSH checkpoint, slot 3, no fusion)
    vs
    sacm_official (seeds 0-3, 20 protocol datasets)

The official eval tables must already exist under
``results/runs/sacm_official/default/seed_<s>/eval/tables/``.  This script
never trains or evaluates the official model; it only reads those tables and
pairs them with the new fixed3 evaluations.

Default layout:

    <fixed3-root>/eval/seed_<s>                       # fixed3
    <official-root>/seed_<s>/eval                     # sacm_official

The report always contains Dice, IoU, clDice and HD95 (HD95 lower is better),
per-role and per-seed macro tables, per-dataset rows, paired fixed3-official
wins, and descriptive Wilcoxon p-values.

Usage:
    python scripts/pipeline/summarize_fixed3_seeds.py \
        --fixed3-root results/runs/sacm_stc_fixed3_800 \
        --official-root results/runs/sacm_official/default \
        --seeds 0 1 2 3 \
        --output-root results/ablations/sacm_stc_fixed3_800
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

METRICS = (
    ('dice', 'Dice'),
    ('iou', 'IoU'),
    ('cldice', 'clDice'),
    ('hd95', 'HD95'),
)
METRIC_KEYS = tuple(item[0] for item in METRICS)
ROLES = ('All', 'Seen', 'Unseen', 'Novel')
MODEL_LABELS = {
    'fixed3': 'fixed3 (slot3, no fusion)',
    'official': 'SACM official',
}
EPS = 1e-12


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def _check_manifest(eval_root: Path, checkpoint_name: str | None) -> str | None:
    """Return an error when the eval manifest proves the wrong checkpoint."""
    manifest = _read_json(eval_root / 'run_manifest.json')
    if not isinstance(manifest, dict):
        return 'missing run_manifest.json'
    status = manifest.get('status')
    if status not in (None, 'done'):
        return f'evaluation status={status!r}'
    if checkpoint_name:
        recorded = manifest.get('checkpoint')
        if not recorded or Path(str(recorded)).name != checkpoint_name:
            return (f'manifest checkpoint={Path(str(recorded)).name!r}, '
                    f'expected {checkpoint_name!r}')
    return None


def read_macro(eval_root: Path, expected_datasets: int,
               checkpoint_name: str | None) -> dict:
    error = _check_manifest(eval_root, checkpoint_name)
    if error:
        raise ValueError(error)
    path = eval_root / 'tables' / 'macro_metrics.csv'
    if not path.is_file():
        raise ValueError(f'missing {path}')
    values = {}
    all_n = None
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            role = str(row.get('role', '')).strip()
            metric = str(row.get('metric', '')).strip().lower()
            if role not in ROLES or metric not in METRIC_KEYS:
                continue
            try:
                value = float(row.get('mean'))
                n_datasets = int(float(row.get('n_datasets')))
            except (TypeError, ValueError):
                continue
            if role == 'All' and metric == 'dice':
                all_n = n_datasets
            values[(role, metric)] = value
    if all_n is None or all_n < int(expected_datasets):
        raise ValueError(
            f'{path}: All role covers {all_n} datasets, '
            f'expected at least {expected_datasets}')
    missing = [metric for metric in METRIC_KEYS
               if ('All', metric) not in values]
    if missing:
        raise ValueError(f'{path}: missing All metrics {missing}')
    return values


def read_per_dataset(eval_root: Path, expected_datasets: int,
                     checkpoint_name: str | None) -> dict:
    error = _check_manifest(eval_root, checkpoint_name)
    if error:
        raise ValueError(error)
    path = eval_root / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        raise ValueError(f'missing {path}')
    datasets = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get('dataset', '')).strip()
            if not dataset:
                continue
            entry = {'role': str(row.get('role', '')).strip()}
            ok = True
            for metric in METRIC_KEYS:
                try:
                    entry[metric] = float(row.get(f'{metric}_mean'))
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                datasets[dataset] = entry
    if len(datasets) < int(expected_datasets):
        raise ValueError(
            f'{path}: found {len(datasets)} datasets, '
            f'expected at least {expected_datasets}')
    return datasets


def load_eval(seeds, template: str, root: Path, expected_datasets: int,
              checkpoint_name: str | None):
    data = {}
    errors = {}
    for seed in seeds:
        eval_root = Path(template.format(
            root=root, fixed3_root=root, official_root=root,
            seed=seed)).resolve()
        if not eval_root.is_dir():
            errors[seed] = f'missing eval directory: {eval_root}'
            continue
        try:
            macro = read_macro(
                eval_root, expected_datasets, checkpoint_name)
            per_dataset = read_per_dataset(
                eval_root, expected_datasets, checkpoint_name)
        except (OSError, ValueError, csv.Error) as exc:
            errors[seed] = str(exc)
            continue
        manifest = _read_json(eval_root / 'run_manifest.json') or {}
        data[seed] = {
            'eval_root': str(eval_root),
            'checkpoint': manifest.get('checkpoint'),
            'checkpoint_sha256': manifest.get('checkpoint_sha256'),
            'macro': macro,
            'per_dataset': per_dataset,
        }
    return data, errors


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _aggregate(values):
    values = [float(value) for value in values if _finite(value)]
    if not values:
        return None, None, 0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std, len(values)


def _fmt_value(metric: str, value) -> str:
    if not _finite(value):
        return 'n/a'
    if metric in ('dice', 'iou', 'cldice'):
        return f'{float(value) * 100.0:.2f}'
    return f'{float(value):.2f}'


def _fmt_delta(metric: str, value) -> str:
    if not _finite(value):
        return 'n/a'
    if metric in ('dice', 'iou', 'cldice'):
        return f'{float(value) * 100.0:+.2f}'
    return f'{float(value):+.2f}'


def _fmt_mean_std(metric: str, mean, std, n: int,
                  requested: int) -> str:
    if mean is None:
        return '—'
    text = _fmt_value(metric, mean)
    if std is not None:
        text += '±' + _fmt_value(metric, std)
    if n < requested:
        text += f' (n={n})'
    return text


def _dataset_role(entry: dict) -> str:
    role = str(entry.get('role', '')).strip()
    return role if role in ROLES else 'All'


def _better(metric: str, delta: float) -> bool:
    if abs(delta) <= EPS:
        return False
    return delta < 0 if metric == 'hd95' else delta > 0


def _is_tie(delta: float) -> bool:
    return abs(delta) <= EPS


def _wilcoxon(first: list[float], second: list[float]):
    diffs = [a - b for a, b in zip(first, second)]
    diffs = [value for value in diffs if math.isfinite(value)]
    if not diffs:
        return None
    if all(abs(value) <= EPS for value in diffs):
        return 1.0
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        return float(wilcoxon(
            diffs, zero_method='wilcox', alternative='two-sided').pvalue)
    except (ValueError, ZeroDivisionError):
        return None


def _seed_payload(entry: dict) -> dict:
    """Return a JSON-serializable copy of one seed's eval data."""
    return {
        'eval_root': entry['eval_root'],
        'checkpoint': entry.get('checkpoint'),
        'checkpoint_sha256': entry.get('checkpoint_sha256'),
        'macro': [
            {'role': role, 'metric': metric, 'value': value}
            for (role, metric), value in entry['macro'].items()
        ],
        'per_dataset': entry['per_dataset'],
    }


def _per_seed_dataset_stats(seed, first: dict, second: dict) -> dict:
    if seed not in first or seed not in second:
        return {}
    a_data = first[seed]['per_dataset']
    b_data = second[seed]['per_dataset']
    shared = sorted(set(a_data) & set(b_data))
    stats = {}
    for metric, _ in METRICS:
        wins = ties = losses = total = 0
        xs, ys = [], []
        for dataset in shared:
            a = a_data[dataset].get(metric)
            b = b_data[dataset].get(metric)
            if not (_finite(a) and _finite(b)):
                continue
            delta = float(a) - float(b)
            xs.append(float(a))
            ys.append(float(b))
            total += 1
            if _is_tie(delta):
                ties += 1
            elif _better(metric, delta):
                wins += 1
            else:
                losses += 1
        stats[metric] = {
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'total': total,
            'wilcoxon_p': _wilcoxon(xs, ys),
        }
    return stats


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        '| ' + ' | '.join(header) + ' |',
        '|' + '|'.join(['---'] * len(header)) + '|',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def build_summary(args) -> tuple[dict, str, str]:
    seeds = sorted(set(int(seed) for seed in args.seeds))
    epochs = int(args.epochs)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fixed3_root = Path(args.fixed3_root).resolve()
    official_root = Path(args.official_root).resolve()

    fixed3_data, fixed3_errors = load_eval(
        seeds, args.fixed3_eval_template, fixed3_root,
        int(args.expected_datasets), args.fixed3_checkpoint_name)
    official_data, official_errors = load_eval(
        seeds, args.official_eval_template, official_root,
        int(args.expected_datasets), args.official_checkpoint_name)
    data = {'fixed3': fixed3_data, 'official': official_data}
    errors = {'fixed3': fixed3_errors, 'official': official_errors}

    missing_required = []
    for name in ('fixed3', 'official'):
        for seed in seeds:
            if seed not in data[name]:
                reason = errors[name].get(seed, 'missing eval')
                missing_required.append(f'{name}/seed_{seed}: {reason}')

    # Macro mean ± std across seeds.
    macro_rows = []
    for name in ('fixed3', 'official'):
        for role in ROLES:
            for metric, _ in METRICS:
                values = [
                    entry['macro'].get((role, metric))
                    for entry in data[name].values()
                ]
                mean, std, n = _aggregate(values)
                if n == 0:
                    continue
                macro_rows.append({
                    'model': name,
                    'role': role,
                    'metric': metric,
                    'mean': mean,
                    'std': std,
                    'n_seeds': n,
                    'n_seeds_requested': len(seeds),
                })

    macro_table_rows = []
    for role in ROLES:
        for metric, label in METRICS:
            cells = [role, label]
            for name in ('fixed3', 'official'):
                row = next(
                    (item for item in macro_rows
                     if item['model'] == name
                     and item['role'] == role
                     and item['metric'] == metric), None)
                if row is None:
                    cells.append('—')
                else:
                    cells.append(_fmt_mean_std(
                        metric, row['mean'], row['std'],
                        int(row['n_seeds']), len(seeds)))
            macro_table_rows.append(cells)

    # Per-seed All-role macro table.
    per_seed_table_rows = []
    run_info = {}
    for seed in seeds:
        entry = fixed3_data.get(seed, {})
        checkpoint = entry.get('checkpoint')
        if checkpoint:
            run_info[seed] = _read_json(
                Path(str(checkpoint)).parent / 'run_info.json') or {}
        else:
            run_info[seed] = {}
    official_run_info = {
        seed: _read_json(
            official_root / f'seed_{seed}' / 'run_info.json') or {}
        for seed in seeds
    }
    official_epochs = sorted({
        int(info['epochs']) for info in official_run_info.values()
        if info.get('epochs') is not None
    })
    mixed_budget = bool(official_epochs) and any(
        value != epochs for value in official_epochs)
    for seed in seeds:
        for name in ('fixed3', 'official'):
            if seed not in data[name]:
                continue
            entry = data[name][seed]
            cells = [str(seed), MODEL_LABELS[name]]
            for metric, _ in METRICS:
                cells.append(_fmt_value(
                    metric, entry['macro'].get(('All', metric))))
            if name == 'fixed3':
                cells.append(str(run_info[seed].get('best_epoch', '—')))
            else:
                cells.append(str(
                    official_run_info[seed].get('best_epoch', '—')))
            per_seed_table_rows.append(cells)
    per_seed_header = [
        'Seed', 'Model', 'Dice', 'IoU', 'clDice', 'HD95', 'Best epoch']

    # Paired fixed3 vs official, All role.
    pair_rows = []
    pair_table_rows = []
    wilcoxon_rows = []
    wilcoxon_table_rows = []
    aggregate = {
        metric: {'wins': 0, 'ties': 0, 'losses': 0, 'total': 0}
        for metric, _ in METRICS
    }
    per_dataset_rows = []
    for seed in seeds:
        stats = _per_seed_dataset_stats(
            seed, fixed3_data, official_data)
        if not stats:
            continue
        cells = [str(seed)]
        for metric, _ in METRICS:
            a = fixed3_data[seed]['macro'].get(('All', metric))
            b = official_data[seed]['macro'].get(('All', metric))
            delta = (float(a) - float(b)
                     if _finite(a) and _finite(b) else None)
            cells.append(_fmt_delta(metric, delta))
            pair_rows.append({
                'seed': seed,
                'role': 'All',
                'metric': metric,
                'fixed3': a,
                'official': b,
                'delta': delta,
                'unit': 'fraction' if metric != 'hd95' else 'pixels',
            })
        for metric, _ in METRICS:
            item = stats[metric]
            cells.append(f"{item['wins']}/{item['total']}")
            aggregate[metric]['wins'] += item['wins']
            aggregate[metric]['ties'] += item['ties']
            aggregate[metric]['losses'] += item['losses']
            aggregate[metric]['total'] += item['total']
        pair_table_rows.append(cells)

        p_cells = [str(seed)]
        for metric, _ in METRICS:
            item = stats[metric]
            p_value = item['wilcoxon_p']
            p_cells.append('n/a' if p_value is None else f'{p_value:.4g}')
            wilcoxon_rows.append({
                'seed': seed,
                'role': 'All',
                'metric': metric,
                'p_value': p_value,
                'wins': item['wins'],
                'ties': item['ties'],
                'losses': item['losses'],
                'total': item['total'],
            })
        wilcoxon_table_rows.append(p_cells)

        for dataset in sorted(
                set(fixed3_data[seed]['per_dataset'])
                & set(official_data[seed]['per_dataset'])):
            a_entry = fixed3_data[seed]['per_dataset'][dataset]
            b_entry = official_data[seed]['per_dataset'][dataset]
            for metric, _ in METRICS:
                a = a_entry.get(metric)
                b = b_entry.get(metric)
                delta = (float(a) - float(b)
                         if _finite(a) and _finite(b) else None)
                per_dataset_rows.append({
                    'seed': seed,
                    'dataset': dataset,
                    'role': _dataset_role(a_entry),
                    'metric': metric,
                    'fixed3': a,
                    'official': b,
                    'delta': delta,
                    'fixed3_better': (
                        None if delta is None
                        else _better(metric, delta)),
                })

    pair_header = [
        'Seed', 'ΔDice (pp)', 'ΔIoU (pp)', 'ΔclDice (pp)', 'ΔHD95',
        'Wins Dice', 'Wins IoU', 'Wins clDice', 'Wins HD95',
    ]
    aggregate_rows = []
    for metric, label in METRICS:
        item = aggregate[metric]
        aggregate_rows.append([
            label, str(item['wins']), str(item['ties']),
            str(item['losses']), str(item['total']),
        ])

    fixed3_checkpoints = {}
    for seed in seeds:
        if seed in fixed3_data:
            fixed3_checkpoints[str(seed)] = fixed3_data[seed].get(
                'checkpoint')
        else:
            fixed3_checkpoints[str(seed)] = None

    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'comparison': 'fixed3 vs sacm_official',
        'fixed3_root': str(fixed3_root),
        'official_root': str(official_root),
        'output_root': str(output_root),
        'seeds': seeds,
        'epochs': epochs,
        'official_epochs': official_epochs,
        'mixed_budget': mixed_budget,
        'expected_datasets': int(args.expected_datasets),
        'fixed3_config': {
            'head_select': 'fixed',
            'deployment_head': 3,
            'head_fusion': 0,
            'note': 'derived from the SSH best checkpoint; no retraining',
        },
        'official_note': (
            'official eval tables were read from disk; the official model '
            'was not trained or evaluated by this report'),
        'fixed3_checkpoints': fixed3_checkpoints,
        'official_run_info': {
            str(seed): {
                'epochs': info.get('epochs'),
                'best_epoch': info.get('best_epoch'),
            }
            for seed, info in official_run_info.items()
        },
        'models': {
            name: {
                'eval_template': (
                    args.fixed3_eval_template if name == 'fixed3'
                    else args.official_eval_template),
                'root': str(fixed3_root if name == 'fixed3'
                            else official_root),
                'seeds': {
                    str(seed): _seed_payload(entry)
                    for seed, entry in data[name].items()
                },
                'errors': {
                    str(seed): message
                    for seed, message in errors[name].items()
                },
            }
            for name in ('fixed3', 'official')
        },
        'missing_required': missing_required,
        'macro_mean_std': macro_rows,
        'paired_fixed3_vs_official': pair_rows,
        'wilcoxon_fixed3_vs_official': wilcoxon_rows,
        'aggregate_wins_fixed3_vs_official': aggregate,
    }

    # Machine-readable outputs.
    macro_csv = output_root / 'macro_mean_std.csv'
    _write_csv(macro_csv, macro_rows, [
        'model', 'role', 'metric', 'mean', 'std', 'n_seeds',
        'n_seeds_requested'])
    pair_csv = output_root / 'paired_fixed3_vs_official.csv'
    _write_csv(pair_csv, pair_rows, [
        'seed', 'role', 'metric', 'fixed3', 'official', 'delta', 'unit'])
    per_dataset_csv = output_root / 'per_dataset_fixed3_vs_official.csv'
    _write_csv(per_dataset_csv, per_dataset_rows, [
        'seed', 'dataset', 'role', 'metric', 'fixed3', 'official', 'delta',
        'fixed3_better'])
    wilcoxon_csv = output_root / 'wilcoxon_fixed3_vs_official.csv'
    _write_csv(wilcoxon_csv, wilcoxon_rows, [
        'seed', 'role', 'metric', 'p_value', 'wins', 'ties', 'losses',
        'total'])
    summary_json = output_root / 'summary.json'
    summary_json.write_text(
        json.dumps(summary, indent=2, default=str) + '\n')

    # Markdown report.
    lines = [
        f'# SACM-STC fixed3 vs sacm_official — {epochs}-epoch summary',
        '',
        f'Generated: `{summary["generated_at"]}`',
        '',
        f'- Seeds: `{seeds}`',
        f'- Epoch budget: {epochs}',
        f'- Test datasets per eval: {int(args.expected_datasets)}',
        '- fixed3: SSH best checkpoint with `head_select=fixed`, '
        '`deployment_head=3`, `head_fusion=0`; **no retraining**.',
        '- sacm_official: the already-completed official eval tables; '
        'this report does not train or evaluate it.',
        '- Metrics: Dice, IoU, clDice, HD95; HD95 lower is better. '
        'Dice/IoU/clDice are percentages.',
        '',
    ]
    if missing_required:
        lines.extend([
            '## Missing or incomplete',
            '',
        ])
        for item in missing_required:
            lines.append(f'- {item}')
        lines.append('')
    lines.extend([
        '## Macro metrics (mean ± std across seeds)',
        '',
        _markdown_table(
            ['Role', 'Metric', 'fixed3', 'SACM official'],
            macro_table_rows),
        '',
        '## Per-seed All-role macro',
        '',
        _markdown_table(per_seed_header, per_seed_table_rows),
        '',
        '## fixed3 − official, per seed (All role)',
        '',
        _markdown_table(pair_header, pair_table_rows),
        '',
        '## Wilcoxon over the 20 test datasets (All role)',
        '',
        _markdown_table(
            ['Seed', 'Dice p', 'IoU p', 'clDice p', 'HD95 p'],
            wilcoxon_table_rows),
        '',
        '## Per-dataset wins, aggregated across seeds',
        '',
        _markdown_table(
            ['Metric', 'fixed3 better', 'ties', 'official better', 'total'],
            aggregate_rows),
        '',
        '## Caveats',
        '',
        '- Each fixed3 seed deploys one checkpoint selected on validation '
        'Dice; the test protocol is evaluated once, with no test-time tuning.',
        '- fixed3 is an inference-only deployment of the SSH checkpoint. '
        'The official rows come from the already-completed official eval '
        'tables, not from a rerun in this script.',
        ('- WARNING: official run_info epochs=' + str(official_epochs) +
         f' differ from fixed3 epochs={epochs}; this is a mixed-budget '
         'comparison, not a same-budget claim.'
         if mixed_budget else
         f'- Both fixed3 and official training budgets are {epochs} epochs.'),

        '- HD95 is reported in pixels and lower is better. A fixed3 HD95 '
        'difference may be non-inferiority, not superiority; check the '
        'per-dataset table before claiming all-four-metric dominance.',
        '- Macro Novel is dominated by TopoMortar, whose ground truth has a '
        'known caveat; use the per-dataset CSV for claims.',
        '- The Wilcoxon tests pair datasets within a seed; seeds are not '
        'independent replicates of the same test images. Treat p-values '
        'as descriptive.',
        '',
        '## Files',
        '',
        f'- `{macro_csv}`',
        f'- `{pair_csv}`',
        f'- `{per_dataset_csv}`',
        f'- `{wilcoxon_csv}`',
        f'- `{summary_json}`',
        '',
    ])
    summary_md = output_root / 'summary.md'
    summary_md.write_text('\n'.join(lines))

    return summary, str(summary_md), str(summary_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed3-root', type=Path, required=True,
                        help='root holding fixed3 eval/seed_<s> outputs')
    parser.add_argument(
        '--official-root', type=Path,
        default=Path('results/runs/sacm_official/default'),
        help='root holding official seed_<s>/eval outputs')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=800,
                        help='training budget used for the title/caveats')
    parser.add_argument('--expected-datasets', type=int, default=20)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--fixed3-eval-template',
        default='{fixed3_root}/eval/seed_{seed}',
        help='format template for fixed3 eval roots')
    parser.add_argument(
        '--official-eval-template',
        default='{official_root}/seed_{seed}/eval',
        help='format template for official eval roots')
    parser.add_argument('--fixed3-checkpoint-name',
                        default='best_model_fixed3_nofusion.pth')
    parser.add_argument('--official-checkpoint-name', default='best_model.pth')
    args = parser.parse_args(argv)

    summary, summary_md, summary_json = build_summary(args)
    print(f'summary: {summary_md}')
    print(f'json:    {summary_json}')
    print(f'seeds:   {summary["seeds"]}')
    for model in ('fixed3', 'official'):
        count = len(summary['models'][model].get('seeds', {}))
        print(f'{model}: {count}/{len(summary["seeds"])} seed evals')
    if summary['missing_required']:
        for item in summary['missing_required']:
            print(f'MISSING: {item}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
