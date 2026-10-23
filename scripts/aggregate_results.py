"""Aggregate results_summary.txt files into CSV and LaTeX tables.

Scans <root>/<preset>/<dataset>/results_summary.txt and builds:

    <out_dir>/all_metrics.csv       - one row per (preset, dataset, role)
    <out_dir>/macro_metrics.csv     - Seen/Unseen/Novel macro averages
    <out_dir>/table_<metric>.tex    - per-dataset tables, mean±std
    <out_dir>/table_macro_<metric>.tex - role-level macro tables

Macro averaging: first average within each dataset, then average the
per-dataset means (equal dataset weight, not equal image weight).  The
reported macro std is the sample std of the per-dataset means.

Usage:
    python aggregate_results.py --root results --out results/tables
"""

import argparse
import csv
import json
import math
import os
import re
import statistics

METRIC_RE = re.compile(
    r'^Average\s+(\w+):\s*([\d.eE+-]+|nan)\s*±\s*([\d.eE+-]+|nan)')
MAIN_METRICS = ['dice', 'iou', 'cldice', 'hd95']
# All reported metrics, in output order (AUC keys appear only when the
# test ran with --auc; parsing is tolerant to missing entries)
ALL_METRICS = ['dice', 'iou', 'precision', 'recall', 'sensitivity',
               'specificity', 'accuracy', 'mcc', 'cldice', 'hd', 'hd95',
               'assd', 'asd', 'ravd', 'nsd', 'betti', 'dice_auc',
               'cldice_auc', 'betti_matching', 'topograph_error']

ROLE_BY_DATASET = {
    # Seen (training domains; CORN included in the v1 protocol)
    'DRIVE': 'Seen', 'CHASEDB1': 'Seen', 'DCA1': 'Seen',
    'CREMI': 'Seen', 'CrackTree200': 'Seen', 'CORN': 'Seen',
    # Unseen (active protocol only)
    'FIVES': 'Unseen', 'STARE': 'Unseen', 'XCAD': 'Unseen',
    'DeepCrack': 'Unseen', 'HRF': 'Unseen', 'LES-AV': 'Unseen',
    'CHCrack5k': 'Unseen', 'OCTA-500': 'Unseen',
    # Novel
    'ROAD': 'Novel', 'LEAF': 'Novel', 'TYRE': 'Novel', 'WIRE': 'Novel',
    'DeepGlobe': 'Novel', 'TopoMortar': 'Novel',
}
ROLE_ORDER = ['Seen', 'Unseen', 'Novel', 'Other']


def dataset_role(name):
    return ROLE_BY_DATASET.get(name, 'Other')


def parse_summary(path):
    """Return {metric: (mean, std)} from a results_summary.txt file."""
    metrics = {}
    with open(path) as f:
        for line in f:
            m = METRIC_RE.match(line.strip())
            if m:
                name, mean, std = m.group(1).lower(), m.group(2), m.group(3)
                metrics[name] = (float(mean), float(std))
    return metrics


def parse_num_samples(path):
    """Read the 'Num Samples: N' line, if the evaluator wrote one."""
    with open(path) as f:
        for line in f:
            if line.strip().lower().startswith('num samples:'):
                try:
                    return int(line.split(':', 1)[1].strip())
                except ValueError:
                    return None
    return None


def load_expected_counts(path):
    """Return {dataset: test_count} from the frozen protocol summary."""
    if not path or not os.path.exists(path):
        print(f"[warn] protocol summary not found: {path}")
        return None
    with open(path) as f:
        summary = json.load(f)
    expected = {}
    for role in ('seen', 'unseen', 'novel'):
        for split, datasets in summary.get('counts', {}).get(role, {}).items():
            if split.endswith('_test'):
                expected.update(datasets)
    return expected


def collect(root, expected_counts=None, strict=False):
    """Scan for (preset, dataset) -> metrics. Skips missing summaries."""
    rows = []
    if not os.path.isdir(root):
        raise ValueError(f"Results root not found: {root}")
    skip_dirs = {'predictions', 'paper', 'visualizations', 'tables',
                 'torchrun_logs', 'logs'}
    for preset in sorted(os.listdir(root)):
        if preset in skip_dirs:
            continue
        preset_dir = os.path.join(root, preset)
        if not os.path.isdir(preset_dir):
            continue
        for dataset in sorted(os.listdir(preset_dir)):
            dataset_dir = os.path.join(preset_dir, dataset)
            if not os.path.isdir(dataset_dir):
                # e.g. results/<preset>/best_model.pth — not a dataset.
                continue
            summary_path = os.path.join(
                dataset_dir, 'results_summary.txt')
            if not os.path.exists(summary_path):
                msg = f"[missing summary] {preset}/{dataset}"
                if strict:
                    raise SystemExit(msg)
                print(f"[skip] no summary: {preset}/{dataset}")
                continue
            n = parse_num_samples(summary_path)
            expected = (expected_counts or {}).get(dataset)
            if expected is not None and n is None:
                msg = (f'[protocol unverifiable] {preset}/{dataset}: '
                       'results_summary.txt has no Num Samples line')
                if strict:
                    raise SystemExit(msg)
                print(msg + ' — accepted, count unchecked')
            if expected is not None and n is not None and n != expected:
                msg = (f'[protocol mismatch] {preset}/{dataset}: '
                       f'{n} images, frozen protocol expects {expected}')
                if strict:
                    raise SystemExit(msg)
                print(msg + ' — skipped')
                continue
            if expected_counts is not None and expected is None:
                msg = f'[protocol mismatch] {preset}/{dataset}: dataset not in frozen protocol'
                if strict:
                    raise SystemExit(msg)
                print(msg + ' — skipped')
                continue
            metrics = parse_summary(summary_path)
            row = {
                'preset': preset,
                'dataset': dataset,
                'role': dataset_role(dataset),
            }
            for name in ALL_METRICS:
                if name in metrics:
                    row[name + '_mean'], row[name + '_std'] = metrics[name]
            rows.append(row)
    if strict and expected_counts:
        for preset in sorted({row['preset'] for row in rows}):
            present = {row['dataset'] for row in rows
                       if row['preset'] == preset}
            missing = sorted(set(expected_counts) - present)
            if missing:
                raise SystemExit(
                    f'{preset}: missing expected datasets: {missing}')
    return rows


def write_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ['preset', 'dataset', 'role']
    for name in ALL_METRICS:
        fieldnames += [name + '_mean', name + '_std']
    path = os.path.join(out_dir, 'all_metrics.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def _fmt(mean, std):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return '--'
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def write_latex(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    presets = sorted({r['preset'] for r in rows})
    datasets = sorted({r['dataset'] for r in rows})

    for metric in MAIN_METRICS:
        mean_key, std_key = metric + '_mean', metric + '_std'
        lines = ["\\begin{table}[t]", "\\centering"]
        title = metric.upper() if metric != 'cldice' else 'clDice'
        note = 'Lower is better' if metric == 'hd95' else 'Higher is better'
        lines.append(f"\\caption{{{title} on all datasets ({note}).}}")
        header = ' & '.join(['Dataset'] + presets) + ' \\\\'
        lines.append("\\begin{tabular}{l" + "c" * len(presets) + "}")
        lines.append("\\hline")
        lines.append(header)
        lines.append("\\hline")
        for ds in datasets:
            cells = [ds]
            for preset in presets:
                row = next((r for r in rows
                            if r['preset'] == preset and r['dataset'] == ds),
                           None)
                cells.append(
                    _fmt(row.get(mean_key), row.get(std_key))
                    if row else '--')
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\label{tab:" + metric + "}")
        lines.append("\\end{table}")

        path = os.path.join(out_dir, f'table_{metric}.tex')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"LaTeX: {path}")


def write_macro(rows, out_dir):
    """Seen/Unseen/Novel equal-dataset macro averages."""
    os.makedirs(out_dir, exist_ok=True)
    presets = sorted({r['preset'] for r in rows})
    roles = [role for role in ROLE_ORDER
             if any(r['role'] == role for r in rows)]

    macro_rows = []
    for preset in presets:
        for role in roles:
            group = [r for r in rows
                     if r['preset'] == preset and r['role'] == role]
            if not group:
                continue
            for metric in ALL_METRICS:
                key = metric + '_mean'
                vals = [r[key] for r in group
                        if key in r and math.isfinite(r[key])]
                if not vals:
                    continue
                macro_rows.append({
                    'preset': preset,
                    'role': role,
                    'metric': metric,
                    'n_datasets': len(vals),
                    'mean': f'{statistics.fmean(vals):.6f}',
                    'std': (f'{statistics.stdev(vals):.6f}'
                            if len(vals) > 1 else '0.000000'),
                })

    path = os.path.join(out_dir, 'macro_metrics.csv')
    fields = ['preset', 'role', 'metric', 'n_datasets', 'mean', 'std']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(macro_rows)
    print(f"CSV: {path}")

    for metric in MAIN_METRICS:
        lines = ["\\begin{table}[t]", "\\centering"]
        title = metric.upper() if metric != 'cldice' else 'clDice'
        lines.append(f"\\caption{{Macro {title} (equal dataset weight).}}")
        header = ' & '.join(['Split'] + presets) + ' \\\\'
        lines.append("\\begin{tabular}{l" + "c" * len(presets) + "}")
        lines.append("\\hline")
        lines.append(header)
        lines.append("\\hline")
        for role in roles:
            cells = [role]
            for preset in presets:
                row = next((r for r in macro_rows
                            if r['preset'] == preset
                            and r['role'] == role
                            and r['metric'] == metric), None)
                if row is None:
                    cells.append('--')
                else:
                    cells.append(f"{float(row['mean']):.2f}"
                                 f"±{float(row['std']):.2f}")
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\label{tab:macro_" + metric + "}")
        lines.append("\\end{table}")
        path = os.path.join(out_dir, f'table_macro_{metric}.tex')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"LaTeX: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment results into tables')
    parser.add_argument('--root', type=str, required=True,
                        help='Results root (run_experiments --output_root)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir for CSV/LaTeX (default: <root>/tables)')
    parser.add_argument('--protocol-summary', type=str,
                        default='data/sacm_3shot/protocol_summary.json',
                        help='frozen protocol counts for result validation')
    parser.add_argument('--strict-protocol', action='store_true',
                        help='fail instead of skipping a protocol mismatch')
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.root, 'tables')

    expected = load_expected_counts(args.protocol_summary)
    if args.strict_protocol and expected is None:
        raise SystemExit(
            f'--strict-protocol requires {args.protocol_summary}')
    rows = collect(args.root, expected_counts=expected,
                   strict=args.strict_protocol)
    if not rows:
        print("No results found.")
        return 1
    write_csv(rows, out_dir)
    write_latex(rows, out_dir)
    write_macro(rows, out_dir)
    print(f"Aggregated {len(rows)} (preset, dataset) rows.")


if __name__ == '__main__':
    raise SystemExit(main())
