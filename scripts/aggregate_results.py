"""Aggregate results_summary.txt files into CSV and LaTeX tables.

Scans <root>/<preset>/<dataset>/results_summary.txt and builds:
    <out_dir>/all_metrics.csv     - one row per (preset, dataset)
    <out_dir>/table_dice.tex      - LaTeX tables, one per metric
    <out_dir>/table_iou.tex
    <out_dir>/table_cldice.tex
    <out_dir>/table_hd95.tex

Usage:
    python aggregate_results.py --root results --out results/tables
"""

import argparse
import csv
import os
import re

METRIC_RE = re.compile(r'^Average\s+(\w+):\s*([\d.eE+-]+|nan)\s*±\s*([\d.eE+-]+|nan)')
MAIN_METRICS = ['dice', 'iou', 'cldice', 'hd95']
# All reported metrics, in output order (AUC keys appear only when the
# test ran with --auc; parsing is tolerant to missing entries)
ALL_METRICS = ['dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity',
               'accuracy', 'mcc', 'cldice', 'hd', 'hd95', 'assd', 'asd', 'ravd',
               'nsd', 'betti', 'dice_auc', 'cldice_auc', 'betti_matching']


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


def collect(root):
    """Scan for (preset, dataset) -> metrics. Skips missing summaries."""
    rows = []
    if not os.path.isdir(root):
        raise ValueError(f"Results root not found: {root}")
    for preset in sorted(os.listdir(root)):
        preset_dir = os.path.join(root, preset)
        if not os.path.isdir(preset_dir):
            continue
        for dataset in sorted(os.listdir(preset_dir)):
            summary_path = os.path.join(preset_dir, dataset, 'results_summary.txt')
            if not os.path.exists(summary_path):
                print(f"[skip] no summary: {preset}/{dataset}")
                continue
            metrics = parse_summary(summary_path)
            row = {'preset': preset, 'dataset': dataset}
            for name in ALL_METRICS:
                if name in metrics:
                    row[name + '_mean'], row[name + '_std'] = metrics[name]
            rows.append(row)
    return rows


def write_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = ['preset', 'dataset']
    for name in ALL_METRICS:
        fieldnames += [name + '_mean', name + '_std']
    path = os.path.join(out_dir, 'all_metrics.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def write_latex(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    presets = sorted({r['preset'] for r in rows})
    datasets = sorted({r['dataset'] for r in rows})

    for metric in MAIN_METRICS:
        mean_key, std_key = metric + '_mean', metric + '_std'
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
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
            for p in presets:
                row = next((r for r in rows if r['preset'] == p and r['dataset'] == ds), None)
                if row and mean_key in row:
                    cells.append(f"{row[mean_key]:.2f}")
                else:
                    cells.append('--')
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\label{tab:" + metric + "}")
        lines.append("\\end{table}")

        path = os.path.join(out_dir, f'table_{metric}.tex')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"LaTeX: {path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results into tables')
    parser.add_argument('--root', type=str, required=True, help='Results root (run_experiments --output_root)')
    parser.add_argument('--out', type=str, default=None, help='Output dir for CSV/LaTeX (default: <root>/tables)')
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.root, 'tables')

    rows = collect(args.root)
    if not rows:
        print("No results found.")
        return
    write_csv(rows, out_dir)
    write_latex(rows, out_dir)
    print(f"Aggregated {len(rows)} (preset, dataset) rows.")


if __name__ == '__main__':
    main()
