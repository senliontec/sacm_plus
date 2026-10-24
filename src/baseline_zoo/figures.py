"""Paper figures generated only from the tidy report CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .registry import REPO_ROOT

METRICS = ('dice', 'iou', 'cldice', 'hd95')
LABELS = {'dice': 'Dice', 'iou': 'IoU', 'cldice': 'clDice', 'hd95': 'HD95'}
ROLES = ('Seen', 'Unseen', 'Novel')


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _value(value: float, metric: str) -> float:
    if not math.isfinite(value):
        return float('nan')
    return value if metric == 'hd95' else value * 100.0


def plot_main_by_role(variance: list[dict], path: Path) -> None:
    models = sorted({row['model'] for row in variance})
    roles = [role for role in ROLES
             if any(row['role'] == role for row in variance)]
    by_key = {(row['model'], row['role'], row['metric']): row
              for row in variance}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = plt.get_cmap('tab10').colors
    for ax, metric in zip(axes.ravel(), METRICS):
        width = 0.8 / max(len(models), 1)
        for index, model in enumerate(models):
            means, errors = [], []
            for role in roles:
                item = by_key.get((model, role, metric))
                means.append(_value(float(item['mean']), metric)
                             if item else float('nan'))
                errors.append(float(item['std']) * (1.0 if metric == 'hd95'
                                                     else 100.0)
                              if item else 0.0)
            positions = (np.arange(len(roles)) +
                         (index - (len(models) - 1) / 2) * width)
            ax.bar(positions, means, width, yerr=errors, capsize=2,
                   color=colors[index % len(colors)], label=model)
        ax.set_xticks(np.arange(len(roles)))
        ax.set_xticklabels(roles)
        ax.set_ylabel('pixels' if metric == 'hd95' else '%')
        ax.set_title(f"{LABELS[metric]} "
                     f"({'lower' if metric == 'hd95' else 'higher'} is better)")
        ax.grid(axis='y', alpha=0.3)
    fig.legend(models, loc='lower center', ncol=min(len(models), 5),
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _dataset_order(rows: list[dict]) -> list[str]:
    role_of = {row['dataset']: row['role'] for row in rows}
    return sorted(role_of, key=lambda name: (
        ROLES.index(role_of[name]) if role_of[name] in ROLES else 99, name))


def plot_per_dataset_heatmap(rows: list[dict], path: Path) -> None:
    models = sorted({row['model'] for row in rows})
    datasets = _dataset_order(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['dataset'], row['metric'])].append(
            _float(row['mean']))
    fig, axes = plt.subplots(2, 2, figsize=(max(14, 0.45 * len(datasets)), 9))
    for ax, metric in zip(axes.ravel(), METRICS):
        matrix = np.full((len(models), len(datasets)), np.nan)
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                values = grouped.get((model, dataset, metric), ())
                values = [value for value in values if math.isfinite(value)]
                if values:
                    matrix[i, j] = statistics.fmean(values) * (
                        1.0 if metric == 'hd95' else 100.0)
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto',
                          cmap='viridis')
        ax.set_yticks(np.arange(len(models)), models)
        ax.set_xticks(np.arange(len(datasets)), datasets, rotation=90,
                      fontsize=7)
        ax.set_title(LABELS[metric])
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def generate_figures(tables_dir: Path, out_dir: Path,
                     formats: tuple[str, ...] = ('png', 'pdf')) -> list[Path]:
    variance = load_rows(tables_dir / 'seed_variance.csv')
    rows = load_rows(tables_dir / 'all_metrics.csv')
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        if variance:
            path = out_dir / f'main_results_by_role.{suffix}'
            plot_main_by_role(variance, path)
            written.append(path)
        if rows:
            path = out_dir / f'per_dataset_heatmap.{suffix}'
            plot_per_dataset_heatmap(rows, path)
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', type=Path,
                        default=REPO_ROOT / 'results' / 'paper' / 'tables')
    parser.add_argument('--out-dir', type=Path,
                        default=REPO_ROOT / 'results' / 'paper' / 'figures')
    parser.add_argument('--formats', default='png,pdf')
    args = parser.parse_args(argv)
    formats = tuple(item.strip() for item in args.formats.split(',')
                    if item.strip())
    paths = generate_figures(args.tables_dir, args.out_dir, formats)
    if not paths:
        print('no report tables found; run report first', file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
