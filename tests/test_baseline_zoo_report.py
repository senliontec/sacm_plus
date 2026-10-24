"""Tests for cross-model / cross-seed metric aggregation."""

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.report import (  # noqa: E402
    load_runs,
    write_csv,
    write_report,
)

FIELDS = ['dice_mean', 'dice_std', 'iou_mean', 'iou_std',
          'cldice_mean', 'cldice_std', 'hd95_mean', 'hd95_std']


def _write_sacm_x(root, seed):
    path = root / 'sacm_x' / f'seed_{seed}' / 'tables'
    path.mkdir(parents=True)
    rows = [
        {'dataset': 'DRIVE', 'role': 'Seen',
         'dice_mean': 0.80, 'dice_std': 0.02,
         'iou_mean': 0.70, 'iou_std': 0.02,
         'cldice_mean': 0.84, 'cldice_std': 0.02,
         'hd95_mean': 5.0, 'hd95_std': 1.0},
        {'dataset': 'WIRE', 'role': 'Novel',
         'dice_mean': 0.50, 'dice_std': 0.03,
         'iou_mean': 0.40, 'iou_std': 0.03,
         'cldice_mean': 0.60, 'cldice_std': 0.03,
         'hd95_mean': 60.0, 'hd95_std': 5.0},
    ]
    with open(path / 'all_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'role'] + FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_baseline(root, seed):
    path = root / 'unet' / f'seed_{seed}'
    path.mkdir(parents=True)
    rows = [
        {'method': 'unet', 'dataset': 'DRIVE', 'role': 'Seen',
         'dice_mean': 0.75, 'dice_std': 0.01,
         'iou_mean': 0.66, 'iou_std': 0.01,
         'cldice_mean': 0.80, 'cldice_std': 0.01,
         'hd95_mean': 8.0, 'hd95_std': 1.0},
    ]
    with open(path / 'paper_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'dataset', 'role'] + FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_write_csv_is_atomic_on_failure(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('old\n')
    with pytest.raises(ValueError):
        write_csv(path, ['a'], [{'a': 1}, {'a': 2, 'b': 3}])
    assert path.read_text() == 'old\n'
    assert not path.with_name('table.csv.tmp').exists()


def test_load_runs_reads_both_schemas(tmp_path):
    _write_sacm_x(tmp_path, 0)
    _write_baseline(tmp_path, 0)
    rows, missing = load_runs(['sacm_x', 'unet'], [0], tmp_path)
    assert missing == []
    assert len(rows) == 12  # 3 datasets × 4 metrics
    assert {(row.model, row.role) for row in rows} == {
        ('sacm_x', 'Seen'), ('sacm_x', 'Novel'), ('unet', 'Seen')}


def test_write_report_outputs_tables(tmp_path):
    _write_sacm_x(tmp_path, 0)
    _write_baseline(tmp_path, 0)
    rows, _ = load_runs(['sacm_x', 'unet'], [0], tmp_path)
    out = tmp_path / 'tables'
    paths = write_report(rows, out)

    assert all(path.is_file() for path in paths.values())
    text = paths['main_results'].read_text()
    assert 'sacm_x' in text and 'unet' in text
    assert '80.00' in text

    with open(paths['macro_metrics'], newline='') as f:
        macro = list(csv.DictReader(f))
    assert len(macro) == 12
    assert {'sacm_x', 'unet'} == {row['model'] for row in macro}
