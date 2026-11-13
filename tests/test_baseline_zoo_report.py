"""Tests for cross-model / cross-seed metric aggregation."""

import csv
import json
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


def _write_sacm_gcd(root, seed):
    path = (root / 'sacm_gcd' / 'all_dense' / f'seed_{seed}' /
            'eval' / 'tables')
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
    path = (root / 'unet' / 'default' / f'seed_{seed}' /
            'eval' / 'tables')
    path.mkdir(parents=True)
    rows = [
        {'method': 'unet', 'dataset': 'DRIVE', 'role': 'Seen',
         'dice_mean': 0.75, 'dice_std': 0.01,
         'iou_mean': 0.66, 'iou_std': 0.01,
         'cldice_mean': 0.80, 'cldice_std': 0.01,
         'hd95_mean': 8.0, 'hd95_std': 1.0},
    ]
    with open(path / 'all_metrics.csv', 'w', newline='') as f:
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
    _write_sacm_gcd(tmp_path, 0)
    _write_baseline(tmp_path, 0)
    rows, missing = load_runs(['sacm_gcd', 'unet'], [0], tmp_path)
    assert missing == []
    assert len(rows) == 12  # 3 datasets × 4 metrics
    assert {(row.model, row.role) for row in rows} == {
        ('sacm_gcd', 'Seen'), ('sacm_gcd', 'Novel'), ('unet', 'Seen')}


def test_explicit_preset_selects_sacm_gcd_variant(tmp_path):
    _write_sacm_gcd(tmp_path, 0)
    _write_baseline(tmp_path, 0)
    rows, missing = load_runs(['sacm_gcd'], [0], tmp_path,
                              preset='all_dense')
    assert missing == []
    assert {row.model for row in rows} == {'sacm_gcd'}


def test_row_method_mismatch_raises(tmp_path):
    path = (tmp_path / 'unet' / 'default' / 'seed_0' / 'eval' / 'tables')
    path.mkdir(parents=True)
    row = {'method': 'sacm_gcd', 'dataset': 'DRIVE', 'role': 'Seen',
           'dice_mean': 0.8}
    with open(path / 'all_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(SystemExit, match='row method'):
        load_runs(['unet'], [0], tmp_path)


def test_configured_variant_does_not_fall_back_to_legacy_default(tmp_path):
    """A configured all_dense run must not silently read sacm_gcd/seed_0."""
    _write_sacm_gcd(tmp_path, 0)
    legacy = tmp_path / 'sacm_gcd' / 'seed_0' / 'eval' / 'tables'
    legacy.mkdir(parents=True)
    with open(legacy / 'all_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['dataset', 'role'] + FIELDS)
        writer.writeheader()
        writer.writerow({'dataset': 'DRIVE', 'role': 'Seen',
                         'dice_mean': 0.9})
    # Remove the canonical run so only the legacy directory remains.
    for path in sorted(
            (tmp_path / 'sacm_gcd' / 'all_dense').rglob('*'), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    rows, missing = load_runs(['sacm_gcd'], [0], tmp_path)
    assert rows == []
    assert missing == [
        str(tmp_path / 'sacm_gcd' / 'all_dense' / 'seed_0')]


def test_manifest_dataset_mismatch_warns(tmp_path, capsys):
    _write_sacm_gcd(tmp_path, 0)
    manifest_path = (tmp_path / 'sacm_gcd' / 'all_dense' / 'seed_0' /
                     'eval' / 'run_manifest.json')
    manifest_path.write_text(json.dumps({
        'status': 'done', 'variant': 'all_dense',
        'datasets': ['DRIVE', 'WIRE', 'MISSING'],
    }))
    rows, missing = load_runs(['sacm_gcd'], [0], tmp_path)
    assert missing == []
    assert len(rows) == 8
    out = capsys.readouterr().out
    assert 'dataset set differs' in out
    assert 'MISSING' in out


def test_manifest_variant_mismatch_raises(tmp_path):
    _write_sacm_gcd(tmp_path, 0)
    manifest_path = (tmp_path / 'sacm_gcd' / 'all_dense' / 'seed_0' /
                     'eval' / 'run_manifest.json')
    manifest_path.write_text(json.dumps({
        'status': 'done', 'variant': 'default',
        'datasets': ['DRIVE', 'WIRE'],
    }))
    with pytest.raises(SystemExit, match='manifest variant'):
        load_runs(['sacm_gcd'], [0], tmp_path)


def test_write_report_outputs_tables(tmp_path):
    _write_sacm_gcd(tmp_path, 0)
    _write_baseline(tmp_path, 0)
    rows, _ = load_runs(['sacm_gcd', 'unet'], [0], tmp_path)
    out = tmp_path / 'tables'
    paths = write_report(rows, out)

    assert all(path.is_file() for path in paths.values())
    text = paths['main_results'].read_text()
    assert 'sacm_gcd' in text and 'unet' in text
    assert '80.00' in text

    with open(paths['macro_metrics'], newline='') as f:
        macro = list(csv.DictReader(f))
    assert len(macro) == 12
    assert {'sacm_gcd', 'unet'} == {row['model'] for row in macro}
