"""Tests for the shared protocol_manifest -> test-root parser."""

import csv
import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'pipeline',
                       'protocol_specs.py')
_spec = importlib.util.spec_from_file_location('protocol_specs', _SCRIPT)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def _write_manifest(root):
    (root / 'DRIVE' / 'masks').mkdir(parents=True)
    (root / 'WIRE' / 'masks').mkdir(parents=True)
    rows = [
        {'dataset': 'DRIVE', 'split': 'seen_test',
         'mask_path': 'DRIVE/masks/01.png'},
        {'dataset': 'DRIVE', 'split': 'train',
         'mask_path': 'DRIVE/masks/02.png'},
        {'dataset': 'WIRE', 'split': 'novel_test',
         'mask_path': 'WIRE/masks/01.png'},
        {'dataset': 'FIVES', 'split': 'unseen_test',
         'mask_path': str(root / 'FIVES' / 'masks' / '01.png')},
    ]
    path = root / 'protocol_manifest.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dataset_specs_uses_split_test_rows_and_roots(tmp_path):
    manifest = _write_manifest(tmp_path)
    specs = ps.dataset_specs(manifest, repo=tmp_path)

    assert specs == [
        f'DRIVE:{tmp_path / "DRIVE"}',
        f'FIVES:{tmp_path / "FIVES"}',
        f'WIRE:{tmp_path / "WIRE"}',
    ]


def test_dataset_specs_filters_requested_datasets(tmp_path):
    manifest = _write_manifest(tmp_path)
    specs = ps.dataset_specs(manifest, datasets={'DRIVE'}, repo=tmp_path)

    assert specs == [f'DRIVE:{tmp_path / "DRIVE"}']
