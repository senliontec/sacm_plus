"""Tests for the prediction-vs-GT audit script."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / 'scripts' / 'pipeline' / 'audit_predictions.py'
    spec = importlib.util.spec_from_file_location(
        'sacm_audit_predictions_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def audit_mod():
    return _load_module()


def _write_mask(path: Path, mask: np.ndarray, scale: int = 255):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * scale)).save(path)


def _args(tmp_path, **overrides):
    values = dict(
        pred_root=tmp_path / 'pred',
        test_datasets=[f'DRIVE:{tmp_path / "gt" / "DRIVE"}'],
        protocol=None,
        datasets='',
        repo=tmp_path,
        out=tmp_path / 'audit',
        pred_suffix='_result.png',
        limit=0,
        workers=1,
        empty_policy='penalty',
        small_px=20,
        save_error_maps=False,
        save_worst_per_dataset=1,
        error_map_max_side=256,
        strict=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_binary_accepts_0_1_and_0_255(audit_mod, tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    path01 = tmp_path / 'mask01.png'
    path255 = tmp_path / 'mask255.png'
    _write_mask(path01, mask, scale=1)
    _write_mask(path255, mask, scale=255)
    assert audit_mod._load_binary(path01).sum() == 16
    assert audit_mod._load_binary(path255).sum() == 16


def test_prediction_name_matching(audit_mod):
    assert 'foo' in audit_mod._key_candidates(
        'foo_result.png', '_result.png')
    keys = audit_mod._key_candidates('foo.png_result.png', '_result.png')
    assert 'foo.png' in keys
    assert 'foo' in keys


def test_audit_one_computes_pixel_and_topology_fields(
        audit_mod, tmp_path):
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred = gt.copy()
    pred[0, 0] = 1          # isolated FP component
    pred[4:12, 11] = 0      # missing one GT column
    gt_path = tmp_path / 'gt.png'
    pred_path = tmp_path / 'pred.png'
    _write_mask(gt_path, gt)
    _write_mask(pred_path, pred)

    row = audit_mod._audit_one(
        ('DRIVE', gt_path.name, str(pred_path), str(gt_path), 'penalty', 20))
    assert row['status'] == 'ok'
    assert row['tp'] > 0
    assert row['fp'] > 0
    assert row['fn'] > 0
    assert row['components_pred'] == 2
    assert row['components_gt'] == 1
    assert row['component_count_error'] == 1
    assert 0.0 < row['dice'] < 1.0
    assert row['skeleton_precision'] < 1.0
    assert 0.0 <= row['skeleton_recall'] <= 1.0
    assert 'no_overlap' in row
    assert 'fragmented' in row


def test_audit_flags_fragmentation(audit_mod, tmp_path):
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred = gt.copy()
    for i in range(6):
        # Gaps of one pixel keep these six 8-connected components separate.
        pred[20, 20 + 2 * i] = 1
    gt_path = tmp_path / 'gt2.png'
    pred_path = tmp_path / 'pred2.png'
    _write_mask(gt_path, gt)
    _write_mask(pred_path, pred)
    row = audit_mod._audit_one(
        ('DRIVE', gt_path.name, str(pred_path), str(gt_path), 'penalty', 20))
    assert row['components_pred'] == 7
    assert row['fragmented'] == 1
    assert row['small_components_pred'] == 6


def test_run_audit_end_to_end(audit_mod, tmp_path):
    gt_root = tmp_path / 'gt' / 'DRIVE'
    gt = np.zeros((24, 24), dtype=np.uint8)
    gt[6:18, 6:18] = 1
    _write_mask(gt_root / 'masks' / 'foo.png', gt)
    _write_mask(gt_root / 'images' / 'foo.png', gt)

    pred_root = tmp_path / 'pred'
    pred = gt.copy()
    pred[0, 0] = 1
    _write_mask(pred_root / 'DRIVE' / 'foo.png_result.png', pred)

    args = _args(tmp_path)
    rc = audit_mod.run_audit(args)
    assert rc == 0

    out = args.out
    assert (out / 'per_image.csv').is_file()
    assert (out / 'per_dataset.csv').is_file()
    assert (out / 'macro_summary.csv').is_file()
    assert (out / 'report.md').is_file()
    assert (out / 'summary.json').is_file()
    assert (out / 'error_maps' / 'DRIVE' / 'foo_compare.png').is_file()

    with open(out / 'per_image.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]['status'] == 'ok'
    assert rows[0]['components_pred'] == '2'
    assert rows[0]['gt_path'].endswith('foo.png')

    payload = json.loads((out / 'summary.json').read_text())
    assert payload['n_images'] == 1
    assert payload['n_unmatched'] == 0
    assert payload['macro_all']['n_datasets'] == 1


def test_prediction_index_pairs(audit_mod, tmp_path):
    pred_dir = tmp_path / 'pred' / 'DRIVE'
    gt_dir = tmp_path / 'gt' / 'DRIVE' / 'masks'
    pred_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    _write_mask(gt_dir / 'foo.png', np.ones((8, 8), dtype=np.uint8))
    _write_mask(pred_dir / 'foo_result.png', np.ones((8, 8), dtype=np.uint8))
    with (pred_dir / 'prediction_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'image', 'pred_path',
                                'gt_path', 'image_path'])
        writer.writeheader()
        writer.writerow({
            'dataset': 'DRIVE',
            'image': 'foo.png',
            'pred_path': 'foo_result.png',
            'gt_path': str(gt_dir / 'foo.png'),
            'image_path': '',
        })
    pairs, unmatched = audit_mod._pairs_from_prediction_index(pred_dir)
    assert len(pairs) == 1
    assert pairs[0][0].name == 'foo.png'
    assert pairs[0][1].name == 'foo_result.png'
    assert unmatched == []


def test_size_mismatch_is_reported(audit_mod, tmp_path):
    gt_root = tmp_path / 'gt' / 'DRIVE'
    _write_mask(gt_root / 'masks' / 'foo.png',
                np.ones((24, 24), dtype=np.uint8))
    pred_root = tmp_path / 'pred'
    _write_mask(pred_root / 'DRIVE' / 'foo.png_result.png',
                np.ones((16, 16), dtype=np.uint8))

    args = _args(tmp_path, strict=True)
    assert audit_mod.run_audit(args) == 1
    with open(args.out / 'per_image.csv', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['status'] == 'size_mismatch'
