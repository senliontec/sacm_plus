"""Regression tests for the DRIVE test-mask conversion bug.

The official DRIVE test/mask is a FOV mask, not vessel GT. The converter
must never write it to DRIVE_test/masks; it belongs in DRIVE_test/fov.
"""

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                       'organize_datasets.py')
_spec = importlib.util.spec_from_file_location('organize_datasets', _SCRIPT)
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)


def _write_image(path, size=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(path)


def _write_mask(path, ratio, size=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size), dtype=np.uint8)
    arr.flat[:int(round(ratio * size * size))] = 255
    Image.fromarray(arr).save(path)


def _make_drive(tmp_path, valid_test_gt=False):
    src = tmp_path / 'DRIVE'
    _write_image(src / 'training/images/21_training.tif')
    _write_mask(src / 'training/1st_manual/21_manual1.gif', 0.07)
    _write_image(src / 'test/images/01_test.tif')
    _write_mask(src / 'test/mask/01_test_mask.gif', 0.68)   # FOV
    if valid_test_gt:
        _write_mask(src / 'test/1st_manual/01_manual1.gif', 0.07)
    return src


def test_validate_vessel_mask_rejects_fov(tmp_path):
    fov = tmp_path / 'fov.gif'
    _write_mask(fov, 0.68)
    with pytest.raises(ValueError, match='FOV'):
        od.validate_vessel_mask(str(fov))

    vessel = tmp_path / 'vessel.gif'
    _write_mask(vessel, 0.07)
    assert od.validate_vessel_mask(str(vessel)) == pytest.approx(0.07, abs=0.01)


def test_missing_test_gt_keeps_fov_out_of_masks(tmp_path):
    src = _make_drive(tmp_path)
    out = tmp_path / 'out'
    n = od.organize_drive(str(src), str(out))
    assert n == 1

    test = out / 'DRIVE_test'
    assert (test / 'images/01_test.tif').exists()
    assert not (test / 'masks').exists()
    assert (test / 'fov/01_test.png').exists()
    assert (test / 'README.md').exists()


def test_valid_test_gt_builds_masks(tmp_path):
    src = _make_drive(tmp_path, valid_test_gt=True)
    out = tmp_path / 'out'
    stale = out / 'DRIVE_test/README.md'
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('stale: masks absent')

    od.organize_drive(str(src), str(out))

    mask = out / 'DRIVE_test/masks/01_test.png'
    assert mask.exists()
    assert not stale.exists()
    ratio = od.mask_foreground_ratio(str(mask))
    assert 0.005 <= ratio <= 0.30


def test_explicit_fov_gt_is_rejected(tmp_path):
    src = _make_drive(tmp_path)
    out = tmp_path / 'out'
    od.organize_drive(str(src), str(out),
                      drive_test_gt=str(src / 'test/mask'))

    assert not (out / 'DRIVE_test/masks').exists()
    assert (out / 'DRIVE_test/fov/01_test.png').exists()


def test_old_fov_masks_are_quarantined(tmp_path):
    src = _make_drive(tmp_path)
    out = tmp_path / 'out'
    _write_mask(out / 'DRIVE_test/masks/01_test.png', 0.68)

    od.organize_drive(str(src), str(out))

    assert not (out / 'DRIVE_test/masks').exists()
    assert (out / 'DRIVE_test/fov/01_test.png').exists()
