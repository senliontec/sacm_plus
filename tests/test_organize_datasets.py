"""Regression tests for the DRIVE test-mask conversion bug.

The official DRIVE test/mask is a FOV mask, not vessel GT. The converter
must never write it to DRIVE_test/masks; it belongs in DRIVE_test/fov.
"""

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'data',
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


def test_organize_hrf_pairs_masks_and_fov(tmp_path):
    src = tmp_path / 'HRF'
    _write_image(src / 'images/01_h.jpg')
    _write_mask(src / 'masks/01_h.tif', 0.07)
    _write_mask(src / 'roi_masks/01_h_mask.tif', 0.70)

    out = tmp_path / 'out'
    n = od.organize_hrf(str(src), str(out))

    assert n == 1
    assert (out / 'HRF/images/01_h.jpg').exists()
    assert (out / 'HRF/masks/01_h.png').exists()
    assert (out / 'HRF/fov/01_h.png').exists()
    mask = np.array(Image.open(out / 'HRF/masks/01_h.png'))
    assert set(np.unique(mask)) == {0, 255}
    fov = np.array(Image.open(out / 'HRF/fov/01_h.png'))
    assert set(np.unique(fov)) == {0, 255}


def test_organize_lesav_uses_vessel_segmentations(tmp_path):
    src = tmp_path / 'LES-AV'
    _write_image(src / 'images/12.png')
    _write_mask(src / 'masks/12_mask.gif', 0.50)
    _write_mask(src / 'vessel-segmentations/12.png', 0.07)

    out = tmp_path / 'out'
    n = od.organize_lesav(str(src), str(out))

    assert n == 1
    assert (out / 'LES-AV/images/12.png').exists()
    mask = np.array(Image.open(out / 'LES-AV/masks/12.png'))
    assert set(np.unique(mask)) == {0, 255}
    assert od.mask_foreground_ratio(out / 'LES-AV/masks/12.png') == pytest.approx(0.07, abs=0.01)


def test_organize_chcrack5k_skips_blank_gt(tmp_path):
    src = tmp_path / 'CHCrack5k'
    _write_image(src / 'test/image/a.jpg')
    _write_mask(src / 'test/gt/a.png', 0.01)
    _write_image(src / 'test/image/b.jpg')
    _write_mask(src / 'test/gt/b.png', 0.0)

    out = tmp_path / 'out'
    n = od.organize_chcrack5k(str(src), str(out))

    assert n == 1
    assert (out / 'CHCrack5k_test/images/a.jpg').exists()
    assert (out / 'CHCrack5k_test/masks/a.png').exists()
    assert not (out / 'CHCrack5k_test/images/b.jpg').exists()
    assert not (out / 'CHCrack5k_test/masks/b.png').exists()


def test_cremi_skips_blank_raw_slices(tmp_path):
    """Volume C z014/z074 are all-zero in the released raw HDF5.

    Their annotations are non-empty, but an all-black image carries no
    signal. The converter must drop those pairs instead of shipping them
    into Seen evaluation.
    """
    h5py = pytest.importorskip('h5py')
    src = tmp_path / 'CREMI'
    src.mkdir()
    raw = np.zeros((3, 8, 8), dtype=np.uint8)
    raw[1] = 10
    raw[2] = 20
    nid = np.zeros((3, 8, 8), dtype=np.uint64)
    nid[1, :, 4:] = 1
    nid[2, :, 4:] = 1
    with h5py.File(src / 'sample_A_20160501.hdf', 'w') as f:
        f.create_dataset('volumes/raw', data=raw)
        f.create_dataset('volumes/labels/neuron_ids', data=nid)

    out = tmp_path / 'out'
    stale_img = out / 'CREMI/images/sample_A_z000.png'
    stale_mask = out / 'CREMI/masks/sample_A_z000.png'
    stale_img.parent.mkdir(parents=True)
    stale_mask.parent.mkdir(parents=True)
    stale_img.write_bytes(b'stale')
    stale_mask.write_bytes(b'stale')

    n = od.organize_cremi(str(src), str(out))

    assert n == 2
    assert not (out / 'CREMI/images/sample_A_z000.png').exists()
    assert not (out / 'CREMI/masks/sample_A_z000.png').exists()
    assert (out / 'CREMI/images/sample_A_z001.png').exists()
    assert (out / 'CREMI/images/sample_A_z002.png').exists()
    mask = np.array(Image.open(out / 'CREMI/masks/sample_A_z001.png'))
    assert set(np.unique(mask)) <= {0, 255}
