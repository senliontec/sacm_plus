"""Regression tests for the seen/unseen/novel dataset grouping step."""

import importlib.util
import os

import numpy as np
from PIL import Image

_SCRIPT = os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'group_datasets.py'
)
_spec = importlib.util.spec_from_file_location('group_datasets', _SCRIPT)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)


def _pair(root, name, size=8):
    image_dir = root / name / 'images'
    mask_dir = root / name / 'masks'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(
        image_dir / f'{name}.png'
    )
    Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(
        mask_dir / f'{name}.png'
    )


def test_group_flat_layout_and_binarize_novel(tmp_path):
    root = tmp_path / 'datasets'
    legacy = tmp_path / 'datasets_legacy'
    _pair(root, 'DRIVE_train')
    _pair(root, 'DIS5K_train')
    _pair(root, 'HRF')
    _pair(root, 'LES-AV')
    _pair(root, 'CHCrack5k_test')
    _pair(root, 'tyre')

    # Novel source: anti-aliased RGB mask, like the in-house TYRE data.
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 200
    Image.fromarray(np.stack([mask, mask, mask], axis=-1)).save(
        root / 'tyre' / 'masks' / 'tyre.png'
    )

    summary = gd.group_datasets(root, legacy)

    assert (root / 'seen/DRIVE/images/DRIVE_train.png').exists()
    assert not (root / 'DRIVE_train').exists()
    assert (root / 'unseen/HRF/images/HRF.png').exists()
    assert (root / 'unseen/LES-AV/images/LES-AV.png').exists()
    assert (root / 'unseen/CHCrack5k/images/CHCrack5k_test.png').exists()
    assert (legacy / 'DIS5K_train/images/DIS5K_train.png').exists()
    assert not (root / 'DIS5K_train').exists()
    assert not (root / 'tyre').exists()

    novel_mask = Image.open(root / 'novel/TYRE/masks/tyre.png')
    assert novel_mask.mode == 'L'
    assert set(np.unique(np.array(novel_mask)).tolist()) == {0, 255}
    assert summary == {'moved': 4, 'legacy': 1, 'novel': 1}
