"""Lightweight tests for the FoB-SAM protocol adapter.

The full model needs a 54M-parameter ResNet-101 and SAM ViT-H; these tests
cover the protocol/preprocessing pieces that can run on CPU with synthetic
image pairs.  Model forward/backward is exercised by the smoke commands in
``docs/BASELINE_PORTING.md``.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline_zoo.adapters.fob_sam import FoBSAMAdapter


def _write_pair(root: Path, dataset: str, index: int, color=False):
    image_dir = root / 'images'
    mask_dir = root / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{dataset}_{index:03d}'
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[4:12, 4:12] = 200
    if color:
        array[..., 0] = np.arange(16)[None, :] * 10
    Image.fromarray(array, mode='RGB').save(image_dir / f'{stem}.png')
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    Image.fromarray(mask, mode='L').save(mask_dir / f'{stem}.png')


def _adapter(tmp_path: Path, shots=1):
    train = tmp_path / 'train'
    for dataset in ('DRIVE', 'CrackTree200', 'CORN'):
        for index in range(3):
            _write_pair(train, dataset, index)
    args = argparse.Namespace(
        data_root=tmp_path, fob_shots=shots,
        fob_support_dataset='auto')
    adapter = FoBSAMAdapter()
    adapter.args = args
    adapter.device = torch.device('cpu')
    return adapter


def test_support_policy_same_domain_and_donor(tmp_path):
    adapter = _adapter(tmp_path)
    # Seen domains use their own train shots.
    assert adapter._support_dataset('DRIVE') == 'DRIVE'
    assert adapter._support_dataset('CORN') == 'CORN'
    # Unseen crack-like/vessel-like datasets use the structural donor.
    assert adapter._support_dataset('DeepCrack') == 'CrackTree200'
    assert adapter._support_dataset('CHCrack5k') == 'CrackTree200'
    assert adapter._support_dataset('FIVES') == 'DRIVE'
    assert adapter._support_dataset('STARE') == 'DRIVE'
    # Unknown/other unseen datasets fall back to a valid seen domain.
    assert adapter._support_dataset('UnknownDataset') in {'DRIVE', 'CORN',
                                                         'CrackTree200'}


def test_support_shots_are_nonempty_and_capped(tmp_path):
    adapter = _adapter(tmp_path, shots=2)
    pairs = adapter._support_pairs('FIVES')
    assert len(pairs) == 2
    for sample in pairs:
        assert adapter._load_pair(sample)[1].sum() > 0


def test_preprocessing_shapes_and_finiteness(tmp_path):
    adapter = _adapter(tmp_path)
    sample = adapter._support_pairs('DRIVE')[0]
    image, mask = adapter._load_pair(sample)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert torch.isfinite(image).all()
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0


def test_color_and_grayscale_normalization(tmp_path):
    adapter = _adapter(tmp_path)
    gray = Image.new('RGB', (16, 16), (128, 128, 128))
    color = Image.new('RGB', (16, 16), (10, 20, 30))
    gray_tensor = adapter._normalize(gray)
    color_tensor = adapter._normalize(color)
    assert gray_tensor.shape == color_tensor.shape == (3, 256, 256)
    assert torch.isfinite(gray_tensor).all()
    assert torch.isfinite(color_tensor).all()


def test_eval_sample_uses_dummy_query_label(tmp_path):
    adapter = _adapter(tmp_path)
    sample = adapter._support_pairs('DRIVE')[0]
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    assert prepared['query_label'].shape == (1, 256, 256)
    assert torch.all(prepared['query_label'] == 1)
    assert prepared['query_image'].shape == (1, 3, 256, 256)
    assert len(prepared['support_images'][0]) == 1


def test_epoch_budget_matches_official_iterations():
    adapter = FoBSAMAdapter()
    assert adapter.official_epochs * 3000 >= 39000
    assert adapter.official_lr == 1e-4
