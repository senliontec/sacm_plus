"""Tests for deterministic data ordering and worker seeding."""

import random

import numpy as np
from PIL import Image

from core.data import _find_pairs
from core.reproducibility import seed_worker


def test_find_pairs_is_sorted(tmp_path):
    image_dir = tmp_path / 'images'
    mask_dir = tmp_path / 'masks'
    image_dir.mkdir()
    mask_dir.mkdir()
    for name in ('c.png', 'a.png', 'b.png'):
        Image.new('RGB', (2, 2)).save(image_dir / name)
        Image.new('L', (2, 2)).save(mask_dir / name)
    assert _find_pairs(str(image_dir), str(mask_dir)) == [
        'a.png', 'b.png', 'c.png']


def test_seed_worker_is_reproducible():
    random.seed(0)
    np.random.seed(0)
    seed_worker(0)
    first = (random.random(), float(np.random.rand()))
    random.seed(0)
    np.random.seed(0)
    seed_worker(0)
    second = (random.random(), float(np.random.rand()))
    assert first == second
