"""Joint augmentation correctness: image/mask alignment, identity limits,
grayscale color-jitter skipping."""

import numpy as np
import torch

from core.augmentation import JointAugment, elastic_deform


def test_flips_align_image_and_mask():
    aug = JointAugment(p_flip=1.0, p_rot90=0.0, p_elastic=0.0, p_color=0.0)
    rng = np.random.default_rng(0)
    img = torch.tensor(rng.random((3, 32, 32)), dtype=torch.float32)
    mask = torch.tensor((rng.random((1, 32, 32)) > 0.5), dtype=torch.float32)
    img2, mask2 = aug(img, mask)
    # p_flip=1.0 -> horizontal AND vertical flips both applied
    expected_img = torch.flip(torch.flip(img, dims=[-1]), dims=[-2])
    expected_mask = torch.flip(torch.flip(mask, dims=[-1]), dims=[-2])
    assert torch.equal(img2, expected_img)
    assert torch.equal(mask2, expected_mask)


def test_elastic_zero_alpha_is_identity():
    rng = np.random.default_rng(0)
    img = torch.tensor(rng.random((3, 16, 16)), dtype=torch.float32)
    mask = torch.tensor((rng.random((1, 16, 16)) > 0.5), dtype=torch.float32)
    img2, mask2 = elastic_deform(img, mask, alpha=0.0, sigma=3.0)
    assert torch.allclose(img2, img, atol=1e-6)
    assert torch.allclose(mask2, mask, atol=1e-6)


def test_grayscale_skips_color_jitter():
    aug = JointAugment(p_flip=0.0, p_rot90=0.0, p_elastic=0.0, p_color=1.0)
    gray = torch.full((3, 8, 8), 0.5)
    mask = torch.zeros((1, 8, 8))
    img2, mask2 = aug(gray, mask)
    # R==G==B -> color jitter skipped even though p_color=1.0
    assert torch.equal(img2, gray)
    assert torch.equal(mask2, mask)
