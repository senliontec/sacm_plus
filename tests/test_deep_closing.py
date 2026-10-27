"""Fast tests for the Deep Closing post-processing port."""

from __future__ import annotations

import random

import numpy as np
import torch
from scipy.ndimage import label

from core.postprocess.deep_closing import (
    DeepClosingRefiner,
    SimplePointErosion,
    UNetMIM,
    is_simple_patch,
    mim_loss,
    random_block_mask,
    recurrent_tube_scale,
    save_checkpoint,
)


def _components(mask: torch.Tensor) -> int:
    return int(label(
        mask[0, 0].detach().cpu().numpy() > 0.5,
        structure=np.ones((3, 3), dtype=np.uint8),
    )[1])


def test_simple_point_criterion():
    isolated = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    interior = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    endpoint = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    assert not is_simple_patch(isolated)
    assert not is_simple_patch(interior)
    assert is_simple_patch(endpoint)


def test_rspe_preserves_topology():
    spe = SimplePointErosion()
    mask = torch.zeros(1, 1, 21, 21)
    mask[0, 0, 5:16, 10:12] = 1
    mask[0, 0, 10:12, 5:16] = 1
    before = _components(mask)
    eroded = spe.rspe(mask, mask, max_iters=1000)
    assert _components(eroded) == before
    assert float(eroded.sum()) <= float(mask.sum())


def test_simple_component_erosion_keeps_original_prediction():
    spe = SimplePointErosion()
    original = torch.zeros(1, 1, 25, 25)
    original[0, 0, 5:20, 8:10] = 1
    original[0, 0, 5:7, 8:18] = 1
    dilated = original.clone()
    dilated[0, 0, 10:12, 10:12] = 1  # a small added blob
    refined = spe.simple_component_erosion(
        original, dilated, max_iters=20)
    assert refined.shape == original.shape
    assert torch.all(refined >= original)
    assert float(refined.sum()) >= float(original.sum())


def test_mim_loss_backward_and_mask_generator():
    torch.manual_seed(0)
    spe = SimplePointErosion()
    model = UNetMIM(channels=(8, 16, 32))
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8:24, 15:17] = 1
    rng = random.Random(0)
    scaled = recurrent_tube_scale(target, spe, rng, max_k=2)
    masked = random_block_mask(scaled, rng)
    assert masked.shape == target.shape
    assert float(masked.mean()) > 0.0
    prediction = model(scaled * (1.0 - masked))
    loss = mim_loss(prediction, scaled, masked, spe, topo_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.inc.block[0].weight.grad is not None


def test_refiner_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = UNetMIM(channels=(8, 16, 32))
    checkpoint = tmp_path / 'shape_prior.pth'
    save_checkpoint(
        checkpoint, model,
        {
            'model': {
                'in_channels': 1,
                'out_channels': 1,
                'channels': [8, 16, 32],
            },
            'resolution': 32,
            'dilation_threshold': 0.2,
            'max_iters': 20,
        },
        epoch=1,
        val_loss=0.5,
    )
    refiner = DeepClosingRefiner.from_checkpoint(
        checkpoint, device='cpu')
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:20, 15:17] = True
    first = refiner.refine(mask)
    second = refiner.refine(mask)
    assert first.shape == mask.shape
    assert first.dtype == bool
    assert np.array_equal(first, second)
    empty = refiner.refine(np.zeros((8, 8), dtype=bool))
    assert not empty.any()
