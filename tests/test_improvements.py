"""Tests for geometry, boundary loss and the empty-prediction guard."""

import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from core.data import (
    SegmentationDataset,
    TestDataset,
    _random_native_crop,
    _resize_pad_pair,
    letterbox_rect,
)
from core.evaluator import select_mask, select_mask_fixed

pytest.importorskip("scipy")

from core.losses.boundary import BoundaryLoss  # noqa: E402


def test_resize_pad_preserves_vertical_line_and_padding():
    image = Image.new('RGB', (200, 100), (255, 0, 0))
    mask = Image.new('L', (200, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((100, 0, 100, 99), fill=255, width=2)

    out_image, out_mask = _resize_pad_pair(image, mask, size=64)
    arr = np.asarray(out_mask)
    assert out_image.size == (64, 64)
    assert arr.shape == (64, 64)
    rows = np.where(arr.any(axis=1))[0]
    assert rows.min() == 16
    assert rows.max() == 47
    assert arr[0].sum() == 0
    assert arr[-1].sum() == 0
    # The vertical line must stay vertical, not become a filled square.
    assert arr[:, 31:34].sum() > 0
    assert arr[:, 0].sum() == 0


def test_letterbox_rect_matches_resize_pad_geometry():
    assert letterbox_rect(200, 100, 64) == (0, 16, 64, 32)
    assert letterbox_rect(100, 200, 64) == (16, 0, 32, 64)


def test_native_crop_keeps_foreground_pixel():
    image = Image.new('RGB', (200, 120), (0, 0, 0))
    mask = Image.new('L', (200, 120), 0)
    ImageDraw.Draw(mask).point((150, 60), fill=255)

    out_image, out_mask = _random_native_crop(
        image, mask, size=64, foreground_prob=1.0,
        rng=random.Random(0))
    assert out_image.size == (64, 64)
    assert out_mask.size == (64, 64)
    assert np.asarray(out_mask).sum() > 0


def test_segmentation_dataset_native_crop_shapes(tmp_path):
    image_dir = tmp_path / 'train' / 'images'
    mask_dir = tmp_path / 'train' / 'masks'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new('RGB', (200, 120), (10, 20, 30)).save(image_dir / 'x.png')
    mask = Image.new('L', (200, 120), 0)
    ImageDraw.Draw(mask).point((100, 60), fill=255)
    mask.save(mask_dir / 'x.png')

    dataset = SegmentationDataset(
        str(tmp_path), 'train', transform=transforms.ToTensor(),
        input_mode='native_crop', image_size=64, foreground_crop_prob=1.0)
    image, mask = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert tuple(mask.shape) == (1, 64, 64)
    assert float(mask.sum()) > 0


def test_select_mask_fixed_uses_requested_head():
    masks = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(1, 4, 1, 1)
    chosen, head = select_mask_fixed(masks, 2)
    assert head == 2
    assert torch.equal(chosen, masks[:, 2:3])


def test_select_mask_fixed_empty_guard_redirects_to_nonempty_head():
    masks = torch.tensor([-1.0, 1.0, -2.0, -3.0]).view(1, 4, 1, 1)
    chosen, head = select_mask_fixed(masks, 0, empty_guard=True)
    assert head == 1
    assert torch.equal(chosen, masks[:, 1:2])


def test_testdataset_aspect_pad_keeps_original_mask_size(tmp_path):
    image_dir = tmp_path / 'images'
    mask_dir = tmp_path / 'masks'
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.new('RGB', (200, 100), (10, 20, 30)).save(image_dir / 'x.png')
    Image.new('L', (200, 100), 0).save(mask_dir / 'x.png')

    dataset = TestDataset(str(tmp_path), input_mode='aspect_pad',
                          image_size=64)
    model_input, original_mask, original_size, name, fov = dataset[0]
    assert name == 'x.png'
    assert tuple(model_input.shape) == (3, 64, 64)
    assert tuple(original_mask.shape) == (1, 100, 200)
    assert tuple(original_size) == (200, 100)
    assert tuple(fov.shape) == (1, 100, 200)


def test_boundary_loss_is_finite_and_has_gradient():
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 6:18, 10:26] = 1.0
    loss = BoundaryLoss(resolution=32)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_boundary_loss_empty_target_is_zero():
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16)
    loss = BoundaryLoss(resolution=16)(logits, target)
    assert float(loss) == 0.0


def test_empty_guard_switches_to_nonempty_candidate():
    masks = torch.zeros(1, 4, 8, 8)
    masks[:, 2] = 1.0
    iou_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    selected = select_mask(masks, iou_pred, 'iou', empty_guard=False)
    assert float(selected.sum()) == 0.0
    guarded = select_mask(masks, iou_pred, 'iou', empty_guard=True)
    assert float(guarded.sum()) == 64.0


def test_empty_guard_keeps_nonempty_selection():
    masks = torch.zeros(1, 4, 8, 8)
    masks[:, 1] = 1.0
    iou_pred = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    selected = select_mask(masks, iou_pred, 'iou', empty_guard=True)
    assert float(selected.sum()) == 64.0
    assert torch.equal(selected, masks[:, 1:2])
