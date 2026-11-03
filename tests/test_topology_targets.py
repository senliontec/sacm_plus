"""Regression tests: topology losses must stay active on 1-px structures."""

import torch

from core.losses.skeleton_recall import SkeletonRecallLoss
from core.losses.topograph import TopographLossAdapter
from core.losses.topology_utils import resize_target_for_topology


def _one_pixel_line(size: int = 1024) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size)
    mask[..., size // 2, size // 10:9 * size // 10] = 1.0
    return mask


def test_target_resize_preserves_one_pixel_line():
    gt = _one_pixel_line()
    for resolution in (256, 512):
        resized = resize_target_for_topology(gt, resolution)
        assert resized.shape == (1, 1, resolution, resolution)
        # 80% of the width must survive; bilinear + >0.5 gives zero.
        assert int((resized > 0.5).sum()) >= resolution // 2


def test_skeleton_recall_is_active_on_one_pixel_line():
    gt = _one_pixel_line()
    logits = torch.full_like(gt, -8.0)
    for resolution in (256, 512):
        loss = SkeletonRecallLoss(resolution=resolution)(logits, gt)
        # No prediction inside the tube -> recall is ~0 -> loss ~1.
        assert float(loss) > 0.9


def test_topograph_is_active_on_one_pixel_line():
    gt = _one_pixel_line()
    logits = torch.full_like(gt, -8.0)
    loss = TopographLossAdapter(resolution=256)(logits, gt)
    assert float(loss) > 0.5
