"""Loss smoke tests: every registered topology loss must run on a small
input and return a finite scalar; core losses have closed-form checks."""

import numpy as np
import pytest
import torch

from core.losses import TOPOLOGY_LOSS_REGISTRY, build_topology_loss
from core.losses import DiceBCELoss, SoftclDiceLoss


def _make_inputs(size=64):
    logits = torch.randn(1, 1, size, size) * 2.0
    target = (torch.rand(1, 1, size, size) > 0.7).float()
    return logits, target


def _build(name):
    # Engine-based adapters accept a resolution kwarg; the pure-torch
    # losses do not. Fall back gracefully.
    try:
        return build_topology_loss(name, resolution=64)
    except TypeError:
        return build_topology_loss(name)


@pytest.mark.parametrize("name", sorted(TOPOLOGY_LOSS_REGISTRY))
def test_registered_loss_smoke(name):
    loss = _build(name)
    logits, target = _make_inputs(64)
    out = loss(logits, target)
    assert torch.isfinite(out).item(), f"{name} produced a non-finite loss"
    assert out.dim() == 0, f"{name} did not return a scalar"


def test_softcl_dice_zero_for_identity():
    target = (torch.rand(1, 1, 32, 32) > 0.5).float()
    # Saturating logits: sigmoid(±50) is exactly 0/1 in float32, so the
    # soft skeleton matches the target skeleton and the loss is exactly 0.
    # (At ±5 the residual softplus/skeleton mass is ~1e-2, not zero.)
    logits = torch.where(target > 0.5, torch.tensor(50.0), torch.tensor(-50.0))
    loss = SoftclDiceLoss()(logits, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_dice_bce_zero_for_identity():
    target = (torch.rand(1, 1, 32, 32) > 0.5).float()
    # Saturating logits (see above): BCE = softplus(-50) = 0 exactly.
    logits = torch.where(target > 0.5, torch.tensor(50.0), torch.tensor(-50.0))
    loss = DiceBCELoss()(logits, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_registry_contains_none():
    # The default CLI option must always be usable
    assert build_topology_loss('none') is None
