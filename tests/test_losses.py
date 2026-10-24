"""Loss smoke tests: every registered topology loss must run on a small
input and return a finite scalar; core losses have closed-form checks."""

import numpy as np
import pytest
import torch

from core.losses import TOPOLOGY_LOSS_REGISTRY, build_topology_loss
from core.losses import DiceBCELoss, SCNP, SCNPCriterion, SoftclDiceLoss


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


def test_scnp_foreground_takes_min_neighbor_logit():
    logits = torch.zeros(1, 1, 3, 3)
    logits[0, 0, 1, 1] = -5.0
    target = torch.ones_like(logits)
    out = SCNP('2D', 3)(logits, target)
    # Every 3x3 window contains the single minimum foreground logit.
    assert torch.allclose(out, torch.full_like(out, -5.0))


def test_scnp_background_takes_max_neighbor_logit():
    logits = torch.zeros(1, 1, 3, 3)
    logits[0, 0, 1, 1] = 5.0
    target = torch.zeros_like(logits)
    out = SCNP('2D', 3)(logits, target)
    assert torch.allclose(out, torch.full_like(out, 5.0))


def test_scnp_does_not_mix_classes():
    # Center is foreground (-2); every other pixel is background (+10).
    logits = torch.full((1, 1, 5, 5), 10.0)
    logits[0, 0, 2, 2] = -2.0
    target = torch.zeros_like(logits)
    target[0, 0, 2, 2] = 1.0
    out = SCNP('2D', 3)(logits, target)
    # The foreground pixel's window contains only itself (same class), so
    # the 9999 sentinel must keep the background +10 out of its min.
    assert out[0, 0, 2, 2].item() == pytest.approx(-2.0)
    # Background pixels never take the foreground -2 into their max.
    assert out[0, 0, 0, 0].item() == pytest.approx(10.0)
    assert out[0, 0, 1, 2].item() == pytest.approx(10.0)


def test_scnp_criterion_identity_for_constant_logits():
    # With constant logits SCNP is an identity transform, so the wrapped
    # criterion must match the base loss exactly.
    base = DiceBCELoss()
    wrapped = SCNPCriterion(base, neighborhood_size=3)
    logits = torch.full((1, 1, 16, 16), 0.25)
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    assert torch.allclose(wrapped(logits, target), base(logits, target))


def test_scnp_criterion_original_weight():
    base = DiceBCELoss()
    logits = torch.full((1, 1, 16, 16), 0.25)
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    plain = base(logits, target)
    mixed = SCNPCriterion(base, original_weight=1.0)(logits, target)
    # Constant logits: SCNP is identity, so mixed = 2 * plain.
    assert torch.allclose(mixed, 2 * plain)


def test_scnp_rejects_even_window_and_shape_mismatch():
    with pytest.raises(ValueError):
        SCNP('2D', 4)
    with pytest.raises(AssertionError):
        SCNP('2D', 3)(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)[..., :4])


def test_scnp_gradients_are_finite():
    logits = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    loss = SCNPCriterion(DiceBCELoss())(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_softcl_dice_iterations_alias():
    # 回归:训练 CLI 曾用 iterations= 而类只认 num_iter(服务器实测崩溃),
    # 别名 + 正确参数必须都可用
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    logits = torch.randn(1, 1, 16, 16)
    a = SoftclDiceLoss(num_iter=5)(logits, target)
    b = SoftclDiceLoss(iterations=5)(logits, target)
    assert torch.isfinite(a) and torch.isfinite(b)
    assert a.item() == b.item()
