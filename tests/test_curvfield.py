"""CurvField tests: field targets, adapter, hooks and config."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline_zoo.adapters.curvfield import CurvFieldAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.models.convlora import build_convlora
from baseline_zoo.models.curvfield import (
    CurvFieldSAM,
    StructureConditionedAdapter,
    StructureFieldHead,
    field_loss,
    structure_targets,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / 'checkpoints' / 'sam_vit_b_01ec64.pth')


def _args(**overrides):
    values = dict(
        input_size=1024,
        epochs=100,
        convlora_model_type='vit_b',
        convlora_rank=3,
        convlora_alpha=32,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint=CHECKPOINT,
        curvfield_field_hidden=32,
        curvfield_adapter_hidden=16,
        curvfield_adapter_fractions=[0.25, 0.5, 0.75],
        curvfield_k=7,
        curvfield_max_step=4.0,
        curvfield_field_weight=1.0,
        curvfield_new_lr_mult=10.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _mask(size=64):
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, size // 3:2 * size // 3, size // 2 - 1:size // 2 + 1] = 1
    mask[:, :, size // 2 - 1:size // 2 + 1, size // 4:3 * size // 4] = 1
    return mask


def test_registry_and_metadata():
    assert 'curvfield' in available_adapters()
    adapter = load_adapter('curvfield')
    assert isinstance(adapter, CurvFieldAdapter)
    assert adapter.id == 'curvfield'
    assert adapter.display_name == 'CurvField'
    assert any('project extension' in item for item in adapter.deviations)


def test_structure_targets_and_loss():
    mask = _mask()
    targets, foreground = structure_targets(mask, size=64)
    assert targets.shape == (1, 6, 64, 64)
    assert foreground.shape == (1, 1, 64, 64)
    assert torch.isfinite(targets).all()
    assert targets[:, 0].max() > 0.5
    assert targets[:, 4].max() > 0.5
    assert targets[:, 1].max() > 0.0

    head = StructureFieldHead(16, hidden=32)
    features = torch.randn(1, 64, 64, 16)
    pred = head(features)
    assert pred.shape == targets.shape
    loss = field_loss(pred, targets, foreground)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())


def test_adapter_is_identity_at_init_and_uses_field():
    adapter = StructureConditionedAdapter(16, hidden=16, k=7)
    tokens = torch.randn(1, 64, 64, 16)
    field = torch.zeros(1, 6, 64, 64)
    field[:, 0] = 1.0
    field[:, 1] = 0.5
    field[:, 2] = 1.0
    field[:, 5] = 1.0
    out = adapter(tokens, field)
    assert out.shape == tokens.shape
    assert torch.allclose(out, tokens)
    adapter.gamma.data.fill_(1.0)
    out = adapter(tokens, field)
    assert not torch.allclose(out, tokens)


def test_config_roundtrip():
    adapter = CurvFieldAdapter()
    args = _args()
    adapter._apply_arch(args)
    config = adapter.save_config(args)
    target = CurvFieldAdapter()
    restored = _args(
        curvfield_field_hidden=128,
        curvfield_adapter_hidden=64,
        curvfield_adapter_fractions=[0.25],
        curvfield_k=5,
        curvfield_field_weight=0.0,
        curvfield_new_lr_mult=1.0,
    )
    target.restore_config(restored, config)
    assert target._field_hidden == 32
    assert target._adapter_hidden == 16
    assert target._adapter_fractions == (0.25, 0.5, 0.75)
    assert target._k == 7
    assert target._field_weight == pytest.approx(1.0)
    assert target._new_lr_mult == pytest.approx(10.0)


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason='SAM ViT-B checkpoint is not available')
def test_hooks_preserve_baseline_at_zero_gamma():
    base = build_convlora(CHECKPOINT, model_type='vit_b').eval()
    model = CurvFieldSAM(
        base, field_hidden=32, adapter_hidden=16,
        adapter_fractions=(0.25, 0.5, 0.75), k=7).eval()
    images = torch.rand(1, 3, 1024, 1024) * 255
    with torch.no_grad():
        baseline, _ = base(images)
        logits, _, field = model(images)
    assert field is not None and field.shape == (1, 6, 64, 64)
    assert logits.shape == baseline.shape
    assert torch.allclose(logits, baseline, atol=1e-6)
    assert len(model.adapters) == 3
    assert model.adapter_indices == [3, 6, 8]
