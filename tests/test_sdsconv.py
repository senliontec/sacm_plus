"""SDSConv project-extension tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.sdsconv import SDSConvAdapter
from baseline_zoo.models.convlora import build_convlora
from baseline_zoo.models.sdsconv import SDSConv2d, SDSConvResidual, SDSConvSAM

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / 'checkpoints' / 'sam_vit_b_01ec64.pth')


def test_sdsconv_shapes_and_zero_init():
    conv = SDSConv2d(16, kernel_size=5)
    x = torch.randn(1, 16, 8, 8)
    y = conv(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


def test_sdsconv_residual_is_identity_at_init():
    residual = SDSConvResidual(16, hidden=8, kernel_size=5)
    x = torch.randn(1, 16, 8, 8)
    assert torch.allclose(residual(x), x)
    residual.gamma.data.fill_(1.0)
    with torch.no_grad():
        residual.sds.fuse[-1].weight.normal_(0.0, 0.01)
    assert not torch.allclose(residual(x), x)


def test_registry_metadata():
    assert 'sdsconv' in available_adapters()
    adapter = load_adapter('sdsconv')
    assert isinstance(adapter, SDSConvAdapter)
    assert adapter.id == 'sdsconv'
    assert 'project extension' in adapter.official_source


def test_config_roundtrip():
    adapter = SDSConvAdapter()
    args = SimpleNamespace(
        input_size=1024,
        convlora_model_type='vit_b',
        convlora_rank=3,
        convlora_alpha=32,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint=CHECKPOINT,
        sdsconv_kernel_size=7,
        sdsconv_hidden=32,
        sdsconv_new_lr_mult=5.0,
    )
    adapter._apply_sds(args)
    config = adapter.save_config(args)
    target = SDSConvAdapter()
    restored = SimpleNamespace(
        input_size=1024,
        convlora_model_type='vit_b',
        convlora_rank=3,
        convlora_alpha=32,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint=CHECKPOINT,
        sdsconv_kernel_size=5,
        sdsconv_hidden=16,
        sdsconv_new_lr_mult=1.0,
    )
    target.restore_config(restored, config)
    assert target._sds_kernel == 7
    assert target._sds_hidden == 32
    assert target._sds_new_lr_mult == pytest.approx(5.0)


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason='SAM ViT-B checkpoint is not available')
def test_hook_preserves_baseline_at_zero_gamma():
    base = build_convlora(CHECKPOINT, model_type='vit_b').eval()
    model = SDSConvSAM(base, hidden=8, kernel_size=5).eval()
    images = torch.rand(1, 3, 1024, 1024) * 255
    with torch.no_grad():
        baseline, _ = base(images)
        logits, _ = model(images)
    assert logits.shape == baseline.shape
    assert torch.allclose(logits, baseline, atol=1e-6)
