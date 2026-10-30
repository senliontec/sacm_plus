"""Conv-SACM project-extension tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline_zoo.adapters.convsacm import ConvSACMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.models.convlora import build_convlora
from baseline_zoo.models.convsacm import (
    ConvSACMSAM,
    ExternalContextAdapter,
    HighResRefiner,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / 'checkpoints' / 'sam_vit_b_01ec64.pth')


def test_external_context_zero_init_and_shape():
    adapter = ExternalContextAdapter(
        [16, 16], out_dim=8, hidden=8, pool_size=4)
    features = [torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)]
    out = adapter(features, (8, 8))
    assert out.shape == (1, 8, 8, 8)
    assert torch.allclose(out, torch.zeros_like(out))
    with torch.no_grad():
        adapter.fuse[-1].weight.normal_(0.0, 0.1)
    assert not torch.allclose(adapter(features, (8, 8)),
                              torch.zeros_like(out))


def test_high_res_refiner_zero_init_and_shape():
    refiner = HighResRefiner(out_size=512, hidden=4)
    coarse = torch.randn(1, 1, 16, 16)
    images = torch.rand(1, 3, 64, 64) * 255
    out = refiner(coarse, images)
    assert out.shape == (1, 1, 512, 512)
    expected = torch.nn.functional.interpolate(
        coarse, size=(512, 512), mode='bilinear', align_corners=False)
    assert torch.allclose(out, expected)
    with torch.no_grad():
        refiner.fuse[-1].weight.normal_(0.0, 0.01)
    assert not torch.allclose(refiner(coarse, images), expected)


def test_registry_metadata():
    assert 'convsacm' in available_adapters()
    adapter = load_adapter('convsacm')
    assert isinstance(adapter, ConvSACMAdapter)
    assert adapter.id == 'convsacm'
    assert 'project extension' in adapter.official_source


def test_config_roundtrip():
    adapter = ConvSACMAdapter()
    args = SimpleNamespace(
        input_size=1024,
        convlora_model_type='vit_b',
        convlora_rank=3,
        convlora_alpha=32,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint=CHECKPOINT,
        convsacm_external_hidden=32,
        convsacm_external_fractions=[0.25, 0.5],
        convsacm_refiner_hidden=8,
        convsacm_refiner_size=512,
        convsacm_topo_weight=0.1,
        convsacm_topo_warmup=5,
        convsacm_new_lr_mult=5.0,
    )
    adapter._apply_convsacm(args)
    config = adapter.save_config(args)
    target = ConvSACMAdapter()
    restored = SimpleNamespace(**vars(args))
    restored.convsacm_external_hidden = 64
    restored.convsacm_external_fractions = [0.5]
    restored.convsacm_refiner_hidden = 16
    restored.convsacm_topo_weight = 0.0
    restored.convsacm_new_lr_mult = 1.0
    target.restore_config(restored, config)
    assert target._external_hidden == 32
    assert target._external_fractions == (0.25, 0.5)
    assert target._refiner_hidden == 8
    assert target._topo_weight == pytest.approx(0.1)
    assert target._new_lr_mult == pytest.approx(5.0)


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason='SAM ViT-B checkpoint is not available')
def test_full_model_zero_init_matches_convlora():
    base = build_convlora(CHECKPOINT, model_type='vit_b').eval()
    model = ConvSACMSAM(
        base, external_hidden=8, refiner_hidden=4).eval()
    images = torch.rand(1, 3, 1024, 1024) * 255
    with torch.no_grad():
        coarse, high_res, _ = model(images)
    assert coarse.shape == (1, 1, 256, 256)
    assert high_res.shape == (1, 1, 512, 512)
    # The high-res refiner is zero-initialized: stage 2 is the coarse
    # prediction at 512, without changing its values.
    expected = torch.nn.functional.interpolate(
        coarse, size=(512, 512), mode='bilinear', align_corners=False)
    assert torch.allclose(high_res, expected, atol=1e-6)
