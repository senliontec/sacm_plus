"""NativeToken project-extension tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from baseline_zoo.models.convlora import build_convlora
from methods.nativetok import NativeResidualTokenizer, NativeTokenSAM
from methods.nativetok.adapter import NativeTokenAdapter

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / 'checkpoints' / 'sam_vit_b_01ec64.pth')


def _tokenizer() -> NativeResidualTokenizer:
    projection = nn.Conv2d(3, 8, 16, stride=16)
    return NativeResidualTokenizer(
        projection, in_dim=8, scales=(8, 4), hidden=8)


def test_tokenizer_zero_init_and_shape():
    tokenizer = _tokenizer()
    global_tokens = torch.randn(1, 64, 64, 8)
    native = [torch.rand(3, 256, 320) * 255]
    out = tokenizer(global_tokens, native)
    assert out.shape == global_tokens.shape
    assert torch.allclose(out, global_tokens)
    # The frozen patch-embedding copy is a buffer, not a trainable parameter.
    assert not any('patch_weight' in name for name, _ in
                   tokenizer.named_parameters())
    with torch.no_grad():
        tokenizer.alpha.fill_(0.5)
        tokenizer.gate[-1].weight.normal_(0.0, 0.1)
        tokenizer.gate[-1].bias.zero_()
    changed = tokenizer(global_tokens, native)
    assert not torch.allclose(changed, global_tokens)


def test_tokenizer_accepts_varied_native_sizes():
    tokenizer = _tokenizer()
    global_tokens = torch.randn(2, 64, 64, 8)
    native = [
        torch.rand(3, 300, 400) * 255,
        torch.rand(3, 512, 512) * 255,
    ]
    out = tokenizer(global_tokens, native)
    assert out.shape == global_tokens.shape
    assert tokenizer.last_gate is not None
    assert tokenizer.last_gate.shape == (2, 3, 64, 64)


def test_adapter_metadata():
    adapter = NativeTokenAdapter()
    assert adapter.id == 'nativetok'
    assert adapter.default_epochs == 800
    assert 'project extension' in adapter.official_source
    assert 'not an official baseline' in adapter.official_source


def test_config_roundtrip():
    adapter = NativeTokenAdapter()
    args = SimpleNamespace(
        input_size=1024,
        convlora_model_type='vit_b',
        convlora_rank=3,
        convlora_alpha=32,
        convlora_experts=8,
        convlora_moe_weight=1.0,
        convlora_checkpoint=CHECKPOINT,
        nativetok_scales=[8, 4],
        nativetok_hidden=16,
        nativetok_alpha_init=0.0,
        nativetok_new_lr_mult=5.0,
        nativetok_native_max_side=0,
    )
    adapter._apply_nativetok(args)
    config = adapter.save_config(args)
    target = NativeTokenAdapter()
    restored = SimpleNamespace(**vars(args))
    restored.nativetok_scales = [4]
    restored.nativetok_hidden = 32
    restored.nativetok_alpha_init = 1.0
    restored.nativetok_new_lr_mult = 1.0
    restored.nativetok_native_max_side = 2048
    target.restore_config(restored, config)
    assert target._scales == (8, 4)
    assert target._hidden == 16
    assert target._alpha_init == pytest.approx(0.0)
    assert target._new_lr_mult == pytest.approx(5.0)
    assert target._native_max_side == 0


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason='SAM ViT-B checkpoint is not available')
def test_full_model_zero_init_matches_convlora():
    base = build_convlora(CHECKPOINT, model_type='vit_b').eval()
    model = NativeTokenSAM(
        base, scales=(8, 4), hidden=8, alpha_init=0.0).eval()
    images = torch.rand(1, 3, 1024, 1024) * 255
    native = [torch.rand(3, 512, 512) * 255]
    with torch.no_grad():
        logits, moe_loss = model(images, native)
        reference, reference_moe = base(images)
    assert logits.shape == reference.shape
    assert moe_loss.shape == reference_moe.shape
    assert torch.allclose(logits, reference, atol=1e-6)
