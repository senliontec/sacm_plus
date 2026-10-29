"""CWSAM backbone/config regression tests."""

from types import SimpleNamespace

import pytest
import torch

from baseline_zoo.adapters.cwsam import CWSAMAdapter
from baseline_zoo.models.cwsam import build_cwsam


def test_official_vit_b_default_and_forward():
    model = build_cwsam(num_classes=2, inp_size=256, encoder='vit_b').eval()
    assert sum(p.numel() for p in model.parameters()) < 110_000_000
    with torch.no_grad():
        logits = model.infer(torch.zeros(1, 3, 256, 256))
    assert logits.shape == (1, 2, 256, 256)


def test_encoder_config_roundtrip(tmp_path):
    adapter = CWSAMAdapter()
    adapter.input_size = 1024
    args = SimpleNamespace(
        cwsam_encoder='vit_l', sam_checkpoint=tmp_path / 'missing.pth')
    model = adapter.build_model(args)
    assert adapter.encoder == 'vit_l'
    assert model.image_encoder.embed_dim == 1024
    config = adapter.save_config(args)
    assert config['encoder'] == 'vit_l'
    assert config['sam_checkpoint_sha256'] is None

    restored = SimpleNamespace()
    adapter.restore_config(restored, config)
    assert restored.cwsam_encoder == 'vit_l'
    assert restored.input_size == 1024


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError, match='unknown CWSAM encoder'):
        build_cwsam(encoder='vit_h')
