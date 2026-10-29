"""SegDINO freeze-policy/config regression tests."""

from types import SimpleNamespace

import torch

from baseline_zoo.adapters.segdino import SegDINOAdapter
from baseline_zoo.models.segdino import build_segdino


def _model():
    return build_segdino('small', None, True).eval()


def test_default_freezes_backbone_and_trains_decoder():
    model = _model()
    adapter = SegDINOAdapter()
    adapter.args = SimpleNamespace(segdino_train_backbone=False)
    adapter.configure_trainable(model)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_release_profile_trains_backbone():
    model = _model()
    adapter = SegDINOAdapter()
    adapter.args = SimpleNamespace(segdino_train_backbone=True)
    adapter.configure_trainable(model)
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_config_roundtrip_keeps_profile(tmp_path):
    adapter = SegDINOAdapter()
    args = SimpleNamespace(
        dino_size='small', dino_ckpt=None, random_init=False,
        segdino_train_backbone=True)
    config = adapter.save_config(args)
    assert config['train_backbone'] is True
    restored = SimpleNamespace()
    adapter.restore_config(restored, config)
    assert restored.segdino_train_backbone is True
    assert restored.random_init is True
