"""SegDINO freeze-policy/config regression tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline_zoo.adapters.segdino import SegDINOAdapter
from baseline_zoo.models.segdino import (
    _convert_dino_state,
    build_segdino,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_VIT_B = (REPO_ROOT / 'checkpoints' /
             'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')


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


def test_hub_keys_convert_to_timm_names():
    target_keys = {
        'reg_token', 'blocks.0.gamma_1', 'blocks.0.gamma_2',
        'blocks.0.attn.qkv.weight',
    }
    state = {
        'storage_tokens': torch.zeros(1, 4, 8),
        'mask_token': torch.zeros(1, 8),
        'blocks.0.ls1.gamma': torch.zeros(8),
        'blocks.0.ls2.gamma': torch.zeros(8),
        'blocks.0.attn.qkv.weight': torch.zeros(3, 8),
        'blocks.0.attn.qkv.bias': torch.zeros(3),
        'blocks.0.attn.qkv.bias_mask': torch.zeros(3),
        'rope_embed.periods': torch.ones(16),
    }
    converted, periods = _convert_dino_state(state, target_keys)
    assert set(converted) == target_keys
    assert torch.equal(converted['reg_token'], state['storage_tokens'])
    assert torch.equal(converted['blocks.0.gamma_1'],
                       state['blocks.0.ls1.gamma'])
    assert torch.equal(converted['blocks.0.gamma_2'],
                       state['blocks.0.ls2.gamma'])
    assert torch.equal(periods, state['rope_embed.periods'])


@pytest.mark.skipif(not HUB_VIT_B.is_file(),
                    reason='official DINOv3 ViT-B checkpoint is not present')
def test_local_official_hub_vit_b_loads_and_forwards():
    model = build_segdino('base', str(HUB_VIT_B), False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
