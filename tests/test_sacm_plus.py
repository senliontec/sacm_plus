"""Tests for the fixed SACM+ baseline (adapter, decoder, predictor)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import (  # noqa: E402
    available_adapters,
    load_adapter,
)
from baseline_zoo.models.sacm_plus import build_sam_vit_b  # noqa: E402
from baseline_zoo.models.sacm_plus.segment_anything.modeling.mask_decoder import (  # noqa: E402
    MaskDecoder,
)
from baseline_zoo.models.sacm_plus.segment_anything.modeling.transformer import (  # noqa: E402
    TwoWayTransformer,
)
from baseline_zoo.registry import load_models  # noqa: E402


def _tiny_decoder(num_layers=2, adapter_dim=64):
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=1, embedding_dim=256, mlp_dim=512, num_heads=8),
        transformer_dim=256,
        iou_head_depth=2,
        iou_head_hidden_dim=64,
        use_adapter_skip=True,
        adapter_embed_dim=adapter_dim,
        adapter_num_layers=num_layers,
        adapter_proj_dim=8,
        adapter_hidden=16,
    )


def test_sacm_plus_registered():
    assert 'sacm_plus' in available_adapters()
    adapter = load_adapter('sacm_plus')
    assert adapter.id == 'sacm_plus'
    model = load_models()['sacm_plus']
    assert model.available
    assert model.defaults['epochs'] == 800


def test_sacm_plus_all_heads_receive_gradient():
    decoder = _tiny_decoder()
    image = torch.randn(1, 256, 8, 8, requires_grad=True)
    image_pe = torch.randn(1, 256, 8, 8)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 8, 8)
    adapters = [torch.randn(1, 8, 8, 64, requires_grad=True)
                for _ in range(2)]

    stage2, iou_pred, stage1 = decoder.predict_masks(
        image, image_pe, sparse, dense, adapter_features=adapters,
        return_stage1=True)
    loss = (stage2.mean() + stage1.mean()
            + torch.nn.functional.mse_loss(iou_pred, torch.rand(1, 4)))
    loss.backward()

    groups = {
        'stage1': list(decoder.output_hypernetworks_mlps),
        'stage2': list(decoder.output_hypernetworks_mlps_stage2),
        'iou': [decoder.iou_prediction_head],
        'fusion_proj': list(decoder.adapter_weight_proj),
        'fusion_ffn': [decoder.adapter_weight_ffn],
        'aggregator': [decoder.adapter_aggregator],
    }
    for name, modules in groups.items():
        grads = [p.grad for module in modules
                 for p in module.parameters()]
        assert grads, name
        assert all(g is not None for g in grads), name
        assert any(float(g.abs().sum()) > 0 for g in grads), name


def test_sacm_plus_forward_selects_highest_iou(monkeypatch):
    decoder = _tiny_decoder()
    masks = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    masks = masks.expand(1, 4, 4, 4).clone()
    iou = torch.tensor([[0.1, 0.2, 0.9, 0.3]])

    def fake_predict_masks(**kwargs):
        del kwargs
        return masks, iou

    monkeypatch.setattr(decoder, 'predict_masks', fake_predict_masks)
    out_masks, out_iou = decoder.forward(
        image_embeddings=torch.zeros(1, 256, 8, 8),
        image_pe=torch.zeros(1, 256, 8, 8),
        sparse_prompt_embeddings=torch.zeros(1, 0, 256),
        dense_prompt_embeddings=torch.zeros(1, 256, 8, 8),
        multimask_output=False,
        adapter_features=None)
    assert torch.allclose(out_masks[0, 0], masks[0, 2])
    assert torch.allclose(out_iou[0, 0], iou[0, 2])


def test_sacm_plus_preprocess_scales_01_to_sam_range():
    mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    class FakeSam:
        pixel_mean = mean
        pixel_std = std

        @staticmethod
        def preprocess(x):
            return (x - mean) / std

    adapter = load_adapter('sacm_plus')
    images = torch.rand(1, 3, 4, 4)
    out = adapter._sam_preprocess(FakeSam, images)
    assert torch.allclose(out, ((images * 255.0) - mean) / std)
    # Already-normalized inputs (negative values) must not be scaled again.
    normalized = torch.randn(1, 3, 4, 4)
    out2 = adapter._sam_preprocess(FakeSam, normalized)
    assert torch.allclose(out2, (normalized - mean) / std)


def test_sacm_plus_build_wiring_and_predictor(monkeypatch):
    model = build_sam_vit_b(checkpoint=None, use_adapter=True)
    assert model.mask_decoder.adapter_num_layers == 12
    assert model.image_encoder.depth == 12
    clone = build_sam_vit_b(checkpoint=None, use_adapter=True)
    clone.load_state_dict(model.state_dict())

    seen = {}

    def fake_encoder(x, return_features=False):
        del x, return_features
        embeddings = torch.zeros(1, 256, 64, 64)
        adapter_features = [torch.zeros(1, 64, 64, 768)
                            for _ in range(12)]
        return embeddings, None, adapter_features

    def fake_predict_masks(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 4, 256, 256), torch.zeros(1, 4)

    monkeypatch.setattr(model.image_encoder, 'forward', fake_encoder)
    monkeypatch.setattr(model.mask_decoder, 'predict_masks',
                        fake_predict_masks)

    from baseline_zoo.models.sacm_plus.segment_anything.predictor import (
        SamPredictor,
    )
    predictor = SamPredictor(model)
    predictor.set_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
    predictor.predict(point_coords=np.array([[512, 512]]),
                      point_labels=np.array([1]))
    assert seen.get('adapter_features') is not None
    assert len(seen['adapter_features']) == 12


def test_sacm_plus_prediction_skips_missing_base_checkpoint(monkeypatch):
    captured = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        'baseline_zoo.adapters.sacm_plus.build_sam_vit_l', Dummy)
    adapter = load_adapter('sacm_plus')
    adapter.prediction_mode = True
    adapter.build_model(argparse.Namespace(
        sam_checkpoint=Path('/missing/vit_l.pth'),
        adapter_dim_ratio=0.2,
        head_select='auto',
        iou_loss_weight=1.0,
        deep_sup_weight=0.3))
    assert captured['checkpoint'] is None
    assert captured['adapter_dim_ratio'] == 0.2

    adapter.prediction_mode = False
    with pytest.raises(SystemExit):
        adapter.build_model(argparse.Namespace(
            sam_checkpoint=Path('/missing/vit_l.pth'),
            adapter_dim_ratio=0.2,
            head_select='auto',
            iou_loss_weight=1.0,
            deep_sup_weight=0.3))


def test_sacm_plus_config_roundtrip():
    adapter = load_adapter('sacm_plus')
    args = argparse.Namespace(
        adapter_dim_ratio=0.25, bce_weight=0.6, dice_weight=0.4,
        deep_sup_weight=0.1, iou_loss_weight=0.5, head_select='gating',
        input_size=1024)
    config = adapter.save_config(args)
    restored = argparse.Namespace(head_select=None)
    adapter.restore_config(restored, config)
    assert restored.adapter_dim_ratio == 0.25
    assert restored.bce_weight == 0.6
    assert restored.dice_weight == 0.4
    assert restored.deep_sup_weight == 0.1
    assert restored.iou_loss_weight == 0.5
    assert restored.head_select == 'gating'
    assert restored.input_size == 1024
