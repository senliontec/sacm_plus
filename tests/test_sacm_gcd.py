"""Core tests for the cleaned SACM-GCD adapter."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import (  # noqa: E402
    available_adapters,
    load_adapter,
)
from baseline_zoo.adapters.sacm_gcd import SACMGCDAdapter  # noqa: E402
from baseline_zoo.models.sacm_gcd import build_sam_vit_b  # noqa: E402
from baseline_zoo.models.sacm_gcd.fusion import HeadFusion  # noqa: E402
from baseline_zoo.models.sacm_gcd.segment_anything.modeling.mask_decoder import (  # noqa: E402
    MaskDecoder,
)
from baseline_zoo.models.sacm_gcd.segment_anything.modeling.transformer import (  # noqa: E402
    TwoWayTransformer,
)
from baseline_zoo.models.sacm_gcd.topology_losses import (  # noqa: E402
    SkeletonRecallLoss,
    SoftclDiceLoss,
    TverskyLoss,
)
from baseline_zoo.registry import load_models  # noqa: E402
from baseline_zoo.runtime import build_parser  # noqa: E402


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


def _core_args(**overrides):
    values = dict(
        adapter_dim_ratio=0.1,
        bce_weight=0.7,
        dice_weight=0.3,
        bce_pos_weight=2.0,
        binary_targets=0,
        deploy_threshold=0.5,
        deep_sup_weight=0.3,
        specialized_heads=1,
        specialized_head_weight=0.1,
        specialized_topology_weight=0.1,
        head_fusion=0,
        fusion_loss_weight=1.0,
        head_fusion_mode='linear',
        head_fusion_warmup_epochs=0,
        head_select='gating',
        deployment_head=3,
        stage2_loss_scope='all',
        topology_head='all',
        topology_source='native',
        cl_dice_weight=0.1,
        skeleton_recall_weight=0.1,
        topology_resolution=32,
        topology_warmup_epochs=5,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _configured_adapter(**overrides):
    adapter = load_adapter('sacm_gcd')
    args = _core_args(**overrides)
    adapter.args = args
    adapter.head_select = args.head_select
    adapter.deployment_head = args.deployment_head
    adapter.specialized_heads = bool(args.specialized_heads)
    adapter.specialized_head_weight = args.specialized_head_weight
    adapter.specialized_topology_weight = args.specialized_topology_weight
    adapter.head_fusion = bool(args.head_fusion)
    adapter.fusion_loss_weight = args.fusion_loss_weight
    adapter.stage2_loss_scope = args.stage2_loss_scope
    adapter.topology_head = args.topology_head
    adapter.topology_source = args.topology_source
    adapter.cl_dice_weight = args.cl_dice_weight
    adapter.skeleton_recall_weight = args.skeleton_recall_weight
    adapter.topology_resolution = args.topology_resolution
    adapter.topology_warmup_epochs = args.topology_warmup_epochs
    adapter.deep_sup_weight = args.deep_sup_weight
    adapter._epoch = 10
    return adapter


def test_sacm_gcd_registered():
    assert 'sacm_gcd' in available_adapters()
    adapter = load_adapter('sacm_gcd')
    assert adapter.id == 'sacm_gcd'
    model = load_models()['sacm_gcd']
    assert model.available
    assert model.defaults['epochs'] == 200


def test_sacm_gcd_all_heads_receive_gradient():
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


def test_sacm_gcd_forward_selects_highest_iou(monkeypatch):
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


def test_sacm_gcd_preprocess_scales_01_to_sam_range():
    mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    class FakeSam:
        pixel_mean = mean
        pixel_std = std

        @staticmethod
        def preprocess(x):
            return (x - mean) / std

    adapter = load_adapter('sacm_gcd')
    images = torch.rand(1, 3, 4, 4)
    out = adapter._sam_preprocess(FakeSam, images)
    assert torch.allclose(out, ((images * 255.0) - mean) / std)
    normalized = torch.randn(1, 3, 4, 4)
    out2 = adapter._sam_preprocess(FakeSam, normalized)
    assert torch.allclose(out2, (normalized - mean) / std)


def test_sacm_gcd_build_wiring_and_predictor(monkeypatch):
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

    from baseline_zoo.models.sacm_gcd.segment_anything.predictor import (
        SamPredictor,
    )
    predictor = SamPredictor(model)
    predictor.set_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
    predictor.predict(point_coords=np.array([[512, 512]]),
                      point_labels=np.array([1]))
    assert seen.get('adapter_features') is not None
    assert len(seen['adapter_features']) == 12


def test_sacm_gcd_prediction_skips_missing_base_checkpoint(monkeypatch):
    captured = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        'baseline_zoo.adapters.sacm_gcd.build_sam_vit_l', Dummy)
    adapter = load_adapter('sacm_gcd')
    adapter.prediction_mode = True
    args = _core_args(sam_checkpoint=Path('/does/not/exist.pth'))
    adapter.build_model(args)
    assert captured['checkpoint'] is None


def test_sacm_gcd_head_select_defaults_to_checkpoint():
    adapter = load_adapter('sacm_gcd')
    args = build_parser(adapter).parse_args([])
    assert args.head_select is None


def test_sacm_gcd_config_roundtrip():
    adapter = _configured_adapter()
    cfg = adapter.save_config(adapter.args)
    assert cfg['specialized_heads'] == 1
    assert cfg['head_fusion'] == 0
    assert cfg['head_select'] == 'gating'
    assert cfg['deployment_head'] == 3

    restored = argparse.Namespace(
        head_select=None, deployment_head=0, input_size=0)
    adapter.restore_config(restored, cfg)
    assert restored.specialized_heads == 1
    assert restored.deployment_head == 3
    assert restored.head_select == 'gating'
    assert restored.topology_resolution == 32


def test_sacm_gcd_topology_losses_are_finite_and_differentiable():
    target = (torch.rand(1, 1, 32, 32) > 0.5).float()
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    for loss in (SoftclDiceLoss(), SkeletonRecallLoss(),
                 TverskyLoss(alpha=0.3, beta=0.7)):
        value = loss(logits, target)
        assert torch.isfinite(value)
        value.backward(retain_graph=True)


def test_sacm_gcd_fixed_head_selection_resolution():
    adapter = _configured_adapter(head_select='gating')
    assert adapter.selected_candidate_index() == 0
    adapter.head_select = 'fixed'
    adapter.deployment_head = 3
    assert adapter.selected_candidate_index() == 3
    adapter.head_select = 'iou'
    with pytest.raises(ValueError):
        adapter.selected_candidate_index()


def test_sacm_gcd_candidate_scope_indices():
    adapter = _configured_adapter(
        stage2_loss_scope='all', topology_head='all')
    assert adapter._stage2_loss_indices(4) == [0, 1, 2, 3]
    assert adapter._topology_indices(4) == [0, 1, 2, 3]

    adapter.args.stage2_loss_scope = 'selected'
    adapter.args.topology_head = 'selected'
    assert adapter._stage2_loss_indices(4) == [0]
    assert adapter._topology_indices(4) == [0]

    adapter.head_select = 'fixed'
    adapter.deployment_head = 3
    assert adapter._stage2_loss_indices(4) == [3]
    assert adapter._topology_indices(4) == [3]

    adapter.args.topology_head = 'deployment'
    adapter.head_select = 'gating'
    assert adapter._topology_indices(4) == [3]


def test_sacm_gcd_deployment_aggregate_keeps_deployed_weight():
    losses = [torch.tensor(float(i)) for i in range(4)]
    total = SACMGCDAdapter._aggregate_head_losses(
        losses, [0, 1, 2, 3], anchor=0)
    # deployed head 0 at full weight + 0.25 * mean(1, 2, 3)
    assert torch.allclose(total, torch.tensor(0.0 + 0.25 * 2.0))
    selected = SACMGCDAdapter._aggregate_head_losses(
        losses, [2], anchor=0)
    assert torch.allclose(selected, torch.tensor(2.0))


def test_sacm_gcd_topology_loss_uses_native_stage2():
    adapter = _configured_adapter()
    stage2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    masks = (torch.rand(1, 1, 64, 64) > 0.5).float()
    loss = adapter._topology_loss(stage2, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert stage2.grad is not None


def test_sacm_gcd_head_fusion_starts_as_identity():
    heads = torch.randn(1, 4, 8, 8)
    fusion = HeadFusion()
    assert torch.allclose(fusion(heads), heads[:, 0:1])
    with torch.no_grad():
        fusion.fuse.weight.fill_(1.0)
        fusion.fuse.bias.zero_()
    assert not torch.allclose(fusion(heads), heads[:, 0:1])


def test_sacm_gcd_specialized_head_loss_is_finite():
    adapter = _configured_adapter()
    stage2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    masks = (torch.rand(1, 1, 64, 64) > 0.5).float()
    loss = adapter._specialized_head_loss(stage2, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert stage2.grad is not None


def test_sacm_gcd_train_step_loss_is_finite(monkeypatch):
    adapter = _configured_adapter(head_fusion=0)
    adapter.args.bce_weight = 0.7
    adapter.args.dice_weight = 0.3

    def fake_forward(model, images, return_stage1=False):
        del model, images
        stage2 = torch.randn(1, 4, 8, 8, requires_grad=True)
        stage1 = torch.randn(1, 4, 8, 8, requires_grad=True)
        if return_stage1:
            return stage2, torch.rand(1, 4), stage1
        return stage2, torch.rand(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.rand(1, 3, 64, 64)
    masks = (torch.rand(1, 1, 64, 64) > 0.5).float()
    loss = adapter.train_step(model=SimpleNamespace(), batch=(images, masks))
    assert torch.isfinite(loss)
    loss.backward()


def test_sacm_gcd_predict_candidates(tmp_path, monkeypatch):
    adapter = _configured_adapter()
    stage2 = torch.randn(1, 4, 8, 8)

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        return stage2, torch.rand(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    mask_path = tmp_path / 'mask.png'
    Image.fromarray(mask).save(mask_path)

    out = adapter.predict_candidates(
        model=None, prepared=None,
        sample=SimpleNamespace(mask_path=str(mask_path)))
    assert out.shape == (4, 6, 7)


def test_sacm_gcd_predict_selects_gating_and_fixed(tmp_path, monkeypatch):
    adapter = _configured_adapter(head_select='gating')
    adapter.args.deploy_threshold = 0.5
    logits = torch.full((1, 4, 2, 2), -10.0)
    logits[:, 0] = 10.0
    logits[:, 3] = -10.0

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        return logits, torch.zeros(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask_path = tmp_path / 'mask.png'
    Image.fromarray(mask).save(mask_path)
    sample = SimpleNamespace(mask_path=str(mask_path))

    gating = adapter.predict_sample(None, None, sample)
    assert gating.all()

    adapter.head_select = 'fixed'
    adapter.deployment_head = 3
    fixed = adapter.predict_sample(None, None, sample)
    assert not fixed.any()
