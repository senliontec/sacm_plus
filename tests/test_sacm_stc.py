"""Tests for the fixed SACM-STC baseline (adapter, decoder, predictor)."""

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
from baseline_zoo.models.sacm_stc import build_sam_vit_b  # noqa: E402
from baseline_zoo.models.sacm_stc.segment_anything.modeling.mask_decoder import (  # noqa: E402
    MaskDecoder,
)
from baseline_zoo.models.sacm_stc.segment_anything.modeling.transformer import (  # noqa: E402
    TwoWayTransformer,
)
from baseline_zoo.models.sacm_stc.stc import (  # noqa: E402
    STCHead,
    structure_targets,
)
from baseline_zoo.models.sacm_stc.topology_losses import (  # noqa: E402
    SkeletonRecallLoss,
    SoftclDiceLoss,
    TverskyLoss,
)
from baseline_zoo.models.sacm_stc.typed import (  # noqa: E402
    WIDTH_MAX_PX,
    WIDTH_MIN_PX,
    TypedComposer,
    typed_targets,
    width_to_logits,
    width_to_pixels,
)
from baseline_zoo.models.sacm_stc.tpd import (  # noqa: E402
    HeadFusion,
    ImageDensePrompt,
    MorphPrompt,
    TopologyQualityHead,
)
from baseline_zoo.models.sacm_stc.topoqa import (  # noqa: E402
    RoleQualityHead,
    TopologyStability,
    pareto_arbitrate,
    quality_metric_targets,
)
from baseline_zoo.registry import load_models  # noqa: E402


def _tiny_decoder(num_layers=2, adapter_dim=64, stable_stage2=False):
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
        stable_stage2=stable_stage2,
    )


def test_sacm_stc_registered():
    assert 'sacm_stc' in available_adapters()
    adapter = load_adapter('sacm_stc')
    assert adapter.id == 'sacm_stc'
    model = load_models()['sacm_stc']
    assert model.available
    assert model.defaults['epochs'] == 800


def test_sacm_stc_all_heads_receive_gradient():
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


def test_sacm_stc_stable_stage2_forward_and_prior_gradients():
    decoder = _tiny_decoder(stable_stage2=True)
    image = torch.randn(1, 256, 8, 8, requires_grad=True)
    image_pe = torch.randn(1, 256, 8, 8)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 8, 8)
    adapters = [torch.randn(1, 8, 8, 64, requires_grad=True)
                for _ in range(2)]

    stage2, iou_pred, stage1 = decoder.predict_masks(
        image, image_pe, sparse, dense, adapter_features=adapters,
        return_stage1=True)
    assert stage2.shape == (1, 4, 32, 32)
    assert stage1.shape == (1, 4, 32, 32)
    loss = stage2.mean() + stage1.mean() + iou_pred.mean()
    loss.backward()
    assert decoder.stage1_prior[-1].weight.grad is not None
    assert decoder.stage1_prior[-1].weight.grad.abs().sum() > 0
    assert decoder.stage2_type_emb.grad is not None
    assert decoder.stage2_type_emb.grad.abs().sum() > 0


def test_sacm_stc_forward_selects_highest_iou(monkeypatch):
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


def test_sacm_stc_preprocess_scales_01_to_sam_range():
    mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    class FakeSam:
        pixel_mean = mean
        pixel_std = std

        @staticmethod
        def preprocess(x):
            return (x - mean) / std

    adapter = load_adapter('sacm_stc')
    images = torch.rand(1, 3, 4, 4)
    out = adapter._sam_preprocess(FakeSam, images)
    assert torch.allclose(out, ((images * 255.0) - mean) / std)
    # Already-normalized inputs (negative values) must not be scaled again.
    normalized = torch.randn(1, 3, 4, 4)
    out2 = adapter._sam_preprocess(FakeSam, normalized)
    assert torch.allclose(out2, (normalized - mean) / std)


def test_sacm_stc_build_wiring_and_predictor(monkeypatch):
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

    from baseline_zoo.models.sacm_stc.segment_anything.predictor import (
        SamPredictor,
    )
    predictor = SamPredictor(model)
    predictor.set_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
    predictor.predict(point_coords=np.array([[512, 512]]),
                      point_labels=np.array([1]))
    assert seen.get('adapter_features') is not None
    assert len(seen['adapter_features']) == 12


def test_sacm_stc_prediction_skips_missing_base_checkpoint(monkeypatch):
    captured = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        'baseline_zoo.adapters.sacm_stc.build_sam_vit_l', Dummy)
    adapter = load_adapter('sacm_stc')
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


def test_sacm_stc_config_roundtrip():
    adapter = load_adapter('sacm_stc')
    args = argparse.Namespace(
        adapter_dim_ratio=0.25, bce_weight=0.6, dice_weight=0.4,
        deep_sup_weight=0.1, iou_loss_weight=0.5, head_select='fixed',
        input_size=1024, deployment_head=3, stage2_loss_scope='selected',
        stage2_loss_aggregate='mean', stage2_aux_weight=0.4,
        cl_dice_weight=0.2,
        tversky_weight=0.3, skeleton_recall_weight=0.1,
        topology_resolution=512, topology_source='training',
        topology_head='deployment',
        topology_aggregate='mean', topology_aux_weight=0.35,
        topology_warmup_epochs=4, tversky_alpha=0.4, tversky_beta=0.6,
        use_stc=1, stc_resolution=256, stc_width=16,
        stc_dense_weight=1.0, stc_centerline_weight=0.2,
        stc_cl_dice_weight=0.2, stc_boundary_weight=0.1,
        stc_width_weight=0.05, stc_compose_weight=0.5,
        stc_gate_weight=0.05, use_typed_stage2=1, typed_resolution=256,
        typed_centerline_weight=0.3, typed_cl_dice_weight=0.1,
        typed_boundary_weight=0.2, typed_width_weight=0.06,
        typed_compose_weight=0.4, typed_gate_weight=0.07,
        typed_structure_warmup_epochs=3, typed_composer_lr=5e-3,
        typed_gate_temperature=0.03, typed_gate_distill_weight=0.3,
        typed_output='compose', typed_dense_skeleton_weight=0.1,
        typed_dense_cldice_weight=0.2, typed_dense_boundary_weight=0.05)
    config = adapter.save_config(args)
    restored = argparse.Namespace(head_select=None)
    adapter.restore_config(restored, config)
    assert restored.adapter_dim_ratio == 0.25
    assert restored.bce_weight == 0.6
    assert restored.dice_weight == 0.4
    assert restored.deep_sup_weight == 0.1
    assert restored.iou_loss_weight == 0.5
    assert restored.head_select == 'fixed'
    assert restored.deployment_head == 3
    assert restored.stage2_loss_scope == 'selected'
    assert restored.stage2_loss_aggregate == 'mean'
    assert restored.stage2_aux_weight == 0.4
    assert restored.cl_dice_weight == 0.2
    assert restored.tversky_weight == 0.3
    assert restored.skeleton_recall_weight == 0.1
    assert restored.topology_resolution == 512
    assert restored.topology_source == 'training'
    assert restored.topology_head == 'deployment'
    assert restored.topology_aggregate == 'mean'
    assert restored.topology_aux_weight == 0.35
    assert restored.topology_warmup_epochs == 4
    assert restored.tversky_alpha == 0.4
    assert restored.tversky_beta == 0.6
    assert restored.use_stc == 1
    assert restored.stc_resolution == 256
    assert restored.stc_width == 16
    assert restored.stc_dense_weight == 1.0
    assert restored.stc_centerline_weight == 0.2
    assert restored.stc_cl_dice_weight == 0.2
    assert restored.stc_boundary_weight == 0.1
    assert restored.stc_width_weight == 0.05
    assert restored.stc_compose_weight == 0.5
    assert restored.stc_gate_weight == 0.05
    assert restored.use_typed_stage2 == 1
    assert restored.typed_resolution == 256
    assert restored.typed_centerline_weight == 0.3
    assert restored.typed_cl_dice_weight == 0.1
    assert restored.typed_boundary_weight == 0.2
    assert restored.typed_width_weight == 0.06
    assert restored.typed_compose_weight == 0.4
    assert restored.typed_gate_weight == 0.07
    assert restored.typed_structure_warmup_epochs == 3
    assert restored.typed_composer_lr == 5e-3
    assert restored.typed_gate_temperature == 0.03
    assert restored.typed_gate_distill_weight == 0.3
    assert restored.typed_output == 'compose'
    assert restored.typed_dense_skeleton_weight == 0.1
    assert restored.typed_dense_cldice_weight == 0.2
    assert restored.typed_dense_boundary_weight == 0.05
    assert restored.input_size == 1024


def test_sacm_stc_topology_losses_are_finite_and_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 15:17] = 1.0
    losses = (
        SoftclDiceLoss(num_iter=3),
        TverskyLoss(alpha=0.3, beta=0.7),
        SkeletonRecallLoss(tube_radius=1),
    )
    for loss_fn in losses:
        value = loss_fn(logits, target)
        assert torch.isfinite(value), type(loss_fn).__name__
        value.backward(retain_graph=True)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sacm_stc_fixed_head_selection_resolution():
    from baseline_zoo.adapters.sacm_stc import SACMSTCAdapter
    args = argparse.Namespace(head_select='fixed', iou_loss_weight=1.0)
    assert SACMSTCAdapter._resolved_head_select(args) == 'fixed'
    args = argparse.Namespace(head_select='auto', iou_loss_weight=0.0)
    assert SACMSTCAdapter._resolved_head_select(args) == 'gating'


def test_sacm_stc_candidate_scope_indices():
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.deployment_head = 2
    adapter.args = argparse.Namespace(
        stage2_loss_scope='all', topology_head='all')
    assert adapter._stage2_loss_indices(4) == [0, 1, 2, 3]
    assert adapter._topology_indices(4) == [0, 1, 2, 3]

    adapter.args.stage2_loss_scope = 'selected'
    adapter.args.topology_head = 'selected'
    assert adapter._stage2_loss_indices(4) == [0]
    assert adapter._topology_indices(4) == [0]
    assert adapter.selected_candidate_index() == 0

    adapter.head_select = 'fixed'
    adapter.deployment_head = 2
    assert adapter._stage2_loss_indices(4) == [2]
    assert adapter._topology_indices(4) == [2]
    assert adapter.selected_candidate_index() == 2

    adapter.args.topology_head = 'deployment'
    assert adapter._topology_indices(4) == [2]

    adapter.head_select = 'mean'
    with pytest.raises(ValueError):
        adapter.selected_candidate_index()
    adapter.head_select = 'gating_soft'
    with pytest.raises(ValueError):
        adapter.selected_candidate_index()


def test_sacm_stc_selected_loss_scope_uses_only_deployed_slot(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.deployment_head = 2
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.0,
        iou_loss_weight=0.0, use_typed_stage2=0, use_stc=0,
        stage2_loss_scope='selected',
        stage2_loss_aggregate='mean', stage2_aux_weight=0.25,
        cl_dice_weight=0.0,
        tversky_weight=0.0, skeleton_recall_weight=0.0,
        topology_head='all', topology_aggregate='mean',
        topology_aux_weight=0.25, topology_warmup_epochs=0,
        topology_resolution=32)

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        stage2 = stage2.expand(1, 4, 256, 256).clone()
        stage1 = torch.zeros(1, 4, 256, 256)
        return stage2, torch.zeros(1, 4), stage1

    calls = []

    def fake_criterion(logits, target, bce_weight, dice_weight):
        del target, bce_weight, dice_weight
        calls.append(float(logits.mean().item()))
        return logits.mean()

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(adapter, '_mask_criterion', fake_criterion)
    model = torch.nn.Module()
    images = torch.zeros(1, 3, 8, 8)
    masks = torch.zeros(1, 1, 256, 256)

    loss = adapter.train_step(model, (images, masks))
    assert calls == [0.0]
    assert float(loss) == 0.0

    calls.clear()
    adapter.args.stage2_loss_scope = 'all'
    loss = adapter.train_step(model, (images, masks))
    assert calls == [0.0, 1.0, 2.0, 3.0]
    assert abs(float(loss) - 1.5) < 1e-6


def test_sacm_stc_deployment_aggregate_keeps_deployed_weight(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.deployment_head = 2
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.0,
        iou_loss_weight=0.0, use_typed_stage2=0, use_stc=0,
        stage2_loss_scope='all', stage2_loss_aggregate='deployment',
        stage2_aux_weight=0.25, cl_dice_weight=0.0,
        tversky_weight=0.0, skeleton_recall_weight=0.0,
        topology_head='selected', topology_aggregate='deployment',
        topology_aux_weight=0.25, topology_warmup_epochs=0,
        topology_resolution=32)

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        stage2 = stage2.expand(1, 4, 256, 256).clone()
        stage1 = torch.zeros(1, 4, 256, 256)
        return stage2, torch.zeros(1, 4), stage1

    calls = []

    def fake_criterion(logits, target, bce_weight, dice_weight):
        del target, bce_weight, dice_weight
        calls.append(float(logits.mean().item()))
        return logits.mean()

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(adapter, '_mask_criterion', fake_criterion)
    loss = adapter.train_step(
        torch.nn.Module(),
        (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 256, 256)))
    assert calls == [0.0, 1.0, 2.0, 3.0]
    # selected slot 0 keeps full weight; mean(1,2,3)=2 at 0.25.
    assert abs(float(loss) - 0.5) < 1e-6


def test_sacm_stc_topology_uses_native_stage2(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.deployment_head = 2
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.0,
        iou_loss_weight=0.0, use_typed_stage2=0, use_stc=0,
        stage2_loss_scope='selected', stage2_loss_aggregate='deployment',
        stage2_aux_weight=0.25, cl_dice_weight=0.0,
        tversky_weight=0.0, skeleton_recall_weight=0.0,
        topology_head='selected', topology_aggregate='deployment',
        topology_aux_weight=0.25, topology_warmup_epochs=0,
        topology_resolution=8, topology_source='native')

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.zeros(1, 4, 4, 4)
        stage1 = torch.zeros(1, 4, 4, 4)
        return stage2, torch.zeros(1, 4), stage1

    def fake_criterion(logits, target, bce_weight, dice_weight):
        del target, bce_weight, dice_weight
        return logits.sum() * 0.0

    seen = {}

    def fake_topology(stage2, masks):
        del masks
        seen['shape'] = tuple(stage2.shape)
        return stage2.sum() * 0.0

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(adapter, '_mask_criterion', fake_criterion)
    monkeypatch.setattr(adapter, '_topology_loss', fake_topology)
    adapter.train_step(
        torch.nn.Module(),
        (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8)))
    assert seen['shape'] == (1, 4, 4, 4)


def test_sacm_stc_topology_legacy_uses_stretched_stage2(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.deployment_head = 2
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.0,
        iou_loss_weight=0.0, use_typed_stage2=0, use_stc=0,
        stage2_loss_scope='selected', stage2_loss_aggregate='deployment',
        stage2_aux_weight=0.25, cl_dice_weight=0.0,
        tversky_weight=0.0, skeleton_recall_weight=0.0,
        topology_head='selected', topology_aggregate='deployment',
        topology_aux_weight=0.25, topology_warmup_epochs=0,
        topology_resolution=8, topology_source='training')

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.zeros(1, 4, 4, 4)
        stage1 = torch.zeros(1, 4, 4, 4)
        return stage2, torch.zeros(1, 4), stage1

    def fake_criterion(logits, target, bce_weight, dice_weight):
        del target, bce_weight, dice_weight
        return logits.sum() * 0.0

    seen = {}

    def fake_topology(stage2, masks):
        del masks
        seen['shape'] = tuple(stage2.shape)
        return stage2.sum() * 0.0

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(adapter, '_mask_criterion', fake_criterion)
    monkeypatch.setattr(adapter, '_topology_loss', fake_topology)
    adapter.train_step(
        torch.nn.Module(),
        (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8)))
    # Legacy intentionally receives the 1024-res stretched training tensor.
    assert seen['shape'] == (1, 4, 1024, 1024)


def test_sacm_stc_predict_candidates(tmp_path):
    from PIL import Image

    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(mask_path)
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        deploy_threshold=0.5)

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        stage2 = torch.full((1, 4, 2, 2), -10.0)
        stage2[:, 0] = 10.0
        return stage2, torch.zeros(1, 4)

    adapter._forward = fake_forward
    sample = argparse.Namespace(mask_path=mask_path)
    candidates = adapter.predict_candidates(
        torch.nn.Module(), torch.zeros(1, 3, 1024, 1024), sample)
    assert candidates.shape == (4, 3, 4)
    assert candidates[0].all()
    assert not candidates[1:].any()
    assert adapter.selected_candidate_index() == 0


def test_sacm_stc_stable_stage2_builder():
    model = build_sam_vit_b(checkpoint=None, use_adapter=True,
                            stable_stage2=True)
    assert model.mask_decoder.stable_stage2 is True


def test_sacm_stc_stable_stage2_starts_as_identity():
    model = build_sam_vit_b(checkpoint=None, use_adapter=True,
                            stable_stage2=True)
    decoder = model.mask_decoder
    # Type embeddings start at zero so the stable heads begin as the
    # unmodified SAM heads; the Stage-1 prior branch starts as identity.
    assert decoder.stage2_type_emb.shape == (
        decoder.num_mask_tokens, decoder.transformer_dim)
    assert torch.count_nonzero(decoder.stage2_type_emb) == 0
    assert torch.count_nonzero(decoder.stage1_prior[-1].weight) == 0
    assert torch.count_nonzero(decoder.stage1_prior[-1].bias) == 0
    # Width regression starts from a neutral zero logit, not a saturated
    # random projection.
    width_layer = decoder.output_hypernetworks_mlps_stage2[-1].layers[-1]
    assert torch.count_nonzero(width_layer.weight) == 0
    assert torch.count_nonzero(width_layer.bias) == 0

    official = build_sam_vit_b(checkpoint=None, use_adapter=True)
    assert not hasattr(official.mask_decoder, 'stage1_prior')
    assert not hasattr(official.mask_decoder, 'stage2_type_emb')


def test_sacm_stc_gate_loss_ignores_agreed_background():
    adapter = load_adapter('sacm_stc')
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    dense_prob = torch.zeros(1, 1, 8, 8)
    structure = torch.zeros(1, 1, 8, 8)
    # On the foreground structure is right and dense is wrong, so the gate
    # target is open there.  On the background both agree, so those pixels
    # are excluded rather than teaching the gate to close.
    structure[..., 2:6, 2:6] = 1.0
    logits = torch.full((1, 1, 8, 8), 2.0)
    loss = adapter._gate_loss(logits, dense_prob, structure, target)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[..., 2:6, 2:6], torch.ones(1, 1, 4, 4))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_sacm_stc_gate_loss_soft_target_at_tie():
    adapter = load_adapter('sacm_stc')
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    dense_prob = target.clone()
    structure = target.clone()
    logits = torch.zeros(1, 1, 8, 8)
    # Equal errors -> soft target 0.5; BCE(0, 0.5) = ln 2.
    loss = adapter._gate_loss(logits, dense_prob, structure, target)
    assert abs(float(loss) - float(torch.log(torch.tensor(2.0)))) < 1e-6


def test_sacm_stc_composer_has_its_own_lr_group():
    adapter = load_adapter('sacm_stc')

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(4, 4)
            self.typed_composer = TypedComposer()

    model = TinyModel()
    args = argparse.Namespace(lr=3e-4, weight_decay=0.01,
                              typed_composer_lr=3e-3)
    optimizer = adapter.make_optimizer(model, args)
    composer_ids = {id(p) for p in model.typed_composer.parameters()}
    composer_groups = [
        group for group in optimizer.param_groups
        if any(id(p) in composer_ids for p in group['params'])]
    assert len(composer_groups) == 1
    assert composer_groups[0]['lr'] == 3e-3
    assert composer_groups[0]['weight_decay'] == 0.0
    other_groups = [
        group for group in optimizer.param_groups
        if any(id(p) not in composer_ids for p in group['params'])]
    assert other_groups
    assert all(group['lr'] == 3e-4 for group in other_groups)


def test_sacm_stc_composer_lr_requires_official_scheduler():
    adapter = load_adapter('sacm_stc')

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(4, 4)
            self.typed_composer = TypedComposer()

    model = TinyModel()
    args = argparse.Namespace(
        lr=3e-4, weight_decay=0.01, typed_composer_lr=3e-3,
        scheduler='constant')
    # Generic scheduler overrides reset every group to args.lr; refuse to
    # silently drop the composer LR.
    with pytest.raises(SystemExit):
        adapter.make_optimizer(model, args)
    # Explicitly opting into the base rate is allowed.
    args.typed_composer_lr = 3e-4
    optimizer = adapter.make_optimizer(model, args)
    assert any(group['lr'] == 3e-4 for group in optimizer.param_groups)


def test_sacm_stc_typed_composer_initial_blend():
    torch.manual_seed(0)
    composer = TypedComposer()
    dense = torch.randn(1, 1, 32, 32)
    center = torch.randn(1, 1, 32, 32)
    boundary = torch.randn(1, 1, 32, 32)
    width = torch.randn(1, 1, 32, 32)
    out = composer(dense, center, boundary, width, 32)
    assert out['final'].shape == dense.shape
    # The global gate starts at 0.4; the zero spatial residual leaves it
    # unchanged.  This is close to the validation oracle mean and avoids
    # both the closed-gate and the arbitrary 0.5 blend.
    assert torch.allclose(
        out['global_gate'], torch.full_like(out['global_gate'], 0.4))
    assert torch.allclose(
        out['spatial_gate'], torch.full_like(out['spatial_gate'], 0.5))
    assert torch.allclose(out['gate'], torch.full_like(out['gate'], 0.4))
    width_px = width_to_pixels(width)
    assert float(width_px.min()) >= WIDTH_MIN_PX
    assert float(width_px.max()) <= WIDTH_MAX_PX


def test_sacm_stc_oracle_gate_target():
    adapter = load_adapter('sacm_stc')
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0
    empty_logits = torch.full_like(target, -10.0)
    perfect_dense = target * 20.0 - 10.0
    # Structure is perfect and dense is empty -> oracle gate opens fully.
    best_gate, dices = adapter._oracle_gate_target(
        empty_logits, target, target)
    # 0.6 is the first grid point where foreground crosses the threshold;
    # ties are broken toward the smallest (most conservative) blend.
    assert float(best_gate[0]) >= 0.6 - 1e-6
    assert abs(float(dices.max()) - 1.0) < 1e-6
    # Dense is perfect and structure is empty -> the Dice plateau spans the
    # low-gate region, so the mean plateau target stays on the dense side.
    best_gate, _ = adapter._oracle_gate_target(
        perfect_dense, torch.zeros_like(target), target)
    assert float(best_gate[0]) <= 0.5
    assert tuple(dices.shape) == (1, 11)


def test_sacm_stc_composer_blend_override():
    torch.manual_seed(0)
    composer = TypedComposer()
    dense = torch.randn(1, 1, 32, 32)
    center = torch.randn(1, 1, 32, 32)
    boundary = torch.randn(1, 1, 32, 32)
    width = torch.randn(1, 1, 32, 32)
    out_dense = composer(
        dense, center, boundary, width, 32, blend_override=0.0)
    assert torch.allclose(out_dense['final'], dense)
    out_struct = composer(
        dense, center, boundary, width, 32, blend_override=1.0)
    structure_logits = torch.logit(
        out_struct['structure'].clamp(1e-4, 1.0 - 1e-4))
    assert torch.allclose(out_struct['final'], structure_logits, atol=1e-5)


def test_sacm_stc_width_parameterisation_is_bounded():
    logits = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0])
    px = width_to_pixels(logits)
    assert torch.all(px >= WIDTH_MIN_PX)
    assert torch.all(px <= WIDTH_MAX_PX)
    assert px[2] == (WIDTH_MIN_PX + WIDTH_MAX_PX) / 2
    target = torch.tensor([1.0, 3.0, 7.0, 13.0])
    back = width_to_pixels(width_to_logits(target))
    assert torch.allclose(back, target, atol=1e-3)


def test_sacm_stc_typed_targets():
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, 15:17] = 1.0
    skeleton, boundary, width = typed_targets(mask, 32)
    assert skeleton.shape == (1, 1, 32, 32)
    assert boundary.shape == (1, 1, 32, 32)
    assert width.shape == (1, 1, 32, 32)
    assert skeleton.sum() > 0
    assert boundary.sum() > 0
    assert width.max() > 0


def test_sacm_stc_typed_train_step(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.3,
        iou_loss_weight=1.0, deployment_head=2, head_select='iou',
        cl_dice_weight=0.0, tversky_weight=0.0,
        skeleton_recall_weight=0.0, topology_resolution=32,
        topology_head='deployment', topology_warmup_epochs=0,
        tversky_alpha=0.3, tversky_beta=0.7,
        use_stc=0, use_typed_stage2=1, typed_resolution=32,
        typed_centerline_weight=0.2, typed_cl_dice_weight=0.2,
        typed_boundary_weight=0.1, typed_width_weight=0.05,
        typed_compose_weight=0.5, typed_gate_weight=0.05,
        typed_structure_warmup_epochs=0,
        typed_dense_skeleton_weight=0.1, typed_dense_cldice_weight=0.1,
        typed_dense_boundary_weight=0.1)
    adapter.head_select = 'iou'
    adapter.deployment_head = 2
    adapter._epoch = 10

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.typed_composer = TypedComposer()

    model = FakeModel()

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.randn(1, 4, 256, 256, requires_grad=True)
        stage1 = torch.randn(1, 4, 256, 256, requires_grad=True)
        return stage2, torch.rand(1, 4), stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.zeros(1, 3, 8, 8)
    masks = torch.zeros(1, 1, 256, 256)
    masks[..., 10:40, 10:40] = 1.0
    loss = adapter.train_step(model, (images, masks))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None
               for p in model.typed_composer.parameters())


def test_sacm_stc_typed_train_step_keeps_native_stage2(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(use_typed_stage2=1)
    captured = {}

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.randn(1, 4, 256, 256, requires_grad=True)
        stage1 = torch.randn(1, 4, 256, 256, requires_grad=True)
        return stage2, torch.rand(1, 4), stage1

    def fake_typed_loss(model, masks, stage2, stage1):
        del model, masks
        captured['stage2'] = tuple(stage2.shape)
        captured['stage1'] = tuple(stage1.shape)
        return stage2.sum() * 0.0

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(adapter, '_typed_loss', fake_typed_loss)
    loss = adapter.train_step(
        None, (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 256, 256)))
    # Stage-2 stays at 256 and is resampled once inside _typed_loss; only the
    # Stage-1 deep-supervision stream goes to 1024.
    assert captured['stage2'] == (1, 4, 256, 256)
    assert captured['stage1'] == (1, 4, 1024, 1024)
    assert torch.isfinite(loss)


def test_sacm_stc_typed_structure_warmup(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.3,
        iou_loss_weight=1.0, deployment_head=2, head_select='iou',
        cl_dice_weight=0.0, tversky_weight=0.0,
        skeleton_recall_weight=0.0, topology_resolution=32,
        topology_head='deployment', topology_warmup_epochs=0,
        tversky_alpha=0.3, tversky_beta=0.7,
        use_stc=0, use_typed_stage2=1, typed_resolution=32,
        typed_centerline_weight=0.2, typed_cl_dice_weight=0.2,
        typed_boundary_weight=0.1, typed_width_weight=0.05,
        typed_compose_weight=0.5, typed_gate_weight=0.05,
        typed_structure_warmup_epochs=10)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.typed_composer = TypedComposer()

    model = FakeModel()

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.randn(1, 4, 256, 256, requires_grad=True)
        stage1 = torch.randn(1, 4, 256, 256, requires_grad=True)
        return stage2, torch.rand(1, 4), stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.zeros(1, 3, 8, 8)
    masks = torch.zeros(1, 1, 256, 256)
    masks[..., 10:40, 10:40] = 1.0

    # Before the warmup ends only the dense Stage-2 head and the Stage-1
    # deep supervision are trained; structural losses are still off.
    adapter._epoch = 5
    loss = adapter.train_step(model, (images, masks))
    loss.backward()
    assert all(p.grad is None
               for p in model.typed_composer.parameters())

    # Halfway through the ramp the structural losses are active.
    model.zero_grad(set_to_none=True)
    adapter._epoch = 15
    loss = adapter.train_step(model, (images, masks))
    loss.backward()
    assert any(p.grad is not None
               for p in model.typed_composer.parameters())


def test_sacm_stc_stc_identity_initialisation():
    torch.manual_seed(0)
    head = STCHead(feature_ch=16, width=8)
    anchor = torch.randn(1, 4, 32, 32)
    expected = anchor.mean(dim=1, keepdim=True)
    image = torch.rand(1, 3, 32, 32)
    sam_features = torch.randn(1, 16, 8, 8)
    out = head(anchor, image, sam_features)
    assert out['final'].shape == expected.shape
    assert out['centerline'].shape == expected.shape
    assert out['boundary'].shape == expected.shape
    assert out['width_bins'].shape == (1, 3, 32, 32)
    assert torch.allclose(out['refined'], expected)
    assert torch.all(out['gate'] < 0.01)
    # The gate is closed but not exactly zero; the untrained output is the
    # four-head mean up to this small residual.
    assert torch.allclose(out['final'], expected, atol=1e-2, rtol=0.0)


def test_sacm_stc_structure_targets():
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, 15:17] = 1.0
    skeleton, boundary, width_bins = structure_targets(mask, 32)
    assert skeleton.shape == (1, 1, 32, 32)
    assert boundary.shape == (1, 1, 32, 32)
    assert width_bins.shape == (1, 32, 32)
    assert skeleton.sum() > 0
    assert boundary.sum() > 0
    assert width_bins.min() >= 0
    assert width_bins.max() <= 2


def test_sacm_stc_stc_train_step(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.3,
        iou_loss_weight=1.0, deployment_head=2, head_select='fixed',
        cl_dice_weight=0.0, tversky_weight=0.0,
        skeleton_recall_weight=0.0, topology_resolution=32,
        topology_head='deployment', topology_warmup_epochs=0,
        tversky_alpha=0.3, tversky_beta=0.7,
        use_stc=1, stc_resolution=32, stc_width=8,
        stc_dense_weight=1.0, stc_centerline_weight=0.2,
        stc_cl_dice_weight=0.2, stc_boundary_weight=0.1,
        stc_width_weight=0.05, stc_compose_weight=0.5,
        stc_gate_weight=0.05)
    adapter.head_select = 'fixed'
    adapter.deployment_head = 2
    adapter._epoch = 10

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stc_head = STCHead(feature_ch=16, width=8)

    model = FakeModel()

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        adapter._last_sam_features = torch.randn(1, 16, 8, 8)
        stage2 = torch.randn(1, 4, 256, 256, requires_grad=True)
        stage1 = torch.randn(1, 4, 256, 256, requires_grad=True)
        return stage2, torch.rand(1, 4), stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.zeros(1, 3, 8, 8)
    masks = torch.zeros(1, 1, 256, 256)
    masks[..., 10:40, 10:40] = 1.0
    loss = adapter.train_step(model, (images, masks))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.stc_head.parameters())


def test_sacm_stc_optional_topology_train_step(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, deep_sup_weight=0.3,
        iou_loss_weight=1.0, deployment_head=2,
        cl_dice_weight=0.2, tversky_weight=0.3,
        skeleton_recall_weight=0.0, topology_resolution=64,
        topology_head='deployment', topology_warmup_epochs=0,
        tversky_alpha=0.3, tversky_beta=0.7)
    adapter.deployment_head = 2
    adapter._epoch = 10

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.randn(1, 4, 256, 256, requires_grad=True)
        stage1 = torch.randn(1, 4, 256, 256, requires_grad=True)
        iou = torch.rand(1, 4)
        return stage2, iou, stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    images = torch.zeros(1, 3, 8, 8)
    masks = torch.zeros(1, 1, 256, 256)
    masks[..., 10:40, 10:40] = 1.0
    loss = adapter.train_step(None, (images, masks))
    assert torch.isfinite(loss)
    loss.backward()


def test_sacm_stc_bce_pos_weight_changes_mask_criterion():
    adapter = load_adapter('sacm_stc')
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = (torch.rand(1, 1, 8, 8) > 0.6).float()

    adapter.args = argparse.Namespace(bce_pos_weight=1.0)
    unweighted = adapter._mask_criterion(logits, target, 0.7, 0.3)
    adapter.args = argparse.Namespace(bce_pos_weight=2.0)
    weighted = adapter._mask_criterion(logits, target, 0.7, 0.3)

    assert torch.isfinite(unweighted) and torch.isfinite(weighted)
    assert weighted > unweighted
    weighted.backward()
    assert logits.grad is not None


def test_sacm_stc_all_dense_config_roundtrip():
    adapter = load_adapter('sacm_stc')
    args = argparse.Namespace(
        bce_weight=0.7, dice_weight=0.3, bce_pos_weight=2.0,
        binary_targets=0, deploy_threshold=0.4,
        head_select='mean', adapter_dim_ratio=0.1,
        deep_sup_weight=0.3, iou_loss_weight=0.0,
        stage2_loss_aggregate='deployment', stage2_aux_weight=0.25,
        cl_dice_weight=0.1, tversky_weight=0.0,
        skeleton_recall_weight=0.1, topology_resolution=512,
        topology_source='native', topology_head='all',
        topology_aggregate='deployment',
        topology_aux_weight=0.25, topology_warmup_epochs=5,
        tversky_alpha=0.3, tversky_beta=0.7, use_stc=0,
        stc_resolution=512, stc_width=32, stc_dense_weight=1.0,
        stc_centerline_weight=0.2, stc_cl_dice_weight=0.2,
        stc_boundary_weight=0.1, stc_width_weight=0.05,
        stc_compose_weight=0.5, stc_gate_weight=0.05,
        use_typed_stage2=0, typed_resolution=512,
        typed_centerline_weight=0.2, typed_cl_dice_weight=0.2,
        typed_boundary_weight=0.1, typed_width_weight=0.05,
        typed_compose_weight=0.0, typed_gate_weight=0.05,
        typed_structure_warmup_epochs=10, typed_composer_lr=3e-3,
        typed_gate_temperature=0.02, typed_gate_distill_weight=0.0,
        typed_output='dense', typed_dense_skeleton_weight=0.0,
        typed_dense_cldice_weight=0.0, typed_dense_boundary_weight=0.0,
        deployment_head=2)
    adapter.head_select = 'mean'
    config = adapter.save_config(args)

    target = argparse.Namespace(
        **{key: value for key, value in vars(args).items()})
    target.bce_pos_weight = 1.0
    target.binary_targets = 1
    target.deploy_threshold = 0.5
    target.head_select = None
    adapter.restore_config(target, config)

    assert target.bce_pos_weight == 2.0
    assert target.binary_targets == 0
    assert target.deploy_threshold == 0.4
    assert target.head_select == 'mean'
    assert target.stage2_loss_aggregate == 'deployment'
    assert target.stage2_aux_weight == 0.25
    assert target.topology_source == 'native'
    assert target.topology_aggregate == 'deployment'
    assert target.topology_aux_weight == 0.25


def test_sacm_stc_new_cli_choices_parse():
    adapter = load_adapter('sacm_stc')
    parser = argparse.ArgumentParser()
    adapter.add_args(parser)
    for choice in ('gating', 'gating_soft', 'mean', 'iou', 'fixed'):
        args = parser.parse_args(['--head-select', choice])
        assert args.head_select == choice
    args = parser.parse_args(
        ['--binary-targets', '0', '--bce-pos-weight', '2',
         '--deploy-threshold', '0.4'])
    assert args.binary_targets == 0
    assert args.bce_pos_weight == 2.0
    assert args.deploy_threshold == 0.4
    for choice in ('deployment', 'mean'):
        args = parser.parse_args(['--stage2-loss-aggregate', choice])
        assert args.stage2_loss_aggregate == choice
        args = parser.parse_args(['--topology-aggregate', choice])
        assert args.topology_aggregate == choice
    args = parser.parse_args([
        '--stage2-aux-weight', '0.5', '--topology-aux-weight', '0.3'])
    assert args.stage2_aux_weight == 0.5
    assert args.topology_aux_weight == 0.3
    args = parser.parse_args([])
    assert args.topology_source == 'native'
    for choice in ('native', 'training'):
        args = parser.parse_args(['--topology-source', choice])
        assert args.topology_source == choice
    args = parser.parse_args([])
    assert args.image_dense_prompt == 0
    assert args.topology_quality == 0
    args = parser.parse_args(
        ['--image-dense-prompt', '1', '--topology-quality', '1',
         '--quality-loss-weight', '0.2', '--quality-resolution', '32',
         '--image-dense-prompt-mode', 'replace', '--quality-refine', '1'])
    assert args.image_dense_prompt == 1
    assert args.topology_quality == 1
    assert args.quality_loss_weight == 0.2
    assert args.quality_resolution == 32
    assert args.image_dense_prompt_mode == 'replace'
    assert args.quality_refine == 1
    args = parser.parse_args(
        ['--quality-refine', '1', '--quality-refine-stage1', '1',
         '--quality-select-margin', '0.02'])
    assert args.quality_refine_stage1 == 1
    assert args.quality_select_margin == 0.02


def test_sacm_stc_tpd_modules():
    prompt = ImageDensePrompt(embed_dim=8)
    image = torch.randn(2, 8, 4, 4)
    residual = prompt(image)
    assert residual.shape == image.shape
    # The gate starts at 0.1 (LayerScale style) so the enabled branch is
    # active from the first step instead of learning an ~1e-4 no-op.
    assert abs(float(prompt.scale.detach()) - 0.1) < 1e-6
    assert torch.count_nonzero(residual) > 0
    with pytest.raises(ValueError):
        prompt(torch.randn(2, 7, 4, 4))

    projected = ImageDensePrompt(embed_dim=16, prompt_dim=4)
    assert projected(torch.randn(2, 16, 4, 4)).shape == (2, 4, 4, 4)

    quality = TopologyQualityHead(hidden=8)
    candidates = torch.randn(2, 4, 16, 16)
    scores = quality(candidates)
    assert scores.shape == (2, 4)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)

    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    values = TopologyQualityHead.quality_target(
        candidates, target, SoftclDiceLoss(), 0.33, 0.33, 0.34)
    assert values.shape == (2, 4)
    assert torch.all(values >= 0) and torch.all(values <= 1)

    adapter = load_adapter('sacm_stc')
    adapter.quality_refine_topk = 2
    adapter.quality_refine_threshold = 0.7
    adapter.quality_refine_weight = 1.0
    logits = torch.full((1, 4, 4, 4), -10.0)
    logits[:, 1] = 10.0
    scores = torch.tensor([[0.90, 0.80, 0.0, 0.0]])
    refined = adapter._quality_refined_logits(logits, scores)
    assert refined.shape == (1, 1, 4, 4)
    # Candidate 1 supports all pixels and its quality is close to the best,
    # so full-weight refinement fills the missing region.
    assert torch.all(torch.sigmoid(refined) > 0.5)


def test_sacm_stc_tpd_train_step(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.image_dense_prompt = False
    adapter.topology_quality = True
    adapter.quality_resolution = 16
    adapter.quality_warmup_epochs = 0
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        deploy_threshold=0.5, bce_weight=0.7, dice_weight=0.3,
        bce_pos_weight=2.0, stage2_loss_scope='selected',
        stage2_loss_aggregate='deployment', stage2_aux_weight=0.25,
        deep_sup_weight=0.0, iou_loss_weight=0.0,
        cl_dice_weight=0.1, skeleton_recall_weight=0.1,
        tversky_weight=0.0, topology_head='selected',
        topology_aggregate='deployment', topology_aux_weight=0.25,
        topology_warmup_epochs=0, topology_resolution=16,
        topology_source='native', quality_loss_weight=0.1,
        quality_dice_weight=0.5, quality_cldice_weight=0.5)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.topology_quality_head = TopologyQualityHead(hidden=8)

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.zeros(1, 4, 8, 8, requires_grad=True)
        stage1 = torch.zeros(1, 4, 8, 8)
        return stage2, torch.zeros(1, 4), stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(
        adapter, '_mask_criterion',
        lambda logits, target, bce_weight, dice_weight:
        logits.sum() * 0.0)
    monkeypatch.setattr(
        adapter, '_topology_loss',
        lambda stage2, masks: stage2.sum() * 0.0)
    model = FakeModel()
    loss = adapter.train_step(
        model, (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8)))
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in model.topology_quality_head.parameters()
             if p.grad is not None]
    assert grads
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_sacm_stc_tpd_predict_selects_quality(tmp_path, monkeypatch):
    from PIL import Image

    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(mask_path)
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.topology_quality = True
    adapter.quality_warmup_epochs = 0
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        deploy_threshold=0.5, topology_quality=1)

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        stage2 = torch.full((1, 4, 2, 2), -10.0)
        stage2[:, 2] = 10.0
        return stage2, torch.zeros(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(
        adapter, '_quality_scores',
        lambda model, stage2: torch.tensor([[0.1, 0.2, 0.9, 0.3]]))
    sample = argparse.Namespace(mask_path=mask_path)
    mask = adapter.predict_sample(
        torch.nn.Module(), torch.zeros(1, 3, 1024, 1024), sample)
    assert mask.shape == (3, 4)
    assert mask.all()


def test_sacm_stc_tpd_refine_runs_when_quality_keeps_gating(tmp_path,
                                                            monkeypatch):
    """Refinement must not depend on the quality head overriding gating."""
    from PIL import Image

    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(mask_path)
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.topology_quality = True
    adapter.quality_refine = True
    adapter.quality_refine_topk = 2
    adapter.quality_refine_threshold = 0.7
    adapter.quality_refine_weight = 1.0
    adapter.quality_refine_stage1 = False
    adapter.quality_warmup_epochs = 0
    adapter.quality_select_margin = 0.01
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        stage1_fusion_mode='gated', deploy_threshold=0.5,
        topology_quality=1)

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        stage2 = torch.full((1, 4, 2, 2), -10.0)
        stage2[:, 1] = 10.0
        return stage2, torch.zeros(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    # Slot 0 (the gating anchor) is also the quality argmax, so selection
    # keeps gating and reports no override index.  Candidate 1 must still be
    # quality-vetted before it can contribute refinement support.
    monkeypatch.setattr(
        adapter, '_quality_scores',
        lambda model, stage2: torch.tensor([[0.9, 0.8, 0.1, 0.0]]))
    sample = argparse.Namespace(mask_path=mask_path)
    mask = adapter.predict_sample(
        torch.nn.Module(), torch.zeros(1, 3, 1024, 1024), sample)
    # The gating base is empty, but candidate 1 is high-confidence support,
    # so refinement must fill it even without a quality override.
    assert mask.all()


def test_sacm_stc_tpd_refine_uses_deploy_threshold():
    adapter = load_adapter('sacm_stc')
    adapter.quality_refine_topk = 2
    adapter.quality_refine_threshold = 0.7
    adapter.quality_refine_weight = 1.0
    adapter.quality_refine_stage1 = False
    low = float(torch.logit(torch.tensor(0.1)))
    base = float(torch.logit(torch.tensor(0.4)))
    high = float(torch.logit(torch.tensor(0.95)))
    stage2 = torch.full((1, 4, 2, 2), low)
    stage2[:, 0] = base
    stage2[:, 1] = high
    scores = torch.tensor([[0.9, 0.8, 0.0, 0.0]])

    adapter.args = argparse.Namespace(deploy_threshold=0.5)
    filled = torch.sigmoid(adapter._quality_refined_logits(
        stage2, scores, base_index=0))
    assert float(filled.min()) > 0.9

    adapter.args = argparse.Namespace(deploy_threshold=0.3)
    kept = torch.sigmoid(adapter._quality_refined_logits(
        stage2, scores, base_index=0))
    assert torch.allclose(kept, torch.full_like(kept, 0.4), atol=1e-5)


def test_sacm_stc_tpd_refine_stage1_support():
    adapter = load_adapter('sacm_stc')
    adapter.quality_refine_topk = 2
    adapter.quality_refine_threshold = 0.7
    adapter.quality_refine_weight = 1.0
    adapter.quality_refine_stage1 = True
    adapter.args = argparse.Namespace(deploy_threshold=0.5)
    low = float(torch.logit(torch.tensor(0.1)))
    base = float(torch.logit(torch.tensor(0.2)))
    stage2 = torch.full((1, 4, 2, 2), low)
    stage2[:, 0] = base
    scores = torch.tensor([[0.9, 0.8, 0.0, 0.0]])
    stage1 = torch.full((1, 4, 2, 2), 10.0)
    refined = torch.sigmoid(adapter._quality_refined_logits(
        stage2, scores, stage1, base_index=0))
    # Top-k proposals are below threshold; only the Stage-1 support fills.
    assert torch.all(refined > 0.9)


def test_sacm_stc_tpd_quality_margin_keeps_anchor(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.head_select = 'gating'
    adapter.topology_quality = True
    adapter.quality_warmup_epochs = 0
    adapter.quality_select_margin = 0.01
    stage2 = torch.zeros(1, 4, 2, 2)

    def make_scores(anchor, best):
        return torch.tensor([[anchor, best, 0.0, 0.0]])

    monkeypatch.setattr(
        adapter, '_quality_scores',
        lambda model, stage2: make_scores(0.50, 0.505))
    assert adapter._quality_candidate_index(None, stage2) is None

    monkeypatch.setattr(
        adapter, '_quality_scores',
        lambda model, stage2: make_scores(0.50, 0.60))
    assert adapter._quality_candidate_index(None, stage2) == 1


def test_sacm_stc_tpd_refine_ignores_degenerate_low_score_candidate():
    """A low-quality all-foreground proposal must not flood the mask."""
    adapter = load_adapter('sacm_stc')
    adapter.quality_refine_topk = 2
    adapter.quality_refine_threshold = 0.7
    adapter.quality_refine_weight = 1.0
    adapter.quality_refine_stage1 = False
    adapter.quality_refine_min_score_ratio = 0.5
    adapter.args = argparse.Namespace(deploy_threshold=0.5)
    low = float(torch.logit(torch.tensor(0.1)))
    base = float(torch.logit(torch.tensor(0.2)))
    stage2 = torch.full((1, 4, 2, 2), low)
    stage2[:, 0] = base
    stage2[:, 1] = 10.0  # all-foreground degenerate proposal
    # The quality head correctly ranks candidate 1 far below candidate 0.
    scores = torch.tensor([[0.50, 0.02, 0.01, 0.01]])
    refined = torch.sigmoid(adapter._quality_refined_logits(
        stage2, scores, base_index=0))
    # Candidate 1 is below 0.5 * best, so it contributes no support and the
    # deployed mask is unchanged.
    assert torch.allclose(refined, torch.full_like(refined, 0.2), atol=1e-5)


def test_sacm_stc_morph_prompt():
    prompt = MorphPrompt(embed_dim=16, scales=(3, 7), work_size=32,
                         gate_threshold=0.0)
    x = torch.rand(2, 3, 32, 32)
    feats = prompt.morph_features(x)
    feats_inv = prompt.morph_features(1.0 - x)
    assert feats.shape == (2, 16, 32, 32)
    # White/black top-hat swap under inversion; the features are invariant.
    assert torch.allclose(feats, feats_inv, atol=1e-5)
    out = prompt(x)
    assert out.shape == (2, 16, 32, 32)
    assert torch.count_nonzero(out) > 0
    # The initial gate (~0.018) is below the default 0.1 threshold.
    closed = MorphPrompt(embed_dim=16, scales=(3,), work_size=32)
    closed.eval()
    assert torch.count_nonzero(closed(x)) == 0

    adapter = load_adapter('sacm_stc')
    adapter.input_size = 1024
    adapter.morph_prompt = True
    cfg = adapter.save_config(argparse.Namespace())
    assert cfg['morph_prompt'] == 1
    assert cfg['morph_prompt_gate_threshold'] == 0.1
    parser = argparse.ArgumentParser()
    adapter.add_args(parser)
    ns = parser.parse_args(
        ['--morph-prompt', '1', '--polarity-augment', '1'])
    assert ns.morph_prompt == 1 and ns.polarity_augment == 1


def test_sacm_stc_morph_prompt_forward():
    adapter = load_adapter('sacm_stc')
    adapter.morph_prompt = True
    adapter.image_dense_prompt = False

    class FakePE:
        def __call__(self, points=None, boxes=None, masks=None):
            return torch.zeros(1, 4, 256), torch.zeros(1, 256, 8, 8)

        def get_dense_pe(self):
            return torch.zeros(1, 256, 8, 8)

    class FakeIE:
        embed_dim = 256

        def __call__(self, images, return_features=True):
            return torch.zeros(images.shape[0], 256, 8, 8), []

    class FakeMD:
        def predict_masks(self, **kwargs):
            return kwargs['dense_prompt_embeddings']

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_encoder = FakePE()
            self.image_encoder = FakeIE()
            self.mask_decoder = FakeMD()
            self.morph_prompt = MorphPrompt(
                embed_dim=256, scales=(3,), work_size=16,
                gate_threshold=0.0)

        def preprocess(self, images):
            return images

    model = FakeModel()
    with_morph = adapter._forward(model, torch.rand(1, 3, 16, 16))
    assert with_morph.shape == (1, 256, 8, 8)
    assert torch.count_nonzero(with_morph) > 0
    adapter.morph_prompt = False
    without_morph = adapter._forward(model, torch.rand(1, 3, 16, 16))
    assert torch.count_nonzero(without_morph) == 0
    # With the safety gate closed at inference, the no-POMP path is
    # reproduced exactly.
    adapter.morph_prompt = True
    model.morph_prompt.gate_threshold = 0.1
    model.eval()
    closed = adapter._forward(model, torch.rand(1, 3, 16, 16))
    assert torch.count_nonzero(closed) == 0


def test_sacm_stc_head_fusion():
    fusion = HeadFusion()
    assert fusion.mode == 'linear'
    heads = torch.randn(2, 4, 8, 8)
    out = fusion(heads)
    assert out.shape == (2, 1, 8, 8)
    # Zero-initialised fusion reproduces the deployed head exactly.
    assert torch.allclose(out, heads[:, 0:1])
    out.sum().backward()
    assert fusion.fuse.weight.grad is not None
    assert float(fusion.fuse.weight.grad.abs().sum()) > 0

    adapter = load_adapter('sacm_stc')
    adapter.input_size = 1024
    adapter.head_fusion = True
    adapter.specialized_heads = True
    adapter.head_fusion_mode = 'image_gated'
    adapter.head_fusion_warmup_epochs = 10
    adapter.specialized_topology_mode = 'both'
    adapter.specialized_topology_weight = 0.2
    cfg = adapter.save_config(argparse.Namespace())
    assert cfg['head_fusion'] == 1 and cfg['specialized_heads'] == 1
    assert cfg['head_fusion_mode'] == 'image_gated'
    assert cfg['head_fusion_warmup_epochs'] == 10
    assert cfg['specialized_topology_mode'] == 'both'
    assert cfg['specialized_topology_weight'] == 0.2
    parser = argparse.ArgumentParser()
    adapter.add_args(parser)
    ns = parser.parse_args(
        ['--head-fusion', '1', '--specialized-heads', '1',
         '--head-fusion-mode', 'image_gated',
         '--head-fusion-warmup-epochs', '10',
         '--specialized-topology-mode', 'both',
         '--specialized-topology-weight', '0.2'])
    assert ns.head_fusion == 1 and ns.specialized_heads == 1
    assert ns.head_fusion_mode == 'image_gated'
    assert ns.head_fusion_warmup_epochs == 10
    assert ns.specialized_topology_mode == 'both'
    assert ns.specialized_topology_weight == 0.2
    restored = argparse.Namespace()
    adapter.restore_config(restored, cfg)
    assert restored.head_fusion_mode == 'image_gated'
    assert restored.head_fusion_warmup_epochs == 10
    assert restored.specialized_topology_mode == 'both'
    assert restored.specialized_topology_weight == 0.2


def test_sacm_stc_head_fusion_image_gated():
    fusion = HeadFusion(mode='image_gated')
    heads = torch.randn(2, 4, 8, 8)
    out = fusion(heads)
    assert out.shape == (2, 1, 8, 8)
    # Zero-initialised gate also reproduces the deployed head exactly.
    assert torch.allclose(out, heads[:, 0:1])
    out.sum().backward()
    assert fusion.gate[-1].weight.grad is not None
    assert float(fusion.gate[-1].weight.grad.abs().sum()) > 0
    # With a non-zero gate the output departs from the base head and the
    # per-image weights depend on the input statistics.
    with torch.no_grad():
        fusion.gate[-1].weight.fill_(0.1)
        fusion.gate[-1].bias.fill_(0.1)
    fused = fusion(heads)
    assert not torch.allclose(fused, heads[:, 0:1])

    def _gate_weights(value):
        stats = torch.cat(
            [value.mean(dim=(2, 3)), value.std(dim=(2, 3))], dim=1)
        return torch.tanh(fusion.gate(stats))

    swapped = heads.clone()
    swapped[:, 1:] = swapped[:, 1:].flip(1)
    assert not torch.allclose(_gate_weights(heads), _gate_weights(swapped))
    assert not torch.allclose(fused, fusion(swapped))
    # The warmup scale linearly interpolates between the deployed head and
    # the fully gated fusion.
    base = heads[:, 0:1]
    assert torch.allclose(fusion(heads, scale=0.0), base)
    half = fusion(heads, scale=0.5)
    assert torch.allclose(
        half, base + 0.5 * (fusion(heads, scale=1.0) - base), atol=1e-6)
    with pytest.raises(ValueError):
        HeadFusion(mode='unknown')


def test_sacm_stc_head_fusion_warmup_scale():
    adapter = load_adapter('sacm_stc')
    fusion = HeadFusion(mode='image_gated')
    with torch.no_grad():
        fusion.gate[-1].weight.fill_(0.1)
        fusion.gate[-1].bias.fill_(0.1)
    heads = torch.randn(2, 4, 8, 8)
    model = argparse.Namespace(head_fusion=fusion)
    adapter.head_fusion_warmup_epochs = 10
    adapter._epoch = 5
    adapter.args = argparse.Namespace(stage='train')
    warm = adapter._fused_heads(model, heads)
    assert torch.allclose(warm, fusion(heads, scale=0.5))
    # Final prediction uses the fully trained gate, not the warmup scale.
    adapter.args.stage = 'predict'
    assert torch.allclose(adapter._fused_heads(model, heads), fusion(heads))
    # warmup=0 preserves the pre-warmup behaviour.
    adapter.head_fusion_warmup_epochs = 0
    adapter.args.stage = 'train'
    assert torch.allclose(adapter._fused_heads(model, heads), fusion(heads))


def test_sacm_stc_specialized_topology_modes():
    adapter = load_adapter('sacm_stc')
    adapter.args = argparse.Namespace(topology_resolution=32)
    adapter.specialized_head_weight = 0.1
    stage2 = torch.randn(2, 4, 16, 16)
    masks = torch.zeros(2, 1, 16, 16)
    masks[:, :, 8, :] = 1.0
    values = {}
    for mode in ('cldice', 'skeleton', 'both'):
        adapter.specialized_topology_mode = mode
        loss = adapter._specialized_head_loss(stage2, masks)
        assert torch.isfinite(loss)
        assert float(loss) > 0
        values[mode] = float(loss)
    # Both adds the skeleton term on top of soft-clDice; each mode must be
    # at least as large as the single-objective variants.
    assert values['both'] >= values['cldice'] - 1e-6
    assert values['both'] >= values['skeleton'] - 1e-6
    # The dedicated topology weight amplifies only head 1's topology term.
    adapter.specialized_topology_weight = 0.2
    stronger = adapter._specialized_head_loss(stage2, masks)
    assert float(stronger) > values['both']


def test_sacm_stc_boundary_weighted_loss():
    adapter = load_adapter('sacm_stc')
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = adapter._boundary_weighted_loss(logits, target)
    assert torch.isfinite(loss)


def test_sacm_stc_decoder_exposes_topoqa_tokens():
    decoder = _tiny_decoder()
    image = torch.randn(1, 256, 8, 8)
    image_pe = torch.randn(1, 256, 8, 8)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 8, 8)
    adapters = [torch.randn(1, 8, 8, 64) for _ in range(2)]
    decoder.predict_masks(
        image, image_pe, sparse, dense, adapter_features=adapters)
    assert decoder.last_image_embedding.shape == (1, 256, 8, 8)
    assert decoder.last_sorted_tokens.shape == (1, 4, 256)


def test_sacm_stc_topoqa_modules():
    head = RoleQualityHead(embed_dim=8, hidden=16, role_dim=4)
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    global_feature = torch.randn(2, 8, requires_grad=True)
    quality = head(tokens, global_feature)
    assert quality.shape == (2, 4, 2)
    assert torch.all(quality >= 0) and torch.all(quality <= 1)
    quality.mean().backward()
    assert tokens.grad is not None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in head.parameters())

    logits = torch.randn(2, 4, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    iou, cldice = quality_metric_targets(logits, target)
    assert iou.shape == (2, 4) and cldice.shape == (2, 4)
    assert torch.all(iou >= 0) and torch.all(iou <= 1)
    assert torch.all(cldice >= 0) and torch.all(cldice <= 1)

    perfect = torch.full((1, 4, 8, 8), 10.0)
    full = torch.ones(1, 1, 8, 8)
    iou, cldice = quality_metric_targets(perfect, full)
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)
    assert torch.all(cldice > 0.95)

    empty = torch.full((1, 4, 8, 8), -10.0)
    iou, cldice = quality_metric_targets(
        empty, torch.zeros(1, 1, 8, 8))
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)
    assert torch.allclose(cldice, torch.ones_like(cldice), atol=1e-5)

    stability = TopologyStability(num_iter=2, delta=0.1)
    scores = stability(logits)
    assert scores.shape == (2, 4)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)
    confident = stability(torch.full((1, 4, 8, 8), 10.0))
    assert torch.all(confident > 0.95)


def test_sacm_stc_topoqa_arbitration():
    champion = torch.tensor([[[0.8, 0.6], [0.797, 0.7],
                              [0.7, 0.2], [0.6, 0.1]]])
    stable = torch.ones(1, 4)
    assert pareto_arbitrate(champion, stable).tolist() == [1]
    # IoU drop beyond the tolerance is rejected even when clDice rises.
    guarded = torch.tensor([[[0.8, 0.6], [0.7, 0.7],
                             [0.6, 0.2], [0.5, 0.1]]])
    assert pareto_arbitrate(guarded, stable).tolist() == [0]
    # No eligible candidate + unstable champion -> TSS rescue.
    rescue = torch.tensor([[[0.8, 0.6], [0.797, 0.605],
                            [0.7, 0.2], [0.6, 0.1]]])
    unstable = torch.tensor([[0.5, 0.95, 0.5, 0.5]])
    assert pareto_arbitrate(rescue, unstable).tolist() == [1]
    # Highest clDice among eligible alternatives wins.
    multiple = torch.tensor([[[0.8, 0.6], [0.7, 0.7],
                              [0.797, 0.8], [0.6, 0.1]]])
    assert pareto_arbitrate(multiple, stable).tolist() == [2]
    # Champion is kept when no alternative passes the clDice margin.
    quiet = torch.tensor([[[0.8, 0.6], [0.79, 0.605],
                           [0.78, 0.5], [0.77, 0.4]]])
    assert pareto_arbitrate(quiet, stable).tolist() == [0]


def test_sacm_stc_topoqa_train_step(monkeypatch):
    adapter = load_adapter('sacm_stc')
    adapter.topoqa = True
    adapter.topoqa_quality_weight = 0.1
    adapter.topoqa_rank_weight = 0.1
    adapter.topoqa_temperature = 0.1
    adapter.topoqa_resolution = 8
    adapter.topoqa_warmup_epochs = 0
    adapter.topoqa_detach = True
    adapter.topoqa_alpha = 0.5
    adapter.topoqa_skeleton_iters = 2
    adapter.topoqa_arbitration = 'none'
    adapter.head_select = 'gating'
    adapter.topology_quality = False
    adapter.specialized_heads = False
    adapter.head_fusion = False
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        deploy_threshold=0.5, bce_weight=0.7, dice_weight=0.3,
        bce_pos_weight=2.0, stage2_loss_scope='all',
        stage2_loss_aggregate='deployment', stage2_aux_weight=0.25,
        deep_sup_weight=0.0, iou_loss_weight=0.0,
        cl_dice_weight=0.0, skeleton_recall_weight=0.0,
        tversky_weight=0.0, topology_head='all',
        topology_aggregate='deployment', topology_aux_weight=0.25,
        topology_warmup_epochs=0, topology_resolution=8,
        topology_source='native', topoqa=1)

    class FakeDecoder:
        last_sorted_tokens = torch.randn(1, 4, 256)
        last_image_embedding = torch.randn(1, 256, 8, 8)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.topoqa_quality_head = RoleQualityHead(
                embed_dim=256, hidden=16, role_dim=4)
            self.mask_decoder = FakeDecoder()

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.zeros(1, 4, 8, 8, requires_grad=True)
        stage1 = torch.zeros(1, 4, 8, 8)
        return stage2, torch.zeros(1, 4), stage1

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(
        adapter, '_mask_criterion',
        lambda logits, target, bce_weight, dice_weight:
        logits.sum() * 0.0)
    monkeypatch.setattr(
        adapter, '_topology_loss',
        lambda stage2, masks: stage2.sum() * 0.0)
    model = FakeModel()
    loss = adapter.train_step(
        model, (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8)))
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in model.topoqa_quality_head.parameters()
             if p.grad is not None]
    assert grads
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_sacm_stc_topoqa_predict_switches_expert(tmp_path, monkeypatch):
    from PIL import Image

    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(mask_path)
    adapter = load_adapter('sacm_stc')
    adapter.topoqa = True
    adapter.topoqa_arbitration = 'pareto'
    adapter.topoqa_warmup_epochs = 0
    adapter.topoqa_margin_cldice = 0.01
    adapter.topoqa_margin_iou = 0.005
    adapter.topoqa_stability_threshold = 0.9
    adapter.topoqa_skeleton_iters = 2
    adapter.topoqa_stability_delta = 0.1
    adapter._topoqa_stability = TopologyStability(
        num_iter=2, delta=0.1)
    adapter.head_select = 'gating'
    adapter.head_fusion = False
    adapter.topology_quality = False
    adapter.args = argparse.Namespace(
        use_typed_stage2=0, use_stc=0, stage1_fusion_weight=0.0,
        deploy_threshold=0.5, topoqa=1, head_select='gating')

    def fake_forward(model, prepared, return_stage1=False):
        del model, prepared, return_stage1
        stage2 = torch.full((1, 4, 2, 2), -10.0)
        stage2[:, 2] = 10.0
        return stage2, torch.zeros(1, 4)

    monkeypatch.setattr(adapter, '_forward', fake_forward)
    monkeypatch.setattr(
        adapter, '_topoqa_quality',
        lambda model: torch.tensor([[[0.8, 0.6], [0.7, 0.61],
                                     [0.797, 0.8], [0.6, 0.1]]]))
    sample = argparse.Namespace(mask_path=mask_path)
    mask = adapter.predict_sample(
        torch.nn.Module(), torch.zeros(1, 3, 1024, 1024), sample)
    assert mask.shape == (3, 4)
    assert mask.all()


def test_sacm_stc_topoqa_args_and_config():
    adapter = load_adapter('sacm_stc')
    parser = argparse.ArgumentParser()
    adapter.add_args(parser)
    args = parser.parse_args([
        '--topoqa', '1', '--topoqa-quality-weight', '0.2',
        '--topoqa-rank-weight', '0.3', '--topoqa-temperature', '0.2',
        '--topoqa-resolution', '128', '--topoqa-warmup-epochs', '5',
        '--topoqa-detach', '0', '--topoqa-arbitration', 'none',
        '--topoqa-margin-cldice', '0.02', '--topoqa-margin-iou', '0.01',
        '--topoqa-stability-delta', '0.2',
        '--topoqa-stability-threshold', '0.8',
        '--topoqa-alpha', '0.6', '--topoqa-hidden', '32',
        '--topoqa-role-dim', '8', '--topoqa-skeleton-iters', '3'])
    assert args.topoqa == 1
    assert args.topoqa_quality_weight == 0.2
    assert args.topoqa_rank_weight == 0.3
    assert args.topoqa_temperature == 0.2
    assert args.topoqa_resolution == 128
    assert args.topoqa_warmup_epochs == 5
    assert args.topoqa_detach == 0
    assert args.topoqa_arbitration == 'none'
    assert args.topoqa_margin_cldice == 0.02
    assert args.topoqa_margin_iou == 0.01
    assert args.topoqa_stability_delta == 0.2
    assert args.topoqa_stability_threshold == 0.8
    assert args.topoqa_alpha == 0.6
    assert args.topoqa_hidden == 32
    assert args.topoqa_role_dim == 8
    assert args.topoqa_skeleton_iters == 3

    adapter.topoqa = True
    for key in ('topoqa_quality_weight', 'topoqa_rank_weight',
                'topoqa_temperature', 'topoqa_resolution',
                'topoqa_warmup_epochs', 'topoqa_detach',
                'topoqa_arbitration', 'topoqa_margin_cldice',
                'topoqa_margin_iou', 'topoqa_stability_delta',
                'topoqa_stability_threshold', 'topoqa_alpha',
                'topoqa_hidden', 'topoqa_role_dim',
                'topoqa_skeleton_iters'):
        setattr(adapter, key, getattr(args, key))
    config = adapter.save_config(args)
    restored = argparse.Namespace(head_select=None)
    adapter.restore_config(restored, config)
    assert restored.topoqa == 1
    assert restored.topoqa_quality_weight == 0.2
    assert restored.topoqa_rank_weight == 0.3
    assert restored.topoqa_temperature == 0.2
    assert restored.topoqa_resolution == 128
    assert restored.topoqa_warmup_epochs == 5
    assert restored.topoqa_detach == 0
    assert restored.topoqa_arbitration == 'none'
    assert restored.topoqa_margin_cldice == 0.02
    assert restored.topoqa_margin_iou == 0.01
    assert restored.topoqa_stability_delta == 0.2
    assert restored.topoqa_stability_threshold == 0.8
    assert restored.topoqa_alpha == 0.6
    assert restored.topoqa_hidden == 32
    assert restored.topoqa_role_dim == 8
    assert restored.topoqa_skeleton_iters == 3


def test_sacm_stc_topoqa_build_model(monkeypatch):
    class DummyDecoder:
        num_mask_tokens = 4
        transformer_dim = 256

    class DummyModel:
        def __init__(self, **kwargs):
            del kwargs
            self.mask_decoder = DummyDecoder()

    monkeypatch.setattr(
        'baseline_zoo.adapters.sacm_stc.build_sam_vit_l',
        lambda **kwargs: DummyModel(**kwargs))
    adapter = load_adapter('sacm_stc')
    adapter.prediction_mode = True
    args = argparse.Namespace(
        sam_checkpoint=Path('/missing/vit_l.pth'),
        adapter_dim_ratio=0.2, head_select='gating',
        iou_loss_weight=0.0, deep_sup_weight=0.3,
        topoqa=1, topoqa_hidden=32, topoqa_role_dim=8,
        topoqa_skeleton_iters=3, topoqa_stability_delta=0.1,
        stage2_loss_scope='all', use_stc=0, use_typed_stage2=0,
        head_fusion=0)
    model = adapter.build_model(args)
    assert hasattr(model, 'topoqa_quality_head')
    assert adapter.topoqa is True
    assert adapter._topoqa_stability is not None

    bad = argparse.Namespace(**vars(args))
    bad.topology_quality = 1
    with pytest.raises(SystemExit):
        adapter.build_model(bad)


def test_sacm_stc_topoqa_load_state_dict_allows_fresh_head():
    adapter = load_adapter('sacm_stc')
    adapter.topoqa = True

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.topoqa_quality_head = RoleQualityHead(
                embed_dim=8, hidden=8, role_dim=4)
            self.other = torch.nn.Linear(2, 2)

    model = Tiny()
    state = {key: value.clone()
             for key, value in model.state_dict().items()
             if not key.startswith('topoqa_quality_head.')}
    adapter.load_state_dict(model, state)
    state.pop('other.weight')
    with pytest.raises(RuntimeError):
        adapter.load_state_dict(model, state)
