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
        topology_resolution=512, topology_head='deployment',
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
        topology_resolution=8)

    def fake_forward(model, images, return_stage1=True):
        del model, images, return_stage1
        stage2 = torch.zeros(1, 4, 8, 8)
        stage1 = torch.zeros(1, 4, 8, 8)
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
    assert seen['shape'] == (1, 4, 8, 8)


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
        topology_head='all', topology_aggregate='deployment',
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
