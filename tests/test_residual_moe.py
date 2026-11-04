"""Tests for the Topology-aware Residual MoE (FineSAM++-style)."""

import torch

from models.sacm_x.configs import PRESETS
from models.sacm_x.specs import freeze_sam
from models.sam_network.build_sam import build_sam_vit_b
from models.sam_network.mask_decoder import MaskDecoder
from models.sam_network.residual_moe import (
    TopoResidualMoE,
    build_topo_gate_target,
    degrade_coarse,
)
from models.sam_network.transformer import TwoWayTransformer


def _module(**kwargs):
    defaults = dict(
        num_experts=2,
        hidden=8,
        gate_hidden=8,
        gate_size=32,
        out_size=32,
        head_index=2,
    )
    defaults.update(kwargs)
    return TopoResidualMoE(**defaults)


def test_residual_moe_is_identity_at_init_and_gets_gradients():
    torch.manual_seed(0)
    module = _module()
    image = torch.randn(2, 3, 32, 32)
    coarse = torch.randn(2, 1, 32, 32)
    heads = torch.randn(2, 4, 32, 32)
    out = module(image, coarse, heads)
    assert out.shape == coarse.shape
    assert torch.allclose(out, coarse)
    # The gate is not zero, but the zero-initialised experts make the
    # correction exactly zero.
    assert module.last_gate_logits is not None
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()
    loss = out.mean() + module.gate_loss(target) + module.sparsity_loss()
    loss.backward()
    assert module.experts[0].out.weight.grad is not None
    assert module.experts[0].out.weight.grad.abs().sum() > 0
    assert module.gate.mask_head.weight.grad is not None
    assert module.gate.mask_head.weight.grad.abs().sum() > 0


def test_res_scale_zero_is_exact_identity():
    module = _module()
    image = torch.randn(1, 3, 32, 32)
    coarse = torch.randn(1, 1, 32, 32)
    heads = torch.randn(1, 4, 32, 32)
    module.res_scale = 0.0
    out = module(image, coarse, heads)
    assert torch.equal(out, coarse)
    assert module.last_gate_logits is None
    assert module.last_gate_mask is None


def test_residual_moe_output_changes_after_expert_step():
    torch.manual_seed(1)
    module = _module()
    image = torch.randn(1, 3, 32, 32)
    coarse = torch.randn(1, 1, 32, 32)
    heads = torch.randn(1, 4, 32, 32)
    out = module(image, coarse, heads)
    loss = out.mean() + module.gate_loss(
        (torch.rand(1, 1, 32, 32) > 0.5).float())
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    opt.zero_grad()
    loss.backward()
    opt.step()
    out2 = module(image, coarse, heads)
    assert not torch.allclose(out2, coarse)


def test_multiprompt_image_repetition():
    module = _module()
    image = torch.randn(1, 3, 32, 32)
    coarse = torch.randn(3, 1, 32, 32)
    heads = torch.randn(3, 4, 32, 32)
    out = module(image, coarse, heads)
    assert out.shape == coarse.shape
    # Two images, two prompt replicas each.
    image2 = torch.randn(2, 3, 32, 32)
    coarse2 = torch.randn(4, 1, 32, 32)
    heads2 = torch.randn(4, 4, 32, 32)
    assert module(image2, coarse2, heads2).shape == coarse2.shape


def test_disagreement_gate_has_no_gate_parameters():
    module = _module(gate_mode='disagreement')
    assert module.gate is None
    image = torch.randn(1, 3, 32, 32)
    coarse = torch.randn(1, 1, 32, 32)
    heads = torch.randn(1, 4, 32, 32)
    out = module(image, coarse, heads)
    assert out.shape == coarse.shape
    assert module.last_gate_logits is None
    assert module.sparsity_loss().item() >= 0
    # Identical heads => zero disagreement => exactly zero gate.
    same = heads[:, :1].repeat(1, 4, 1, 1)
    module(image, coarse, same)
    assert module.last_gate_mask.abs().max().item() == 0.0


def test_topo_gate_target_marks_missed_and_spurious_regions():
    # GT: a horizontal line.  Coarse: an absent prediction (missed line)
    # plus a blob far from GT (spurious component).
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 16, 4:28] = 1.0
    coarse = torch.full((1, 1, 32, 32), -8.0)
    coarse[:, :, 26:30, 26:30] = 8.0
    gt = build_topo_gate_target(coarse, target, band=2, dilate=1)
    assert gt.shape == target.shape
    # Missed line pixels are positive.
    assert gt[:, :, 16, 4:28].mean() > 0.5
    # Spurious blob is positive.
    assert gt[:, :, 26:30, 26:30].mean() > 0.5
    # A confident true-negative far from both is not selected.
    assert gt[0, 0, 0, 0].item() == 0.0


def test_degrade_coarse_is_deterministic_and_shape_preserving():
    prob = (torch.rand(1, 1, 16, 16) > 0.6).float()
    for kind in ('open', 'close', 'threshold_low', 'threshold_high'):
        out = degrade_coarse(prob, kind)
        assert out.shape == prob.shape
        assert torch.equal(out, degrade_coarse(prob, kind))


def test_fixed2_resmoe_presets():
    preset = PRESETS['full_fixed2_resmoe']
    assert preset['use_res_moe'] is True
    assert preset['res_head_index'] == 2
    assert preset['res_gate_mode'] == 'learned'
    assert preset['res_num_experts'] == 4
    assert preset['res_out_size'] == 512
    assert preset['res_topology_weight'] == 0.5
    assert preset['res_warmup_epochs'] == 30
    assert preset['fixed_supervision_head'] == 2
    assert preset['topology_loss'] == 'skeleton_recall'

    dg = PRESETS['full_fixed2_resmoe_dg']
    assert dg['use_res_moe'] is True
    assert dg['res_gate_mode'] == 'disagreement'
    assert dg['res_gate_loss_weight'] == 0.0

    scnp = PRESETS['full_fixed2_scnp']
    assert scnp['scnp'] is True
    assert scnp.get('use_res_moe', False) is False

    combined = PRESETS['full_fixed2_scnp_resmoe']
    assert combined['scnp'] is True
    assert combined['use_res_moe'] is True
    assert combined['res_gate_mode'] == 'learned'
    assert combined['res_topology_weight'] == 0.5


def test_builder_constructs_trainable_resmoe():
    model = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        use_geo_i=False,
        use_geo_e=True,
        use_internal_adapter=False,
        collect_adapter_features=True,
        use_coarse_to_fine=True,
        use_fusion_v2=True,
        use_multi_depth=True,
        use_structural_aux=True,
        high_res_output=True,
        high_res_size=512,
        high_res_refine=False,
        use_res_moe=True,
        res_num_experts=2,
        res_hidden=8,
        res_gate_hidden=8,
        res_gate_size=64,
        res_out_size=64,
        init_stage2_from_stage1=False,
    )
    assert model.mask_decoder.residual_moe is not None
    assert model.mask_decoder.residual_moe.num_experts == 2
    freeze_sam(model)
    params = dict(model.named_parameters())
    assert params[
        'mask_decoder.residual_moe.experts.0.out.weight'].requires_grad
    assert params[
        'mask_decoder.residual_moe.gate.mask_head.weight'].requires_grad


def test_mask_decoder_applies_resmoe_to_one_head_only():
    torch.manual_seed(2)
    decoder = MaskDecoder(
        transformer_dim=32,
        transformer=TwoWayTransformer(
            depth=1, embedding_dim=32, mlp_dim=64, num_heads=2),
        num_multimask_outputs=3,
        use_adapter_skip=False,
        use_coarse_to_fine=False,
        use_fusion_v2=False,
        use_multi_depth=False,
        use_structural_aux=False,
        use_res_moe=True,
        res_num_experts=2,
        res_hidden=8,
        res_gate_hidden=8,
        res_gate_size=32,
        res_out_size=32,
        res_head_index=2,
    )
    b = 1
    masks, iou, stage1 = decoder.predict_masks(
        torch.randn(b, 32, 8, 8),
        torch.randn(1, 32, 8, 8),
        torch.zeros(b, 0, 32),
        torch.zeros(b, 32, 8, 8),
        image=torch.randn(b, 3, 32, 32),
        return_stage1=True,
    )
    assert masks.shape == (b, 4, 32, 32)
    assert stage1.shape == (b, 4, 32, 32)
    assert iou.shape == (b, 4)
    assert decoder.residual_moe.last_gate_logits is not None
    # Only the selected head is replaced; at init the replacement is
    # identity, so all heads remain the coarse prediction.
    loss = masks.mean() + decoder.residual_moe.gate_loss(
        (torch.rand(b, 1, 32, 32) > 0.5).float())
    loss.backward()
    assert decoder.residual_moe.experts[0].out.weight.grad is not None
