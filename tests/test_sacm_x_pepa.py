"""Tests for the optional PEPA (TCSU + TADT) post-encoder adapters."""

import torch
from torch import nn

from core.losses import TOPOLOGY_LOSS_REGISTRY, build_topology_loss
from models.sacm_x.configs import PRESETS
from models.sacm_x.specs import freeze_sam
from models.sam_network.build_sam import build_sam_vit_b
from models.sam_network.detail_stem import DetailStem
from models.sam_network.mask_decoder import MaskDecoder
from models.sam_network.pepa import (
    CandidateRanker,
    SnakeUpsample2d,
    TargetAdaptiveThreshold,
)


class _DummyTransformer(nn.Module):
    def forward(self, src, pos_src, tokens):
        b, c, h, w = src.shape
        return tokens, src.flatten(2).transpose(1, 2)


def _decoder(use_tcsu: bool, use_tadt: bool,
             use_ranker: bool = False,
             use_detail_stem: bool = False,
             high_res_output: bool = False,
             high_res_size: int = 128,
             high_res_refine: bool = False,
             high_res_refine_size: int = 128,
             single_head: bool = False,
             use_coarse_to_fine: bool = False) -> MaskDecoder:
    return MaskDecoder(
        transformer_dim=256,
        transformer=_DummyTransformer(),
        num_multimask_outputs=3,
        use_adapter_skip=False,
        use_coarse_to_fine=use_coarse_to_fine,
        use_fusion_v2=False,
        use_multi_depth=False,
        use_tcsu=use_tcsu,
        tcsu_mid_channels=8,
        tcsu_max_length=5,
        use_tadt=use_tadt,
        tadt_hidden=8,
        use_ranker=use_ranker,
        ranker_hidden=8,
        ranker_pair_hidden=8,
        use_detail_stem=use_detail_stem,
        detail_channels=32,
        high_res_output=high_res_output,
        high_res_size=high_res_size,
        high_res_refine=high_res_refine,
        high_res_refine_size=high_res_refine_size,
        single_head=single_head,
    )


def _inputs():
    image = torch.randn(1, 256, 16, 16)
    pe = torch.zeros(1, 256, 16, 16)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 16, 16)
    return image, pe, sparse, dense


def test_tcsu_is_zero_initialized_and_receives_gradient():
    module = SnakeUpsample2d(
        16, 16, scale=4, mid_channels=8, max_length=5, target_dim=32)
    feature = torch.randn(1, 16, 8, 8)
    target = torch.randn(1, 32)
    out = module(feature, target)
    assert out.shape == (1, 16, 32, 32)
    assert torch.count_nonzero(out).item() == 0
    out.sum().backward()
    assert module.fuse.weight.grad is not None
    assert torch.count_nonzero(module.fuse.weight.grad).item() > 0


def test_tadt_is_bounded_and_centered():
    module = TargetAdaptiveThreshold(8, hidden=8, t_min=-4.0, t_max=4.0)
    descriptors = torch.randn(2, 4, 8)
    logits = torch.randn(2, 4, 5, 5)
    adjusted, threshold = module(descriptors, logits)
    assert adjusted.shape == logits.shape
    assert threshold.shape == (2, 4)
    assert torch.all(threshold >= -4.0)
    assert torch.all(threshold <= 4.0)
    # Centering removes the spatial mean, so the adjusted mean is -t.
    assert torch.allclose(
        adjusted.mean(dim=(2, 3)) + threshold, torch.zeros_like(threshold),
        atol=1e-5)


def test_decoder_wiring_shapes_and_gradients():
    decoder = _decoder(use_tcsu=True, use_tadt=True)
    image, pe, sparse, dense = _inputs()
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense, return_stage1=True)
    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)
    assert stage1.shape == (1, 4, 64, 64)
    assert decoder.last_tadt_threshold is not None
    (masks.sum() + iou.sum() + stage1.sum()).backward()
    assert decoder.tcsu.fuse.weight.grad is not None
    assert decoder.tadt.mlp[-1].weight.grad is not None


def test_default_decoder_keeps_pepa_off():
    decoder = _decoder(use_tcsu=False, use_tadt=False)
    assert decoder.tcsu is None
    assert decoder.tadt is None
    image, pe, sparse, dense = _inputs()
    masks, iou = decoder.predict_masks(image, pe, sparse, dense)
    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)


def test_full_pepa_preset_exists():
    preset = PRESETS['full_pepa']
    assert preset['use_tcsu'] is True
    assert preset['use_tadt'] is True
    assert preset['tadt_consistency_weight'] > 0


def test_pepa_topology_presets_exist():
    for name in ('full_pepa_topo', 'full_pepa_cape', 'full_pepa_skel',
                 'full_pepa_topo_cape'):
        assert name in PRESETS
        assert PRESETS[name]['use_tcsu'] is True
        assert PRESETS[name]['use_tadt'] is True


def test_candidate_ranker_shape_and_gradient():
    ranker = CandidateRanker(16, hidden=8, pair_hidden=8)
    descriptors = torch.randn(2, 4, 16, requires_grad=True)
    scores = ranker(descriptors)
    assert scores.shape == (2, 4)
    scores.sum().backward()
    assert descriptors.grad is not None
    assert ranker.utility[1].weight.grad is not None
    assert ranker.pair[1].weight.grad is not None


def test_decoder_ranker_replaces_iou_output():
    decoder = _decoder(use_tcsu=False, use_tadt=False, use_ranker=True)
    image, pe, sparse, dense = _inputs()
    masks, iou = decoder.predict_masks(image, pe, sparse, dense)
    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)
    assert decoder.last_rank_scores is not None
    (masks.sum() + iou.sum()).backward()
    assert decoder.ranker.utility[1].weight.grad is not None


def test_full_pepa_rank_presets_exist():
    assert PRESETS['full_pepa_rank']['use_ranker'] is True
    assert PRESETS['full_pepa_rank']['rank_loss_weight'] > 0
    assert PRESETS['full_pepa_all']['use_ranker'] is True
    assert PRESETS['full_pepa_all']['topology_loss'] == 'topograph_cape'


def test_detail_stem_shapes_and_high_pass():
    stem = DetailStem(in_channels=3, channels=8, out_channels=32)
    image = torch.randn(1, 3, 128, 128)
    detail_256, detail_512 = stem(image)
    assert detail_256.shape == (1, 32, 32, 32)
    assert detail_512.shape == (1, 32, 64, 64)
    (detail_256.sum() + detail_512.sum()).backward()
    assert stem.out_512.weight.grad is not None


def test_decoder_high_res_output_and_detail_fusion():
    decoder = _decoder(
        use_tcsu=False, use_tadt=False,
        use_detail_stem=True, high_res_output=True, high_res_size=128)
    image, pe, sparse, dense = _inputs()
    detail_256 = torch.randn(1, 32, 64, 64)
    detail_512 = torch.randn(1, 32, 128, 128)
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense,
        detail_features=(detail_256, detail_512),
        return_stage1=True)
    assert masks.shape == (1, 4, 128, 128)
    assert iou.shape == (1, 4)
    assert stage1.shape == (1, 4, 64, 64)
    (masks.sum() + iou.sum() + stage1.sum()).backward()
    assert decoder.detail_fuse_512.weight.grad is not None


def test_decoder_image_guided_refinement():
    decoder = _decoder(
        use_tcsu=False, use_tadt=False,
        use_detail_stem=True, high_res_output=True, high_res_size=64,
        high_res_refine=True, high_res_refine_size=128)
    image, pe, sparse, dense = _inputs()
    refine_image = torch.randn(1, 3, 128, 128)
    detail_256 = torch.randn(1, 32, 64, 64)
    detail_512 = torch.randn(1, 32, 64, 64)
    masks, iou = decoder.predict_masks(
        image, pe, sparse, dense,
        detail_features=(detail_256, detail_512),
        image=refine_image)
    assert masks.shape == (1, 4, 128, 128)
    assert iou.shape == (1, 4)
    (masks.sum() + iou.sum()).backward()
    assert decoder.final_refine.weight.grad is not None


def test_full_pepa_hr_presets_exist():
    assert PRESETS['full_pepa_hr']['use_detail_stem'] is True
    assert PRESETS['full_pepa_hr']['high_res_output'] is True
    assert PRESETS['full_pepa_hr']['high_res_size'] == 512
    assert PRESETS['full_pepa_hr']['high_res_refine'] is True
    assert PRESETS['full_pepa_hr']['high_res_refine_size'] == 1024
    assert PRESETS['full_pepa_all_hr']['use_ranker'] is True
    assert PRESETS['full_pepa_all_hr']['topology_loss'] == 'topograph_cape'


def test_detail_stem_is_trainable_after_freeze():
    model = build_sam_vit_b(
        checkpoint=None, use_adapter=True, use_detail_stem=True)
    freeze_sam(model)
    assert all(
        p.requires_grad for p in model.image_encoder.detail_stem.parameters())
    assert not model.image_encoder.patch_embed.proj.weight.requires_grad


def test_single_head_v2_decoder():
    decoder = _decoder(
        use_tcsu=False, use_tadt=True,
        use_detail_stem=True, high_res_output=True, high_res_size=64,
        high_res_refine=True, high_res_refine_size=128,
        single_head=True)
    image, pe, sparse, dense = _inputs()
    refine_image = torch.randn(1, 3, 128, 128)
    detail_256 = torch.randn(1, 32, 64, 64)
    detail_512 = torch.randn(1, 32, 64, 64)
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense,
        detail_features=(detail_256, detail_512),
        image=refine_image, return_stage1=True)
    assert masks.shape == (1, 1, 128, 128)
    assert iou.shape == (1, 1)
    assert stage1.shape == (1, 1, 64, 64)
    (masks.pow(2).mean() + iou.mean() + stage1.pow(2).mean()).backward()
    assert decoder.final_refine.weight.grad is not None
    assert decoder.tadt.mlp[-1].weight.grad is not None


def test_coarse_to_fine_refiner_is_in_high_res_stage2_path():
    # Regression: mask_features used to be built before the C2F refiner, so
    # Stage-2 consumed the pre-refinement feature map and the refiner got no
    # gradient in high-res mode.
    decoder = _decoder(
        use_tcsu=False, use_tadt=False,
        use_detail_stem=True, high_res_output=True, high_res_size=64,
        single_head=True, use_coarse_to_fine=True)
    image, pe, sparse, dense = _inputs()
    detail_256 = torch.randn(1, 32, 64, 64)
    detail_512 = torch.randn(1, 32, 64, 64)
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense,
        detail_features=(detail_256, detail_512),
        return_stage1=True)
    (masks.pow(2).mean() + iou.mean() + stage1.pow(2).mean()).backward()
    grad = decoder.coarse_refiner.conv.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_coarse_to_fine_refiner_is_in_plain_stage2_path():
    decoder = _decoder(
        use_tcsu=False, use_tadt=False, use_coarse_to_fine=True)
    image, pe, sparse, dense = _inputs()
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense, return_stage1=True)
    (masks.pow(2).mean() + iou.mean() + stage1.pow(2).mean()).backward()
    grad = decoder.coarse_refiner.conv.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_full_v2_preset_fields():
    preset = PRESETS['full_v2']
    assert preset['single_head'] is True
    assert preset['use_geo_i'] is False
    assert preset['use_multi_depth'] is False
    assert preset['use_detail_stem'] is True
    assert preset['high_res_refine_size'] == 1024
    assert preset['use_tadt'] is True
    assert preset['topology_loss'] == 'topograph_skeleton'
    assert 'topograph_skeleton' in TOPOLOGY_LOSS_REGISTRY


def test_new_topology_losses_are_registered_and_differentiable():
    assert 'cape' in TOPOLOGY_LOSS_REGISTRY
    assert 'skeleton_recall' in TOPOLOGY_LOSS_REGISTRY
    assert 'topograph_cape' in TOPOLOGY_LOSS_REGISTRY
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 30:34, 10:54] = 1.0
    for name in ('cape', 'skeleton_recall', 'topograph_cape',
                 'topograph_skeleton'):
        loss_fn = build_topology_loss(name, resolution=64)
        logits = (target * 10.0 - 5.0).clone().requires_grad_(True)
        loss = loss_fn(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
