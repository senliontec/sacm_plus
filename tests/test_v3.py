"""Tests for the v3 dense structural decoder."""

import torch

from core.losses.structural_targets import build_structural_targets
from models.sacm_x.configs import PRESETS
from models.sam_network.build_sam import build_sam_vit_b
from models.sam_network.mask_decoder import MaskDecoder
from models.sam_network.structural_decoder import StructuralDecoderV3


class _DummyTransformer(torch.nn.Module):
    def forward(self, src, pos_src, tokens):
        return tokens, src.flatten(2).transpose(1, 2)


def _decoder() -> StructuralDecoderV3:
    return StructuralDecoderV3(
        encoder_embed_dim=1024,
        decoder_dim=256,
        feature_dim=32,
        mid_dim=24,
        fine_dim=16,
        detail_channels=8,
        output_size=128,
        refine_size=256,
        refine_channels=8,
    )


def _inputs():
    embeddings = torch.randn(1, 256, 16, 16)
    intermediates = [torch.randn(1, 16, 16, 1024) for _ in range(4)]
    detail_256 = torch.randn(1, 8, 64, 64)
    detail_512 = torch.randn(1, 8, 128, 128)
    image = torch.randn(1, 3, 256, 256)
    return embeddings, intermediates, detail_256, detail_512, image


def test_v3_decoder_shapes_and_gradients():
    decoder = _decoder()
    embeddings, intermediates, d256, d512, image = _inputs()
    masks, iou, coarse = decoder.predict_masks(
        embeddings, intermediate_features=intermediates,
        detail_features=(d256, d512), image=image, return_stage1=True)
    assert masks.shape == (1, 1, 256, 256)
    assert iou.shape == (1, 1)
    assert coarse.shape == (1, 1, 64, 64)

    aux = decoder.last_aux_outputs
    loss = (masks.pow(2).mean() + iou.mean()
            + coarse.pow(2).mean()
            + aux['centerline'].mean()
            + aux['boundary'].mean()
            + aux['width'].mean())
    loss.backward()
    for name in (
        'v3_mask_out.weight',
        'v3_refine.0.weight',
        'v3_fuse512.conv.weight',
        'v3_lat3.weight',
        'v3_gate512.weight',
        'v3_centerline_out.weight',
        'v3_boundary_out.weight',
        'v3_width_out.weight',
    ):
        grad = decoder.get_parameter(name).grad
        assert grad is not None, name
        assert grad.abs().sum() > 0, name


def test_v3_builder_uses_structural_decoder():
    model = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        use_geo_i=False,
        use_geo_e=True,
        use_internal_adapter=False,
        collect_adapter_features=False,
        use_v3=True,
        use_detail_stem=True,
        high_res_output=True,
        high_res_size=512,
        high_res_refine=True,
        high_res_refine_size=1024,
        single_head=True,
        init_stage2_from_stage1=False,
    )
    assert isinstance(model.mask_decoder, StructuralDecoderV3)
    assert model.image_encoder.use_internal_adapter is False
    assert model.image_encoder.collect_adapter_features is False
    assert sum(p.numel() for p in model.mask_decoder.parameters()) > 0


def test_structural_targets_preserve_one_pixel_line():
    gt = torch.zeros(1, 1, 1024, 1024)
    gt[..., 512, 100:900] = 1.0
    targets = build_structural_targets(gt, resolution=512, width_max=16.0)
    assert targets['centerline'].shape == (1, 1, 512, 512)
    assert float(targets['centerline'].sum()) > 0
    assert float(targets['boundary'].sum()) > 0
    assert float(targets['width'].sum()) > 0
    assert float(targets['width'].max()) <= 1.0


def test_structural_targets_empty_mask():
    gt = torch.zeros(1, 1, 1024, 1024)
    targets = build_structural_targets(gt, resolution=256, width_max=16.0)
    for value in targets.values():
        assert float(value.sum()) == 0.0


def test_full_v3_preset_fields():
    preset = PRESETS['full_v3']
    assert preset['use_v3'] is True
    assert preset['use_internal_adapter'] is False
    assert preset['collect_adapter_features'] is False
    assert preset['use_coarse_to_fine'] is False
    assert preset['use_fusion_v2'] is False
    assert preset['use_multi_depth'] is False
    assert preset['single_head'] is True
    assert preset['use_detail_stem'] is True
    assert preset['high_res_size'] == 512
    assert preset['high_res_refine_size'] == 1024
    assert preset['topology_loss'] == 'skeleton_recall'
    assert preset['topology_resolution'] == 512
    assert preset['v3_centerline_weight'] > 0
    assert preset['tversky_weight'] == 0.5


def test_prompt_decoder_structural_aux_heads():
    decoder = MaskDecoder(
        transformer_dim=64,
        transformer=_DummyTransformer(),
        num_multimask_outputs=0,
        use_adapter_skip=False,
        use_coarse_to_fine=False,
        use_fusion_v2=False,
        use_multi_depth=False,
        use_detail_stem=True,
        detail_channels=8,
        high_res_output=True,
        high_res_size=64,
        single_head=True,
        use_structural_aux=True,
    )
    embeddings = torch.randn(1, 64, 8, 8)
    image = torch.randn(1, 3, 64, 64)
    detail = (torch.randn(1, 8, 16, 16), torch.randn(1, 8, 64, 64))
    masks, iou = decoder.predict_masks(
        embeddings, image_pe=torch.zeros(1, 64, 8, 8),
        sparse_prompt_embeddings=torch.zeros(1, 0, 64),
        dense_prompt_embeddings=torch.zeros(1, 64, 8, 8),
        detail_features=detail, image=image)
    assert masks.shape == (1, 1, 64, 64)
    assert iou.shape == (1, 1)
    aux = decoder.last_aux_outputs
    assert aux['centerline'].shape == (1, 1, 64, 64)
    assert aux['boundary'].shape == (1, 1, 64, 64)
    assert aux['width'].shape == (1, 1, 64, 64)
    loss = (masks.pow(2).mean() + iou.mean()
            + aux['centerline'].mean() + aux['boundary'].mean()
            + aux['width'].mean())
    loss.backward()
    for name in ('structural_aux_centerline.weight',
                 'structural_aux_boundary.weight',
                 'structural_aux_width.weight'):
        grad = decoder.get_parameter(name).grad
        assert grad is not None and grad.abs().sum() > 0


def test_full_v2_skel_aux_preset_fields():
    preset = PRESETS['full_v2_skel_aux']
    assert preset['use_structural_aux'] is True
    assert preset.get('use_v3', False) is False
    assert preset['single_head'] is True
    assert preset['topology_loss'] == 'skeleton_recall'
    assert preset['v3_centerline_weight'] > 0


def test_prompt_decoder_fixed2_structural_aux_heads():
    decoder = MaskDecoder(
        transformer_dim=64,
        transformer=_DummyTransformer(),
        num_multimask_outputs=3,
        use_adapter_skip=False,
        use_coarse_to_fine=True,
        use_fusion_v2=False,
        use_multi_depth=False,
        use_detail_stem=True,
        detail_channels=8,
        high_res_output=True,
        high_res_size=64,
        single_head=False,
        use_structural_aux=True,
    )
    embeddings = torch.randn(1, 64, 8, 8)
    image = torch.randn(1, 3, 64, 64)
    detail = (torch.randn(1, 8, 16, 16), torch.randn(1, 8, 64, 64))
    masks, iou, stage1 = decoder.predict_masks(
        embeddings, image_pe=torch.zeros(1, 64, 8, 8),
        sparse_prompt_embeddings=torch.zeros(1, 0, 64),
        dense_prompt_embeddings=torch.zeros(1, 64, 8, 8),
        detail_features=detail, image=image, return_stage1=True)
    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)
    assert stage1.shape == (1, 4, 32, 32)
    aux = decoder.last_aux_outputs
    assert aux['centerline'].shape == (1, 1, 64, 64)
    loss = (masks.pow(2).mean() + iou.mean()
            + aux['centerline'].mean() + aux['boundary'].mean()
            + aux['width'].mean())
    loss.backward()
    for name in ('structural_aux_centerline.weight',
                 'structural_aux_boundary.weight',
                 'structural_aux_width.weight'):
        grad = decoder.get_parameter(name).grad
        assert grad is not None and grad.abs().sum() > 0


def test_full_fixed2_aux_preset_fields():
    preset = PRESETS['full_fixed2_aux']
    assert preset['use_structural_aux'] is True
    assert preset.get('single_head', False) is False
    assert preset['use_coarse_to_fine'] is True
    assert preset['use_fusion_v2'] is True
    assert preset['use_multi_depth'] is True
    assert preset['use_geo_i'] is True
    assert preset['use_geo_e'] is True
    assert preset['topology_loss'] == 'skeleton_recall'
    assert preset['topology_resolution'] == 512
    assert preset['tversky_weight'] == 0.3
    assert preset.get('use_detail_stem', False) is False

    hr = PRESETS['full_fixed2_hr_aux']
    assert hr['use_detail_stem'] is True
    assert hr['high_res_size'] == 512
    assert hr['high_res_refine_size'] == 1024
