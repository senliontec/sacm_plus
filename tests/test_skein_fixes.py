"""Regression tests for Skein initialization/checkpoint compatibility."""

import torch

from models.sam_network import build_sam_vit_b


def _stage1_state(module):
    return module.mask_decoder.output_hypernetworks_mlps[0].state_dict()


def _stage2_state(module):
    return module.mask_decoder.output_hypernetworks_mlps_stage2[0].state_dict()


def test_stage2_copied_after_base_checkpoint_load(tmp_path):
    """Stage-2 must be copied from the loaded Stage-1, not random __init__ weights."""
    source = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        init_stage2_from_stage1=False,
    )
    # Simulate a base SAM checkpoint: no Stage-2 keys at all.
    state = {
        key: value for key, value in source.state_dict().items()
        if 'output_hypernetworks_mlps_stage2' not in key
    }
    checkpoint = tmp_path / 'base_sam.pth'
    torch.save(state, checkpoint)

    restored = build_sam_vit_b(
        checkpoint=str(checkpoint),
        use_adapter=True,
        init_stage2_from_stage1=True,
    )
    stage1 = _stage1_state(restored)
    stage2 = _stage2_state(restored)
    assert all(torch.equal(stage1[key], stage2[key]) for key in stage1)


def test_legacy_shared_adapter_norm_is_migrated():
    """Old checkpoints with one adapter_norm must load into per-block norms."""
    base = build_sam_vit_b(checkpoint=None, use_adapter=True)
    state = {
        key: value for key, value in base.state_dict().items()
        if not key.startswith('image_encoder.adapter_norms.')
    }
    dim = base.image_encoder.adapter_norms[0].normalized_shape[0]
    state['image_encoder.adapter_norm.weight'] = torch.ones(dim)
    state['image_encoder.adapter_norm.bias'] = torch.zeros(dim)

    target = build_sam_vit_b(checkpoint=None, use_adapter=True)
    result = target.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for norm in target.image_encoder.adapter_norms:
        assert torch.allclose(norm.weight, torch.ones(dim))
        assert torch.allclose(norm.bias, torch.zeros(dim))
