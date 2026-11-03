"""Tests for the optional dynamic strip branches."""

import torch

from models.sacm_x.configs import PRESETS
from models.sam_network.build_sam import build_sam_vit_b
from models.sam_network.dynamic_strip import (
    DynamicStripBranch,
    DynamicStripConv2d,
)
from models.sam_network.image_encoder import AdapterEGlobal, GeoAdapterModule


def test_dynamic_strip_parameter_shapes_and_zero_offsets():
    conv = DynamicStripConv2d(32, kernel_size=9, orientation='v',
                              offset_groups=8)
    assert conv.weight.shape == (32, 1, 9, 1)
    assert conv.bias.shape == (32,)
    assert torch.count_nonzero(conv.offset_conv.weight).item() == 0
    assert torch.count_nonzero(conv.offset_conv.bias).item() == 0

    branch = DynamicStripBranch(32, kernel_size=9, offset_groups=8)
    assert branch.strip_v.weight.shape == (32, 1, 9, 1)
    assert branch.strip_h.weight.shape == (32, 1, 1, 9)


def test_dynamic_adapters_are_zero_initialized():
    geo = GeoAdapterModule(64, strip_dim=16, dynamic_strip=True)
    assert geo.dynamic_strip
    assert geo.strip_branch is not None
    assert geo.strip_v is None
    assert geo.strip_h is None
    assert torch.count_nonzero(geo.gamma).item() == 0

    external = AdapterEGlobal(64, dynamic_strip=True, dynamic_kernel=9)
    assert external.dynamic_strip
    assert external.strip_branch is not None
    assert external.strip_v is None
    assert external.strip_h is None
    assert torch.count_nonzero(external.eps_strip).item() == 0


def test_dynamic_preset_and_builder_wiring():
    assert PRESETS['full_dynamic_strip']['use_dynamic_strip'] is True

    model = build_sam_vit_b(
        checkpoint=None,
        use_adapter=True,
        use_dynamic_strip=True,
    )
    encoder = model.image_encoder
    assert encoder.use_dynamic_strip
    assert encoder.blocks[0].adapter.dynamic_strip
    assert any(
        getattr(adapter, 'dynamic_strip', False)
        for adapter in encoder.external_adapters
    )
