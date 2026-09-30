"""Config/preset integrity tests."""

import argparse

import pytest

from models.sacm.configs import PRESETS, apply_preset, str2bool

REQUIRED_KEYS = {
    "use_geo_i",
    "use_geo_e",
    "use_coarse_to_fine",
    "use_fusion_v2",
    "use_multi_depth",
    "deep_sup_weight",
    "cl_dice_weight",
    "iou_loss_weight",
}


def test_all_presets_complete():
    for name, preset in PRESETS.items():
        if name == "none":
            continue
        assert REQUIRED_KEYS.issubset(preset.keys()), f"preset {name!r} missing keys"


def test_str2bool():
    assert str2bool(True) is True
    assert str2bool("true") is True
    assert str2bool("1") is True
    assert str2bool("False") is False
    assert str2bool("0") is False
    assert str2bool("off") is False


def test_apply_preset_overrides():
    args = argparse.Namespace(use_geo_i=False, deep_sup_weight=0.0)
    apply_preset(args, "stage2")
    assert args.use_geo_i is True
    assert args.deep_sup_weight == pytest.approx(0.3)
    # 'none' is a no-op
    apply_preset(args, "none")
    assert args.use_geo_i is True
