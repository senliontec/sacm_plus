"""Config/preset integrity tests."""

import argparse

import pytest

from models.sacm_x.configs import (
    PRESETS,
    apply_preset,
    apply_yaml_presets,
    str2bool,
)

REQUIRED_KEYS = {
    "use_geo_i",
    "use_geo_e",
    "use_coarse_to_fine",
    "use_fusion_v2",
    "use_multi_depth",
    "deep_sup_weight",
    "cl_dice_weight",
    "iou_loss_weight",
    "stage2_supervision",
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


def test_scnp_preset_is_opt_in():
    assert PRESETS["full_scnp"]["scnp"] is True
    assert "scnp" not in PRESETS["full"]


def test_d8_supervision_modes_are_explicit():
    assert PRESETS["full"]["stage2_supervision"] == "soft"
    assert PRESETS["full_soft"]["stage2_supervision"] == "soft"
    assert PRESETS["full_allhead"]["stage2_supervision"] == "allhead"
    assert PRESETS["full_wta"]["stage2_supervision"] == "wta"
    assert PRESETS["sacm"]["stage2_supervision"] == "wta"
    assert PRESETS["stage1"]["stage2_supervision"] == "soft"
    assert PRESETS["stage2"]["stage2_supervision"] == "soft"


def test_yaml_uses_python_presets_as_source_of_truth(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('preset: sacm\nuse_geo_i: true\n')
    args = argparse.Namespace(
        use_geo_i=False, deep_sup_weight=0.0,
        stage2_supervision='soft')
    apply_yaml_presets(args, str(config))
    # Python PRESETS supplied the D8 WTA baseline...
    assert args.stage2_supervision == 'wta'
    # ...and the top-level YAML key overrode it afterwards.
    assert args.use_geo_i is True
