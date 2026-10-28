# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the vendored FSSAM SAM2 model without global Hydra state.

The upstream FSSAM package registered ``sam2_configs`` as a global Hydra
config module at import time and composed named configs.  This port loads
the YAML directly with OmegaConf and instantiates it with the stateless
``hydra.utils.instantiate``.  The FSSAM-specific hydra overrides are applied
to the loaded config where they matter for this port.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "sam2_configs"


def _load_cfg(config_file: str):
    config_path = CONFIG_DIR / Path(config_file).name
    if not config_path.is_file():
        raise FileNotFoundError(f"missing SAM2 config: {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg


def build_sam2(
    config_file: str = "sam2_hiera_s.yaml",
    ckpt_path: str | Path | None = None,
    device: str = "cpu",
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
):
    cfg = _load_cfg(config_file)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file: str = "sam2_hiera_s.yaml",
    ckpt_path: str | Path | None = None,
    device: str = "cpu",
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
):
    cfg = _load_cfg(config_file)
    cfg.model._target_ = (
        "baseline_zoo.models.fssam.sam2.sam2_video_predictor."
        "SAM2VideoPredictor")
    # The released builder passes these as hydra ++ overrides.  They are
    # inert in FSSAM's mask-prompt path, but are kept so the constructed
    # module matches the upstream defaults.
    if apply_postprocessing:
        cfg.model.binarize_mask_from_pts_for_mem_enc = True
        cfg.model.fill_hole_area = 8
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing SAM2 checkpoint: {path}")
    # SAM2 checkpoints are dictionaries with a "model" state dict; newer
    # PyTorch versions default to weights_only=True, which cannot load them.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            f"{path}: expected a SAM2 checkpoint containing a 'model' "
            "state dict")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: state dict does not match the vendored config "
            f"({len(missing)} missing, {len(unexpected)} unexpected); use "
            "the original SAM2 (not SAM2.1) checkpoint that matches the "
            f"config. First missing: {missing[:5]}; first unexpected: "
            f"{unexpected[:5]}")
