# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the vendored SAM2 model without global Hydra state.

The upstream ``build_sam2`` composed a named config from the global Hydra
config module.  This port loads the YAML directly and uses the stateless
``hydra.utils.instantiate``; no ``GlobalHydra`` initialization is required.
Only the Hiera-L config used by SAM2-UNet is vendored.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "sam2_configs"


def build_sam2(
    config_file: str = "sam2_hiera_l.yaml",
    ckpt_path: str | Path | None = None,
    device: str = "cpu",
    mode: str = "eval",
):
    """Instantiate the full SAM2 base model and optionally load a checkpoint.

    ``SAM2UNet`` deletes most of the returned modules after construction, but
    the official checkpoint is loaded into the full model first, so the whole
    ``SAM2Base`` import/construction closure is required.
    """
    config_path = CONFIG_DIR / Path(config_file).name
    if not config_path.is_file():
        raise FileNotFoundError(f"missing SAM2 config: {config_path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path: str | Path | None) -> None:
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
            f"{path}: state dict does not match the vendored Hiera-L config "
            f"({len(missing)} missing, {len(unexpected)} unexpected); use the "
            "original SAM2 (not SAM2.1) sam2_hiera_large.pt. "
            f"First missing: {missing[:5]}; first unexpected: "
            f"{unexpected[:5]}")
