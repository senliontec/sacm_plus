"""Run the matched-inference DCD control.

Usage (prediction only):

    python -m methods.tube_token.run_control --stage predict \
        --checkpoint results/runs_local/dcd_reference_control/seed_0/best_model.pth \
        --workdir results/runs_local/dcd_reference_control/seed_0 \
        --pred-root results/runs_local/dcd_reference_control/tta \
        --data-root data/sacm_3shot_seed0 --seed 0 --device cuda:0 \
        --test-datasets NAME:DIR ... --tta-flip

The checkpoint must carry ``adapter='dcd_reference_control'``.  Create it
with ``prepare_checkpoint`` below; the model weights are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import baseline_zoo.runtime as runtime

from .control import DCDReferenceControl


def prepare_checkpoint(source: Path, target: Path) -> Path:
    import torch

    payload = torch.load(source, map_location='cpu', weights_only=False)
    payload = dict(payload)
    payload['adapter'] = DCDReferenceControl.id
    config = dict(payload.get('config') or {})
    adapter_config = dict(config.get('adapter_config') or {})
    # The canonical fixed3 derivation already sets these; enforce them so a
    # gating checkpoint can never silently deploy a different slot.
    adapter_config['head_select'] = 'fixed'
    adapter_config['deployment_head'] = 3
    adapter_config['head_fusion'] = 0
    config['adapter_config'] = adapter_config
    payload['config'] = config
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def main(argv: list[str] | None = None) -> int:
    adapter = DCDReferenceControl()
    original = runtime._configured_epochs

    def _configured_epochs(_adapter):
        return 800

    runtime._configured_epochs = _configured_epochs
    try:
        return runtime.main_for(
            adapter, sys.argv[1:] if argv is None else argv)
    finally:
        runtime._configured_epochs = original


if __name__ == '__main__':
    raise SystemExit(main())
