#!/usr/bin/env python3
"""Thin CLI for the unified U-Net adapter.

All shared protocol logic lives in ``baseline_zoo.runtime``; the official
model/preprocessing/loss logic lives in ``baseline_zoo.adapters.unet``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.unet import UNetAdapter  # noqa: E402
from baseline_zoo.runtime import main_for  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main_for(UNetAdapter()))
