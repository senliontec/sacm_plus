# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SAM network implementation used by SACM-X.

This package contains only the network modules (encoder, prompt encoder,
mask decoder, adapters, fusion, predictor).  The algorithm layer — model
registration, freezing, parameter groups and presets — lives in
``models.sacm_x``.
"""

# SACM re-exports: the builders live in build_sam.py; specs.py and
# external callers import them from the package.
from .build_sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l  # noqa: F401
