# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vendored SAM2 runtime for the SAM2-UNet appendix baseline.

The upstream package initialized the global Hydra config module
``sam2_configs`` at import time.  This port loads the config directly from
``sam2_configs/sam2_hiera_l.yaml`` in ``build_sam2`` instead, so importing
this package has no global side effects.
"""
