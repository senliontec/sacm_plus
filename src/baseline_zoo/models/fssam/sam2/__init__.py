# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vendored SAM 2 runtime with FSSAM's memory-attention modifications.

The upstream FSSAM package initialized the global Hydra config module
``sam2_configs`` at import time.  This port loads the YAML directly in
``build_sam`` instead, so importing this package has no global side effects.
"""
