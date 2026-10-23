"""SACM — prompt-free curvilinear structure segmentation on SAM.

One of the project's segmentation models. Importing this module
registers the SAM-family ModelSpecs (sam_l / sam_b / sam_h) into the
framework registry and exposes the SACM-specific configuration:

    from models.sacm_x.configs import PRESETS
    from models.sam_network import build_sam_vit_l
    from models.sacm_x.diagnose import main as diagnose

Structure (SAM the foundation model lives in models/sam_network):
  - specs.py:    ModelSpecs (freeze protocol, param groups, forward)
  - configs.py:  SACM ablation presets
  - diagnose.py: head-supervision / gating diagnostics (paper motivation)
"""

from . import specs  # noqa: F401  (registers sam_l/sam_b/sam_h)
from . import configs  # noqa: F401
