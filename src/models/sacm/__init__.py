"""SACM — prompt-free curvilinear structure segmentation on SAM.

One of the project's segmentation models. Importing this module
registers the SAM-family ModelSpecs (sam_l / sam_b / sam_h) into the
framework registry and exposes the SACM-specific configuration:

    from models.sacm.configs import PRESETS
    from models.sam import build_sam_vit_l
    from models.sacm.diagnose import main as diagnose

Structure (the SAM architecture lives in models/sam — a sibling module):
  - specs.py:    ModelSpecs (freeze protocol, param groups, forward)
  - configs.py:  SACM ablation presets
  - diagnose.py: head-supervision / gating diagnostics (paper motivation)
"""

from . import specs  # noqa: F401  (registers sam_l/sam_b/sam_h)
from . import configs  # noqa: F401
