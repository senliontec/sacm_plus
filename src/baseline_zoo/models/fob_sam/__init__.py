"""Self-contained FoB-SAM baseline (CVPR 2026).

Official code: ``baselines/FoB_SAM`` (Apache-2.0).  The package contains
the official ResNet-101 encoder, background-prompt generator and frozen
SAM point-prompt wrapper; the adapter owns episode construction and the
20-dataset protocol adaptation.
"""

from .encoder import Res101Encoder
from .model import FewShotSeg
from .sam import SAM

__all__ = ['Res101Encoder', 'FewShotSeg', 'SAM']
