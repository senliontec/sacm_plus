"""Self-contained A2SAM port (Medical Image Analysis 2026).

A2SAM adapts SAM ViT-B to medical tubular segmentation with three pieces
from the paper:

* a trained probabilistic U-Net that supplies a coarse likelihood map and
  structural uncertainty (K=5 prior samples);
* a structural prompt generator that turns the likelihood map into a pruned
  skeleton/Morse complex, learns branch gates and fuses the structural map
  with likelihood/uncertainty into a dense SAM prompt;
* contrastive parameter selection (CPS) that trains the top 3.5% of SAM
  elements by a squared-gradient influence / magnitude-preservation score,
  plus the released thickness-aware loss.

The released repository contains SAM ViT-B and a detach-only prompt fusion
network but not the prior or the paper's SPG; those are reimplemented here
and documented in ``UPSTREAM.md``.

Source: https://github.com/YongChen-Exact/A2SAM
"""

from .cps import (
    apply_cps_masks,
    contrastive_scores,
    select_contrastive_masks,
)
from .model import A2SAMModel, build_a2sam
from .prior import ProbabilisticUNet
from .spg import StructuralPromptGenerator
from .topology import StructuralPoints, extract_structural_points

__all__ = [
    'A2SAMModel',
    'ProbabilisticUNet',
    'StructuralPoints',
    'StructuralPromptGenerator',
    'apply_cps_masks',
    'build_a2sam',
    'contrastive_scores',
    'extract_structural_points',
    'select_contrastive_masks',
]
