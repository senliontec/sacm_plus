"""Self-contained A2SAM port (Medical Image Analysis 2026).

Official components: vendored SAM ViT-B, the two-branch prompt fusion, CPS
and the thickness-aware loss.  Paper-level reconstruction: coarse U-Net,
persistence-threshold prior, exact H0 persistence / Morse-watershed branches
and F_MS sampling.  See ``UPSTREAM.md`` and ``docs/A2SAM_FIDELITY.md``.
"""

from .cps import (
    contrastive_scores,
    sam_cps_candidates,
    select_freeze_masks,
    zero_cps_gradients,
)
from .loss import IOU, ThicknessAwareLoss
from .model import SAM_CHECKPOINT_EXCLUDED, A2SAMModel, build_a2sam
from .prior import PersistencePrior
from .prompt import PromptFusion, PromptGenerator
from .topology import (
    MorseComplex,
    PersistenceBars,
    build_morse_complex,
    compute_persistence,
    dice_optimal_xi,
    f_ms,
    uncertainty_from_samples,
)
from .unet import CoarseUNet

__all__ = [
    'A2SAMModel',
    'CoarseUNet',
    'IOU',
    'MorseComplex',
    'PersistenceBars',
    'PersistencePrior',
    'PromptFusion',
    'PromptGenerator',
    'SAM_CHECKPOINT_EXCLUDED',
    'ThicknessAwareLoss',
    'build_a2sam',
    'build_morse_complex',
    'compute_persistence',
    'contrastive_scores',
    'dice_optimal_xi',
    'f_ms',
    'sam_cps_candidates',
    'select_freeze_masks',
    'uncertainty_from_samples',
    'zero_cps_gradients',
]
