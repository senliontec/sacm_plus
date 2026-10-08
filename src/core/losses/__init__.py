"""Loss function package.

Importing this package registers every available topology loss in
TOPOLOGY_LOSS_REGISTRY (losses whose hard dependencies are missing are
skipped gracefully) and re-exports the public API.

Direct usage:

    from losses import DiceBCELoss, SoftclDiceLoss          # core losses
    from losses import TOPOLOGY_LOSS_REGISTRY, build_topology_loss
    from losses import EndpointDistanceLossAverage           # any official port
    from losses.betti_matching import BettiMatching         # engines
"""

from .core import DiceLoss, DiceBCELoss, soft_dice, SoftSkeletonize, SoftclDiceLoss
from .registry import TOPOLOGY_LOSS_REGISTRY, build_topology_loss, register

# Import all loss modules: side effect = registration into the registry
from . import decl, centerline_ce, betti_matching, satloss, topograph, warping, topo_losses  # noqa: F401
from . import euler_refine  # noqa: F401

from .decl import EndpointDistanceLossAverage
from .centerline_ce import (
    DiceCenterlineCELoss,
    DiceCLDiceLoss,
    CECLDiceLoss,
    CECLCELoss,
)
from .betti_matching import (
    BettiMatching,
    BettiMatchingLoss,
    DiceBettiMatchingLoss,
    WassersteinMatching,
    ComposedWassersteinMatching,
    DiceWassersteinLoss,
    DiceComposedWassersteinLoss,
    betti_number_error_metric,
)
from .satloss import PDMatchingLoss, SATLossAdapter
from .topograph import (
    TopographLoss,
    DiceTopographLoss,
    ExactTopographLoss,
    Multiclass_CLDice,
    topograph_error_metric,
)
