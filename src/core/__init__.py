"""core — algorithm-agnostic segmentation framework.

The framework provides the engines and registries shared by every
segmentation algorithm in this project:

    from core import Trainer, Evaluator          # engines
    from core import MODEL_REGISTRY, get_model_spec
    from core.losses import ...                  # topology losses
    from core.metrics import compute_metrics     # metrics

Models live in the `models/` package (one module each) and
register their ModelSpecs into MODEL_REGISTRY on import.
"""

from .io import get_prompt_embeddings, predict_all  # noqa: F401
from .data import SegmentationDataset, TestDataset  # noqa: F401
from .trainer import Trainer  # noqa: F401
from .evaluator import Evaluator  # noqa: F401
from .registry import (  # noqa: F401
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    register_model,
)
from . import losses, metrics  # noqa: F401  (registries)
