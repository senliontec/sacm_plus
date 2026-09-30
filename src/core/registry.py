"""Model registry: every model the framework can train/evaluate is a
ModelSpec — a self-describing bundle of build/freeze/param-groups/
preprocess/prompt/forward functions. The Trainer and Evaluator only
speak ModelSpec; model modules register their specs (see
models/sacm/models.py for the reference implementation).

See docs/MODEL_MANAGEMENT.md for the full design and the step-by-step
recipe for adding a model.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ModelSpec:
    """A self-describing model bundle.

    All callables share the convention that the model instance is the
    first positional argument.
    """

    name: str
    build: Callable[..., Any]                    # (**overrides) -> nn.Module
    freeze: Callable[[Any], None]                # (model) -> None
    param_groups: Callable[[Any, Any], list]     # (model, args) -> optimizer groups
    preprocess: Callable[[Any, Any], Any]        # (model, images) -> images
    prompt_builder: Callable[[Any, int], tuple]  # (model, batch_size) -> (sparse, dense)
    forward: Callable[..., Any]                  # (model, images, sparse, dense, return_stage1) -> outputs
    algorithm: str = ""
    input_size: int = 1024
    description: str = ""
    default_config: dict = field(default_factory=dict)


MODEL_REGISTRY: dict = {}


def register_model(name):
    def _wrap(spec: ModelSpec):
        spec.name = name
        MODEL_REGISTRY[name] = spec
        return spec
    return _wrap


def get_model_spec(name):
    """Return the ModelSpec for a registered name (raises with the
    available names on unknown input)."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]
