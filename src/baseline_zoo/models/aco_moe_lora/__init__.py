"""Self-contained ACO-MoE-LoRA port (ICML 2026).

The released repository provides a SAM2 Hiera-L encoder with dynamic LoRA in
the attention projections, a four-level ACO-ConvLoRA decoder, and an ACO/MoE
search over rank, alpha and top-k.  The official code constructs full SAM2 and
then deletes all unused components; this port builds the same Hiera-L trunk
directly and keeps the official ACO modules and search.

Source: https://github.com/lky0234/ACO-MoE-LoRA
"""

from .model import (
    ACOConvLoRA,
    ACOMoeGate,
    ACOMoELoRA,
    AdaptiveExpert,
    AdaptiveTrainingScheduler,
    DynamicLoRALayer,
    DynamicLoRAWrapper,
    build_hiera_trunk,
)

__all__ = [
    'ACOConvLoRA',
    'ACOMoeGate',
    'ACOMoELoRA',
    'AdaptiveExpert',
    'AdaptiveTrainingScheduler',
    'DynamicLoRALayer',
    'DynamicLoRAWrapper',
    'build_hiera_trunk',
]
