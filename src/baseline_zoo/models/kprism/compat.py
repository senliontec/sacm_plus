"""Minimal replacements for the detectron2/fvcore helpers used upstream.

K-Prism vendors detectron2 for a handful of utility types.  The benchmark
port only uses the semantic query branch, so these small equivalents keep
the model self-contained without adding the detectron2 dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

# detectron2's Conv2d is a thin nn.Conv2d wrapper with optional norm/act.
# Every upstream call in the copied files uses the plain convolution form.
Conv2d = nn.Conv2d


class ShapeSpec(SimpleNamespace):
    """The subset of detectron2's ShapeSpec used by UTNet/SegHead."""

    def __init__(self, channels: int, stride: int, **kwargs):
        super().__init__(channels=channels, stride=stride, **kwargs)


def c2_xavier_fill(module: nn.Module) -> None:
    """detectron2's ``c2_xavier_fill`` for a Conv2d projection.

    Caffe2's XavierFill is implemented as ``kaiming_uniform_(a=1)`` in
    PyTorch, not ``xavier_uniform_``; the bias is zeroed.
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
