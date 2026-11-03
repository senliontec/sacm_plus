"""Helpers shared by the topology losses.

Curvilinear targets are often one pixel wide.  Bilinear downsampling of a
binary mask followed by ``> 0.5`` erases such a line at every scale below
the input resolution: a 1-px line at 1024 resized to 512 has maximum value
0.5 (strictly greater than 0.5 is never true) and at 256 it can be 0.0.
That silently disables Skeleton Recall and Topograph on exactly the thin
structures they are meant to supervise (CREMI membranes, OCTA capillaries).

Max pooling preserves a foreground pixel if any pixel in its pooling cell
is foreground, so a 1-px line stays 1-px after integer downsampling.  For
non-integer ratios use nearest-neighbour, which also never averages a line
away.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_target_for_topology(
    target: torch.Tensor,
    size: int | None,
) -> torch.Tensor:
    """Downsample a binary/soft target without erasing thin structures.

    Args:
        target: ``[B, 1, H, W]`` tensor in ``[0, 1]``.
        size: target spatial resolution; ``None`` keeps the input.
    Returns:
        ``[B, 1, size, size]`` tensor (or the input when size is None).
    """
    if size is None:
        return target
    if target.ndim != 4:
        raise ValueError('target must be [B, C, H, W]')
    if target.shape[-2:] == (size, size):
        return target

    h, w = target.shape[-2:]
    if h % size == 0 and w % size == 0:
        factor = h // size
        if factor > 1:
            return F.max_pool2d(target, kernel_size=factor, stride=factor)

    # Non-integer ratio (or upsampling): nearest-neighbour keeps the
    # sparse foreground instead of averaging it into the background.
    return F.interpolate(
        target, size=(size, size), mode='nearest')
