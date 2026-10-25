"""Self-contained DSCNet 2D model package."""

from .losses import cross_loss
from .model import DSCNet


def build_dscnet(n_channels: int = 1, n_classes: int = 1,
                 basic_layer: int = 16, kernel_size: int = 9,
                 extend_scope: int = 1, if_offset: bool = True,
                 dim: int = 1) -> DSCNet:
    """Build the official DSCNet 2D model (DRIVE settings by default)."""
    return DSCNet(
        n_channels=n_channels,
        n_classes=n_classes,
        kernel_size=kernel_size,
        extend_scope=extend_scope,
        if_offset=if_offset,
        device=None,
        number=basic_layer,
        dim=dim,
    )


__all__ = ['DSCNet', 'build_dscnet', 'cross_loss']
