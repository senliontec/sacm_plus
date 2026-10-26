"""Official class head (synthetic calibration case)."""
from torch import nn

NUM_CLASSES = 8


def build_head(in_channels=64):
    return nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
