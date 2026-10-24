"""nnWNet official nnU-Net trainer plugin (lazy import).

The trainer imports ``nnunetv2`` which reads the nnU-Net path environment
variables at import time, so callers must set those variables first.
"""


def get_trainer_class():
    from .trainer import nnUNetTrainer_WNet2D
    return nnUNetTrainer_WNet2D


__all__ = ['get_trainer_class']
