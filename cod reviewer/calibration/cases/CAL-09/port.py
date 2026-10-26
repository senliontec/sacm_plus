"""Ported inference threshold (synthetic calibration case)."""
import torch


def predict(logits):
    """Port: apply sigmoid before the 0.5 threshold."""
    return (torch.sigmoid(logits) > 0.5).float()
