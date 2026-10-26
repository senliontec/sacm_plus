"""Ported device transfer (synthetic calibration case)."""
import torch


def move(tensor, device):
    return tensor.to(device)
