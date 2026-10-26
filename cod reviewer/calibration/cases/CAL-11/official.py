"""Official device transfer (synthetic calibration case)."""
import torch


def move(tensor):
    return tensor.cuda()
