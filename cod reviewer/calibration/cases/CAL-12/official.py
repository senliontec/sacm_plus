"""Official support encoding (synthetic calibration case)."""
import torch


def support_embedding(model, support):
    with torch.no_grad():
        return model.encode(support)
