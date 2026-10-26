"""Ported support encoding (synthetic calibration case)."""
import torch


def support_embedding(model, support):
    return model.encode(support)
