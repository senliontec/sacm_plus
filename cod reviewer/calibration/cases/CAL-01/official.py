"""Official normalization helper (synthetic calibration case)."""
import numpy as np


def normalize(image, mean, std):
    return (image - mean) / std


def run(image):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    return normalize(image, mean, std)
