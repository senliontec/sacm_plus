"""Ported normalization helper (synthetic calibration case)."""
import numpy as np

PIXEL_MEAN = np.array([0.5, 0.5, 0.5])
PIXEL_STD = np.array([0.25, 0.25, 0.25])


def normalize_image(value, center, scale):
    return (value - center) / scale


def run(image):
    return normalize_image(image, PIXEL_MEAN, PIXEL_STD)
