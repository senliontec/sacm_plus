"""MaskVSC masked-vessel corruption, extracted verbatim from upstream.

Source: ``baselines/maskvsc/lib/datasets.py`` (``ImageDataset_Fundus``).
The only addition is an early return when the skeleton has no pixels; the
upstream code would raise inside ``skan`` for an empty mask.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from skan import Skeleton, csr
from skimage import morphology


def mask_vessel_corrupt(image: np.ndarray, label: np.ndarray,
                        ratio: float) -> np.ndarray:
    if ratio <= 0:
        return image
    skeleton = morphology.thin(label).astype(np.uint8)
    if skeleton.sum() == 0:
        return image
    skeleton_copy = skeleton.copy()
    graph = Skeleton(skeleton)
    branch_data = csr.summarize(graph)
    length = branch_data.shape[0]
    count = int(ratio * length)
    if count:
        deleted = random.sample(range(0, length), count)
        for index in deleted:
            coords = tuple(
                graph.path_coordinates(index)[1:-1].T.astype(int))
            skeleton[coords] = False
    removed = skeleton_copy - skeleton
    kernel = np.ones((7, 7), np.uint8)
    masked = cv2.dilate(removed, kernel, iterations=1) * 255

    height, width = label.shape
    gauss_r = np.random.normal(0, 20, (height, width))
    gauss_r = np.where(masked == 255, gauss_r, masked)
    gauss_g = np.random.normal(0, 20, (height, width))
    gauss_g = np.where(masked == 255, gauss_g, masked)
    gauss_b = np.random.normal(0, 20, (height, width))
    gauss_b = np.where(masked == 255, gauss_b, masked)

    mask_rgb = np.zeros((height, width, 3))
    gauss_rgb = np.zeros((height, width, 3))
    mask_rgb[:, :, 0] = masked
    mask_rgb[:, :, 1] = masked
    mask_rgb[:, :, 2] = masked
    gauss_rgb[:, :, 0] = gauss_r
    gauss_rgb[:, :, 1] = gauss_g
    gauss_rgb[:, :, 2] = gauss_b

    result = np.uint8(np.add(255 - image, mask_rgb))
    result = np.uint8(np.add(result, gauss_rgb))
    return 255 - result
