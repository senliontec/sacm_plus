"""Local Intensity Order Transformation (LIOT).

The official implementation in ``3rd/LIOT/LIOT.py`` walks every pixel with
numba.  This module is a vectorised NumPy translation of
``distance_weight_binary_pattern_faster``; the bit ordering and the four
direction channels are unchanged.  The only functional additions are:

* a grayscale fallback for non-RGB inputs (the release assumes RGB); and
* an explicit polarity switch.  LIOT assumes the curvilinear structure is
  darker than its surroundings.  For datasets whose foreground is bright,
  the adapter inverts the green/gray channel before applying the transform
  and uses the same switch at inference time.
"""

from __future__ import annotations

import numpy as np

# Foreground brighter than local background in the protocol's datasets.  The
# table is a manually curated task adaptation (retinal vessels dark; cracks/
# roads/wires bright); global mask means are misleading on fundus images
# because the dark outside-FOV border inflates the background.  It is not
# fitted to test labels and can be overridden with --liot-polarity.
BRIGHT_FOREGROUND_DATASETS = frozenset({
    'CORN', 'DeepGlobe', 'ROAD', 'TopoMortar', 'WIRE',
})

DARK_FOREGROUND_DATASETS = frozenset({
    'CREMI', 'CrackTree200', 'DCA1', 'CHCrack5k', 'DeepCrack', 'LEAF',
    'TYRE', 'XCAD',
})


def polarity_for_dataset(dataset: str, mode: str = 'auto') -> bool:
    """Return whether the LIOT input should be inverted for ``dataset``.

    ``mode='auto'`` uses the protocol table above; ``mode='dark'`` disables
    inversion and ``mode='bright'`` always inverts.  Unknown datasets follow
    the paper's default (dark foreground) and are therefore not inverted.
    """
    if mode == 'dark':
        return False
    if mode == 'bright':
        return True
    if mode != 'auto':
        raise ValueError(f'unknown LIOT polarity mode: {mode!r}')
    return dataset in BRIGHT_FOREGROUND_DATASETS


def _gray_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        # Official LIOT uses the green channel for RGB retinal images.  The
        # same index is green in both RGB and BGR arrays, so this is stable
        # regardless of the loader's channel order.
        return image[:, :, 1]
    raise ValueError(
        f'LIOT expects a 2-D or HxWx3+ image, got {image.shape}')


def local_intensity_order(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Apply the official 4-channel LIOT transform.

    Parameters
    ----------
    image:
        ``HxW`` or ``HxWxC`` integer/float array.  For colour inputs the green
        channel is used, matching ``3rd/LIOT/LIOT.py``.
    invert:
        Invert the gray channel before computing the intensity order.  Use
        this for bright curvilinear structures.

    Returns
    -------
    ``HxWx4`` uint8 array with the right/left/up/down 8-bit intensity-order
    codes in channels 0..3.
    """
    array = np.asarray(image)
    gray = _gray_channel(array)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if invert:
        gray = 255 - gray

    height, width = gray.shape
    padded = np.pad(gray, ((8, 8), (8, 8)), mode='constant', constant_values=0)
    codes = np.zeros((height, width, 4), dtype=np.uint8)

    # Direction order is the release's: 0=right/down, 1=left/up,
    # 2=up/right, 3=down/left.  Bits 0..7 correspond to the eight neighbours
    # at distances 1..8; the slice offsets below reproduce the official
    # padded-array indexing exactly.
    for direction in range(4):
        accumulator = np.zeros((height, width), dtype=np.uint16)
        for position in range(8):
            if direction == 0:
                neighbour = padded[position + 9:position + 9 + height,
                                   8:8 + width]
            elif direction == 1:
                neighbour = padded[position:position + height,
                                   8:8 + width]
            elif direction == 2:
                neighbour = padded[8:8 + height,
                                   position + 9:position + 9 + width]
            else:
                neighbour = padded[8:8 + height,
                                   position:position + width]
            accumulator += (
                (gray > neighbour).astype(np.uint16) << position)
        codes[:, :, direction] = accumulator.astype(np.uint8)
    return codes
