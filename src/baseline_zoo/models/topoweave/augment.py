"""Deterministic augmentation for the pooled 20-domain TopoWeave task."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image


def _center_crop_or_pad(image: Image.Image, label: Image.Image,
                        width: int, height: int):
    def place(source: Image.Image, fill):
        if source.width > width:
            left = (source.width - width) // 2
            source = source.crop((left, 0, left + width, source.height))
        if source.height > height:
            top = (source.height - height) // 2
            source = source.crop((0, top, source.width, top + height))
        canvas = Image.new(source.mode, (width, height), fill)
        left = (width - source.width) // 2
        top = (height - source.height) // 2
        canvas.paste(source, (left, top))
        return canvas

    return (place(image, (0, 0, 0)), place(label, 0))


def augment_pair(image: Image.Image, label: Image.Image,
                 rng: random.Random, strong: bool = True):
    """Return augmented image/label with identical geometry."""
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)

    if not strong:
        return image, label

    angle = rng.uniform(-30.0, 30.0)
    image = image.rotate(
        angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    label = label.rotate(
        angle, resample=Image.NEAREST, expand=False, fillcolor=0)

    scale = rng.uniform(0.9, 1.1)
    if abs(scale - 1.0) > 1e-3:
        width, height = image.size
        new_size = (max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))))
        image = image.resize(new_size, resample=Image.BILINEAR)
        label = label.resize(new_size, resample=Image.NEAREST)
        image, label = _center_crop_or_pad(image, label, width, height)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)
    mean = pixels.mean()
    pixels = (pixels - mean) * contrast + mean
    pixels = pixels * brightness
    gamma = rng.uniform(0.8, 1.2)
    pixels = np.clip(pixels, 0.0, 1.0) ** gamma
    noise_rng = np.random.default_rng(rng.randrange(0, 2 ** 32))
    pixels = pixels + noise_rng.normal(0.0, 0.01, size=pixels.shape)
    pixels = np.clip(pixels, 0.0, 1.0)
    image = Image.fromarray(
        np.uint8(np.round(pixels * 255.0)), mode='RGB')
    return image, label
