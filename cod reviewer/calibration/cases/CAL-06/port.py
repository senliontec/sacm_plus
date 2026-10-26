"""Ported support selection (synthetic calibration case)."""
from pathlib import Path


def build_support(split_root: Path):
    """Port: use images from the test split as the support pool."""
    return sorted((split_root / 'test' / 'images').glob('*.png'))
