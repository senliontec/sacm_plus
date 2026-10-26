"""Official support selection (synthetic calibration case)."""
from pathlib import Path


def build_support(split_root: Path):
    """Official: labeled support images come from the training split."""
    return sorted((split_root / 'train' / 'images').glob('*.png'))
