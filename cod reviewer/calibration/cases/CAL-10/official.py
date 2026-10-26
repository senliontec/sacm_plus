"""Official model loading (synthetic calibration case)."""
from pathlib import Path

CHECKPOINT = Path('assets/official.pth')


def build_model(loader):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    return loader(CHECKPOINT)
