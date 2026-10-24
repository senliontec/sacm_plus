"""Raw protocol data shared by all baseline adapters.

Adapters load images as original-resolution PIL objects so each method can
apply its own official resize/colour/normalization.  Masks are always read
as boolean arrays at the original GT resolution; predictions are compared at
that resolution regardless of the model's internal input size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    dataset: str
    stem: str


def list_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, mask)`` pairs under ``root/{images,masks}``."""
    image_dir, mask_dir = root / 'images', root / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f'expected images/ and masks/ under {root}')
    masks = {path.stem: path for path in sorted(mask_dir.iterdir())
             if path.suffix.lower() in IMAGE_EXTS}
    pairs = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        mask = masks.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))
    if not pairs:
        raise SystemExit(f'no image/mask pairs under {root}')
    return pairs


def dataset_of(image_path: Path) -> str:
    """Protocol convention: ``<DATASET>_<case>`` -> ``<DATASET>``."""
    return image_path.stem.split('_', 1)[0]


class RawPairDataset(Dataset):
    """Original-resolution image/mask pairs with no preprocessing."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.samples = [
            Sample(image, mask, dataset_of(image), image.stem)
            for image, mask in list_pairs(self.root)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    @property
    def datasets(self) -> list[str]:
        return sorted({sample.dataset for sample in self.samples})


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask:
        return np.asarray(mask.convert('L')) > 127


def sample_specs(root: Path) -> list[Sample]:
    return RawPairDataset(root).samples
