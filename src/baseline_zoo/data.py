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


class SampleListDataset(Dataset):
    """Minimal Dataset over an explicit, already-validated sample list."""

    def __init__(self, samples: list[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    @property
    def datasets(self) -> list[str]:
        return sorted({sample.dataset for sample in self.samples})


def load_split_samples(root: Path, split: str,
                       require_manifest: bool = True) -> list[Sample]:
    """Load a train/val split and verify it against ``split_manifest.csv``.

    Directory-only loading silently changes the 18/6 protocol when a stray
    image or missing mask appears.  The manifest's ``output`` column is the
    authoritative membership list; every expected stem must exist, and no
    extra pair may be present.
    """
    import csv

    root = Path(root)
    dataset = RawPairDataset(root / split)
    manifest = root / 'split_manifest.csv'
    if not manifest.is_file():
        if require_manifest:
            raise SystemExit(
                f'missing split manifest: {manifest}; refusing to infer '
                f'the {split} membership from directory contents')
        return dataset.samples

    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = [row['output'] for row in rows if row.get('split') == split]
    if not expected:
        raise SystemExit(
            f'{manifest}: no rows with split={split!r}')
    if len(expected) != len(set(expected)):
        raise SystemExit(f'{manifest}: duplicate {split} output stems')

    found = {sample.stem: sample for sample in dataset.samples}
    missing = [stem for stem in expected if stem not in found]
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f'missing {len(missing)} manifest rows '
                           f'(first: {missing[0]})')
        if extra:
            details.append(f'{len(extra)} unlisted pairs '
                           f'(first: {extra[0]})')
        raise SystemExit(
            f'{root / split} does not match {manifest}: ' + '; '.join(details))
    return [found[stem] for stem in expected]
