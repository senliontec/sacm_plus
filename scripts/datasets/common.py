"""Shared helpers for the per-dataset preparation scripts.

Each ``prepare_<dataset>.py`` script writes the same layout:

    <out>/images/
    <out>/masks/          single-channel PNG, 0/255
    <out>/fov/            optional field-of-view masks
    <out>/manifest.csv    source_image,image,mask,fov,fg_ratio

The scripts write directly to a final role path by default (for example
``datasets/seen/DRIVE``).  An existing images/masks directory is never
replaced unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT.parent / 'datasets'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.ppm'}
MASK_EXTS = IMAGE_EXTS | {'.gif'}
MANIFEST_FIELDS = ['source_image', 'image', 'mask', 'fov', 'fg_ratio']


def build_parser(description: str, src_name: str, role_path: str,
                 extra=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--src', type=Path, default=RAW_ROOT / src_name,
        help=f'raw dataset directory (default: {RAW_ROOT / src_name})',
    )
    parser.add_argument(
        '--out', type=Path, default=REPO_ROOT / 'datasets' / role_path,
        help=f'prepared output directory (default: datasets/{role_path})',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='replace existing images/, masks/, fov/ and manifest.csv',
    )
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='only prepare the first N pairs (debug/smoke use)',
    )
    if extra is not None:
        extra(parser)
    return parser


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f'{label} not found: {path}')
    return path


def prepare_output(out: Path, overwrite: bool) -> Path:
    out = out.resolve()
    existing = any(
        (out / sub).is_dir() and any((out / sub).iterdir())
        for sub in ('images', 'masks', 'fov')
    )
    if existing and not overwrite:
        raise SystemExit(
            f'{out} already contains prepared data; pass --overwrite to replace it'
        )
    if existing:
        for sub in ('images', 'masks', 'fov'):
            path = out / sub
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    manifest = out / 'manifest.csv'
    if manifest.exists():
        manifest.unlink()
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    return out


def iter_files(directory: Path, exts=IMAGE_EXTS):
    if not directory.is_dir():
        return []
    return [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


def index_by_stem(directory: Path, exts=IMAGE_EXTS) -> dict[str, Path]:
    return {p.stem: p for p in iter_files(directory, exts)}


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_rgb_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert('RGB').save(dst)


def save_binary_mask(src: Path, dst: Path, threshold: int = 0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = open_image(src) if src.suffix.lower() == '.gz' else Image.open(src)
    arr = np.array(image.convert('L'))
    Image.fromarray(((arr > threshold) * 255).astype(np.uint8)).save(dst)


def mask_foreground_ratio(path: Path) -> float:
    arr = np.array(Image.open(path).convert('L'))
    if arr.size == 0:
        return float('nan')
    return float((arr > 0).mean())


def validate_vessel_mask(path: Path, lo: float = 0.005,
                         hi: float = 0.30) -> float:
    ratio = mask_foreground_ratio(path)
    if not np.isfinite(ratio) or not (lo <= ratio <= hi):
        raise SystemExit(
            f'{path}: foreground ratio {ratio:.1%} outside vessel range '
            f'{lo:.1%}-{hi:.1%}'
        )
    return ratio


def open_image(path: Path):
    if path.suffix.lower() == '.gz':
        with gzip.open(path, 'rb') as f:
            return Image.open(io.BytesIO(f.read()))
    return Image.open(path)


def add_row(rows: list[dict], src_image: Path, image: Path, mask: Path,
            fov: Path | None = None) -> None:
    rows.append({
        'source_image': str(src_image),
        'image': image.name,
        'mask': mask.name,
        'fov': fov.name if fov is not None else '',
        'fg_ratio': f'{mask_foreground_ratio(mask):.6f}',
    })


def finish(out: Path, rows: list[dict], tag: str) -> int:
    if not rows:
        raise SystemExit(f'[{tag}] no image/mask pairs were produced')
    with open(out / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'[{tag}] {len(rows)} pairs -> {out}')
    return 0


def limit(items, max_images: int | None):
    if max_images is None:
        return items
    if max_images < 0:
        raise SystemExit('--max-images must be >= 0')
    return items[:max_images]


def prepare_ready(src: Path, out: Path, overwrite: bool,
                  max_images: int | None, tag: str) -> int:
    """Copy an already-prepared images/ + masks/ dataset, binarizing masks."""
    src_images = require_dir(src / 'images', f'{tag} images')
    src_masks = require_dir(src / 'masks', f'{tag} masks')
    out = prepare_output(out, overwrite)
    image_index = index_by_stem(src_images)
    mask_index = index_by_stem(src_masks)
    rows = []
    for stem in limit(sorted(set(image_index) & set(mask_index)), max_images):
        image = image_index[stem]
        image_dst = out / 'images' / image.name
        mask_dst = out / 'masks' / f'{stem}.png'
        copy_image(image, image_dst)
        # The in-house TYRE/WIRE masks are anti-aliased RGB.  The frozen
        # protocol binarizes them at 0.5 (>127), not at >0, so low-intensity
        # interpolation fringe is not treated as foreground.
        save_binary_mask(mask_index[stem], mask_dst, threshold=127)
        add_row(rows, image, image_dst, mask_dst)
    return finish(out, rows, tag)
