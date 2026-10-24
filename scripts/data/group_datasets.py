#!/usr/bin/env python3
"""Group flat organized datasets into seen/unseen/novel/legacy directories.

`organize_datasets.py` keeps its historical flat output (one directory per
source dataset). Run this script afterwards to get the protocol layout:

    datasets/seen/{DRIVE,CHASEDB1,DCA1,CREMI,CrackTree200}
    datasets/unseen/{FIVES,STARE,XCAD,DeepCrack,HRF,LES-AV,CHCrack5k}
    datasets/novel/{ROAD,TYRE,WIRE}
    datasets_legacy/{...}

The TYRE/WIRE sources are RGB with anti-aliased masks; this script writes
single-channel 0/255 masks for them.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_GROUPS = {
    'DRIVE_train': 'seen/DRIVE',
    'CHASEDB1': 'seen/CHASEDB1',
    'DCA1': 'seen/DCA1',
    'CREMI': 'seen/CREMI',
    'CrackTree200': 'seen/CrackTree200',
    'FIVES_test': 'unseen/FIVES',
    'STARE': 'unseen/STARE',
    'XCAD_test': 'unseen/XCAD',
    'DeepCrack_test': 'unseen/DeepCrack',
    'HRF': 'unseen/HRF',
    'LES-AV': 'unseen/LES-AV',
    'CHCrack5k_test': 'unseen/CHCrack5k',
    'ROAD_test': 'novel/ROAD',
}

LEGACY_NAMES = (
    'DRIVE_test',
    'FIVES_train',
    'DeepCrack_train',
    'DIS5K_train',
    'DIS5K_val',
    'DIS5K_test',
    'ThinObject5K',
)

INHOUSE_NOVEL = {
    'tyre': 'TYRE',
    'wire': 'WIRE',
    'leaf': 'LEAF',
}


def _replace_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _mask_path(mask_dir: Path, stem: str) -> Path | None:
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        p = mask_dir / f'{stem}{ext}'
        if p.is_file():
            return p
    return None


def _copy_ready_binary(src: Path, dst: Path) -> int:
    """Copy a ready dataset and binarize anti-aliased masks."""
    image_dir = src / 'images'
    mask_dir = src / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"ready dataset missing images/ or masks/: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'images').mkdir(parents=True)
    (dst / 'masks').mkdir(parents=True)
    n = 0
    for name in sorted(os.listdir(image_dir)):
        image_path = image_dir / name
        if not image_path.is_file():
            continue
        stem = Path(name).stem
        mask_path = _mask_path(mask_dir, stem)
        if mask_path is None:
            continue
        shutil.copy2(image_path, dst / 'images' / name)
        arr = np.array(Image.open(mask_path).convert('L'))
        binary = ((arr > 127) * 255).astype(np.uint8)
        Image.fromarray(binary).save(dst / 'masks' / f'{stem}.png')
        n += 1
    return n


def group_datasets(root: Path, legacy_root: Path) -> dict:
    summary = {'moved': 0, 'legacy': 0, 'novel': 0}
    for source_name, relative in DATASET_GROUPS.items():
        src = root / source_name
        if not src.is_dir():
            continue
        _replace_move(src, root / relative)
        print(f"[group] {source_name} -> {relative}")
        summary['moved'] += 1

    for name in LEGACY_NAMES:
        src = root / name
        if not src.is_dir():
            continue
        _replace_move(src, legacy_root / name)
        print(f"[group] {name} -> {legacy_root / name}")
        summary['legacy'] += 1

    for source_name, target_name in INHOUSE_NOVEL.items():
        src = root / source_name
        if not src.is_dir():
            continue
        dst = root / 'novel' / target_name
        n = _copy_ready_binary(src, dst)
        shutil.rmtree(src)
        print(f"[group] {source_name} -> novel/{target_name} ({n} pairs)")
        summary['novel'] += 1

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Group flat dataset roots into seen/unseen/novel/legacy.'
    )
    parser.add_argument(
        '--root', default='datasets',
        help='organized dataset root (default: datasets)',
    )
    parser.add_argument(
        '--legacy-root', default=None,
        help='destination for legacy datasets (default: <root>/../datasets_legacy)',
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    legacy_root = (
        Path(args.legacy_root).resolve()
        if args.legacy_root
        else root.parent / 'datasets_legacy'
    )
    if not root.is_dir():
        raise SystemExit(f"dataset root does not exist: {root}")
    summary = group_datasets(root, legacy_root)
    print(
        f"[group] done: moved={summary['moved']} "
        f"legacy={summary['legacy']} novel={summary['novel']}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
