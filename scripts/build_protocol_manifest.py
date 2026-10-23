#!/usr/bin/env python3
"""Build protocol_manifest.csv / protocol_summary.json for a frozen split.

The split root must contain the prepare_data output:
    train/{images,masks}, val/{images,masks}, test/<dataset>/{images,masks},
    split_manifest.csv

External evaluation roots are added from datasets/unseen and datasets/novel
using the active protocol lists below (candidate datasets are excluded).

Usage:
    python scripts/build_protocol_manifest.py \
        --split-root data/sacm_3shot --datasets-root datasets
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = [
    'protocol_version', 'role', 'dataset', 'split', 'source_image',
    'image_path', 'mask_path', 'group', 'width', 'height', 'fg_ratio',
    'image_sha256', 'mask_sha256',
]
ACTIVE_UNSEEN = ['FIVES', 'STARE', 'XCAD', 'DeepCrack', 'HRF', 'LES-AV',
                 'CHCrack5k']
ACTIVE_NOVEL = ['ROAD', 'LEAF', 'TYRE', 'WIRE']
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(version, role, dataset, split, source_image, image, mask, group):
    with Image.open(image) as im:
        width, height = im.size
    arr = np.array(Image.open(mask).convert('L'))
    return {
        'protocol_version': version,
        'role': role,
        'dataset': dataset,
        'split': split,
        'source_image': source_image,
        'image_path': str(image),
        'mask_path': str(mask),
        'group': group,
        'width': width,
        'height': height,
        'fg_ratio': f'{float((arr > 0).mean()):.6f}',
        'image_sha256': sha256(image),
        'mask_sha256': sha256(mask),
    }


def split_rows(root: Path, version: str):
    rows = []
    manifest = root / 'split_manifest.csv'
    with open(manifest) as f:
        for item in csv.DictReader(f):
            split = item['split']
            dataset = item['dataset']
            stem = item['output']
            if split == 'test':
                base = root / 'test' / dataset / 'images'
            else:
                base = root / split / 'images'
            matches = sorted(base.glob(stem + '.*'))
            if not matches:
                raise SystemExit(f'missing image for manifest row {item}')
            image = matches[0]
            mask = (root / split / 'masks' / f'{stem}.png'
                    if split != 'test' else
                    root / 'test' / dataset / 'masks' / f'{stem}.png')
            if not mask.is_file():
                raise SystemExit(f'missing mask for manifest row {item}')
            role = 'seen'
            split_label = {'train': 'train', 'val': 'val',
                           'test': 'seen_test'}[split]
            rows.append(record(version, role, dataset, split_label,
                               item['source_image'], image, mask,
                               item['group']))
    return rows


def external_rows(datasets_root: Path, version: str, role: str, names,
                  split_label: str):
    rows = []
    for name in names:
        base = datasets_root / role / name
        image_dir = base / 'images'
        mask_dir = base / 'masks'
        if not image_dir.is_dir() or not mask_dir.is_dir():
            print(f'[warn] missing external dataset: {base}')
            continue
        masks = {p.stem: p for p in sorted(mask_dir.iterdir())
                 if p.suffix.lower() in IMAGE_EXTS}
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            mask = masks.get(image.stem)
            if mask is None:
                raise SystemExit(f'{name}: no mask for {image}')
            rows.append(record(version, role, name, split_label,
                               image.name, image, mask, name))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path,
                        default=Path('data/sacm_3shot'))
    parser.add_argument('--datasets-root', type=Path, default=Path('datasets'))
    parser.add_argument('--version', default=None,
                        help='protocol_version (default: split root name)')
    args = parser.parse_args()

    # Keep the paths as supplied (relative by default) so the manifest is
    # portable, keeping the supplied relative paths (data/... , datasets/...).
    root = args.split_root
    datasets_root = args.datasets_root
    version = args.version or root.name
    rows = split_rows(root, version)
    rows += external_rows(datasets_root, version, 'unseen', ACTIVE_UNSEEN,
                          'unseen_test')
    rows += external_rows(datasets_root, version, 'novel', ACTIVE_NOVEL,
                          'novel_test')

    manifest_path = root / 'protocol_manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r['role'], r['split'], r['dataset']) for r in rows)
    summary = {
        'protocol_version': version,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': 42,
        'counts': {
            role: {
                split: {
                    dataset: n for (r, s, dataset), n in counts.items()
                    if r == role and s == split
                }
                for split in sorted({s for (r, s, _) in counts if r == role})
            }
            for role in ('seen', 'unseen', 'novel')
        },
        'totals': {
            'rows': len(rows),
            'images': len(rows),
            'unique_datasets': len({r['dataset'] for r in rows}),
        },
        'manifest_sha256': sha256(manifest_path),
    }
    summary_path = root / 'protocol_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(f'{len(rows)} rows -> {manifest_path}')
    print(f'summary -> {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
