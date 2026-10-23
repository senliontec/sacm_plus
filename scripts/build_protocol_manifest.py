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
                 'CHCrack5k', 'OCTA-500']
ACTIVE_NOVEL = ['ROAD', 'LEAF', 'TYRE', 'WIRE', 'DeepGlobe', 'TopoMortar']
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')

# Group semantics for the training domains.  Subject-level groups must not
# cross train/val/test, and train may contain at most one image per group.
# Image-level datasets have no public patient ID; each image is its own group
# and this limitation is recorded in the summary.
GROUP_POLICIES = {
    'CHASEDB1': 'subject',
    'CORN': 'batch',
    'CREMI': 'volume',
    'DRIVE': 'image',
    'DCA1': 'image',
    'CrackTree200': 'image',
}
# CREMI has only A/B/C; requiring three distinct train volumes plus val and
# test volumes is impossible.  Train therefore uses several B slices while
# val uses A and test uses C.  The exception is explicit and checked.
ALLOWED_TRAIN_REPEATS = {
    'CREMI': 'only three volumes A/B/C; train B slices, val A, test C',
}

# Frozen protocol cardinalities.  A missing/changed pool must fail the build
# instead of silently shrinking the manifest or accepting DeepGlobe --full.
EXPECTED_TEST_COUNTS = {
    'DRIVE': 16, 'CHASEDB1': 20, 'DCA1': 130, 'CREMI': 123,
    'CrackTree200': 202, 'CORN': 152,
    'FIVES': 200, 'STARE': 20, 'XCAD': 126, 'DeepCrack': 237,
    'HRF': 45, 'LES-AV': 22, 'CHCrack5k': 990, 'OCTA-500': 300,
    'ROAD': 49, 'LEAF': 31, 'TYRE': 36, 'WIRE': 31,
    'DeepGlobe': 500, 'TopoMortar': 350,
}
EXPECTED_TRAIN_COUNTS = {
    'DRIVE': 3, 'CHASEDB1': 3, 'DCA1': 3, 'CREMI': 3,
    'CrackTree200': 3, 'CORN': 3,
}
EXPECTED_VAL_COUNTS = {
    'DRIVE': 1, 'CHASEDB1': 1, 'DCA1': 1, 'CREMI': 1,
    'CrackTree200': 1, 'CORN': 1,
}


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
                mask_dir = root / 'test' / dataset / 'masks'
            else:
                base = root / split / 'images'
                mask_dir = root / split / 'masks'
            matches = sorted(base.glob(stem + '.*'))
            if not matches:
                raise SystemExit(f'missing image for manifest row {item}')
            image = matches[0]
            mask = next(
                (mask_dir / f'{stem}{ext}' for ext in IMAGE_EXTS
                 if (mask_dir / f'{stem}{ext}').is_file()),
                None)
            if mask is None:
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
            raise SystemExit(
                f'missing active external dataset: {base}; '
                'prepare it instead of silently shrinking the protocol')
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


def validate_group_policy(rows):
    """Validate subject/volume/batch isolation for the Seen splits.

    Rules:
      * a group may not appear in more than one Seen split;
      * train may contain at most one image per group, except for datasets
        listed in ALLOWED_TRAIN_REPEATS (CREMI's three-volume constraint).
    External Unseen/Novel rows are test-only, so repeated groups there do
    not create train leakage and are not counted.
    """
    seen = [row for row in rows if row['role'] == 'seen']
    group_splits = {}
    train_counts = {}
    for row in seen:
        key = (row['dataset'], row['group'])
        group_splits.setdefault(key, set()).add(row['split'])
        if row['split'] == 'train':
            train_counts[key] = train_counts.get(key, 0) + 1

    cross_split = [
        {'dataset': dataset, 'group': group, 'splits': sorted(splits)}
        for (dataset, group), splits in sorted(group_splits.items())
        if len(splits) > 1
    ]
    duplicates = []
    unexpected = []
    for (dataset, group), count in sorted(train_counts.items()):
        if count <= 1:
            continue
        item = {'dataset': dataset, 'group': group, 'count': count}
        duplicates.append(item)
        if dataset not in ALLOWED_TRAIN_REPEATS:
            unexpected.append(item)

    datasets = sorted({row['dataset'] for row in rows})
    policies = {dataset: GROUP_POLICIES.get(dataset, 'image')
                for dataset in datasets}
    return {
        'ok': not cross_split and not unexpected,
        'group_policy': policies,
        'cross_split_groups': cross_split,
        'train_duplicate_groups': duplicates,
        'unexpected_train_duplicate_groups': unexpected,
        'allowed_exceptions': dict(ALLOWED_TRAIN_REPEATS),
    }


def validate_expected_counts(rows):
    """Fail if a pool silently shrank/grew relative to the frozen protocol."""
    got_train = Counter(row['dataset'] for row in rows
                        if row['split'] == 'train')
    got_val = Counter(row['dataset'] for row in rows
                      if row['split'] == 'val')
    got_test = Counter(row['dataset'] for row in rows
                       if row['split'].endswith('_test'))
    errors = []

    def compare(label, expected, got):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        changed = sorted(
            dataset for dataset in set(expected) & set(got)
            if expected[dataset] != got[dataset])
        if missing:
            errors.append(f'{label}: missing datasets {missing}')
        if extra:
            errors.append(f'{label}: unexpected datasets {extra}')
        if changed:
            errors.append(
                f'{label}: count mismatch ' +
                ', '.join(f'{d}={got[d]}(expected {expected[d]})'
                          for d in changed))

    compare('train', EXPECTED_TRAIN_COUNTS, got_train)
    compare('val', EXPECTED_VAL_COUNTS, got_val)
    compare('test', EXPECTED_TEST_COUNTS, got_test)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path,
                        default=Path('data/sacm_3shot'))
    parser.add_argument('--datasets-root', type=Path, default=Path('datasets'))
    parser.add_argument('--version', default=None,
                        help='protocol_version (default: split root name)')
    parser.add_argument('--seed', type=int, default=42,
                        help='split seed recorded in the summary')
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

    count_errors = validate_expected_counts(rows)
    checks = validate_group_policy(rows)
    if count_errors or not checks['ok']:
        for error in count_errors:
            print(f'[error] {error}')
        if not checks['ok']:
            print(f"  cross-split groups: {checks['cross_split_groups']}")
            print(f"  unexpected train duplicates: "
                  f"{checks['unexpected_train_duplicate_groups']}")
        print('[error] refusing to overwrite the frozen manifest')
        return 2

    manifest_path = root / 'protocol_manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r['role'], r['split'], r['dataset']) for r in rows)
    summary = {
        'protocol_version': version,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
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
        'group_policy': checks['group_policy'],
        'group_checks': checks,
    }
    summary_path = root / 'protocol_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(f'{len(rows)} rows -> {manifest_path}')
    print(f'summary -> {summary_path}')
    print(f"group check: {'ok' if checks['ok'] else 'FAILED'}")
    if not checks['ok']:
        print(f"  cross-split groups: {checks['cross_split_groups']}")
        print(f"  unexpected train duplicates: "
              f"{checks['unexpected_train_duplicate_groups']}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
