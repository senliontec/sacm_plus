"""Build the few-shot training split for the curvilinear protocol.

Layout expected by SegmentationDataset (train_sam.py):

    <out_dir>/train/images/*  +  <out_dir>/train/masks/*
    <out_dir>/val/images/*    +  <out_dir>/val/masks/*

Each source dataset directory must contain images/ and masks/ subdirs
(the Curvilinear_Structure_Datasets layout). Test datasets keep their own
layout and are passed directly to test_sam.py --data_root.

The split is recorded in <out_dir>/split_manifest.csv for reproducibility.

Usage:
    python prepare_data.py \
        --train_dirs <d1> <d2> <d3> <d4> <d5> <d6> \
        --shots 3 --val_shots 1 \
        --out_dir data/sacm_3shot --seed 42
"""

import argparse
import csv
import os
import random
import re
import shutil

SUPPORTED = ('.png', '.jpg', '.jpeg', '.tif', '.tiff',
             '.bmp', '.ppm', '.pgm')

# Built-in patient/volume patterns for the local SACM-protocol datasets.
# --group_regex NAME:REGEX overrides these.
DEFAULT_GROUP_REGEXES = {
    'CHASEDB1': r'Image_(\d+)[LR]',
    'CREMI': r'sample_([ABC])',
}


def find_pairs(src_dir):
    """List (image_path, mask_path) pairs in a dataset directory."""
    image_dir = os.path.join(src_dir, 'images')
    mask_dir = os.path.join(src_dir, 'masks')
    pairs = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.lower().endswith(SUPPORTED):
            base = os.path.splitext(fn)[0]
            for ext in SUPPORTED:
                mask_path = os.path.join(mask_dir, base + ext)
                if os.path.exists(mask_path):
                    pairs.append((os.path.join(image_dir, fn), mask_path))
                    break
    return pairs


def parse_group_regexes(specs):
    """Parse NAME:REGEX entries, e.g. CHASEDB1:Image_(\\d+)[LR]."""
    out = {}
    for spec in specs or []:
        if ':' not in spec:
            raise ValueError(f"--group_regex entries must be NAME:REGEX, got {spec!r}")
        name, regex = spec.split(':', 1)
        if not name or not regex:
            raise ValueError(f"empty name/regex in --group_regex {spec!r}")
        out[name] = regex
    return out


def group_pairs(pairs, dataset, regex=None):
    """Group image pairs by patient/volume ID parsed from the image stem.

    Without a regex every image is its own group (backward-compatible
    image-level split, which can leak patients). With a regex the first
    capture group is the group ID; a non-match is a hard error so a typo
    cannot silently disable grouping.
    """
    if regex is None:
        return {f'{dataset}/__img_{i:06d}': [p]
                for i, p in enumerate(pairs)}
    try:
        pat = re.compile(regex)
    except re.error as e:
        raise ValueError(f"invalid group regex {regex!r}: {e}") from e
    if pat.groups < 1:
        raise ValueError(f"group regex {regex!r} needs one capture group")
    groups = {}
    for img, mask in pairs:
        stem = os.path.splitext(os.path.basename(img))[0]
        m = pat.search(stem)
        if not m:
            raise ValueError(
                f"{dataset}: group regex {regex!r} did not match {stem!r}; "
                "check the --group_regex pattern")
        key = f'{dataset}/{m.group(1)}'
        groups.setdefault(key, []).append((img, mask))
    return groups


def select_group_split(groups, shots, val_shots):
    """Pick exactly shots/val_shots images with disjoint group IDs."""
    keys = list(groups)
    random.shuffle(keys)
    train, val = [], []
    used = set()
    for key in keys:
        if len(train) >= shots:
            break
        for img, mask in groups[key]:
            if len(train) >= shots:
                break
            train.append((img, mask, key))
            used.add(key)
    for key in keys:
        if len(val) >= val_shots:
            break
        if key in used:
            continue
        for img, mask in groups[key]:
            if len(val) >= val_shots:
                break
            val.append((img, mask, key))
    if len(train) < shots or len(val) < val_shots:
        raise ValueError(
            f"not enough disjoint groups: train={len(train)}/{shots}, "
            f"val={len(val)}/{val_shots}; add groups or reduce shots")
    return train, val


def main():
    parser = argparse.ArgumentParser(description='Build the few-shot (or full) training split')
    parser.add_argument('--train_dirs', nargs='+', required=True,
                        help='Source dataset dirs (each with images/ and masks/); '
                             '3-shot per dataset is sampled from the first 6')
    parser.add_argument('--shots', type=int, default=3, help='Train images per dataset (few-shot mode)')
    parser.add_argument('--val_shots', type=int, default=1, help='Val images per dataset (few-shot mode)')
    parser.add_argument('--use_all', action='store_true',
                        help='全量模式(架构研究层): 不做 few-shot 采样, 按 --val_ratio 切 train/val')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='全量模式下验证集比例(默认 0.2)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output root')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--group_regex', action='append', default=[],
                        help='NAME:REGEX (repeatable, one capture group), e.g. '
                             'CHASEDB1:Image_(\\d+)[LR] or CREMI:sample_([ABC])')
    parser.add_argument('--clean', action='store_true',
                        help='Clear existing train/val dirs before writing')
    parser.add_argument('--make_test', action='store_true',
                        help='Copy all unused groups to test/ for Seen evaluation '
                             '(group-disjoint from train/val)')
    args = parser.parse_args()

    random.seed(args.seed)
    group_regexes = parse_group_regexes(args.group_regex)
    if not group_regexes:
        print("[info] --group_regex not set; using built-in CHASEDB1/CREMI "
              "patterns; other datasets are image-level")
    if args.use_all and args.make_test:
        print("[warn] --use_all ignores --make_test "
              "(full mode has no held-out Seen split)")
    if args.clean:
        for split in ('train', 'val'):
            d = os.path.join(args.out_dir, split)
            if os.path.isdir(d):
                shutil.rmtree(d)
    split_names = ['train', 'val'] + (['test'] if args.make_test else [])
    for split in split_names:
        os.makedirs(os.path.join(args.out_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, split, 'masks'), exist_ok=True)

    manifest = []
    for src in args.train_dirs:
        name = os.path.basename(os.path.normpath(src))
        pairs = find_pairs(src)
        if not pairs:
            raise ValueError(f"No image-mask pairs found in {src}")
        regex = group_regexes.get(name, DEFAULT_GROUP_REGEXES.get(name))
        if args.use_all:
            n_val = max(1, int(len(pairs) * args.val_ratio))
            if regex is None:
                random.shuffle(pairs)
                subsets = [
                    ('train', pairs[n_val:]),
                    ('val', pairs[:n_val]),
                ]
            else:
                # Keep whole patients/volumes on one side even in full mode.
                groups = group_pairs(pairs, name, regex)
                keys = list(groups)
                random.shuffle(keys)
                val_items, train_items = [], []
                for key in keys:
                    target = val_items if len(val_items) < n_val else train_items
                    for img, mask in groups[key]:
                        target.append((img, mask, key))
                subsets = [('train', train_items), ('val', val_items)]
        else:
            if regex is None:
                print(f"[warn] {name}: no group regex; image-level split "
                      "may leak patients/volumes into val")
            groups = group_pairs(pairs, name, regex)
            train, val = select_group_split(
                groups, args.shots, args.val_shots)
            subsets = [('train', train), ('val', val)]
            if args.make_test:
                used = {g for _, _, g in train} | {g for _, _, g in val}
                test = [(img, mask, key)
                        for key, group in groups.items() if key not in used
                        for img, mask in group]
                subsets.append(('test', test))
        for split, subset in subsets:
            for item in subset:
                img_path, mask_path = item[0], item[1]
                group = item[2] if len(item) > 2 else ''
                img_ext = os.path.splitext(img_path)[1]
                mask_ext = os.path.splitext(mask_path)[1]
                # Prefix the dataset name to avoid collisions across sources
                new_base = f"{name}_{os.path.splitext(os.path.basename(img_path))[0]}"
                new_img = os.path.join(args.out_dir, split, 'images', new_base + img_ext)
                new_mask = os.path.join(args.out_dir, split, 'masks', new_base + mask_ext)
                shutil.copy2(img_path, new_img)
                shutil.copy2(mask_path, new_mask)
                manifest.append({
                    'dataset': name,
                    'split': split,
                    'source_image': os.path.basename(img_path),
                    'output': new_base,
                    'group': group,
                })
                print(f"[{split}] {name}: {new_base} (group={group})")

    manifest_path = os.path.join(args.out_dir, 'split_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['dataset', 'split', 'source_image', 'output', 'group'])
        writer.writeheader()
        writer.writerows(manifest)

    counts = {}
    for m in manifest:
        counts[m['split']] = counts.get(m['split'], 0) + 1
    print(f"Done: {counts.get('train', 0)} train / {counts.get('val', 0)} val"
          f" / {counts.get('test', 0)} test images -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
