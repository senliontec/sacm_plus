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
        --train_dirs <d1> ... <d6> \
        --shots 3 --val_shots 1 \
        --out_dir data/sacm_3shot --seed 42

Official splits (e.g. CORN) can be named and supplied separately:

    python prepare_data.py \
        --train_dirs datasets/seen/DRIVE ... \
                     CORN:datasets/seen/CORN/train \
        --val_dirs CORN:datasets/seen/CORN/val \
        --test_dirs CORN:datasets/seen/CORN/test \
        --shots 3 --val_shots 1 --make_test \
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
    # CORN filenames are batch_<patient>_<index>.tif; the official split is
    # batch-disjoint, and 3-shot sampling must stay within distinct batches.
    'CORN': r'batch_(\d+)_',
}
# CREMI is the documented volume-limited exception.  Pin the assignment so
# the frozen protocol does not silently move val/test to a different volume
# when unrelated sampling code changes.
PREFERRED_GROUP_ORDER = {
    'CREMI': ('CREMI/B', 'CREMI/A', 'CREMI/C'),
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


def parse_named_dirs(specs):
    """Parse ``NAME:DIR`` (or plain ``DIR``) entries into name -> path.

    Named entries let one dataset pool live under a descriptive directory
    (e.g. ``CORN:datasets/seen/CORN/train``) while keeping the protocol name
    ``CORN`` in manifests and output filenames.
    """
    out = {}
    for spec in specs or []:
        if ':' in spec:
            name, path = spec.split(':', 1)
        else:
            path = spec
            name = os.path.basename(os.path.normpath(path))
        if not name or not path:
            raise ValueError(f"invalid NAME:DIR entry {spec!r}")
        if name in out:
            raise ValueError(f"duplicate dataset name {name!r} in {specs!r}")
        out[name] = path
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


def select_group_images(groups, count):
    """Pick exactly ``count`` images, preferring one per distinct group.

    Used when validation comes from an official directory, so the training
    sampler must not reserve a group for validation.  A second pass fills
    from groups only when the pool has fewer groups than requested (e.g.
    CREMI has three volumes); CORN has nine training batches, so its 3-shot
    sample comes from three distinct batches.
    """
    keys = list(groups)
    random.shuffle(keys)
    picked = []
    seen = set()
    # First pass: one image per group.
    for key in keys:
        if len(picked) >= count:
            break
        img, mask = random.choice(groups[key])
        picked.append((img, mask, key))
        seen.add((img, mask))
    # Second pass: fill from groups if the pool has too few groups.
    if len(picked) < count:
        for key in keys:
            for img, mask in groups[key]:
                if len(picked) >= count:
                    break
                if (img, mask) in seen:
                    continue
                picked.append((img, mask, key))
                seen.add((img, mask))
            if len(picked) >= count:
                break
    if len(picked) < count:
        raise ValueError(
            f"not enough groups: picked={len(picked)}/{count}; "
            "add groups or reduce shots")
    return picked


def _legacy_group_split(groups, shots, val_shots, keys):
    """Old image-filling policy, kept for image-level and volume-limited data.

    One group may contribute several train images.  This is correct for
    image-level datasets (each group has one image) and for CREMI, where
    only three volumes exist: train uses slices from one volume, val takes
    another volume, and test takes the remaining volume.
    """
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


def select_group_split(groups, shots, val_shots, preferred_order=None):
    """Pick exactly shots/val_shots images with disjoint group IDs.

    When the pool has enough distinct groups, train gets ``shots`` images
    from distinct groups and val gets ``val_shots`` images from distinct
    groups; the remaining groups are left for Seen test.  This prevents a
    patient's two eyes (CHASEDB1) from both entering the 3-shot train set.

    Image-level datasets (every group has one image) keep the legacy
    selection so their frozen split does not change.  Volume-limited
    datasets such as CREMI also use the legacy fallback: with only A/B/C,
    requiring three train volumes plus a val volume plus a test volume is
    impossible, so train uses several slices from one volume while val and
    test use the other volumes.
    """
    keys = list(groups)
    if preferred_order:
        rank = {key: index for index, key in enumerate(preferred_order)}
        keys.sort(key=lambda key: rank.get(key, len(rank)))
    else:
        random.shuffle(keys)
    all_singletons = all(len(items) == 1 for items in groups.values())
    enough_groups = len(keys) >= shots + val_shots + 1

    if all_singletons or not enough_groups:
        return _legacy_group_split(groups, shots, val_shots, keys)

    # Reserve distinct groups for val and leave the last group for test.
    val = select_group_images(
        {key: groups[key] for key in keys[:val_shots]}, val_shots)
    train_pool = {
        key: groups[key] for key in keys[val_shots:-1]
    }
    train = select_group_images(train_pool, shots)
    return train, val


def main():
    parser = argparse.ArgumentParser(description='Build the few-shot (or full) training split')
    parser.add_argument('--train_dirs', nargs='+', required=True,
                        help='Source dataset roots (each with images/ and masks/). '
                             'Entries may be NAME:DIR to set the protocol name while '
                             'pointing at a split-specific pool (e.g. '
                             'CORN:datasets/seen/CORN/train).')
    parser.add_argument('--val_dirs', nargs='+', default=[],
                        help='Optional NAME:DIR official validation roots; when '
                             'present, val_shots are sampled from this directory '
                             'instead of the training pool.')
    parser.add_argument('--test_dirs', nargs='+', default=[],
                        help='Optional NAME:DIR official test roots; when present '
                             'with --make_test, the official test images are copied '
                             'to test/<NAME>/ instead of the held-out complement.')
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
                        help='Clear existing train/val/test dirs before writing')
    parser.add_argument('--make_test', action='store_true',
                        help='Write Seen test roots. Uses test_dirs official '
                             'splits when provided, otherwise copies all unused '
                             'groups (group-disjoint from train/val).')
    args = parser.parse_args()

    random.seed(args.seed)
    group_regexes = parse_group_regexes(args.group_regex)
    if not group_regexes:
        print("[info] --group_regex not set; using built-in CHASEDB1/CREMI/CORN "
              "patterns; other datasets are image-level")
    train_specs = parse_named_dirs(args.train_dirs)
    val_specs = parse_named_dirs(args.val_dirs)
    test_specs = parse_named_dirs(args.test_dirs)
    unknown = (set(val_specs) | set(test_specs)) - set(train_specs)
    if unknown:
        raise ValueError(
            "--val_dirs/--test_dirs names are not in --train_dirs: "
            f"{sorted(unknown)}")
    if args.use_all and (val_specs or test_specs):
        print("[warn] --use_all ignores --val_dirs/--test_dirs "
              "(full mode has no fixed official split)")
        val_specs, test_specs = {}, {}
    if args.use_all and args.make_test:
        print("[warn] --use_all ignores --make_test "
              "(full mode has no held-out Seen split)")
    if args.clean:
        for split in ('train', 'val', 'test'):
            d = os.path.join(args.out_dir, split)
            if os.path.isdir(d):
                shutil.rmtree(d)
        # Never leave manifests that describe the previous split.
        for name in ('protocol_manifest.csv', 'protocol_summary.json'):
            path = os.path.join(args.out_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    split_names = ['train', 'val']
    for split in split_names:
        os.makedirs(os.path.join(args.out_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, split, 'masks'), exist_ok=True)

    manifest = []
    for name, src in train_specs.items():
        pairs = find_pairs(src)
        if not pairs:
            raise ValueError(f"No image-mask pairs found in {src}")
        regex = group_regexes.get(name, DEFAULT_GROUP_REGEXES.get(name))
        source_dirs = {'train': src, 'val': src, 'test': src}
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
            if name in val_specs:
                # Official validation split: train may use any shots without
                # reserving a group for validation.
                train = select_group_images(groups, args.shots)
                val_pairs = find_pairs(val_specs[name])
                if not val_pairs:
                    raise ValueError(
                        f"No image-mask pairs found in {val_specs[name]}")
                val_groups = group_pairs(val_pairs, name, regex)
                val = select_group_images(val_groups, args.val_shots)
                train_stems = {
                    os.path.splitext(os.path.basename(img))[0]
                    for img, _, _ in train
                }
                overlap = [
                    img for img, _, _ in val
                    if os.path.splitext(os.path.basename(img))[0] in train_stems
                ]
                if overlap:
                    raise ValueError(
                        f"{name}: official val overlaps train: {overlap[:3]}")
                train_groups = {group for _, _, group in train}
                val_groups = {group for _, _, group in val}
                overlap_groups = sorted(train_groups & val_groups)
                if overlap_groups:
                    raise ValueError(
                        f"{name}: official val shares subject/volume groups "
                        f"with train: {overlap_groups[:5]}")
                source_dirs['val'] = val_specs[name]
            else:
                train, val = select_group_split(
                    groups, args.shots, args.val_shots,
                    preferred_order=PREFERRED_GROUP_ORDER.get(name))
            subsets = [('train', train), ('val', val)]
            if args.make_test and name in test_specs:
                test_pairs = find_pairs(test_specs[name])
                if not test_pairs:
                    raise ValueError(
                        f"No image-mask pairs found in {test_specs[name]}")
                used_stems = {
                    os.path.splitext(os.path.basename(img))[0]
                    for _, items in subsets for img, _, _ in items
                }
                test = []
                for img, mask in test_pairs:
                    stem = os.path.splitext(os.path.basename(img))[0]
                    if stem in used_stems:
                        raise ValueError(
                            f"{name}: official test contains a train/val image: {img}")
                    group = f'{name}/official_test'
                    if regex is not None:
                        match = re.search(regex, stem)
                        if match:
                            group = f'{name}/{match.group(1)}'
                    test.append((img, mask, group))
                subsets.append(('test', test))
                source_dirs['test'] = test_specs[name]
                print(f"[info] {name}: using official test split "
                      f"({len(test)} pairs) from {test_specs[name]}")
            elif args.make_test:
                used = {g for _, _, g in train} | {g for _, _, g in val}
                test = [(img, mask, key)
                        for key, group in groups.items() if key not in used
                        for img, mask in group]
                subsets.append(('test', test))
        for split, subset in subsets:
            # Seen test sets are per-dataset roots: test/<dataset>/images.
            split_root = (os.path.join(args.out_dir, 'test', name)
                          if split == 'test' else os.path.join(args.out_dir, split))
            os.makedirs(os.path.join(split_root, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_root, 'masks'), exist_ok=True)
            for item in subset:
                img_path, mask_path = item[0], item[1]
                group = item[2] if len(item) > 2 else ''
                img_ext = os.path.splitext(img_path)[1]
                mask_ext = os.path.splitext(mask_path)[1]
                # Prefix the dataset name to avoid collisions across sources
                new_base = f"{name}_{os.path.splitext(os.path.basename(img_path))[0]}"
                new_img = os.path.join(split_root, 'images', new_base + img_ext)
                new_mask = os.path.join(split_root, 'masks', new_base + mask_ext)
                shutil.copy2(img_path, new_img)
                shutil.copy2(mask_path, new_mask)

                # Optional FOV: keep it aligned with the split (train/val/test).
                fov_src_dir = os.path.join(
                    source_dirs.get(split, src), 'fov')
                if os.path.isdir(fov_src_dir):
                    src_base = os.path.splitext(os.path.basename(img_path))[0]
                    fov_path = None
                    for ext in SUPPORTED:
                        cand = os.path.join(fov_src_dir, src_base + ext)
                        if os.path.exists(cand):
                            fov_path = cand
                            break
                    if fov_path is not None:
                        fov_dir = os.path.join(split_root, 'fov')
                        os.makedirs(fov_dir, exist_ok=True)
                        # Preserve the source extension: a .tif renamed to
                        # .png is content-misleading even if PIL sniffs it.
                        fov_ext = os.path.splitext(fov_path)[1]
                        shutil.copy2(
                            fov_path,
                            os.path.join(fov_dir, new_base + fov_ext))

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
