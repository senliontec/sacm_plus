#!/usr/bin/env python
"""Offline label-free topology ranking over saved typed-head candidates.

Reads the PNG candidates written by ``evaluate.py --save_candidate_masks``
(and optionally the matching probability maps written by
``--save_candidate_probs``) and reports how the topology ranker compares
with the IoU/quality/fixed-head selectors and with the per-image oracle.

Example:

    python scripts/sacm_x/rank_candidates_topo.py \
        --candidates-dir results/eval/candidates \
        --probs-dir results/eval/candidates \
        --images-dir data/sacm_3shot/test/WIRE/images \
        --masks-dir data/sacm_3shot/test/WIRE/masks \
        --fov-dir data/sacm_3shot/test/WIRE/fov \
        --output-dir results/eval/topo_ranker

This script is analysis-only: it never trains, never edits checkpoints, and
the ranker itself is label-free.  GT is used only to report Dice/clDice and
the oracle head for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from core.metrics.topology import betti_error, cldice_score  # noqa: E402
from core.topology_ranker import (  # noqa: E402
    RankWeights,
    rank_candidates,
    topology_descriptors,
)

_HEAD_RE = re.compile(r'^(?P<stem>.+)_head(?P<head>\d+)\.(?P<ext>png|npy)$')
_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _index_files(root: Path, exts: Iterable[str]) -> dict[str, Path]:
    """Map stem -> path, refusing to silently pick one of duplicate stems."""
    exts = tuple(ext.lower() for ext in exts)
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in exts:
            found.setdefault(path.stem, []).append(path)
    index: dict[str, Path] = {}
    duplicates = 0
    for stem, paths in found.items():
        if len(paths) > 1:
            duplicates += 1
            print(
                f'[warn] duplicate stem {stem!r}: using {paths[0]}, '
                f'ignoring {len(paths) - 1} other(s)', file=sys.stderr)
        index[stem] = paths[0]
    if duplicates:
        print(
            f'[warn] {duplicates} duplicate stem(s); provide one dataset '
            'per candidates directory to avoid ambiguity', file=sys.stderr)
    return index


def _group_candidates(
    root: Path,
    extension: str,
) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(root.rglob(f'*.{extension}')):
        match = _HEAD_RE.match(path.name)
        if match is None:
            continue
        stem = match.group('stem')
        head = int(match.group('head'))
        groups.setdefault(stem, {})[head] = path
    return groups


def _load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert('L')) > 127


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert('L'), dtype=np.float32) / 255.0


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    denominator = int(pred.sum()) + int(gt.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.count_nonzero(pred & gt) / denominator)


def _safe_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        value = float(cldice_score(pred, gt))
    except Exception:
        return float('nan')
    return value if math.isfinite(value) else float('nan')


def _load_head_audit(path: Path | None) -> dict[tuple[str, int], float]:
    if path is None or not path.is_file():
        return {}
    scores: dict[tuple[str, int], float] = {}
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                stem = Path(str(row.get('image', ''))).stem
                head = int(row['head'])
                score = float(row['iou_head_score'])
            except (KeyError, TypeError, ValueError):
                continue
            scores[(stem, head)] = score
    return scores


def _resolve_output_dir(explicit: str | None, candidates_dir: Path) -> Path:
    if explicit:
        return Path(explicit)
    return candidates_dir.parent / 'topo_ranker'


def _mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float('nan')
    return float(np.mean(finite))


def _format(value: float) -> str:
    return 'nan' if not math.isfinite(value) else f'{value:.6f}'


def run(args: argparse.Namespace) -> int:
    candidates_dir = Path(args.candidates_dir)
    if not candidates_dir.is_dir():
        raise SystemExit(f'candidates directory not found: {candidates_dir}')
    probs_dir = Path(args.probs_dir) if args.probs_dir else None
    if probs_dir is not None and not probs_dir.is_dir():
        raise SystemExit(f'probs directory not found: {probs_dir}')

    candidates = _group_candidates(candidates_dir, 'png')
    probabilities = (
        _group_candidates(probs_dir, 'npy') if probs_dir else {})

    image_index = (
        _index_files(Path(args.images_dir), _IMAGE_EXTS)
        if args.images_dir else {})
    mask_index = (
        _index_files(Path(args.masks_dir), _IMAGE_EXTS)
        if args.masks_dir else {})
    fov_index = (
        _index_files(Path(args.fov_dir), _IMAGE_EXTS)
        if args.fov_dir else {})

    head_audit = _load_head_audit(
        Path(args.head_audit) if args.head_audit else
        candidates_dir.parent / 'head_audit.csv')

    output_dir = _resolve_output_dir(args.output_dir, candidates_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = RankWeights(
        stability=args.w_stab,
        widest_path=args.w_wprf,
        evidence=args.w_ev,
        fragmentation=args.w_frag,
        components=args.w_comp,
        iou=args.w_iou,
    )
    weights.validate()

    candidate_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    skipped = 0
    processed = 0

    for stem in sorted(candidates):
        if args.limit is not None and processed >= args.limit:
            break
        heads = candidates[stem]
        head_ids = sorted(heads)
        if len(head_ids) < 2:
            skipped += 1
            print(
                f'[warn] {stem}: only {len(head_ids)} candidate(s); skipping',
                file=sys.stderr)
            continue

        masks = [_load_binary(heads[head]) for head in head_ids]
        shapes = {mask.shape for mask in masks}
        if len(shapes) != 1:
            skipped += 1
            print(
                f'[warn] {stem}: candidate shape mismatch {shapes}; skipping',
                file=sys.stderr)
            continue
        height, width = masks[0].shape

        prob_maps = None
        if probs_dir is not None:
            prob_heads = probabilities.get(stem, {})
            prob_maps = []
            for head in head_ids:
                path = prob_heads.get(head)
                if path is None:
                    print(
                        f'[warn] {stem}: missing probability for head '
                        f'{head}; falling back to binary stability',
                        file=sys.stderr)
                    prob_maps = None
                    break
                prob = np.load(path).astype(np.float32)
                if prob.shape != (height, width):
                    print(
                        f'[warn] {stem}: probability shape {prob.shape} '
                        f'!= mask shape {(height, width)}; skipping probs',
                        file=sys.stderr)
                    prob_maps = None
                    break
                prob_maps.append(prob)

        image = None
        if args.images_dir:
            image_path = image_index.get(stem)
            if image_path is None:
                print(
                    f'[warn] {stem}: image not found; evidence=neutral',
                    file=sys.stderr)
            else:
                image = _load_gray(image_path)
                if image.shape != (height, width):
                    print(
                        f'[warn] {stem}: image shape {image.shape} != '
                        f'candidate shape {(height, width)}; '
                        'evidence=neutral', file=sys.stderr)
                    image = None

        gt = None
        if args.masks_dir:
            mask_path = mask_index.get(stem)
            if mask_path is None:
                print(
                    f'[warn] {stem}: GT mask not found; metrics omitted',
                    file=sys.stderr)
            else:
                gt = _load_binary(mask_path)
                if gt.shape != (height, width):
                    print(
                        f'[warn] {stem}: GT shape {gt.shape} != candidate '
                        f'shape {(height, width)}; metrics omitted',
                        file=sys.stderr)
                    gt = None

        fov = None
        if args.fov_dir:
            fov_path = fov_index.get(stem)
            if fov_path is None:
                print(
                    f'[warn] {stem}: FOV mask not found; metrics are '
                    'unmasked', file=sys.stderr)
            else:
                fov = _load_binary(fov_path)
                if fov.shape != (height, width):
                    print(
                        f'[warn] {stem}: FOV shape {fov.shape} != candidate '
                        f'shape {(height, width)}; metrics are unmasked',
                        file=sys.stderr)
                    fov = None
        if fov is not None:
            masks = [mask & fov for mask in masks]
            if gt is not None:
                gt = gt & fov
            if prob_maps is not None:
                prob_maps = [
                    prob * fov.astype(np.float32) for prob in prob_maps]

        iou_pred = None
        if head_audit:
            iou_pred = [
                head_audit.get((stem, head), 0.0) for head in head_ids]
            if not any(value != 0.0 for value in iou_pred):
                iou_pred = None

        ranking = rank_candidates(
            masks,
            probs=prob_maps,
            image=image,
            iou_pred=iou_pred,
            weights=weights,
            thresholds=args.thresholds,
            stride=args.stride,
        )

        per_head_metrics: dict[int, dict[str, float]] = {}
        for index, head in enumerate(head_ids):
            dice = _dice(masks[index], gt) if gt is not None else float('nan')
            cldice = (
                _safe_cldice(masks[index], gt)
                if gt is not None else float('nan'))
            desc = topology_descriptors(masks[index])
            betti = (
                float(betti_error(masks[index], gt))
                if gt is not None else float('nan'))
            per_head_metrics[head] = {
                'dice': dice,
                'cldice': cldice,
                'betti_error': betti,
            }
            feature = ranking.features[index]
            candidate_rows.append({
                'image': stem,
                'head': head,
                'selected': int(index == ranking.selected),
                'score': feature['score'],
                's_stab': feature['s_stab'],
                's_wprf': feature['s_wprf'],
                's_ev': feature['s_ev'],
                'stab_mode': feature['stab_mode'],
                'frag_skel': feature['frag_skel'],
                'b0': desc.b0,
                'b1': desc.b1,
                'endpoints': desc.endpoints,
                'branchpoints': desc.branchpoints,
                'skeleton_length': desc.skeleton_length,
                'largest_component_skeleton_share':
                    desc.largest_component_skeleton_share,
                'small_component_rate': desc.small_component_rate,
                'dice': dice,
                'cldice': cldice,
                'betti_error': betti,
            })

        selected_head = head_ids[ranking.selected]
        selected_metrics = per_head_metrics[selected_head]
        oracle_head = None
        oracle_metrics = {
            'dice': float('nan'),
            'cldice': float('nan'),
            'betti_error': float('nan'),
        }
        if gt is not None:
            valid = [
                head for head in head_ids
                if math.isfinite(per_head_metrics[head]['cldice'])]
            if valid:
                oracle_head = max(
                    valid, key=lambda head: (
                        per_head_metrics[head]['cldice'],
                        per_head_metrics[head]['dice'],
                        -head,
                    ))
            else:
                oracle_head = max(
                    head_ids,
                    key=lambda head: (
                        per_head_metrics[head]['dice'], -head))
            oracle_metrics = per_head_metrics[oracle_head]

        image_rows.append({
            'image': stem,
            'selected_head': selected_head,
            'oracle_head': '' if oracle_head is None else oracle_head,
            'score': ranking.scores[ranking.selected],
            **{f'selected_{key}': value
               for key, value in selected_metrics.items()},
            **{f'oracle_{key}': value
               for key, value in oracle_metrics.items()},
            'head_count': len(head_ids),
        })
        processed += 1

    if not image_rows:
        raise SystemExit('no complete candidate groups found')

    candidate_path = output_dir / 'per_candidate.csv'
    with candidate_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)

    image_path = output_dir / 'per_image.csv'
    with image_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    selected_heads = [int(row['selected_head']) for row in image_rows]
    oracle_heads = [
        int(row['oracle_head']) for row in image_rows
        if str(row['oracle_head']) != '']
    summary = {
        'images': len(image_rows),
        'skipped': skipped,
        'candidate_heads': sorted({
            int(row['head']) for row in candidate_rows}),
        'selected_head_counts': {
            str(head): selected_heads.count(head)
            for head in sorted(set(selected_heads))},
        'oracle_head_counts': {
            str(head): oracle_heads.count(head)
            for head in sorted(set(oracle_heads))},
        'selected_dice': _mean(row['selected_dice'] for row in image_rows),
        'selected_cldice': _mean(
            row['selected_cldice'] for row in image_rows),
        'selected_betti_error': _mean(
            row['selected_betti_error'] for row in image_rows),
        'oracle_dice': _mean(row['oracle_dice'] for row in image_rows),
        'oracle_cldice': _mean(
            row['oracle_cldice'] for row in image_rows),
        'oracle_betti_error': _mean(
            row['oracle_betti_error'] for row in image_rows),
        'mean_score': _mean(row['score'] for row in image_rows),
        'weights': {
            'stability': weights.stability,
            'widest_path': weights.widest_path,
            'evidence': weights.evidence,
            'fragmentation': weights.fragmentation,
            'components': weights.components,
            'iou': weights.iou,
        },
    }
    for head in summary['candidate_heads']:
        summary[f'fixed_head_{head}_dice'] = _mean(
            row['dice'] for row in candidate_rows
            if int(row['head']) == head)
        summary[f'fixed_head_{head}_cldice'] = _mean(
            row['cldice'] for row in candidate_rows
            if int(row['head']) == head)

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')

    print(f'processed {len(image_rows)} image(s), skipped {skipped}')
    print(f'wrote {candidate_path}')
    print(f'wrote {image_path}')
    print(f'wrote {summary_path}')
    print(
        'selected: Dice={} clDice={} | oracle: Dice={} clDice={}'.format(
            _format(summary['selected_dice']),
            _format(summary['selected_cldice']),
            _format(summary['oracle_dice']),
            _format(summary['oracle_cldice']),
        ))
    for head in summary['candidate_heads']:
        fixed_dice = summary[f'fixed_head_{head}_dice']
        fixed_cldice = summary[f'fixed_head_{head}_cldice']
        print(
            f'fixed head {head}: Dice={_format(fixed_dice)} '
            f'clDice={_format(fixed_cldice)}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Label-free topology ranking of saved typed candidates')
    parser.add_argument(
        '--candidates-dir', required=True,
        help='Directory containing {stem}_head{k}.png files')
    parser.add_argument(
        '--probs-dir', default=None,
        help='Optional directory containing {stem}_head{k}.npy '
             'probability maps')
    parser.add_argument(
        '--images-dir', default=None,
        help='Optional original images for skeleton-evidence scoring')
    parser.add_argument(
        '--masks-dir', default=None,
        help='Optional GT masks for analysis-only Dice/clDice/oracle')
    parser.add_argument(
        '--fov-dir', default=None,
        help='Optional FOV masks; applied to candidates and GT so metrics '
             'match the evaluator when --no-fov is not used')
    parser.add_argument(
        '--head-audit', default=None,
        help='Optional head_audit.csv with iou_head_score per image/head')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument(
        '--thresholds', nargs='+', type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--w-stab', type=float, default=0.35)
    parser.add_argument('--w-wprf', type=float, default=0.35)
    parser.add_argument('--w-ev', type=float, default=0.10)
    parser.add_argument('--w-frag', type=float, default=0.15)
    parser.add_argument('--w-comp', type=float, default=0.05)
    parser.add_argument('--w-iou', type=float, default=0.05)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
