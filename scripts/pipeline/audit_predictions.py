#!/usr/bin/env python3
"""Audit saved prediction masks against ground truth at original resolution.

This is a diagnostic companion to ``src/eval/evaluate.py``.  It does NOT
recompute the paper table: it loads the already-saved prediction PNGs and GT
masks and reports where the prediction is wrong, including topology failures
that Dice/clDice/HD95 can hide:

* pixel confusion (TP/FP/FN/TN) and precision/recall;
* connected-component explosion and small-component rates;
* hole-count / Betti errors;
* skeleton precision/recall (where the predicted centerline is wrong);
* failure flags: empty prediction, no overlap, over/under-segmentation,
  mixed error, fragmentation, hole mismatch;
* optional side-by-side GT / prediction / pixel-error / skeleton-error PNGs.

The official four paper metrics are recomputed with the project's canonical
``core.metrics.binary_metrics`` under an explicit empty-mask policy.  Use
``--empty-policy nan`` to reproduce the raw official semantics.

Prediction layout is auto-detected from one of::

    <pred-root>/<DATASET>/*_result.png
    <pred-root>/<DATASET>/predictions/*_result.png
    <pred-root>/predictions/<DATASET>/*_result.png
    <pred-root>/eval/<DATASET>/*_result.png

Typical use::

    python scripts/pipeline/audit_predictions.py \
        --pred-root results/runs/skein/full_typed/seed_0/eval \
        --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
        --out results/audit/skein_full_typed_seed0

The script is read-only with respect to the evaluation tree.  It never
resizes a prediction to the GT shape; size mismatches are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as nd_label

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(SRC), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.data import SUPPORTED_EXTENSIONS  # noqa: E402
from core.metrics.binary import binary_metrics  # noqa: E402
from core.metrics.common import skeletonize  # noqa: E402
from core.metrics.components import component_metrics  # noqa: E402
from core.metrics.topology import betti_error  # noqa: E402
from protocol_specs import dataset_specs  # noqa: E402

IMAGE_FIELDS = [
    'dataset', 'image', 'status', 'pred_path', 'gt_path', 'height', 'width',
    'tp', 'fp', 'fn', 'tn', 'pred_area', 'gt_area', 'pred_area_ratio',
    'precision', 'recall', 'specificity', 'f1',
    'dice', 'iou', 'cldice', 'hd95', 'empty_pred', 'empty_gt',
    'components_pred', 'components_gt', 'component_count_error',
    'components_ratio', 'small_components_pred', 'small_component_rate',
    'small_components_gt', 'largest_component_ratio',
    'fp_area_ratio', 'fn_area_ratio',
    'holes_pred', 'holes_gt', 'hole_count_error', 'betti',
    'skeleton_tp', 'skeleton_fp', 'skeleton_fn',
    'skeleton_precision', 'skeleton_recall',
    'no_overlap', 'overseg', 'underseg', 'mixed_error', 'fragmented',
    'hole_mismatch', 'compare_png', 'error',
]

DATASET_METRICS = [
    'dice', 'iou', 'cldice', 'hd95', 'precision', 'recall', 'specificity',
    'f1', 'betti', 'skeleton_precision', 'skeleton_recall',
    'components_pred', 'components_gt', 'component_count_error',
    'components_ratio', 'small_components_pred', 'small_component_rate',
    'pred_area_ratio', 'fp_area_ratio', 'fn_area_ratio',
    'holes_pred', 'holes_gt', 'hole_count_error',
]
DATASET_RATES = [
    'empty_pred', 'empty_gt', 'no_overlap', 'overseg', 'underseg',
    'mixed_error', 'fragmented', 'hole_mismatch',
]

DATASET_FIELDS = (
    ['dataset', 'role', 'n_images', 'n_ok', 'n_size_mismatch',
     'n_read_error', 'n_metric_error', 'n_missing_pred', 'n_unmatched_pred',
     'components_pred_total', 'components_gt_total',
     'components_total_ratio']
    + [f'{key}_mean' for key in DATASET_METRICS]
    + [f'{key}_std' for key in DATASET_METRICS]
    + [f'{key}_rate' for key in DATASET_RATES]
    + ['worst_image', 'worst_dice']
)

MACRO_FIELDS = (
    ['scope', 'n_datasets']
    + [f'{key}_mean' for key in DATASET_METRICS]
    + [f'{key}_std' for key in DATASET_METRICS]
    + [f'{key}_rate' for key in DATASET_RATES]
)

UNMATCHED_FIELDS = ['dataset', 'image', 'reason', 'path']


# ---------------------------------------------------------------------------
# mask loading / matching
# ---------------------------------------------------------------------------

def _load_binary(path: Path) -> np.ndarray:
    """Load a 0/255 or literal 0/1 PNG as a boolean mask."""
    values = np.asarray(Image.open(path).convert('L'))
    if values.size and int(values.max()) <= 1:
        return values > 0
    return values > 127


def _count_holes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    lab, count = nd_label(
        ~mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    border = np.unique(np.concatenate([
        lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1],
    ]))
    border = border[border != 0]
    return int(max(count - len(border), 0))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _skeleton_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    tp = int(np.count_nonzero(sp & sg))
    fp = int(np.count_nonzero(sp & ~sg))
    fn = int(np.count_nonzero(sg & ~sp))
    return {
        'skeleton_tp': tp,
        'skeleton_fp': fp,
        'skeleton_fn': fn,
        'skeleton_precision': _safe_ratio(tp, tp + fp),
        'skeleton_recall': _safe_ratio(tp, tp + fn),
    }


def _key_candidates(filename: str, suffix: str) -> list[str]:
    """Return plausible GT names/stems for one prediction filename.

    Handles both evaluator namings:
      foo_result.png       -> raw foo
      foo.png_result.png   -> raw foo.png, stem foo
    """
    if suffix and filename.endswith(suffix):
        raw = filename[:-len(suffix)]
    else:
        raw = Path(filename).stem
    keys = [raw]
    stem = Path(raw).stem
    if stem not in keys:
        keys.append(stem)
    return keys


def _gt_index(mask_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        by_name.setdefault(path.name, path)
        by_stem.setdefault(path.stem, path)
    return by_name, by_stem


def _match_predictions(
    pred_files: Sequence[Path],
    gt_by_name: dict[str, Path],
    gt_by_stem: dict[str, Path],
    suffix: str,
) -> tuple[list[tuple[Path, Path]], list[dict]]:
    pairs: list[tuple[Path, Path]] = []
    unmatched: list[dict] = []
    used_gt: set[Path] = set()
    for pred in sorted(pred_files):
        match = None
        for key in _key_candidates(pred.name, suffix):
            match = gt_by_name.get(key) or gt_by_stem.get(key)
            if match is not None:
                break
        if match is None:
            unmatched.append({
                'image': pred.name,
                'reason': 'missing_gt',
                'path': str(pred),
            })
            continue
        if match in used_gt:
            unmatched.append({
                'image': pred.name,
                'reason': 'duplicate_gt',
                'path': str(pred),
            })
            continue
        used_gt.add(match)
        pairs.append((match, pred))
    for gt in gt_by_name.values():
        if gt not in used_gt:
            unmatched.append({
                'image': gt.name,
                'reason': 'missing_prediction',
                'path': str(gt),
            })
    return pairs, unmatched


def _pairs_from_prediction_index(
    pred_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[dict]] | None:
    """Use prediction_index.csv written by evaluate.py/baseline runtime.

    This is authoritative when present: it removes filename-matching
    ambiguity (``foo_result.png`` vs ``foo.png_result.png``) and records the
    exact GT path used for each prediction.
    """
    index_path = pred_dir / 'prediction_index.csv'
    if not index_path.is_file():
        return None
    pairs: list[tuple[Path, Path]] = []
    unmatched: list[dict] = []
    with open(index_path, newline='') as handle:
        for row in csv.DictReader(handle):
            pred_raw = str(row.get('pred_path', '')).strip()
            gt_raw = str(row.get('gt_path', '')).strip()
            image = str(row.get('image', '')).strip()
            if not pred_raw or not gt_raw:
                unmatched.append({
                    'image': image or '*',
                    'reason': 'index_incomplete',
                    'path': str(index_path),
                })
                continue
            pred = Path(pred_raw)
            gt = Path(gt_raw)
            if not pred.is_absolute():
                pred = pred_dir / pred
            if not gt.is_absolute():
                gt = pred_dir / gt
            if not pred.is_file():
                unmatched.append({
                    'image': image or pred.name,
                    'reason': 'missing_prediction_file',
                    'path': str(pred),
                })
                continue
            if not gt.is_file():
                unmatched.append({
                    'image': image or gt.name,
                    'reason': 'missing_gt_file',
                    'path': str(gt),
                })
                continue
            pairs.append((gt, pred))
    return pairs, unmatched


def _resolve_gt_dir(test_root: Path) -> Path | None:
    candidates = [
        test_root / 'masks',
        test_root / 'test' / 'masks',
        test_root,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for path in candidate.iterdir():
            if (path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS):
                return candidate
    return None


def _has_prediction_files(directory: Path, suffix: str) -> bool:
    if not directory.is_dir() or not suffix:
        return False
    return any(
        path.is_file() and path.name.endswith(suffix)
        for path in directory.iterdir()
    )


def _resolve_pred_dir(
    pred_root: Path, dataset: str, suffix: str, allow_root: bool,
) -> tuple[Path | None, list[Path]]:
    candidates = [
        pred_root / dataset,
        pred_root / dataset / 'predictions',
        pred_root / 'predictions' / dataset,
        pred_root / 'eval' / dataset,
        pred_root / 'eval' / dataset / 'predictions',
    ]
    # Raw Skein eval roots are <root>/<preset>/<DATASET>; accept one
    # intermediate directory (e.g. pass .../eval while a preset dir exists).
    if pred_root.is_dir():
        for child in sorted(pred_root.iterdir()):
            if child.is_dir():
                candidates.append(child / dataset)
                candidates.append(child / dataset / 'predictions')
    if allow_root and _has_prediction_files(pred_root, suffix):
        candidates.insert(0, pred_root)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        has_index = (candidate / 'prediction_index.csv').is_file()
        if not _has_prediction_files(candidate, suffix) and not has_index:
            continue
        files = sorted(
            path for path in candidate.iterdir()
            if path.is_file() and path.name.endswith(suffix)
        ) if suffix else []
        return candidate, files
    return None, []


# ---------------------------------------------------------------------------
# per-image audit
# ---------------------------------------------------------------------------

def _audit_one(task: tuple) -> dict:
    (dataset, image_name, pred_path, gt_path, empty_policy,
     small_px) = task
    row = {
        'dataset': dataset,
        'image': image_name,
        'status': 'ok',
        'pred_path': str(pred_path),
        'gt_path': str(gt_path),
        'compare_png': '',
        'error': '',
    }
    try:
        pred = _load_binary(pred_path)
        gt = _load_binary(gt_path)
    except Exception as exc:  # unreadable/corrupt PNG
        row['status'] = 'read_error'
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row

    if pred.shape != gt.shape:
        row.update({
            'status': 'size_mismatch',
            'height': int(gt.shape[0]),
            'width': int(gt.shape[1]),
            'error': (f'prediction {pred.shape} != GT {gt.shape}; '
                      'no resize was applied'),
        })
        return row

    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    metrics = binary_metrics(pred, gt, empty_policy=empty_policy)
    comp = component_metrics(pred, gt, small_px=int(small_px))
    holes_p = _count_holes(pred)
    holes_g = _count_holes(gt)

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    both_nonempty = pred_area > 0 and gt_area > 0

    no_overlap = both_nonempty and tp == 0
    overseg = both_nonempty and recall >= 0.70 and precision < 0.50
    underseg = both_nonempty and precision >= 0.70 and recall < 0.50
    mixed_error = (
        both_nonempty and not overseg and not underseg
        and float(metrics['dice']) < 0.50
    )
    components_pred = int(comp['components_pred'])
    components_gt = int(comp['components_gt'])
    fragmented = (
        components_pred >= max(5, 3 * max(components_gt, 1))
        and components_pred - components_gt >= 5
    )
    hole_error = abs(holes_p - holes_g)

    row.update({
        'height': int(gt.shape[0]),
        'width': int(gt.shape[1]),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pred_area': pred_area, 'gt_area': gt_area,
        'pred_area_ratio': _safe_ratio(pred_area, max(gt_area, 1)),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'dice': float(metrics['dice']),
        'iou': float(metrics['iou']),
        'cldice': float(metrics['cldice']),
        'hd95': float(metrics['hd95']),
        'empty_pred': int(pred_area == 0),
        'empty_gt': int(gt_area == 0),
        **{key: comp[key] for key in comp},
        'components_ratio': _safe_ratio(
            components_pred, max(components_gt, 1)),
        'holes_pred': holes_p,
        'holes_gt': holes_g,
        'hole_count_error': hole_error,
        'betti': float(betti_error(pred, gt)),
        **_skeleton_stats(pred, gt),
        'no_overlap': int(no_overlap),
        'overseg': int(overseg),
        'underseg': int(underseg),
        'mixed_error': int(mixed_error),
        'fragmented': int(fragmented),
        'hole_mismatch': int(hole_error >= 2),
    })
    return row


def _safe_audit(task: tuple) -> dict:
    """Never let one corrupt/unusual image abort the whole dataset."""
    try:
        return _audit_one(task)
    except Exception as exc:
        dataset, image_name, pred_path, gt_path = task[:4]
        return {
            'dataset': dataset,
            'image': image_name,
            'status': 'metric_error',
            'pred_path': str(pred_path),
            'gt_path': str(gt_path),
            'compare_png': '',
            'error': f'{type(exc).__name__}: {exc}',
        }


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _finite_values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float('nan')


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _float(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _dataset_summary(
    dataset: str, role: str, rows: list[dict],
    unmatched: list[dict],
) -> dict:
    ok = [row for row in rows if row.get('status') == 'ok']
    summary = {
        'dataset': dataset,
        'role': role,
        'n_images': len(rows),
        'n_ok': len(ok),
        'n_size_mismatch': sum(
            row.get('status') == 'size_mismatch' for row in rows),
        'n_read_error': sum(
            row.get('status') == 'read_error' for row in rows),
        'n_metric_error': sum(
            row.get('status') == 'metric_error' for row in rows),
        'n_missing_pred': sum(
            str(item.get('reason', '')).startswith('missing_prediction')
            for item in unmatched),
        'n_unmatched_pred': sum(
            str(item.get('reason', '')) in (
                'missing_gt', 'missing_gt_file', 'duplicate_gt',
                'index_incomplete')
            for item in unmatched),
        'components_pred_total': sum(
            int(row.get('components_pred', 0)) for row in ok),
        'components_gt_total': sum(
            int(row.get('components_gt', 0)) for row in ok),
    }
    summary['components_total_ratio'] = _safe_ratio(
        summary['components_pred_total'],
        max(summary['components_gt_total'], 1))
    for key in DATASET_METRICS:
        values = _finite_values(ok, key)
        summary[f'{key}_mean'] = _mean(values)
        summary[f'{key}_std'] = _std(values)
    for key in DATASET_RATES:
        summary[f'{key}_rate'] = _mean(_finite_values(ok, key))
    if ok:
        worst = min(
            ok,
            key=lambda row: (
                _float(row.get('dice'), math.inf),
                -_float(row.get('component_count_error'), 0.0),
            ),
        )
        summary['worst_image'] = worst.get('image', '')
        summary['worst_dice'] = _float(worst.get('dice'), float('nan'))
    else:
        summary['worst_image'] = ''
        summary['worst_dice'] = float('nan')
    return summary


def _macro_summary(dataset_rows: list[dict]) -> list[dict]:
    scopes = ['Seen', 'Unseen', 'Novel', 'All']
    present_roles = {str(row.get('role', '')) for row in dataset_rows}
    scopes = [scope for scope in scopes
              if scope == 'All' or scope in present_roles]
    rows = []
    for scope in scopes:
        group = (list(dataset_rows) if scope == 'All'
                 else [row for row in dataset_rows
                       if row.get('role') == scope])
        if not group:
            continue
        item = {'scope': scope, 'n_datasets': len(group)}
        for key in DATASET_METRICS:
            values = _finite_values(group, f'{key}_mean')
            item[f'{key}_mean'] = _mean(values)
            item[f'{key}_std'] = _std(values)
        for key in DATASET_RATES:
            values = _finite_values(group, f'{key}_rate')
            item[f'{key}_rate'] = _mean(values)
        rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# reports / visualization
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    if not math.isfinite(number):
        return '—'
    return f'{number:.{digits}f}'


def _percent(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    if not math.isfinite(number):
        return '—'
    return f'{100.0 * number:.1f}%'


def _panel(mask: np.ndarray, foreground: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = foreground
    return canvas


def _error_panel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[pred & gt] = (0, 200, 0)       # TP: green
    canvas[pred & ~gt] = (255, 60, 60)    # FP: red
    canvas[~pred & gt] = (60, 120, 255)   # FN: blue
    return canvas


def _skeleton_panel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[sg] = (255, 230, 0)            # GT skeleton: yellow
    canvas[sp] = (0, 220, 220)            # predicted skeleton: cyan
    canvas[sp & sg] = (255, 255, 255)     # shared: white
    return canvas


def _resize_panel(image: Image.Image, max_side: int) -> Image.Image:
    if max_side and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        new_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        return image.resize(new_size, Image.NEAREST)
    return image


def _save_compare(
    row: dict, out_dir: Path, max_side: int,
) -> str | None:
    pred_path = Path(row['pred_path'])
    gt_path = Path(row['gt_path'])
    try:
        pred = _load_binary(pred_path)
        gt = _load_binary(gt_path)
    except Exception:
        return None
    if pred.shape != gt.shape:
        return None

    panels = [
        ('GT', _panel(gt, (255, 255, 255))),
        ('Prediction', _panel(pred, (255, 255, 255))),
        ('Pixel error (G=TP R=FP B=FN)', _error_panel(pred, gt)),
        ('Skeleton (Y=GT C=pred W=both)', _skeleton_panel(pred, gt)),
    ]
    images = [
        _resize_panel(Image.fromarray(panel), max_side)
        for _label, panel in panels
    ]
    gap = 8
    header = 18
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images) + header
    canvas = Image.new('RGB', (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (label, _panel_rgb), image in zip(panels, images):
        canvas.paste(image, (x, header))
        draw.text((x + 4, 3), label, fill=(255, 255, 0))
        x += image.width + gap

    dataset = str(row['dataset'])
    stem = Path(str(row['image'])).stem
    target = out_dir / 'error_maps' / dataset / f'{stem}_compare.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    try:
        return str(target.relative_to(out_dir))
    except ValueError:
        return str(target)


def _write_report(
    out_dir: Path,
    per_dataset: list[dict],
    macro: list[dict],
    per_image: list[dict],
    unmatched: list[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        '# Prediction audit',
        '',
        f'- Prediction root: `{args.pred_root}`',
        f'- Ground-truth protocol: '
        f'`{args.protocol if args.protocol else "NAME:DIR specs"}`',
        f'- Empty-mask policy: `{args.empty_policy}`',
        f'- Images audited: {len([r for r in per_image if r.get("status") == "ok"])}',
        f'- Size mismatches: '
        f'{len([r for r in per_image if r.get("status") == "size_mismatch"])}',
        f'- Unmatched files: {len(unmatched)}',
        '',
        '> Diagnostic only. The four paper metrics remain those written by '
        '`src/eval/evaluate.py`; this report is for locating failure modes.',
        '',
        '## Per dataset (worst Dice first)',
        '',
        '| Dataset | Role | n | Dice | IoU | clDice | HD95 | P | R | '
        'Components pred/GT | Frag. ratio | Empty | No overlap | Over | '
        'Under | Fragmented |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(
            per_dataset,
            key=lambda item: _float(item.get('dice_mean'), math.inf)):
        lines.append(
            f"| {row['dataset']} | {row['role'] or '—'} | {row['n_ok']} | "
            f"{_fmt(row.get('dice_mean'))} | {_fmt(row.get('iou_mean'))} | "
            f"{_fmt(row.get('cldice_mean'))} | "
            f"{_fmt(row.get('hd95_mean'))} | "
            f"{_fmt(row.get('precision_mean'))} | "
            f"{_fmt(row.get('recall_mean'))} | "
            f"{_fmt(row.get('components_pred_mean'), 2)}/"
            f"{_fmt(row.get('components_gt_mean'), 2)} | "
            f"{_fmt(row.get('components_total_ratio'), 2)} | "
            f"{_percent(row.get('empty_pred_rate'))} | "
            f"{_percent(row.get('no_overlap_rate'))} | "
            f"{_percent(row.get('overseg_rate'))} | "
            f"{_percent(row.get('underseg_rate'))} | "
            f"{_percent(row.get('fragmented_rate'))} |"
        )

    lines += [
        '',
        '## Macro (equal dataset weight)',
        '',
        '| Scope | Datasets | Dice | IoU | clDice | HD95 | Empty | '
        'No overlap | Fragmented |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in macro:
        lines.append(
            f"| {row['scope']} | {row['n_datasets']} | "
            f"{_fmt(row.get('dice_mean'))} | {_fmt(row.get('iou_mean'))} | "
            f"{_fmt(row.get('cldice_mean'))} | "
            f"{_fmt(row.get('hd95_mean'))} | "
            f"{_percent(row.get('empty_pred_rate'))} | "
            f"{_percent(row.get('no_overlap_rate'))} | "
            f"{_percent(row.get('fragmented_rate'))} |"
        )

    ok_rows = [row for row in per_image if row.get('status') == 'ok']
    worst = sorted(
        ok_rows,
        key=lambda row: _float(row.get('dice'), math.inf),
    )[:30]
    lines += [
        '',
        '## Worst images',
        '',
        '| Dataset | Image | Dice | clDice | HD95 | Components | Flags |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for row in worst:
        flags = [
            name for name, key in (
                ('empty', 'empty_pred'),
                ('no-overlap', 'no_overlap'),
                ('over', 'overseg'),
                ('under', 'underseg'),
                ('mixed', 'mixed_error'),
                ('fragmented', 'fragmented'),
                ('hole', 'hole_mismatch'),
            ) if int(row.get(key, 0))
        ]
        lines.append(
            f"| {row['dataset']} | {row['image']} | "
            f"{_fmt(row.get('dice'))} | {_fmt(row.get('cldice'))} | "
            f"{_fmt(row.get('hd95'))} | "
            f"{row.get('components_pred', '')}/"
            f"{row.get('components_gt', '')} | {', '.join(flags) or '—'} |"
        )

    if unmatched:
        lines += [
            '',
            '## Unmatched files',
            '',
            f'{len(unmatched)} prediction/GT files could not be paired. '
            f'See `unmatched.csv` for details.',
        ]

    lines += [
        '',
        '## Failure definitions',
        '',
        '- **no overlap**: non-empty pred and GT, TP = 0.',
        '- **over-seg**: recall ≥ 0.70 and precision < 0.50.',
        '- **under-seg**: precision ≥ 0.70 and recall < 0.50.',
        '- **mixed**: Dice < 0.50 and neither over- nor under-seg.',
        '- **fragmented**: predicted 8-connected components ≥ max(5, '
        '3 × GT components) and at least 5 more components than GT.',
        '- **hole mismatch**: absolute hole-count difference ≥ 2.',
        '- Compare PNGs: GT / prediction / pixel error / skeleton error; '
        'pixel error colors are TP green, FP red, FN blue.',
        '',
    ]
    (out_dir / 'report.md').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_specs(raw_specs: Sequence[str]) -> list[tuple[str, Path]]:
    specs = []
    seen: set[str] = set()
    for item in raw_specs:
        name, sep, path = item.partition(':')
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            raise ValueError(f'expected NAME:DIR, got {item!r}')
        if name in seen:
            raise ValueError(f'duplicate dataset name: {name!r}')
        seen.add(name)
        specs.append((name, Path(path)))
    return specs


def _roles_from_protocol(protocol: Path | None) -> dict[str, str]:
    roles: dict[str, str] = {}
    if protocol is None or not protocol.is_file():
        return roles
    with open(protocol, newline='') as handle:
        for row in csv.DictReader(handle):
            if not str(row.get('split', '')).endswith('_test'):
                continue
            name = str(row.get('dataset', '')).strip()
            role = str(row.get('role', '')).strip()
            if name and role:
                roles[name] = role.capitalize()
    return roles


def _selected_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.test_datasets:
        specs = _parse_specs(args.test_datasets)
    elif args.protocol:
        selected = set(args.datasets.split()) if args.datasets.strip() else None
        raw = dataset_specs(
            args.protocol, selected, repo=args.repo)
        specs = _parse_specs(raw)
    elif getattr(args, 'data_manifest', None):
        payload = json.loads(Path(args.data_manifest).read_text())
        specs = []
        for name, info in sorted(
                (payload.get('test_datasets') or {}).items()):
            root = info.get('root')
            if root:
                specs.append((name, Path(root)))
    else:
        raise ValueError(
            'pass --test_datasets, --protocol, or --data-manifest')
    if args.datasets.strip():
        selected = set(args.datasets.split())
        specs = [(name, root) for name, root in specs if name in selected]
    return specs


def run_audit(args: argparse.Namespace) -> int:
    specs = _selected_specs(args)
    if not specs:
        raise SystemExit('no datasets selected')
    roles = _roles_from_protocol(args.protocol)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_unmatched: list[dict] = []
    per_dataset: list[dict] = []
    single_spec = len(specs) == 1

    for dataset, test_root in specs:
        pred_dir, pred_files = _resolve_pred_dir(
            Path(args.pred_root), dataset, args.pred_suffix, single_spec)
        gt_dir = _resolve_gt_dir(test_root)
        print(f'[{dataset}] test_root={test_root}')
        if pred_dir is None:
            print(f'[{dataset}] no *{args.pred_suffix} predictions found')
            missing = []
            if gt_dir is not None:
                gt_by_name, _ = _gt_index(gt_dir)
                missing = [{'reason': 'missing_prediction'}
                           for _name in gt_by_name]
            per_dataset.append(_dataset_summary(
                dataset, roles.get(dataset, ''), [], missing))
            all_unmatched.append({
                'dataset': dataset,
                'image': '*',
                'reason': 'prediction_directory_missing',
                'path': str(args.pred_root),
            })
            continue
        indexed = _pairs_from_prediction_index(pred_dir)
        if indexed is not None:
            pairs, unmatched = indexed
            gt_count = len(pairs) + len(unmatched)
            source = 'prediction_index.csv'
        else:
            if gt_dir is None:
                print(f'[{dataset}] no GT mask directory under {test_root}')
                per_dataset.append(_dataset_summary(
                    dataset, roles.get(dataset, ''), [], []))
                all_unmatched.append({
                    'dataset': dataset,
                    'image': '*',
                    'reason': 'gt_directory_missing',
                    'path': str(test_root),
                })
                continue
            gt_by_name, gt_by_stem = _gt_index(gt_dir)
            pairs, unmatched = _match_predictions(
                pred_files, gt_by_name, gt_by_stem, args.pred_suffix)
            gt_count = len(gt_by_name)
            source = 'filename matching'
        if args.limit:
            pairs = pairs[:int(args.limit)]
        for item in unmatched:
            item['dataset'] = dataset
            all_unmatched.append(item)
        print(
            f'[{dataset}] predictions={len(pred_files)} '
            f'GT={gt_count} matched={len(pairs)} '
            f'unmatched={len(unmatched)} source={source}')
        if not pairs:
            per_dataset.append(_dataset_summary(
                dataset, roles.get(dataset, ''), [], unmatched))
            continue

        tasks = [
            (dataset, gt_path.name, str(pred_path), str(gt_path),
             args.empty_policy, int(args.small_px))
            for gt_path, pred_path in pairs
        ]
        workers = int(args.workers) if args.workers > 0 else min(
            8, os.cpu_count() or 1)
        if workers > 1 and len(tasks) > 1:
            # Linux fork keeps __main__ task functions directly usable and
            # avoids re-importing the CLI module in every worker.
            try:
                context = mp.get_context('fork')
            except ValueError:
                context = None
            with ProcessPoolExecutor(
                    max_workers=workers, mp_context=context) as executor:
                rows = list(executor.map(_safe_audit, tasks, chunksize=1))
        else:
            rows = [_safe_audit(task) for task in tasks]
        all_rows.extend(rows)
        summary = _dataset_summary(
            dataset, roles.get(dataset, ''), rows, unmatched)
        per_dataset.append(summary)

        # Save comparison PNGs for all images or the worst N per dataset.
        if args.save_error_maps:
            selected = [row for row in rows if row.get('status') == 'ok']
        else:
            selected = sorted(
                (row for row in rows if row.get('status') == 'ok'),
                key=lambda row: (
                    _float(row.get('dice'), math.inf),
                    -_float(row.get('component_count_error'), 0.0),
                ),
            )[:max(0, int(args.save_worst_per_dataset))]
        for row in selected:
            relative = _save_compare(row, out_dir, int(args.error_map_max_side))
            if relative:
                row['compare_png'] = relative

    if not all_rows and not per_dataset:
        print('No predictions were audited.')
        return 1

    _write_csv(out_dir / 'per_image.csv', IMAGE_FIELDS, all_rows)
    _write_csv(out_dir / 'per_dataset.csv', DATASET_FIELDS, per_dataset)
    macro = _macro_summary(per_dataset)
    _write_csv(out_dir / 'macro_summary.csv', MACRO_FIELDS, macro)
    _write_csv(out_dir / 'unmatched.csv', UNMATCHED_FIELDS, all_unmatched)
    _write_report(
        out_dir, per_dataset, macro, all_rows, all_unmatched, args)

    all_summary = next(
        (row for row in macro if row.get('scope') == 'All'), {})
    payload = {
        'pred_root': str(args.pred_root),
        'protocol': str(args.protocol) if args.protocol else None,
        'empty_policy': args.empty_policy,
        'n_images': len([row for row in all_rows
                         if row.get('status') == 'ok']),
        'n_size_mismatch': len([row for row in all_rows
                                if row.get('status') == 'size_mismatch']),
        'n_read_error': len([row for row in all_rows
                             if row.get('status') == 'read_error']),
        'n_metric_error': len([row for row in all_rows
                               if row.get('status') == 'metric_error']),
        'n_unmatched': len(all_unmatched),
        'macro_all': all_summary,
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(payload, indent=2, default=float) + '\n')

    print(f'\nWrote: {out_dir}/per_image.csv')
    print(f'Wrote: {out_dir}/per_dataset.csv')
    print(f'Wrote: {out_dir}/macro_summary.csv')
    print(f'Wrote: {out_dir}/report.md')
    print(f'Images audited: {payload["n_images"]} '
          f'(size mismatch={payload["n_size_mismatch"]}, '
          f'unmatched={payload["n_unmatched"]})')

    if args.strict and (
            payload['n_size_mismatch'] or payload['n_read_error']
            or payload['n_metric_error'] or payload['n_unmatched']):
        print('Strict mode: unmatched/size-mismatch/read errors found.',
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, required=True,
                        help='Root containing <DATASET> prediction dirs')
    parser.add_argument('--test_datasets', nargs='+', default=None,
                        help='NAME:DIR ground-truth roots (same as evaluate.py)')
    parser.add_argument('--protocol', type=Path, default=None,
                        help='protocol_manifest.csv; derives test roots and roles')
    parser.add_argument('--data-manifest', type=Path, default=None,
                        help='provenance/data_manifest.json from a training '
                             'run; derives test roots when no protocol is given')
    parser.add_argument('--datasets', default='',
                        help='space-separated dataset filter')
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='base for relative mask paths in the protocol')
    parser.add_argument('--out', type=Path,
                        default=Path('results/audit/predictions'),
                        help='output directory (read-only source tree is not touched)')
    parser.add_argument('--pred-suffix', default='_result.png',
                        help='prediction filename suffix (default: _result.png)')
    parser.add_argument('--limit', type=int, default=0,
                        help='audit at most N images per dataset (0 = all)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel processes; 0 = min(8, CPU count)')
    parser.add_argument('--empty-policy', choices=['penalty', 'nan'],
                        default='penalty',
                        help='empty-mask metric convention (default: penalty)')
    parser.add_argument('--small-px', type=int, default=20,
                        help='small-component threshold in pixels')
    parser.add_argument('--save-error-maps', action='store_true',
                        help='save a GT/pred/error/skeleton PNG for every image')
    parser.add_argument('--save-worst-per-dataset', type=int, default=5,
                        help='save comparison PNGs for the worst N images '
                             'per dataset (default: 5; 0 disables)')
    parser.add_argument('--error-map-max-side', type=int, default=1024,
                        help='downscale comparison PNG long side (0 = original)')
    parser.add_argument('--strict', action='store_true',
                        help='return nonzero on unmatched/size-mismatch/read errors')
    args = parser.parse_args()
    if not args.test_datasets and not args.protocol and not args.data_manifest:
        parser.error(
            'pass --test_datasets, --protocol, or --data-manifest')
    return run_audit(args)


if __name__ == '__main__':
    raise SystemExit(main())
