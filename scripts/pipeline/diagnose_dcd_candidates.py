#!/usr/bin/env python3
"""Audit the four Stage-2 candidates of a DCD checkpoint.

This is an analysis-only tool.  It never trains and it never changes the
deployed prediction rule.  For each image it runs one forward pass, exports
the four post-gating Stage-2 masks at GT resolution, and reports:

* per-candidate Dice / IoU / clDice / HD95;
* the deployed candidate (slot 0 for gating, ``deployment-head`` for fixed);
* the per-image oracle and worst candidate under Dice;
* selection regret, candidate spread, gating top-1 accuracy;
* pairwise candidate disagreement and mean IoU.

The default sample is the full validation split plus an evenly spaced
stratified subset of each test dataset.  Use ``--max-per-dataset 0`` for all
test images, or ``--splits val`` for the faster go/no-go audit.

Example:
    PYTHONPATH=src python scripts/pipeline/diagnose_dcd_candidates.py \
      --checkpoint results/runs/dcd/all_dense/seed_0/best_model.pth \
      --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
      --seed 0 --splits val test --max-per-dataset 20 \
      --output-dir results/ablations/dcd/candidate_audit/seed_0

Inputs are the canonical protocol manifest and a training checkpoint.  The
checkpoint must use the gating or fixed Stage-2 selector; aggregate selectors
(mean/gating_soft/iou) do not expose one deployed candidate and are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.data import Sample, load_mask  # noqa: E402
from baseline_zoo.runtime import binary_metrics, build_parser  # noqa: E402

METRICS = ('dice', 'iou', 'cldice', 'hd95')
ROLE_ORDER = ('Seen', 'Unseen', 'Novel')
SELECTED_COLUMNS = [
    'split', 'role', 'dataset', 'image', 'selected_candidate',
    'best_candidate', 'worst_candidate', 'gating_top1',
    'pairwise_disagreement', 'pairwise_iou',
    *[f'selected_{metric}' for metric in METRICS],
    *[f'oracle_{metric}' for metric in METRICS],
    *[f'worst_{metric}' for metric in METRICS],
    *[f'oracle_minus_selected_{metric}' for metric in METRICS],
    'dice_regret', 'dice_spread',
]
CANDIDATE_COLUMNS = [
    'split', 'role', 'dataset', 'candidate', 'n', 'selected',
    *[f'{metric}_mean' for metric in METRICS],
    *[f'{metric}_std' for metric in METRICS],
]


def _resolve(path_text: str) -> Path:
    path = Path(str(path_text))
    return path if path.is_absolute() else REPO_ROOT / path


def _role(value: str) -> str:
    text = (value or '').strip().lower()
    return {'seen': 'Seen', 'unseen': 'Unseen', 'novel': 'Novel'}.get(
        text, value or 'Unknown')


def _split_name(value: str) -> str:
    """Normalize manifest splits to the two audit partitions."""
    text = (value or '').strip().lower()
    if text == 'val':
        return 'val'
    if text.endswith('_test') or text == 'test':
        return 'test'
    return text or 'unknown'


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None]
    if not values:
        return float('nan')
    return float(np.nanmean(values))


def _std(values) -> float:
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return float('nan')
    return float(np.nanstd(values, ddof=1))


def _finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _seed_from_protocol(protocol: Path) -> int:
    summary = protocol.parent / 'protocol_summary.json'
    if summary.is_file():
        try:
            seed = json.loads(summary.read_text()).get('seed')
            if seed is not None:
                return int(seed)
        except (OSError, ValueError, TypeError):
            pass
    match = re.search(r'seed[_-]?(\d+)', str(protocol.parent))
    return int(match.group(1)) if match else 0


def _read_manifest(protocol: Path) -> list[dict]:
    with open(protocol, newline='') as handle:
        return list(csv.DictReader(handle))


def _split_filter(rows: list[dict], splits: tuple[str, ...]) -> list[dict]:
    wanted = {str(split).strip().lower() for split in splits}
    selected = []
    for row in rows:
        split = str(row.get('split', '')).strip().lower()
        if 'val' in wanted and split == 'val':
            selected.append(row)
        elif 'test' in wanted and split.endswith('_test'):
            selected.append(row)
        elif 'train' in wanted and split == 'train':
            selected.append(row)
    return selected


def _stratified_limit(rows: list[dict], max_per_dataset: int) -> list[dict]:
    """Keep at most N evenly spaced rows per dataset, deterministically."""
    if max_per_dataset <= 0:
        return list(rows)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row['dataset']].append(row)
    kept = []
    for dataset in sorted(by_dataset):
        items = sorted(by_dataset[dataset],
                       key=lambda item: item.get('source_image', ''))
        if len(items) <= max_per_dataset:
            kept.extend(items)
            continue
        indices = np.linspace(
            0, len(items) - 1, max_per_dataset).round().astype(int)
        kept.extend(items[index] for index in sorted(set(indices)))
    return kept


def _sample_from_row(row: dict) -> Sample:
    image_path = _resolve(row['image_path'])
    mask_path = _resolve(row['mask_path'])
    return Sample(
        image_path=image_path,
        mask_path=mask_path,
        dataset=str(row['dataset']),
        stem=image_path.stem,
    )


def _load_checkpoint(adapter, checkpoint: Path, data_root: Path, seed: int,
                     device: torch.device, protocol: Path):
    if not checkpoint.is_file():
        raise SystemExit(f'missing checkpoint: {checkpoint}')
    parser = build_parser(adapter)
    args = parser.parse_args([
        '--stage', 'predict',
        '--checkpoint', str(checkpoint),
        '--workdir', str(checkpoint.parent),
        '--data-root', str(data_root),
        '--device', str(device),
        '--seed', str(seed),
        '--use_wandb', 'false',
    ])
    payload = torch.load(checkpoint, map_location='cpu',
                         weights_only=False)
    if not isinstance(payload.get('model_state_dict'), dict):
        raise SystemExit(f'checkpoint {checkpoint} has no model_state_dict')

    adapter.args = args
    adapter.device = device
    adapter.begin_prediction()
    adapter.checkpoint_config = payload.get('config', {}) or {}
    adapter_config = (
        payload.get('adapter_config')
        or adapter.checkpoint_config.get('adapter_config', {})
        or {})
    # Match run_prediction: a checkpoint's selector must not be overwritten
    # by the CLI default when the checkpoint recorded its own value.
    if 'head_select' in adapter_config:
        args.head_select = None
    adapter.restore_config(args, adapter_config)
    model = adapter.build_model(args).to(device)
    adapter.load_state_dict(model, payload['model_state_dict'])
    model.eval()
    return adapter, model, args, payload, protocol


def _pairwise_stats(masks: np.ndarray) -> tuple[float, float]:
    disagreements = []
    ious = []
    count = masks.shape[0]
    for left in range(count):
        for right in range(left + 1, count):
            a = masks[left].astype(bool)
            b = masks[right].astype(bool)
            union = np.logical_or(a, b)
            intersection = np.logical_and(a, b).sum()
            disagreements.append(float(np.logical_xor(a, b).mean()))
            ious.append(
                1.0 if not union.any()
                else float(intersection / union.sum()))
    return _mean(disagreements), _mean(ious)


def _candidate_indices(dice: list[float], selected: int) -> tuple[int, int]:
    """Return (best, worst) by Dice, with selected winning exact ties."""
    best_value = max(dice)
    if dice[selected] >= best_value - 1e-12:
        best = selected
    else:
        best = int(np.argmax(dice))
    worst = int(np.argmin(dice))
    return best, worst


def _selection_row(split: str, role: str, dataset: str, image: str,
                   selected: int, values: list[dict]) -> dict:
    dice = [row['dice'] for row in values]
    best, worst = _candidate_indices(dice, selected)
    disagreement, agreement = _pairwise_stats(
        np.stack([row['mask'] for row in values]))
    row = {
        'split': split,
        'role': role,
        'dataset': dataset,
        'image': image,
        'selected_candidate': selected,
        'best_candidate': best,
        'worst_candidate': worst,
        'gating_top1': int(best == selected),
        'pairwise_disagreement': disagreement,
        'pairwise_iou': agreement,
        'dice_regret': float(dice[best] - dice[selected]),
        'dice_spread': float(dice[best] - dice[worst]),
    }
    for metric in METRICS:
        row[f'selected_{metric}'] = _finite(values[selected][metric])
        row[f'oracle_{metric}'] = _finite(values[best][metric])
        row[f'worst_{metric}'] = _finite(values[worst][metric])
        row[f'oracle_minus_selected_{metric}'] = (
            None if row[f'oracle_{metric}'] is None
            or row[f'selected_{metric}'] is None
            else row[f'oracle_{metric}'] - row[f'selected_{metric}'])
    return row


def _candidate_row(split: str, role: str, dataset: str,
                   candidate: int, selected: int, rows: list[dict]) -> dict:
    row = {
        'split': split,
        'role': role,
        'dataset': dataset,
        'candidate': candidate,
        'n': len(rows),
        'selected': int(candidate == selected),
    }
    for metric in METRICS:
        values = [row_[metric] for row_ in rows]
        row[f'{metric}_mean'] = _mean(values)
        row[f'{metric}_std'] = _std(values)
    return row


def _macro_candidate_rows(split: str, grouped: dict, selected: int,
                          roles: dict[str, str]) -> list[dict]:
    """Equal-dataset macro candidate rows for each role and All."""
    by_role = defaultdict(list)
    for dataset, rows in grouped.items():
        role = roles.get(dataset, 'Unknown')
        per_role = [
            _candidate_row(split, role, dataset, candidate, selected,
                           rows[candidate])
            for candidate in range(4)
        ]
        by_role[role].extend(per_role)
    output = []
    role_sets = list(ROLE_ORDER) + sorted(
        set(by_role) - set(ROLE_ORDER))
    for role in role_sets:
        rows = by_role.get(role, [])
        if not rows:
            continue
        for candidate in range(4):
            selected_rows = [row for row in rows
                             if row['candidate'] == candidate]
            macro = {
                'split': split,
                'role': role,
                'candidate': candidate,
                'n': len(selected_rows),
                'selected': int(candidate == selected),
            }
            for metric in METRICS:
                macro[f'{metric}_mean'] = _mean(
                    [row[f'{metric}_mean'] for row in selected_rows])
                macro[f'{metric}_std'] = _std(
                    [row[f'{metric}_mean'] for row in selected_rows])
            output.append(macro)
    # All = equal weight over every dataset, not the average of role means.
    all_rows = [row for rows in by_role.values() for row in rows]
    for candidate in range(4):
        selected_rows = [row for row in all_rows
                         if row['candidate'] == candidate]
        macro = {
            'split': split,
            'role': 'All',
            'candidate': candidate,
            'n': len(selected_rows),
            'selected': int(candidate == selected),
        }
        for metric in METRICS:
            macro[f'{metric}_mean'] = _mean(
                [row[f'{metric}_mean'] for row in selected_rows])
            macro[f'{metric}_std'] = _std(
                [row[f'{metric}_mean'] for row in selected_rows])
        output.append(macro)
    return output


def _macro_selection_rows(split: str, grouped: dict,
                          roles: dict[str, str]) -> list[dict]:
    by_role = defaultdict(list)
    for dataset, rows in grouped.items():
        role = roles.get(dataset, 'Unknown')
        by_role[role].extend(rows)
    output = []
    role_sets = list(ROLE_ORDER) + sorted(
        set(by_role) - set(ROLE_ORDER))
    for role in role_sets:
        rows = by_role.get(role, [])
        if not rows:
            continue
        macro = {
            'split': split,
            'role': role,
            'n_datasets': len({row['dataset'] for row in rows}),
            'n_images': len(rows),
        }
        for key in (
                'gating_top1', 'dice_regret', 'dice_spread',
                'pairwise_disagreement', 'pairwise_iou',
                *[f'selected_{metric}' for metric in METRICS],
                *[f'oracle_{metric}' for metric in METRICS],
                *[f'worst_{metric}' for metric in METRICS],
                *[f'oracle_minus_selected_{metric}' for metric in METRICS]):
            macro[f'{key}_mean'] = _mean([row[key] for row in rows])
            macro[f'{key}_std'] = _std([row[key] for row in rows])
        output.append(macro)
    all_rows = [row for rows in by_role.values() for row in rows]
    macro = {
        'split': split,
        'role': 'All',
        'n_datasets': len({row['dataset'] for row in all_rows}),
        'n_images': len(all_rows),
    }
    for key in (
            'gating_top1', 'dice_regret', 'dice_spread',
            'pairwise_disagreement', 'pairwise_iou',
            *[f'selected_{metric}' for metric in METRICS],
            *[f'oracle_{metric}' for metric in METRICS],
            *[f'worst_{metric}' for metric in METRICS],
            *[f'oracle_minus_selected_{metric}' for metric in METRICS]):
        macro[f'{key}_mean'] = _mean([row[key] for row in all_rows])
        macro[f'{key}_std'] = _std([row[key] for row in all_rows])
    output.append(macro)
    return output


def evaluate_candidates(adapter, model, rows: list[dict],
                        device: torch.device, empty_policy: str,
                        selected: int) -> tuple[list[dict], list[dict],
                                                 dict[str, list[dict]]]:
    """Run one forward per image and return per-image candidate/selection rows."""
    candidate_rows = []
    selection_rows = []
    roles = {}
    for index, row in enumerate(rows, 1):
        sample = _sample_from_row(row)
        split = _split_name(str(row['split']))
        role = _role(row.get('role', 'Unknown'))
        roles[sample.dataset] = role
        gt = load_mask(sample.mask_path)
        prepared = adapter.prepare_eval_sample(sample, device)
        masks = adapter.predict_candidates(model, prepared, sample)
        masks = np.asarray(masks, dtype=bool)
        if masks.shape[0] != 4:
            raise RuntimeError(
                f'{sample.image_path}: expected 4 candidates, '
                f'got {masks.shape[0]}')
        values = []
        for candidate in range(4):
            values.append({
                'mask': masks[candidate],
                **binary_metrics(
                    masks[candidate], gt, empty_policy=empty_policy),
            })
        for candidate, value in enumerate(values):
            candidate_rows.append({
                'split': split,
                'role': role,
                'dataset': sample.dataset,
                'image': sample.stem,
                'candidate': candidate,
                'selected': int(candidate == selected),
                'dice': _finite(value['dice']),
                'iou': _finite(value['iou']),
                'cldice': _finite(value['cldice']),
                'hd95': _finite(value['hd95']),
                'empty': int(value['empty']),
            })
        selection_rows.append(_selection_row(
            split, role, sample.dataset, sample.stem, selected, values))
        if index % 100 == 0:
            print(f'  candidates: {index}/{len(rows)} images', flush=True)

    by_split_candidates = defaultdict(lambda: defaultdict(
        lambda: defaultdict(list)))
    for row in candidate_rows:
        by_split_candidates[row['split']][row['dataset']][
            row['candidate']].append(row)
    by_split_selection = defaultdict(lambda: defaultdict(list))
    for row in selection_rows:
        by_split_selection[row['split']][row['dataset']].append(row)

    candidate_macro = []
    selection_macro = []
    for split in sorted(by_split_candidates):
        candidate_macro.extend(_macro_candidate_rows(
            split, by_split_candidates[split], selected, roles))
        selection_macro.extend(_macro_selection_rows(
            split, by_split_selection[split], roles))

    candidate_by_dataset = []
    for split in sorted(by_split_candidates):
        for dataset in sorted(by_split_candidates[split]):
            for candidate in range(4):
                items = by_split_candidates[split][dataset][candidate]
                candidate_by_dataset.append(_candidate_row(
                    split, roles.get(dataset, 'Unknown'), dataset,
                    candidate, selected, items))
    selection_by_dataset = []
    for split in sorted(by_split_selection):
        for dataset in sorted(by_split_selection[split]):
            items = by_split_selection[split][dataset]
            row = {
                'split': split,
                'role': roles.get(dataset, 'Unknown'),
                'dataset': dataset,
                'n': len(items),
            }
            for key in (
                    'gating_top1', 'dice_regret', 'dice_spread',
                    'pairwise_disagreement', 'pairwise_iou',
                    *[f'selected_{metric}' for metric in METRICS],
                    *[f'oracle_{metric}' for metric in METRICS],
                    *[f'worst_{metric}' for metric in METRICS],
                    *[f'oracle_minus_selected_{metric}' for metric in METRICS]):
                row[f'{key}_mean'] = _mean([item[key] for item in items])
                row[f'{key}_std'] = _std([item[key] for item in items])
            selection_by_dataset.append(row)
    return candidate_rows, selection_rows, {
        'candidate_by_dataset': candidate_by_dataset,
        'candidate_by_role': candidate_macro,
        'selection_by_dataset': selection_by_dataset,
        'selection_by_role': selection_macro,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _selection_metric_keys() -> tuple[str, ...]:
    return (
        'gating_top1', 'dice_regret', 'dice_spread',
        'pairwise_disagreement', 'pairwise_iou',
        *[f'selected_{metric}' for metric in METRICS],
        *[f'oracle_{metric}' for metric in METRICS],
        *[f'worst_{metric}' for metric in METRICS],
        *[f'oracle_minus_selected_{metric}' for metric in METRICS],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test', 'train'])
    parser.add_argument('--max-per-dataset', type=int, default=20,
                        help='stratified test images per dataset; 0 = all')
    parser.add_argument('--empty-metric-policy', default='penalty',
                        choices=['penalty', 'nan'])
    parser.add_argument('--device', default='cuda'
                        if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', default=None)
    args = parser.parse_args(argv)

    protocol = args.protocol
    if protocol is None:
        seed = int(args.seed or 0)
        protocol = (REPO_ROOT / 'data' / f'sacm_3shot_seed{seed}' /
                    'protocol_manifest.csv')
    protocol = protocol if protocol.is_absolute() else REPO_ROOT / protocol
    if not protocol.is_file():
        raise SystemExit(f'missing protocol manifest: {protocol}')
    seed = int(args.seed if args.seed is not None
               else _seed_from_protocol(protocol))
    data_root = args.data_root
    if data_root is None:
        data_root = protocol.parent
    elif not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    manifest_rows = _split_filter(_read_manifest(protocol), tuple(args.splits))
    if not manifest_rows:
        raise SystemExit(
            f'no rows for splits={args.splits} in {protocol}')
    # Validation is tiny (6 images); the limit is intended for test.  Keep
    # val complete unless the caller explicitly asks for a smaller subset.
    limited = []
    for split in args.splits:
        split_rows = [row for row in manifest_rows
                      if (row['split'] == 'val' and split == 'val')
                      or (row['split'].endswith('_test') and split == 'test')
                      or (row['split'] == 'train' and split == 'train')]
        if split == 'val':
            limited.extend(split_rows)
        else:
            limited.extend(_stratified_limit(
                split_rows, int(args.max_per_dataset)))
    manifest_rows = limited

    device = torch.device(args.device)
    adapter = load_adapter('dcd')
    adapter, model, _model_args, payload, _ = _load_checkpoint(
        adapter, checkpoint, data_root, seed, device, protocol)
    if adapter.head_select not in ('gating', 'fixed'):
        raise SystemExit(
            'candidate diagnostics require a single deployment slot; '
            f'head_select={adapter.head_select!r}')
    selected = adapter.selected_candidate_index()

    print(f'candidate audit: {checkpoint}')
    print(f'  protocol={protocol} seed={seed} device={device}')
    print(f'  splits={args.splits} images={len(manifest_rows)} '
          f'selected_slot={selected} head_select={adapter.head_select}')
    candidate_rows, selection_rows, tables = evaluate_candidates(
        adapter, model, manifest_rows, device,
        args.empty_metric_policy, selected)

    write_csv(output_dir / 'per_image_candidates.csv', candidate_rows,
              ['split', 'role', 'dataset', 'image', 'candidate', 'selected',
               *[f'{metric}' for metric in METRICS], 'empty'])
    write_csv(output_dir / 'per_image_selection.csv', selection_rows,
              SELECTED_COLUMNS)
    write_csv(output_dir / 'candidate_by_dataset.csv',
              tables['candidate_by_dataset'], CANDIDATE_COLUMNS)
    write_csv(output_dir / 'candidate_by_role.csv',
              tables['candidate_by_role'], CANDIDATE_COLUMNS)
    selection_columns = [
        'split', 'role', 'dataset', 'n',
        *[f'{key}_mean' for key in _selection_metric_keys()],
        *[f'{key}_std' for key in _selection_metric_keys()],
    ]
    write_csv(output_dir / 'selection_by_dataset.csv',
              tables['selection_by_dataset'], selection_columns)
    selection_role_columns = [
        'split', 'role', 'n_datasets', 'n_images',
        *[f'{key}_mean' for key in _selection_metric_keys()],
        *[f'{key}_std' for key in _selection_metric_keys()],
    ]
    write_csv(output_dir / 'selection_by_role.csv',
              tables['selection_by_role'], selection_role_columns)

    split_counts = defaultdict(int)
    for row in manifest_rows:
        split_counts[_split_name(str(row['split']))] += 1
    summary = {
        'name': args.name,
        'checkpoint': str(checkpoint),
        'checkpoint_epoch': payload.get('epoch'),
        'protocol': str(protocol),
        'seed': seed,
        'data_root': str(data_root),
        'splits': list(args.splits),
        'max_per_dataset': int(args.max_per_dataset),
        'selected_candidate': selected,
        'head_select': adapter.head_select,
        'deployment_head': getattr(adapter, 'deployment_head', None),
        'split_counts': dict(split_counts),
        'n_images': len(manifest_rows),
        'outputs': {
            'per_image_candidates': str(output_dir / 'per_image_candidates.csv'),
            'per_image_selection': str(output_dir / 'per_image_selection.csv'),
            'candidate_by_dataset': str(output_dir / 'candidate_by_dataset.csv'),
            'candidate_by_role': str(output_dir / 'candidate_by_role.csv'),
            'selection_by_dataset': str(output_dir / 'selection_by_dataset.csv'),
            'selection_by_role': str(output_dir / 'selection_by_role.csv'),
        },
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(f'wrote candidate audit to {output_dir}')
    print('  selected Slot Metrics:', {
        row['role']: {metric: round(row[f'selected_{metric}_mean'], 4)
                      for metric in METRICS}
        for row in tables['selection_by_role']
        if row['role'] in ('Seen', 'Unseen', 'Novel', 'All')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
