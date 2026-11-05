#!/usr/bin/env python3
"""Measure gate structural-error localization on all protocol test datasets.

This is a diagnostic pass over an existing final checkpoint.  It does NOT
train, does NOT run component pruning, and does NOT use target labels at
inference: target pixels are used only to score the gate after prediction.

For every dataset it reports:
  * AUROC / AUPRC against hard errors (|pred-GT| > 0.5)
  * precision / recall / F1 / IoU at fixed gate threshold 0.5
  * best-threshold F1 / IoU (oracle diagnostic, not deployable)
  * AUROC / AUPRC / F1@0.5 against the topology-aware gate target
    (missed/spurious/boundary errors)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# The gate utilities live next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from component_repair_eval import (  # noqa: E402
    forward,
    gate_localization_scores,
    load_model,
    protocol_test_specs,
    topology_gate_target,
)
from core.data import TestDataset  # noqa: E402

HARD_KEYS = (
    'auc', 'auprc', 'f1_0_5', 'precision_0_5', 'recall_0_5',
    'iou_0_5', 'best_f1', 'best_iou', 'best_threshold',
)
TOPO_KEYS = ('auc', 'auprc', 'f1_0_5')


def mean(values):
    vals = [float(v) for v in values if v is not None
            and isinstance(v, (int, float))
            and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else float('nan')


def aggregate(rows, keys, prefix):
    return {
        f'{prefix}{key}': mean([row[f'{prefix}{key}'] for row in rows])
        for key in keys
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-root', type=Path,
                    default=Path('data/sacm_3shot_seed0'))
    ap.add_argument('--protocol', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--expected-datasets', type=int, default=20)
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    protocol = args.protocol or (args.data_root / 'protocol_manifest.csv')
    if not protocol.is_file():
        raise SystemExit(f'protocol not found: {protocol}')
    specs = protocol_test_specs(protocol)
    if args.expected_datasets and len(specs) != args.expected_datasets:
        raise SystemExit(
            f'protocol {protocol} has {len(specs)} test datasets, '
            f'expected {args.expected_datasets}')
    for spec in specs:
        if (not (spec['root'] / 'images').is_dir()
                or not (spec['root'] / 'masks').is_dir()):
            raise SystemExit(
                f"missing images/masks for {spec['dataset']} at "
                f"{spec['root']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, rm, payload = load_model(args.preset, args.checkpoint, device)
    if rm is None:
        raise SystemExit(f'{args.preset} has no residual gate')
    print(f'preset={args.preset} checkpoint={args.checkpoint} '
          f'epoch={payload.get("epoch_completed")} '
          f'datasets={len(specs)}', flush=True)

    rows = []
    for spec in specs:
        dataset = spec['dataset']
        ds = TestDataset(str(spec['root']), split=None,
                         input_mode='resize', image_size=1024)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0)
        hard_metrics, topo_metrics = [], []
        for i, (image, gt, _, _, _) in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            image = image.to(device)
            p0, gate, _ = forward(model, rm, image)
            gt512 = F.adaptive_max_pool2d(gt.to(device), (512, 512))
            hard_target = (p0[0, 0] - gt512[0, 0]).abs() > 0.5
            hard_metrics.append(
                gate_localization_scores(gate[0, 0], hard_target))
            topo_metrics.append(gate_localization_scores(
                gate[0, 0], topology_gate_target(p0, gt512)))
            torch.cuda.empty_cache()

        row = {
            'model': args.run_name or args.preset,
            'preset': args.preset,
            'dataset': dataset,
            'role': spec['role'],
            'n_test': len(hard_metrics),
        }
        row.update(aggregate(hard_metrics, HARD_KEYS, 'gate_'))
        row.update(aggregate(topo_metrics, TOPO_KEYS, 'gate_topo_'))
        rows.append(row)
        print(json.dumps(row), flush=True)

    if not rows:
        raise SystemExit('no test rows evaluated')
    with (args.output_dir / 'per_dataset.csv').open(
            'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for role in ('Seen', 'Unseen', 'Novel'):
        sub = [r for r in rows if r['role'] == role]
        if not sub:
            continue
        summary[role] = {
            'n_datasets': len(sub),
            **aggregate(sub, HARD_KEYS, 'gate_'),
            **aggregate(sub, TOPO_KEYS, 'gate_topo_'),
        }
    summary['All'] = {
        'n_datasets': len(rows),
        **aggregate(rows, HARD_KEYS, 'gate_'),
        **aggregate(rows, TOPO_KEYS, 'gate_topo_'),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print('SUMMARY ' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
