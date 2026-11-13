#!/usr/bin/env python
"""Evaluate DCD checkpoints on the validation split with a chosen head rule.

This is a diagnostic for the multi-candidate experiments.  It supports the
non-typed Stage-2 path (four native dense candidates) with:

  gating : official order, slot 0 (top-gated candidate)
  mean   : average the four candidate logits
  iou    : argmax of the IoU prediction head
  fixed  : deployment-head
  config : use the checkpoint's stored head-select rule

The typed Stage-2 path is evaluated through its configured deployment output
(``typed_output``); a head-select override is ignored there.

Usage:
  PYTHONPATH=src python scripts/eval_dcd_val.py \
    --checkpoint results/diagnostics/x/best_model.pth \
    --data-root data/sacm_3shot_seed0 --head-select mean --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.data import load_mask, load_split_samples  # noqa: E402
from baseline_zoo.runtime import binary_metrics  # noqa: E402


def load_model(adapter, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    stored = payload.get('adapter')
    if stored and stored != adapter.id:
        raise SystemExit(
            f'{checkpoint}: adapter {stored!r} != {adapter.id!r}')
    args = argparse.Namespace(
        stage='predict', head_select=None, sam_checkpoint=None)
    adapter.prediction_mode = True
    adapter.restore_config(
        args, (payload.get('config') or {}).get('adapter_config', {}))
    args.input_size = int(payload.get('input_size', 1024))
    adapter.input_size = args.input_size
    adapter.args = args
    adapter.device = device
    model = adapter.build_model(args)
    state = payload.get('model_state_dict') or {}
    adapter.load_state_dict(model, state)
    model.to(device).eval()
    return model, payload


def evaluate(adapter, model, samples, device: torch.device):
    overall = {'dice': [], 'iou': [], 'cldice': [], 'hd95': []}
    per_dataset = {}
    with torch.no_grad():
        for sample in samples:
            prepared = adapter.prepare_eval_sample(sample, device)
            pred = adapter.predict_sample(model, prepared, sample)
            mask = load_mask(sample.mask_path)
            values = binary_metrics(pred, mask, empty_policy='penalty')
            bucket = per_dataset.setdefault(
                sample.dataset, {key: [] for key in overall})
            for key in overall:
                overall[key].append(values[key])
                bucket[key].append(values[key])
    means = {key: float(np.nanmean(values)) if values else float('nan')
             for key, values in overall.items()}
    dataset_means = {
        name: {key: float(np.nanmean(values))
               for key, values in values_by_key.items()}
        for name, values_by_key in per_dataset.items()
    }
    return means, dataset_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--data-root', default='data/sacm_3shot_seed0')
    parser.add_argument('--split', default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--head-select',
        choices=['config', 'gating', 'gating_soft', 'mean', 'iou', 'fixed'],
        default='config')
    parser.add_argument('--threshold', type=float, default=None,
                        help='override the deployment probability threshold')
    parser.add_argument('--stage1-fusion-weight', type=float, default=None,
                        help='override the Stage-1/Stage-2 logit blend')
    parser.add_argument('--stage1-fusion-mode', choices=['gated', 'mean'],
                        default=None)
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    device = torch.device(args.device)
    adapter = load_adapter('dcd')
    samples = load_split_samples(Path(args.data_root), args.split)
    print(f'val samples: {len(samples)} '
          f'datasets={sorted({s.dataset for s in samples})}', flush=True)

    report = {}
    for raw in args.checkpoint:
        checkpoint = Path(raw)
        model, payload = load_model(adapter, checkpoint, device)
        typed = int(getattr(adapter.args, 'use_typed_stage2', 0))
        if args.head_select != 'config':
            if typed:
                print(f'{checkpoint}: typed path ignores '
                      f'--head-select {args.head_select}', flush=True)
            else:
                adapter.head_select = args.head_select
        if args.threshold is not None:
            adapter.args.deploy_threshold = float(args.threshold)
        if args.stage1_fusion_weight is not None:
            adapter.args.stage1_fusion_weight = float(
                args.stage1_fusion_weight)
        if args.stage1_fusion_mode is not None:
            adapter.args.stage1_fusion_mode = args.stage1_fusion_mode
        means, per_dataset = evaluate(adapter, model, samples, device)
        report[str(checkpoint)] = {
            'head_select': (adapter.head_select if not typed
                            else getattr(adapter.args, 'typed_output',
                                         'compose')),
            'threshold': float(getattr(
                adapter.args, 'deploy_threshold', 0.5)),
            'overall': means,
            'per_dataset': per_dataset,
        }
        print(f'{checkpoint.name}: dice={means["dice"]:.4f} '
              f'iou={means["iou"]:.4f} cldice={means["cldice"]:.4f} '
              f'hd95={means["hd95"]:.2f}', flush=True)
        for name in sorted(per_dataset):
            row = per_dataset[name]
            print(f'  {name:14s} dice={row["dice"]:.4f} '
                  f'cldice={row["cldice"]:.4f} hd95={row["hd95"]:.2f}',
                  flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print('wrote', args.output)


if __name__ == '__main__':
    main()
