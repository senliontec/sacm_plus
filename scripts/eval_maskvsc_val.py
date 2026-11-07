#!/usr/bin/env python
"""Evaluate a MaskVSC checkpoint on the validation split.

Diagnostic helper used to compare the released per-image min-max threshold
with the raw sigmoid threshold, and to inspect stage-1 vs stage-2 outputs.

Usage:
  PYTHONPATH=src python scripts/eval_maskvsc_val.py \
    --checkpoint results/diagnostics/x/stage2_best.pth \
    --data-root data/sacm_3shot_seed0 --device cuda:0 --normalize none
"""
from __future__ import annotations

import argparse
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
    args = argparse.Namespace(stage='predict')
    adapter.prediction_mode = True
    adapter.restore_config(
        args, (payload.get('config') or {}).get('adapter_config', {}))
    args.input_size = int(payload.get('input_size', 512))
    adapter.input_size = args.input_size
    adapter.args = args
    adapter.device = device
    model = adapter.build_model(args)
    adapter.load_state_dict(model, payload.get('model_state_dict') or {})
    model.to(device).eval()
    return adapter, model, payload


def evaluate(adapter, model, samples, device):
    overall = {key: [] for key in ('dice', 'iou', 'cldice', 'hd95')}
    per_dataset = {}
    with torch.no_grad():
        for sample in samples:
            prepared = adapter.prepare_eval_sample(sample, device)
            pred = adapter.predict_sample(model, prepared, sample)
            gt = load_mask(sample.mask_path)
            values = binary_metrics(pred, gt, empty_policy='penalty')
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
        '--normalize', choices=['config', 'minmax', 'none'],
        default='config')
    args = parser.parse_args()
    device = torch.device(args.device)
    adapter = load_adapter('maskvsc')
    samples = load_split_samples(Path(args.data_root), args.split)
    print(f'val samples: {len(samples)}', flush=True)
    for raw in args.checkpoint:
        checkpoint = Path(raw)
        adapter, model, payload = load_model(adapter, checkpoint, device)
        if args.normalize != 'config':
            adapter._normalize = args.normalize
        means, per_dataset = evaluate(adapter, model, samples, device)
        stage = (payload.get('config') or {}).get('selected_stage', '?')
        use_vig = (payload.get('config') or {}).get(
            'adapter_config', {}).get('use_vig', '?')
        print(f'{checkpoint.name}: selected_stage={stage} use_vig={use_vig} '
              f'normalize={adapter._normalize} '
              f'dice={means["dice"]:.4f} iou={means["iou"]:.4f} '
              f'cldice={means["cldice"]:.4f} hd95={means["hd95"]:.2f}',
              flush=True)
        for name in sorted(per_dataset):
            row = per_dataset[name]
            print(f'  {name:14s} dice={row["dice"]:.4f} '
                  f'iou={row["iou"]:.4f} cldice={row["cldice"]:.4f} '
                  f'hd95={row["hd95"]:.2f}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
