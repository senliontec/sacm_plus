#!/usr/bin/env python
"""Select the deployment blend for SACM-STC checkpoints on the val split.

For every checkpoint and every fixed blend g, this evaluates the GT-resolution
Dice/clDice/HD95 of

    final = dense + g * (structure - dense)

The best (checkpoint, g) pair can be written into a calibrated checkpoint
(``config.adapter_config.calibrated_blend_g``), which ``predict_sample`` then
uses instead of the learned gate.  g=0 is a legal and often optimal choice:
the dense head alone is sometimes stronger than any composition.

Usage:
  PYTHONPATH=src python scripts/calibrate_sacm_stc.py \
    --checkpoint results/runs/sacm_stc/default/seed_0/best_model.pth \
                 results/runs/sacm_stc/default/seed_0/checkpoints/epoch_0300.pth \
    --data-root data/sacm_3shot_seed0 --device cuda:0 \
    --metric dice --save-calibrated results/runs/sacm_stc/default/seed_0/calibrated_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    if int(getattr(args, 'use_typed_stage2', 0)) != 1:
        raise SystemExit(
            f'{checkpoint}: blend calibration is defined only for the '
            'typed Stage-2 prototype (use_typed_stage2=1). The shipped '
            'all_dense preset deploys the gating slot directly and has no '
            'typed_composer; evaluate its fixed candidates on the val split '
            'instead.')
    args.input_size = int(payload.get('input_size', 1024))
    adapter.input_size = args.input_size
    adapter.args = args
    adapter.device = device
    model = adapter.build_model(args)
    state = payload.get('model_state_dict') or {}
    adapter.load_state_dict(model, state)
    model.to(device).eval()
    return model, payload


def evaluate_g(adapter, model, samples, g: float, resolution: int,
               device: torch.device):
    overall = {'dice': [], 'iou': [], 'cldice': [], 'hd95': []}
    per_dataset = {}
    for sample in samples:
        prepared = adapter.prepare_eval_sample(sample, device)
        with torch.no_grad():
            stage2, _ = adapter._forward(
                model, prepared, return_stage1=False)
            heads = [
                F.interpolate(
                    stage2[:, k:k + 1], (resolution, resolution),
                    mode='bilinear', align_corners=False)
                for k in range(4)
            ]
            outputs = model.typed_composer(
                *heads, resolution, blend_override=float(g))
        mask = load_mask(sample.mask_path)
        height, width = mask.shape
        logits = F.interpolate(
            outputs['final'], size=(height, width), mode='bilinear',
            align_corners=False)
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
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


def selection_value(overall, metric: str) -> float:
    if metric == 'hd95':
        return -float(overall['hd95'])
    if metric == 'cldice':
        return float(overall['cldice'])
    if metric == 'combo':
        return (float(overall['dice']) + float(overall['cldice'])
                - float(overall['hd95']) / 100.0)
    return float(overall['dice'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--data-root', default='data/sacm_3shot_seed0')
    parser.add_argument('--split', default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--blend-grid', nargs='+', type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
        '--metric', choices=['dice', 'cldice', 'hd95', 'combo'],
        default='dice')
    parser.add_argument('--output', default='')
    parser.add_argument(
        '--save-calibrated', default='',
        help='write a copy of the best checkpoint with calibrated_blend_g')
    args = parser.parse_args()

    device = torch.device(args.device)
    adapter = load_adapter('sacm_stc')
    samples = load_split_samples(Path(args.data_root), args.split)

    report = {}
    best = None
    for checkpoint in args.checkpoint:
        checkpoint = Path(checkpoint)
        model, payload = load_model(adapter, checkpoint, device)
        resolution = int(getattr(adapter.args, 'typed_resolution', 512))
        per_g = {}
        for g in args.blend_grid:
            means, dataset_means = evaluate_g(
                adapter, model, samples, g, resolution, device)
            per_g[str(float(g))] = {
                'overall': means,
                'per_dataset': dataset_means,
            }
        chosen = max(per_g, key=lambda key: selection_value(
            per_g[key]['overall'], args.metric))
        report[str(checkpoint)] = {
            'resolution': resolution,
            'blend_grid': list(args.blend_grid),
            'metric': args.metric,
            'chosen_g': float(chosen),
            'results': per_g,
        }
        score = per_g[chosen]['overall']
        print(f'{checkpoint}: g={float(chosen):.2f} '
              f'dice={score["dice"]:.4f} '
              f'cldice={score["cldice"]:.4f} '
              f'hd95={score["hd95"]:.3f}')
        value = selection_value(score, args.metric)
        if best is None or value > best['value']:
            best = {
                'checkpoint': str(checkpoint),
                'g': float(chosen),
                'score': score,
                'value': value,
                'payload': payload,
            }
        del model
        torch.cuda.empty_cache()

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print('wrote', args.output)

    if best is None:
        raise SystemExit('no checkpoint evaluated')
    print(f'BEST: {best["checkpoint"]} g={best["g"]:.2f} '
          f'dice={best["score"]["dice"]:.4f} '
          f'cldice={best["score"]["cldice"]:.4f} '
          f'hd95={best["score"]["hd95"]:.3f}')
    if args.save_calibrated:
        out = Path(args.save_calibrated)
        payload = best['payload']
        config = payload.setdefault('config', {})
        adapter_config = config.setdefault('adapter_config', {})
        adapter_config['calibrated_blend_g'] = float(best['g'])
        adapter_config['typed_output'] = 'compose'
        payload['calibration'] = {
            'source_checkpoint': best['checkpoint'],
            'blend_g': float(best['g']),
            'metric': args.metric,
            'score': best['score'],
            'blend_grid': list(args.blend_grid),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        print('saved calibrated checkpoint ->', out)


if __name__ == '__main__':
    main()
