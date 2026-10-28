"""Read-only A2SAM branch-cap and DCA1 diagnostics.

This tool loads an A2SAM ``best_model.pth`` checkpoint, runs the frozen
U-Net/prior on train/val samples, and reports:

* coarse probability statistics (p.max, foreground counts by threshold);
* persistence bar counts and Morse branch counts, including whether the
  configured branch cap is hit;
* the Dice-optimal xi target and the resulting structural uncertainty ``u``;
* optional full A2SAM predictions for selected datasets (default DCA1).

It never trains, never writes test predictions, and never imports ``3rd/``.
The output is intended for one-seed triage before a paper campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from baseline_zoo.adapters.a2sam import A2SAMAdapter, _sample_seed
from baseline_zoo.data import load_mask, load_split_samples
from baseline_zoo.models.a2sam import (
    build_a2sam,
    build_morse_complex,
    compute_persistence,
    dice_optimal_xi,
)


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred, target).sum()) / denom


def _load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f'{path}: unsupported checkpoint')
    stage_snapshot = (
        'unet_state_dict' in payload and 'prior_state_dict' in payload)
    if stage_snapshot:
        adapter_config = payload.get('config', {})
        config = {'adapter_config': adapter_config,
                  'seed': adapter_config.get('seed', 0)}
    else:
        if 'model_state_dict' not in payload:
            raise SystemExit(f'{path}: missing model_state_dict')
        config = payload.get('config', {})
        adapter_config = config.get('adapter_config', {})
    model = build_a2sam(
        image_size=1024,
        prompt_size=256,
        prior_min_sigma=float(adapter_config.get('prior_min_sigma', 0.05)),
    )
    if stage_snapshot:
        model.unet.load_state_dict(payload['unet_state_dict'], strict=True)
        model.prior.load_state_dict(payload['prior_state_dict'], strict=True)
    else:
        model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device).eval()
    return payload, config, adapter_config, model, stage_snapshot


def _args(adapter_config: dict, seed: int, workdir: str) -> Namespace:
    return Namespace(
        seed=int(seed),
        workdir=workdir,
        a2sam_prior_samples=int(adapter_config.get('prior_samples', 5)),
        a2sam_prior_min_sigma=float(
            adapter_config.get('prior_min_sigma', 0.05)),
        a2sam_coarse_threshold=float(
            adapter_config.get('coarse_threshold', 0.5)),
        a2sam_topology_backend=str(
            adapter_config.get('topology_backend', 'auto')),
        a2sam_min_persistence=float(
            adapter_config.get('min_persistence', 1e-3)),
        a2sam_max_branches=int(adapter_config.get('max_branches', 256)),
        a2sam_prior_grid_size=int(
            adapter_config.get('prior_grid_size', 32)),
        a2sam_fusion_inputs=str(
            adapter_config.get('fusion_inputs', 'release')),
        a2sam_prompt_detach=bool(
            adapter_config.get('prompt_detach', False)),
        a2sam_release_parity=bool(
            adapter_config.get('release_parity', False)),
        a2sam_ta_mode=str(adapter_config.get('ta_mode', 'paper')),
        a2sam_cps_mode=str(adapter_config.get('cps_mode', 'official')),
        input_size=1024,
    )


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def _samples(data_root: Path, split: str, datasets: set[str],
             max_images: int = 0):
    samples = load_split_samples(data_root, split)
    if datasets:
        samples = [s for s in samples if s.dataset in datasets]
    if max_images:
        samples = samples[:max_images]
    return samples


def _branch_sweep(
    probability: np.ndarray,
    *,
    backend: str,
    threshold: float,
    min_values: Iterable[float],
    caps: Iterable[int],
) -> list[dict]:
    rows = []
    for min_persistence in min_values:
        for max_branches in caps:
            morse = build_morse_complex(
                probability,
                backend=backend,
                coarse_threshold=threshold,
                min_persistence=float(min_persistence),
                max_branches=int(max_branches),
            )
            rows.append({
                'min_persistence': float(min_persistence),
                'max_branches': int(max_branches),
                'n_branches': int(morse.n_branches),
                'cap_saturated': bool(morse.n_branches >= int(max_branches)),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('results/paper/a2sam/seed_0/'
                                     'best_model.pth'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('data/sacm_3shot_seed0'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/diagnostics/a2sam_seed_0'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument(
        '--datasets', default='',
        help='comma-separated dataset filter; empty = all train/val')
    parser.add_argument(
        '--a2sam-datasets', default='DCA1',
        help='datasets for full A2SAM prediction; empty = skip')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None,
                        help='sampling seed; default from checkpoint or 0')
    parser.add_argument('--sweep', action='store_true',
                        help='also evaluate persistence/cap combinations')
    parser.add_argument('--sweep-min-persistence', default='0.001,0.01,0.05,0.1')
    parser.add_argument('--sweep-max-branches', default='64,128,256')
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f'missing checkpoint: {args.checkpoint}')
    if not args.data_root.is_dir():
        raise SystemExit(f'missing data root: {args.data_root}')

    device = torch.device(args.device)
    payload, config, adapter_config, model, stage_snapshot = (
        _load_checkpoint(args.checkpoint, device))
    seed = int(args.seed if args.seed is not None else
               config.get('seed', adapter_config.get('seed', 0)))
    if stage_snapshot:
        print('[triage] stage snapshot: A2SAM prediction disabled',
              flush=True)
    adapter = A2SAMAdapter()
    adapter.args = _args(adapter_config, seed, str(args.output))
    adapter.device = device
    adapter.input_size = 1024
    adapter.prediction_mode = True
    adapter._stage = 'a2sam'
    adapter._coarse_threshold = float(
        adapter_config.get('coarse_threshold', 0.5))

    datasets = set(_parse_list(args.datasets))
    a2sam_datasets = set(_parse_list(args.a2sam_datasets))
    if stage_snapshot:
        a2sam_datasets = set()
    backend = str(adapter.args.a2sam_topology_backend)
    threshold = float(adapter.args.a2sam_coarse_threshold)
    min_persistence = float(adapter.args.a2sam_min_persistence)
    max_branches = int(adapter.args.a2sam_max_branches)
    grid_size = int(adapter.args.a2sam_prior_grid_size)
    min_values = [float(item) for item in
                  _parse_list(args.sweep_min_persistence)]
    caps = [int(item) for item in _parse_list(args.sweep_max_branches)]

    args.output.mkdir(parents=True, exist_ok=True)
    stat_path = args.output / 'branch_stats.csv'
    sweep_path = args.output / 'branch_sweep.csv'
    stat_fields = [
        'split', 'dataset', 'stem', 'p_max',
        'pred_fg_0.3', 'pred_fg_0.4', 'pred_fg_0.5',
        'gt_fg_1024', 'gt_fg_orig',
        'n_bars', 'n_finite_bars', 'n_branches', 'cap_hit',
        'n_bars_above_cap',
        'xi_star', 'u_mean', 'u_max',
        'a2sam_dice', 'a2sam_pred_fg', 'a2sam_empty',
    ]
    sweep_fields = [
        'split', 'dataset', 'stem', 'min_persistence', 'max_branches',
        'n_branches', 'cap_saturated',
    ]
    with stat_path.open('w', newline='') as stat_handle, \
            sweep_path.open('w', newline='') as sweep_handle:
        stat_writer = csv.DictWriter(stat_handle, fieldnames=stat_fields)
        sweep_writer = csv.DictWriter(sweep_handle, fieldnames=sweep_fields)
        stat_writer.writeheader()
        sweep_writer.writeheader()
        summary = {'checkpoint': str(args.checkpoint),
                   'data_root': str(args.data_root),
                   'seed': seed, 'rows': 0, 'cap_hit_images': 0,
                   'stage_snapshot': bool(stage_snapshot), 'dca1': {}}
        for split in args.splits:
            samples = _samples(args.data_root, split, datasets,
                               args.max_images)
            print(f'[{split}] {len(samples)} samples', flush=True)
            for sample in samples:
                prepared = adapter.prepare_eval_sample(sample, device)
                image = prepared['image']
                with torch.no_grad():
                    logits, probability, _ = model.coarse(
                        image, threshold=threshold)
                p_np = probability[0, 0].detach().cpu().numpy()
                gt_1024 = adapter._load_mask(sample, 1024)[0].numpy() > 0.5
                gt_orig = load_mask(sample.mask_path)
                morse = build_morse_complex(
                    p_np, backend=backend, coarse_threshold=threshold,
                    min_persistence=min_persistence,
                    max_branches=max_branches)
                bars = compute_persistence(
                    p_np, backend=backend, min_persistence=0.0)
                n_bars = len(bars)
                n_finite = int(bars.finite.sum())
                xi_star = dice_optimal_xi(
                    morse, gt_1024, grid_size=grid_size)
                with torch.no_grad():
                    u_map, _, _ = adapter._sample_uncertainty(
                        model, image, probability,
                        _sample_seed(seed, sample.stem, 'triage'),
                        morse=morse)
                u_np = u_map[0].detach().cpu().numpy()
                row = {
                    'split': split,
                    'dataset': sample.dataset,
                    'stem': sample.stem,
                    'p_max': f'{float(p_np.max()):.6f}',
                    'pred_fg_0.3': int((p_np > 0.3).sum()),
                    'pred_fg_0.4': int((p_np > 0.4).sum()),
                    'pred_fg_0.5': int((p_np > 0.5).sum()),
                    'gt_fg_1024': int(gt_1024.sum()),
                    'gt_fg_orig': int(gt_orig.sum()),
                    'n_bars': n_bars,
                    'n_finite_bars': n_finite,
                    'n_branches': int(morse.n_branches),
                    'cap_hit': int(morse.n_branches >= max_branches),
                    'n_bars_above_cap': int(n_bars > max_branches),
                    'xi_star': f'{float(xi_star):.6f}',
                    'u_mean': f'{float(u_np.mean()):.6f}',
                    'u_max': f'{float(u_np.max()):.6f}',
                    'a2sam_dice': '',
                    'a2sam_pred_fg': '',
                    'a2sam_empty': '',
                }
                predict_a2sam = (
                    'all' in a2sam_datasets
                    or sample.dataset in a2sam_datasets)
                if predict_a2sam:
                    adapter._eval_cache = {}
                    with torch.no_grad():
                        pred = adapter.predict_sample(
                            model, prepared, sample)
                    pred = np.asarray(pred, dtype=bool)
                    row['a2sam_dice'] = f'{_dice(pred, gt_orig):.6f}'
                    row['a2sam_pred_fg'] = int(pred.sum())
                    row['a2sam_empty'] = int(pred.sum() == 0)
                    if sample.dataset == 'DCA1':
                        summary['dca1'][sample.stem] = {
                            'p_max': float(p_np.max()),
                            'pred_fg_0.5': int((p_np > 0.5).sum()),
                            'gt_fg_orig': int(gt_orig.sum()),
                            'a2sam_dice': float(row['a2sam_dice']),
                            'a2sam_pred_fg': int(pred.sum()),
                        }
                stat_writer.writerow(row)
                summary['rows'] += 1
                summary['cap_hit_images'] += int(row['cap_hit'])
                if args.sweep:
                    for sweep_row in _branch_sweep(
                            p_np, backend=backend, threshold=threshold,
                            min_values=min_values, caps=caps):
                        sweep_row.update({
                            'split': split,
                            'dataset': sample.dataset,
                            'stem': sample.stem,
                        })
                        sweep_writer.writerow(sweep_row)
                print(
                    f'  {sample.stem}: p.max={row["p_max"]} '
                    f'branches={morse.n_branches} '
                    f'cap={row["cap_hit"]} u={row["u_mean"]}/'
                    f'{row["u_max"]}', flush=True)
    (args.output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(f'wrote {stat_path}')
    print(f'wrote {sweep_path if args.sweep else "(sweep skipped)"}')
    print(f'wrote {args.output / "summary.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
