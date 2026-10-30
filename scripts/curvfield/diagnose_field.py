#!/usr/bin/env python3
"""Oracle-field causal diagnostic for CurvField.

Compares the trained model's predicted field against a GT-derived field fed
through the same adapter.  This is analysis only: test masks are used to build
the oracle field, never for training, checkpoint selection or hyper-parameters.

The diagnostic fails loudly when swapping in the oracle field does not change
segmentation.  That is the v1 failure mode (gate saturation / dead field path).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.adapters.curvfield import CurvFieldAdapter
from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.curvfield import structure_targets
from core.metrics import cldice_score, dice_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / 'data' / 'sacm_3shot_seed0' / 'protocol_manifest.csv'
DEFAULT_DATASETS = (
    'OCTA-500', 'LEAF', 'TYRE', 'ROAD', 'HRF',
    'DeepCrack', 'XCAD', 'TopoMortar', 'CREMI',
)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _args_from_checkpoint(ck: dict) -> SimpleNamespace:
    cfg = ck['config']['adapter_config']
    return SimpleNamespace(
        input_size=int(cfg['input_size']),
        convlora_model_type=str(cfg['model_type']),
        convlora_rank=int(cfg['rank']),
        convlora_alpha=int(cfg['alpha']),
        convlora_experts=int(cfg['experts']),
        convlora_moe_weight=float(cfg['moe_weight']),
        convlora_checkpoint=Path(cfg['convlora_checkpoint']),
        curvfield_field_hidden=int(cfg['field_hidden']),
        curvfield_field_size=int(cfg.get('field_size', 128)),
        curvfield_adapter_hidden=int(cfg['adapter_hidden']),
        curvfield_adapter_fractions=list(cfg['adapter_fractions']),
        curvfield_film_scale=float(cfg.get('film_scale', 0.5)),
        curvfield_decoder_bias=bool(cfg.get('decoder_bias', True)),
        curvfield_field_weight=float(cfg['field_weight']),
        curvfield_new_lr_mult=float(cfg['new_lr_mult']),
        curvfield_static_field=bool(cfg['static_field']),
    )


def _build_model(ck: dict, device: str):
    adapter = CurvFieldAdapter()
    model = adapter.build_model(_args_from_checkpoint(ck))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.eval().to(device)


def _sample_rows(path: Path, datasets: list[str], n_per: int):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for dataset in datasets:
        candidates = [
            row for row in rows
            if row['dataset'] == dataset
            and row['split'].endswith('_test')
            and float(row['fg_ratio']) > 0.001
        ]
        if not candidates:
            continue
        if len(candidates) > n_per:
            indices = np.linspace(
                0, len(candidates) - 1, n_per).round().astype(int).tolist()
            candidates = [candidates[i] for i in sorted(set(indices))]
        selected[dataset] = candidates
    return selected


def _to_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-3, 1 - 1e-3)
    return torch.log(probability / (1.0 - probability))


def _oracle_field(target: torch.Tensor) -> torch.Tensor:
    """Convert GT field targets to the logits the adapter expects."""
    field = torch.zeros_like(target)
    field[:, 0] = _to_logit((target[:, 0] > 0.5).float())
    field[:, 1] = _to_logit(target[:, 1])
    field[:, 2:4] = target[:, 2:4]
    field[:, 4] = _to_logit((target[:, 4] > 0.5).float())
    field[:, 5] = _to_logit(target[:, 5])
    return field


def _predict(logits: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    logits = F.interpolate(
        logits, size=(height, width), mode='bilinear', align_corners=False)
    return (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy()


def _field_metrics(field: torch.Tensor, target: torch.Tensor,
                   foreground: torch.Tensor) -> dict[str, float]:
    center = torch.sigmoid(field[:, 0])
    radius = torch.sigmoid(field[:, 1])
    coherence = torch.sigmoid(field[:, 5])
    c_t, r_t, k_t = target[:, 0], target[:, 1], target[:, 5]
    fg = foreground[:, 0]
    support = ((c_t > 0.5) | (target[:, 4] > 0.5)).float()
    weight = k_t * support
    pred_vec = torch.stack([field[:, 2], field[:, 3]], dim=1)
    pred_vec = pred_vec / pred_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    target_vec = torch.stack([target[:, 2], target[:, 3]], dim=1)
    target_vec = target_vec / target_vec.norm(
        dim=1, keepdim=True).clamp_min(1e-6)
    angle = torch.acos((pred_vec * target_vec).sum(1).clamp(-1, 1))
    angle = angle * 0.5 * 180.0 / math.pi
    valid = weight > 0.05

    radius_error = (
        ((radius - r_t).abs() * fg).sum() / fg.sum().clamp_min(1.0))
    fg_mask = fg[0] > 0.5
    radius_corr = float('nan')
    if int(fg_mask.sum()) > 2:
        pred_np = radius[0][fg_mask].detach().cpu().numpy()
        target_np = r_t[0][fg_mask].cpu().numpy()
        if pred_np.std() > 1e-9 and target_np.std() > 1e-9:
            radius_corr = float(np.corrcoef(pred_np, target_np)[0, 1])
    return {
        'center_max': float(center.max()),
        'center_frac': float((center > 0.5).float().mean()),
        'radius_mae': float(radius_error),
        'radius_corr': radius_corr,
        'orient_mae_deg': (
            float(angle[valid].mean()) if valid.any() else float('nan')),
        'coherence_mae': float(
            ((coherence - k_t).abs() * fg).sum() /
            fg.sum().clamp_min(1.0)),
    }


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else float('nan')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS))
    parser.add_argument('--n-per', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--min-mean-delta', type=float, default=0.2,
                        help='required mean oracle-vs-predicted Dice gain (pp)')
    parser.add_argument('--min-max-delta', type=float, default=0.5,
                        help='required best-dataset oracle Dice gain (pp)')
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()

    ck = _load_checkpoint(args.checkpoint)
    model = _build_model(ck, args.device)
    field_size = int(model.field_size)
    rows = _sample_rows(args.protocol, args.datasets, args.n_per)
    if not rows:
        raise SystemExit('no test rows selected')
    print(f'checkpoint={args.checkpoint}')
    print(f'field_size={field_size} datasets={",".join(rows)} '
          f'n_per={args.n_per}')
    print('ANALYSIS ONLY: oracle fields use test masks; never used for training')

    results = []
    for dataset, candidates in rows.items():
        values = {
            'dice_pred': [], 'dice_oracle': [],
            'cldice_pred': [], 'cldice_oracle': [],
            'center_max': [], 'center_frac': [],
            'radius_corr': [], 'orient_mae_deg': [],
        }
        for row in candidates:
            image = load_rgb(Path(row['image_path'])).resize(
                (1024, 1024), Image.BICUBIC)
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1).unsqueeze(0).to(args.device)
            mask = load_mask(Path(row['mask_path']))
            mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
            target, foreground = structure_targets(
                F.interpolate(mask_t, size=(1024, 1024), mode='nearest'),
                size=field_size)
            target = target.to(args.device)
            foreground = foreground.to(args.device)

            with torch.no_grad():
                logits, _, field = model(tensor)
            pred = _predict(logits, mask)
            values['dice_pred'].append(dice_score(pred, mask))
            values['cldice_pred'].append(cldice_score(pred, mask))
            metrics = _field_metrics(field, target, foreground)
            for key in ('center_max', 'center_frac', 'radius_corr',
                        'orient_mae_deg'):
                values[key].append(metrics[key])

            oracle = _oracle_field(target)
            original_forward = model.field_head.forward
            model.field_head.forward = lambda _features, gt=oracle: gt
            try:
                with torch.no_grad():
                    oracle_logits, _, _ = model(tensor)
            finally:
                model.field_head.forward = original_forward
            oracle_pred = _predict(oracle_logits, mask)
            values['dice_oracle'].append(dice_score(oracle_pred, mask))
            values['cldice_oracle'].append(cldice_score(oracle_pred, mask))

        result = {'dataset': dataset, 'n': len(candidates)}
        for key, items in values.items():
            result[key] = _mean(items)
        result['dice_delta_pp'] = (
            result['dice_oracle'] - result['dice_pred']) * 100.0
        result['cldice_delta_pp'] = (
            result['cldice_oracle'] - result['cldice_pred']) * 100.0
        results.append(result)

    header = (
        'dataset,n,dice_pred,dice_oracle,dDice_pp,'
        'cldice_pred,cldice_oracle,dclDice_pp,'
        'center_max,center_frac,radius_corr,orient_mae_deg')
    print(header)
    for row in results:
        print(
            f"{row['dataset']},{row['n']},"
            f"{row['dice_pred']:.4f},{row['dice_oracle']:.4f},"
            f"{row['dice_delta_pp']:+.2f},"
            f"{row['cldice_pred']:.4f},{row['cldice_oracle']:.4f},"
            f"{row['cldice_delta_pp']:+.2f},"
            f"{row['center_max']:.4f},{row['center_frac']:.5f},"
            f"{row['radius_corr']:.3f},{row['orient_mae_deg']:.2f}")

    dice_deltas = [row['dice_delta_pp'] for row in results]
    mean_delta = _mean(dice_deltas)
    max_delta = max(dice_deltas)
    passed = (
        math.isfinite(mean_delta)
        and (mean_delta >= args.min_mean_delta
             or max_delta >= args.min_max_delta)
    )
    print(
        f'oracle_gate: mean_dDice={mean_delta:+.3f}pp '
        f'max_dDice={max_delta:+.3f}pp '
        f'thresholds=({args.min_mean_delta},{args.min_max_delta}) '
        f'{"PASS" if passed else "FAIL"}')
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
