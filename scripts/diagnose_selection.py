"""Candidate-level selection diagnostics for prompt-free SACM checkpoints.

Evaluation-only: loads a trained checkpoint, runs the prompt-free decoder on
each test image, and analyses the FOUR Stage-2 candidates it returns.

It answers one question: when the final mask is wrong, is the candidate set
itself bad (oracle is also bad), or did the selector pick the wrong candidate
(oracle is good, selected is bad)?

Per dataset/category it reports:
  - oracle / gating(index 0) / IoU-argmax for each requested metric
  - selection regret = selected - oracle (positive = worse)
  - Spearman correlation between predicted IoU and true candidate quality
  - IoU-best vs topology-best disagreement rate
  - ECE of predicted IoU against true IoU
  - candidate usage distribution / entropy

Example:
  python scripts/diagnose_selection.py \
      --preset full \
      --data_root DRIVE:data/sacm_3shot_v0/test/DRIVE \
                  CHASEDB1:data/sacm_3shot_v0/test/CHASEDB1 \
                  WIRE:datasets/novel/WIRE ROAD:datasets/novel/ROAD \
      --category DRIVE:seen CHASEDB1:seen WIRE:novel ROAD:novel \
      --trained_weights results/full/best_model.pth \
      --output_dir diagnosis/full --metrics dice cldice hd95

Data roots must use the TestDataset layout: <root>/images + <root>/masks.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import models  # noqa: F401,E402  (registers ModelSpecs)
from core.data import TestDataset  # noqa: E402
from core.evaluator import resolve_selection  # noqa: E402
from core.metrics.distance import hd95_score  # noqa: E402
from core.metrics.persistence import betti_matching_score  # noqa: E402
from core.metrics.pixel import dice_score, iou_score  # noqa: E402
from core.metrics.topology import cldice_score  # noqa: E402
from core.registry import MODEL_REGISTRY, get_model_spec  # noqa: E402
from models.sacm.configs import PRESETS, apply_preset, str2bool  # noqa: E402

HIGHER_IS_BETTER = {'dice', 'cldice', 'betti_matching', 'iou'}
LOWER_IS_BETTER = {'hd95'}


def parse_name_path(specs):
    out = []
    for s in specs:
        if ':' in s:
            name, path = s.split(':', 1)
        else:
            path = s
            name = os.path.basename(os.path.normpath(s))
        out.append((name, path))
    return out


def parse_categories(specs):
    out = {}
    for s in specs or []:
        if ':' not in s:
            raise ValueError(f"--category entries must be NAME:CATEGORY, got {s!r}")
        name, cat = s.split(':', 1)
        out[name] = cat
    return out


def best_index(values, higher):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return -1
    arr = np.where(np.isfinite(arr), arr, -np.inf if higher else np.inf)
    return int(np.argmax(arr) if higher else np.argmin(arr))


def regret_value(selected, oracle, higher):
    if not (np.isfinite(selected) and np.isfinite(oracle)):
        return float('nan')
    return float(oracle - selected) if higher else float(selected - oracle)


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float('nan')
    return float(spearmanr(a[mask], b[mask]).correlation)


def expected_calibration_error(pred, true, n_bins=10):
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if pred.size == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (pred >= lo) & ((pred < hi) if i < n_bins - 1 else (pred <= hi))
        if in_bin.sum() == 0:
            continue
        ece += abs(pred[in_bin].mean() - true[in_bin].mean()) * in_bin.sum() / pred.size
    return float(ece)


def usage_entropy(indices, n_candidates):
    idx = [int(i) for i in indices if isinstance(i, (int, np.integer)) and 0 <= i < n_candidates]
    counts = np.bincount(idx, minlength=n_candidates).astype(float)
    if counts.sum() == 0:
        return float('nan'), counts.astype(int)
    p = counts / counts.sum()
    return float(max(0.0, -(p * np.log(p + 1e-12)).sum())), counts.astype(int)


def pairwise_candidate_similarity(cands, threshold):
    """Mean pairwise IoU / clDice between candidates (self-pairs excluded).

    Near 1 means the candidate set is homogeneous (selection has little
    room); near 0 means the candidates are genuinely diverse.
    """
    n = cands.shape[0]
    bins = [(cands[k] > threshold).astype(np.uint8) for k in range(n)]
    ious, clds = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ious.append(iou_score(bins[i], bins[j]))
            clds.append(cldice_score(bins[i], bins[j]))
    return _nanmean(ious), _nanmean(clds)


def compute_candidate_metrics(logits, gt, threshold, metrics):
    pred = (logits > threshold).astype(np.uint8)
    ref = (gt > 0.5).astype(np.uint8)
    out = {'iou': iou_score(pred, ref)}
    for m in metrics:
        if m == 'iou':
            continue
        if m == 'dice':
            out[m] = dice_score(pred, ref)
        elif m == 'cldice':
            out[m] = cldice_score(pred, ref)
        elif m == 'hd95':
            out[m] = hd95_score(pred, ref)
        elif m == 'betti_matching':
            out[m] = betti_matching_score(logits, gt)
        else:
            raise ValueError(f"Unsupported metric {m!r}")
    return out


def build_model(args, device):
    spec = get_model_spec(args.model)
    model = spec.build(
        checkpoint=None,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        geo_e_layers=args.geo_e_layers,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
    )
    checkpoint = torch.load(args.trained_weights, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return spec, model


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float('nan')


def aggregate(rows, metrics, n_candidates):
    summary = []
    datasets = sorted({(r['dataset'], r['category']) for r in rows})
    for dataset, category in datasets:
        rs = [r for r in rows if r['dataset'] == dataset and r['category'] == category]
        if not rs:
            continue
        pred_iou, true_iou = [], []
        for r in rs:
            pred_iou.extend(r['_pred_iou'])
            true_iou.extend(r['_true_iou'])
        ece = expected_calibration_error(pred_iou, true_iou)
        ent, counts = usage_entropy([r['iou_idx'] for r in rs], n_candidates)
        for metric in metrics:
            summary.append({
                'dataset': dataset,
                'category': category,
                'metric': metric,
                'n': len(rs),
                'oracle_mean': _nanmean([r[f'oracle_{metric}'] for r in rs]),
                'gating_mean': _nanmean([r[f'gating_{metric}'] for r in rs]),
                'iou_mean': _nanmean([r[f'iou_{metric}'] for r in rs]),
                'gating_regret': _nanmean([r[f'regret_gating_{metric}'] for r in rs]),
                'iou_regret': _nanmean([r[f'regret_iou_{metric}'] for r in rs]),
                'spearman_mean': _nanmean([r[f'spearman_{metric}'] for r in rs]),
                'disagreement_rate': _nanmean([r[f'disagreement_{metric}'] for r in rs]),
                'pair_iou_mean': _nanmean([r['pair_iou_mean'] for r in rs]),
                'pair_cldice_mean': _nanmean([r['pair_cldice_mean'] for r in rs]),
                'ece_iou': ece,
                'usage_entropy_iou': ent,
                'usage_counts_iou': ' '.join(str(int(c)) for c in counts),
            })
    return summary


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, args):
    with open(path, 'w') as f:
        f.write('# Selection diagnostics\n\n')
        f.write(f'- checkpoint: `{args.trained_weights}`\n')
        f.write(f'- preset: `{args.preset}`\n')
        f.write(f'- threshold: {args.threshold} (logits; 0.0 = probability 0.5)\n')
        f.write(f'- oracle metric: `{args.oracle_metric}`\n')
        f.write(f'- metrics: {", ".join(args.metrics)}\n\n')
        f.write('## Summary\n\n')
        f.write('| dataset | category | metric | n | oracle | gating | IoU | gating regret | IoU regret | Spearman | disagree | pairIoU | paircD | ECE(IoU) |\n')
        f.write('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for s in summary:
            f.write(
                f"| {s['dataset']} | {s['category']} | {s['metric']} | {s['n']} | "
                f"{s['oracle_mean']:.4f} | {s['gating_mean']:.4f} | {s['iou_mean']:.4f} | "
                f"{s['gating_regret']:.4f} | {s['iou_regret']:.4f} | {s['spearman_mean']:.4f} | "
                f"{s['disagreement_rate']:.3f} | {s['pair_iou_mean']:.4f} | "
                f"{s['pair_cldice_mean']:.4f} | {s['ece_iou']:.4f} |\n"
            )
        f.write('\n## How to read\n\n')
        f.write('- `oracle` is the best value among the candidates (candidate ceiling).\n')
        f.write('- `gating` is candidate index 0; `IoU` is argmax of the predicted IoU head.\n')
        f.write('- `regret = selected - oracle` (positive = selector lost quality).\n')
        f.write('- `disagree` = IoU-best candidate differs from topology-best candidate.\n')
        f.write('- `pairIoU` / `paircD` near 1 = candidates are homogeneous (little selection room); near 0 = diverse.\n')
        f.write('- If oracle is high but selected is low and regret is large, the bottleneck is **selection**.\n')
        f.write('- If oracle itself is low, the bottleneck is **candidate quality / adaptation**, not selection.\n')
        f.write('- ECE uses predicted IoU clipped to [0, 1]; raw IoU-head outputs may exceed this range.\n')


def main():
    parser = argparse.ArgumentParser(description='Candidate-level selection diagnostics for SACM checkpoints.')
    parser.add_argument('--data_root', nargs='+', required=True,
                        help='NAME:PATH entries; each PATH has images/ and masks/')
    parser.add_argument('--trained_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='diagnosis/selection')
    parser.add_argument('--model', type=str, default='sam_l', choices=sorted(MODEL_REGISTRY))
    parser.add_argument('--preset', type=str, default='full', choices=sorted(PRESETS))
    parser.add_argument('--selection', type=str, default='auto',
                        choices=['auto', 'iou', 'gating'],
                        help='auto: use iou only when the preset trains the IoU head '
                             '(iou_loss_weight>0), else gating/head 0')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1)
    parser.add_argument('--use_geo_i', type=str2bool, default=True)
    parser.add_argument('--use_geo_e', type=str2bool, default=True)
    parser.add_argument('--geo_e_layers', type=str, default='all', choices=['all', 'shallow', 'deep'])
    parser.add_argument('--use_coarse_to_fine', type=str2bool, default=True)
    parser.add_argument('--use_fusion_v2', type=str2bool, default=True)
    parser.add_argument('--use_multi_depth', type=str2bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Logit threshold; 0.0 = probability 0.5')
    parser.add_argument('--metrics', nargs='+', default=['dice', 'cldice', 'hd95'],
                        choices=['dice', 'cldice', 'hd95', 'betti_matching', 'iou'])
    parser.add_argument('--oracle_metric', type=str, default='cldice',
                        choices=['dice', 'cldice', 'hd95', 'betti_matching', 'iou'])
    parser.add_argument('--category', nargs='*', default=None,
                        help='NAME:CATEGORY entries for Seen/Unseen/Novel grouping')
    parser.add_argument('--limit', type=int, default=0, help='Limit images per dataset (0 = all)')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    apply_preset(args, args.preset)
    args.selection = resolve_selection(
        args.selection, getattr(args, 'iou_loss_weight', 0.0))
    if args.selection == 'gating':
        print("[diagnose] IoU head untrained or gating selected: iou columns "
              "mirror gating and ECE is disabled (NaN).")
    if args.oracle_metric not in args.metrics:
        args.metrics = list(args.metrics) + [args.oracle_metric]

    roots = parse_name_path(args.data_root)
    categories = parse_categories(args.category)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    spec, model = build_model(args, device)
    sparse, dense = spec.prompt_builder(model, 1)

    rows = []
    n_candidates = 4
    for dataset_name, root in roots:
        category = categories.get(dataset_name, 'all')
        dataset = TestDataset(root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
        pbar = tqdm(loader, total=len(loader), desc=f'{dataset_name} [{category}]')
        for idx, (model_input, original_mask, _original_size, img_name,
                  fov_mask) in enumerate(pbar):
            if args.limit and idx >= args.limit:
                break
            model_input = spec.preprocess(model, model_input.to(device))
            with torch.no_grad():
                masks, iou_pred = spec.forward(model, model_input, sparse, dense,
                                               return_stage1=False)
            gt = original_mask[0].cpu().numpy().squeeze()
            fov = fov_mask[0, 0].cpu().numpy() > 0.5
            gt = gt * fov
            h, w = gt.shape
            cands = F.interpolate(masks, size=(h, w), mode='bilinear',
                                  align_corners=False)[0].cpu().numpy()
            cands = cands * fov[None]
            iou_scores = iou_pred[0].detach().cpu().numpy()
            if args.selection == 'iou':
                score_vec = iou_scores
                iou_idx = int(np.argmax(iou_scores)) if iou_scores.size else -1
            else:
                # gating baseline: the IoU head is untrained/disabled, so
                # do not rank candidates or compute ECE with random scores.
                score_vec = np.full_like(iou_scores, np.nan, dtype=float)
                iou_idx = 0 if iou_scores.size else -1
            n_candidates = cands.shape[0]

            cand_metrics = [compute_candidate_metrics(cands[k], gt, args.threshold, args.metrics)
                            for k in range(n_candidates)]
            pair_iou_mean, pair_cldice_mean = pairwise_candidate_similarity(
                cands, args.threshold)
            metric_values = {m: [cand_metrics[k][m] for k in range(n_candidates)]
                             for m in args.metrics}
            gating_idx = 0
            oracle_indices = {
                m: best_index(metric_values[m], higher=(m in HIGHER_IS_BETTER))
                for m in args.metrics
            }
            image_name = img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name)
            row = {
                'dataset': dataset_name,
                'category': category,
                'image': image_name,
                'gating_idx': gating_idx,
                'iou_idx': iou_idx,
                f'oracle_idx_{args.oracle_metric}': oracle_indices[args.oracle_metric],
                'pair_iou_mean': pair_iou_mean,
                'pair_cldice_mean': pair_cldice_mean,
                '_pred_iou': [float(x) for x in score_vec],
                '_true_iou': [float(cand_metrics[k]['iou']) for k in range(n_candidates)],
            }
            for m in args.metrics:
                higher = m in HIGHER_IS_BETTER
                best_idx = oracle_indices[m]
                for k in range(n_candidates):
                    row[f'cand{k}_{m}'] = float(metric_values[m][k])
                row[f'gating_{m}'] = float(metric_values[m][gating_idx])
                row[f'iou_{m}'] = (float(metric_values[m][iou_idx])
                                   if iou_idx >= 0 else float('nan'))
                oracle_val = (float(metric_values[m][best_idx])
                              if best_idx >= 0 else float('nan'))
                row[f'oracle_{m}'] = oracle_val
                row[f'regret_gating_{m}'] = regret_value(
                    metric_values[m][gating_idx], oracle_val, higher)
                row[f'regret_iou_{m}'] = regret_value(row[f'iou_{m}'], oracle_val, higher)
                row[f'disagreement_{m}'] = (float(iou_idx != best_idx)
                                            if best_idx >= 0 else float('nan'))
                row[f'spearman_{m}'] = safe_spearman(score_vec, metric_values[m])
            rows.append(row)

    base_fields = ['dataset', 'category', 'image', 'gating_idx', 'iou_idx',
                   f'oracle_idx_{args.oracle_metric}', 'pair_iou_mean',
                   'pair_cldice_mean']
    metric_fields = []
    for m in args.metrics:
        metric_fields += [f'cand{k}_{m}' for k in range(n_candidates)]
        metric_fields += [f'gating_{m}', f'iou_{m}', f'oracle_{m}',
                          f'regret_gating_{m}', f'regret_iou_{m}',
                          f'disagreement_{m}', f'spearman_{m}']
    fieldnames = base_fields + metric_fields
    public_rows = [{k: r.get(k) for k in fieldnames} for r in rows]
    write_csv(os.path.join(args.output_dir, 'per_image.csv'), public_rows, fieldnames)

    summary = aggregate(rows, args.metrics, n_candidates)
    summary_fields = ['dataset', 'category', 'metric', 'n', 'oracle_mean',
                      'gating_mean', 'iou_mean', 'gating_regret', 'iou_regret',
                      'spearman_mean', 'disagreement_rate', 'pair_iou_mean',
                      'pair_cldice_mean', 'ece_iou', 'usage_entropy_iou',
                      'usage_counts_iou']
    write_csv(os.path.join(args.output_dir, 'summary.csv'), summary, summary_fields)
    write_report(os.path.join(args.output_dir, 'selection_report.md'), summary, args)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump({'args': vars(args), 'summary': summary}, f, indent=2)
    print(f'Wrote {args.output_dir}/per_image.csv, summary.csv, selection_report.md')
    print(f'Images: {len(rows)}; datasets: {len(roots)}')


if __name__ == '__main__':
    main()
