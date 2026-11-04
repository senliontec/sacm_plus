#!/usr/bin/env python3
"""Component-repair evaluation for one SACM-X checkpoint.

Repair policy: support-calibrated pruning of spurious components.  This is
the only local action that passed the Pareto guard; pixel residual repair and
add-missed-component repair were rejected in local probes.

For each seen test dataset the component-spuriousness classifier is trained
on the OTHER seen support datasets and the delete threshold is calibrated on
this dataset's three support images.  Unseen/Novel datasets have no support;
they use the classifier and threshold fitted on all seen support images.
No target labels are used.

The summary's ``coarse_*`` fields mean the checkpoint's final head-2 output
before component pruning; gate-only training freezes residual experts at
zero, so this is also the pre-residual coarse output.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.morphology import skeletonize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.data import TestDataset
from core.io import get_prompt_embeddings, predict_all
from core.metrics import compute_metrics
from core.metrics.components import component_metrics
from models.sacm_x.configs import PRESETS
from models.sacm_x.specs import sam_preprocess
from models.sam_network import build_sam_vit_l

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
FEATURE_NAMES = [
    'area', 'area_ratio', 'skeleton_len', 'skeleton_density',
    'mean_p0', 'max_p0', 'std_p0', 'mean_gate', 'gate_mass',
    'gate_pos_frac', 'other_mean', 'other_agree', 'border_dist',
    'elongation',
]
GATE_FEATURES = {'mean_gate', 'gate_mass', 'gate_pos_frac'}
EPS_DICE = 0.002
EPS_HD = 0.05


def load_model(preset: str, checkpoint: Path, device: torch.device):
    sig = set(inspect.signature(build_sam_vit_l).parameters)
    kwargs = {k: v for k, v in PRESETS[preset].items() if k in sig}
    # PRESETS only stores deviations from the argparse defaults.  The
    # training entrypoint defaults these three switches to True; without
    # restoring them the checkpoint's adapter_norms keys do not exist.
    for key in ('use_adapter', 'use_internal_adapter',
                'collect_adapter_features'):
        if kwargs.get(key) is None:
            kwargs[key] = True
    kwargs['checkpoint'] = None
    model = build_sam_vit_l(**kwargs)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()
    rm = getattr(model.mask_decoder, 'residual_moe', None)
    if rm is not None:
        rm.res_scale = 1.0
    return model, rm, payload


def forward(model, rm, image):
    with torch.no_grad():
        x = sam_preprocess(model, image)
        sparse, dense = get_prompt_embeddings(model, image.shape[0])
        out = predict_all(model, x, sparse, dense, return_stage1=False)
        heads = out[0] if isinstance(out, (tuple, list)) else out
        # Evaluate the checkpoint's actual final head 2, after any residual
        # correction.  Gate-only presets freeze the experts at zero, so this
        # is numerically equivalent to the old pre-residual coarse path;
        # other presets remain correct instead of silently dropping their
        # residual correction.
        if heads.shape[1] > 2:
            p0 = torch.sigmoid(heads[:, 2:3])
        elif rm is not None and rm.last_coarse_logits is not None:
            p0 = torch.sigmoid(rm.last_coarse_logits)
        else:
            p0 = torch.sigmoid(heads[:, :1])
        gate = (rm.last_gate_mask if rm is not None
                and rm.last_gate_mask is not None
                else torch.zeros_like(p0))
        if heads.shape[1] > 3:
            other = torch.sigmoid(heads[:, [0, 1, 3]]).mean(
                1, keepdim=True)
        elif heads.shape[1] > 1:
            other = torch.sigmoid(heads).mean(1, keepdim=True)
        else:
            other = torch.zeros_like(p0)
        if p0.shape[-2:] != (512, 512):
            p0 = F.interpolate(p0, (512, 512), mode='bilinear',
                               align_corners=False)
        if gate.shape[-2:] != (512, 512):
            gate = F.interpolate(gate, (512, 512), mode='bilinear',
                                 align_corners=False)
        if other.shape[-2:] != (512, 512):
            other = F.interpolate(other, (512, 512), mode='bilinear',
                                  align_corners=False)
    return p0, gate, other


def features(mask, p0, gate, other):
    h, w = mask.shape
    lab, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    skel = skeletonize(mask)
    skel_n = np.bincount(lab[skel], minlength=n + 1)
    rows = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area == 0:
            continue
        ys, xs = np.where(comp)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bh, bw = y1 - y0 + 1, x1 - x0 + 1
        vals = [
            float(area), area / float(h * w), float(skel_n[i]),
            skel_n[i] / float(area),
            float(p0[comp].mean()), float(p0[comp].max()),
            float(p0[comp].std()),
            float(gate[comp].mean()), float(gate[comp].sum()),
            float((gate[comp] > 0.5).mean()),
            float(other[comp].mean()),
            float((other[comp] > 0.5).mean()),
            float(min(y0, x0, h - 1 - y1, w - 1 - x1)),
            float(max(bh, bw) / max(1, min(bh, bw))),
        ]
        rows.append({'comp': comp,
                     'feature': np.asarray(vals, dtype=np.float32)})
    return rows


def labels_from_gt(rows, gt512):
    if torch.is_tensor(gt512):
        gt = gt512[0, 0].float().cpu().numpy() > 0.5
    else:
        gt = np.asarray(gt512) > 0.5
    labels = []
    for row in rows:
        comp = row['comp']
        frac = float((comp & gt).sum()) / max(1, int(comp.sum()))
        if frac < 0.20:
            labels.append(1)
        elif frac > 0.50:
            labels.append(0)
        else:
            labels.append(None)
    return labels


def metric_from_prob(p, gt_orig):
    p = p.float()
    z = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    z = F.interpolate(z, size=gt_orig.shape[-2:], mode='bilinear',
                      align_corners=False)
    return compute_metrics(
        z[0], gt_orig[0], threshold=0.0, compute_auc=False,
        compute_betti_matching=False, compute_topograph=False)


def comp_stats(p, gt_orig):
    p = p.float()
    z = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    z = F.interpolate(z, size=gt_orig.shape[-2:], mode='bilinear',
                      align_corners=False)
    return component_metrics((z[0, 0] > 0).cpu().numpy(),
                             (gt_orig[0, 0] > 0.5).cpu().numpy())


def apply_prune(item, rows, scores, tau):
    p = item['p0'].float().clone()
    removed = np.zeros(tuple(p.shape[-2:]), dtype=bool)
    for row, score in zip(rows, scores):
        if score >= tau:
            removed |= row['comp']
    p[0, 0, torch.from_numpy(removed).to(p.device)] = 0
    return p, removed


def support_item(model, rm, image, gt, name, dataset, device):
    image = image.to(device)
    p0, gate, other = forward(model, rm, image)
    gt512 = F.adaptive_max_pool2d(gt.to(device), (512, 512))
    rows = features(p0[0, 0].float().cpu().numpy() > 0.5,
                    p0[0, 0].float().cpu().numpy(),
                    gate[0, 0].float().cpu().numpy(),
                    other[0, 0].float().cpu().numpy())
    labels = labels_from_gt(rows, gt512)
    gate_auc = auc_score(gate[0, 0],
                         (p0[0, 0] - gt512[0, 0]).abs() > 0.5)
    return {'name': name, 'dataset': dataset, 'p0': p0.cpu(),
            'gate': gate.cpu(), 'other': other.cpu(),
            'rows': rows, 'labels': labels, 'gt_orig': gt.cpu(),
            'gt512': gt512.cpu(), 'gate_auc': gate_auc}


def role_map(data_root):
    path = Path(data_root) / 'protocol_summary.json'
    roles = {}
    if path.is_file():
        d = json.loads(path.read_text())
        for role, key, group in (
                ('Seen', 'seen', 'seen_test'),
                ('Unseen', 'unseen', 'unseen_test'),
                ('Novel', 'novel', 'novel_test')):
            for name in d.get('counts', {}).get(key, {}).get(group, {}):
                roles[name] = role
    return roles


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))
            and math.isfinite(v)]
    return statistics.fmean(vals) if vals else float('nan')


def auc_score(score, target):
    s = score.detach().reshape(-1).float().cpu().numpy()
    t = target.detach().reshape(-1).bool().cpu().numpy()
    if t.sum() == 0 or (~t).sum() == 0:
        return float('nan')
    try:
        return float(roc_auc_score(t, s))
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-root', type=Path,
                    default=Path('data/sacm_3shot_seed0'))
    ap.add_argument('--test-root', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--drop-gate-features', action='store_true')
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    test_root = args.test_root or (args.data_root / 'test')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roles = role_map(args.data_root)
    model, rm, payload = load_model(args.preset, args.checkpoint, device)
    has_gate = rm is not None
    use_gate = has_gate and not args.drop_gate_features
    keep_idx = [i for i, n in enumerate(FEATURE_NAMES)
                if use_gate or n not in GATE_FEATURES]
    print(f'preset={args.preset} epoch={payload.get("epoch_completed")} '
          f'gate={has_gate} use_gate_features={use_gate}', flush=True)

    support_ds = TestDataset(str(args.data_root), split='train',
                             input_mode='resize', image_size=1024)
    support = {}
    for image, gt, _, name, _ in torch.utils.data.DataLoader(
            support_ds, batch_size=1, shuffle=False, num_workers=0):
        ds = support_ds.dataset_for(name[0])
        support.setdefault(ds, []).append(
            support_item(model, rm, image, gt, name[0], ds, device))
        torch.cuda.empty_cache()
    print('support', {k: len(v) for k, v in support.items()}, flush=True)

    test_dirs = [d for d in sorted(test_root.iterdir())
                 if d.is_dir() and (d / 'images').is_dir()]
    rows = []
    classifiers = {}
    for ddir in test_dirs:
        target = ddir.name
        role = roles.get(target, 'Seen' if target in support else 'Other')
        seen_target = target in support
        train_items = [it for ds, items in support.items()
                       if ds != target for it in items]
        support_items = support.get(target, [])
        support_source = 'target' if seen_target else 'global'
        if not support_items:
            support_items = [it for items in support.values() for it in items]
        X, y = [], []
        for it in train_items:
            for row, lab in zip(it['rows'], it['labels']):
                if lab is not None:
                    X.append(row['feature'][keep_idx])
                    y.append(lab)
        clf = None
        if X and len(set(y)) > 1:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight='balanced', C=1.0,
                                   max_iter=2000))
            clf.fit(np.stack(X), np.array(y))
            coefs = clf.named_steps['logisticregression'].coef_[0].tolist()
            classifiers[target] = {
                'features': [FEATURE_NAMES[i] for i in keep_idx],
                'coef': coefs, 'n_train': len(X),
                'pos_rate': float(np.mean(y)),
                'support_source': support_source}
        sup_scores = {}
        for it in support_items:
            if clf is None or not it['rows']:
                sup_scores[it['name']] = np.zeros(len(it['rows']))
            else:
                sup_scores[it['name']] = clf.predict_proba(
                    np.stack([r['feature'][keep_idx]
                              for r in it['rows']]))[:, 1]
        base = [metric_from_prob(it['p0'], it['gt_orig'])
                for it in support_items]
        base_m = {k: mean([b[k] for b in base])
                  for k in ('dice', 'cldice', 'hd95')}
        base_m['gate_auc'] = mean(
            [it.get('gate_auc', float('nan')) for it in support_items])
        base_cc = [comp_stats(it['p0'], it['gt_orig'])
                   for it in support_items]
        best = None
        if clf is not None:
            for tau in TAUS:
                vals, comps = [], []
                for it in support_items:
                    p2, _ = apply_prune(
                        it, it['rows'], sup_scores[it['name']], float(tau))
                    vals.append(metric_from_prob(p2, it['gt_orig']))
                    comps.append(comp_stats(p2, it['gt_orig']))
                m = {k: mean([v[k] for v in vals])
                     for k in ('dice', 'cldice', 'hd95')}
                valid = (
                    m['dice'] >= base_m['dice'] - EPS_DICE
                    and (m['cldice'] > base_m['cldice'] + 1e-4
                         or m['hd95'] < base_m['hd95'] - EPS_HD
                         or mean([c['component_count_error']
                                  for c in comps]) <
                         mean([c['component_count_error']
                               for c in base_cc])))
                if valid and (best is None or
                              m['dice'] + m['cldice'] >
                              best[0]['dice'] + best[0]['cldice']):
                    best = (m, float(tau))
        if best is None:
            tau = 1.0
            sup_m = base_m
        else:
            sup_m, tau = best

        ds = TestDataset(str(ddir), split=None, input_mode='resize',
                         image_size=1024)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0)
        coarse_vals, prune_vals, coarse_comps, prune_comps = [], [], [], []
        gate_aucs = []
        removed_n = 0
        total_n = 0
        for i, (image, gt, _, name, _) in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            image = image.to(device)
            p0, gate, other = forward(model, rm, image)
            gt512 = F.adaptive_max_pool2d(gt.to(device), (512, 512))
            gate_aucs.append(auc_score(
                gate[0, 0], (p0[0, 0] - gt512[0, 0]).abs() > 0.5))
            rows_i = features(
                p0[0, 0].float().cpu().numpy() > 0.5,
                p0[0, 0].float().cpu().numpy(),
                gate[0, 0].float().cpu().numpy(),
                other[0, 0].float().cpu().numpy())
            if clf is None or not rows_i:
                scores = np.zeros(len(rows_i))
            else:
                scores = clf.predict_proba(
                    np.stack([r['feature'][keep_idx]
                              for r in rows_i]))[:, 1]
            p2, _ = apply_prune(
                {'p0': p0.cpu(), 'name': name[0]}, rows_i, scores, tau)
            coarse_vals.append(metric_from_prob(p0.cpu(), gt.cpu()))
            prune_vals.append(metric_from_prob(p2, gt.cpu()))
            coarse_comps.append(comp_stats(p0.cpu(), gt.cpu()))
            prune_comps.append(comp_stats(p2, gt.cpu()))
            removed_n += int((scores >= tau).sum())
            total_n += len(scores)
            torch.cuda.empty_cache()

        def mv(vals, key):
            return mean([v[key] for v in vals])
        row = {
            'model': args.run_name or args.preset,
            'preset': args.preset, 'role': role, 'dataset': target,
            'n_test': len(coarse_vals), 'tau': tau,
            'support_source': support_source,
            'support_coarse_dice': base_m['dice'],
            'support_prune_dice': sup_m['dice'],
            'support_coarse_cldice': base_m['cldice'],
            'support_prune_cldice': sup_m['cldice'],
            'support_coarse_hd95': base_m['hd95'],
            'support_prune_hd95': sup_m['hd95'],
            'support_gate_auc': base_m.get('gate_auc', float('nan')),
            'test_coarse_dice': mv(coarse_vals, 'dice'),
            'test_prune_dice': mv(prune_vals, 'dice'),
            'test_coarse_cldice': mv(coarse_vals, 'cldice'),
            'test_prune_cldice': mv(prune_vals, 'cldice'),
            'test_coarse_hd95': mv(coarse_vals, 'hd95'),
            'test_prune_hd95': mv(prune_vals, 'hd95'),
            'test_gate_auc': mean(gate_aucs),
            'test_coarse_components': mean(
                [c['components_pred'] for c in coarse_comps]),
            'test_prune_components': mean(
                [c['components_pred'] for c in prune_comps]),
            'test_coarse_cc_error': mean(
                [c['component_count_error'] for c in coarse_comps]),
            'test_prune_cc_error': mean(
                [c['component_count_error'] for c in prune_comps]),
            'test_removed_frac': removed_n / max(1, total_n),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(args.output_dir / 'per_dataset.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output_dir / 'classifier.json').write_text(
        json.dumps(classifiers, indent=2) + '\n')
    summary = {}
    for role in ('Seen', 'Unseen', 'Novel', 'Other'):
        sub = [r for r in rows if r['role'] == role]
        if not sub:
            continue
        summary[role] = {
            'n_datasets': len(sub),
            'coarse_dice': mean([r['test_coarse_dice'] for r in sub]),
            'prune_dice': mean([r['test_prune_dice'] for r in sub]),
            'coarse_cldice': mean([r['test_coarse_cldice'] for r in sub]),
            'prune_cldice': mean([r['test_prune_cldice'] for r in sub]),
            'coarse_hd95': mean([r['test_coarse_hd95'] for r in sub]),
            'prune_hd95': mean([r['test_prune_hd95'] for r in sub]),
            'coarse_components': mean(
                [r['test_coarse_components'] for r in sub]),
            'prune_components': mean(
                [r['test_prune_components'] for r in sub]),
            'coarse_cc_error': mean(
                [r['test_coarse_cc_error'] for r in sub]),
            'prune_cc_error': mean(
                [r['test_prune_cc_error'] for r in sub]),
            'gate_auc': mean([r['test_gate_auc'] for r in sub]),
        }
    summary['All'] = {
        'n_datasets': len(rows),
        'coarse_dice': mean([r['test_coarse_dice'] for r in rows]),
        'prune_dice': mean([r['test_prune_dice'] for r in rows]),
        'coarse_cldice': mean([r['test_coarse_cldice'] for r in rows]),
        'prune_cldice': mean([r['test_prune_cldice'] for r in rows]),
        'coarse_hd95': mean([r['test_coarse_hd95'] for r in rows]),
        'prune_hd95': mean([r['test_prune_hd95'] for r in rows]),
        'coarse_components': mean(
            [r['test_coarse_components'] for r in rows]),
        'prune_components': mean(
            [r['test_prune_components'] for r in rows]),
        'coarse_cc_error': mean(
            [r['test_coarse_cc_error'] for r in rows]),
        'prune_cc_error': mean(
            [r['test_prune_cc_error'] for r in rows]),
        'gate_auc': mean([r['test_gate_auc'] for r in rows]),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print('SUMMARY', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
