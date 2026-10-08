"""Diagnostic analysis for the paper's motivation section.

Two pieces of evidence over a trained model:

1. Head supervision (gradient norms). With the original SACM
   winner-take-all loss (only the gating-ranked head 0 of Stage-2 is
   supervised), 3 of 4 Stage-2 heads receive zero gradient and the
   Stage-1 hypernetworks receive none at all; with the full loss (deep
   supervision + IoU + clDice) every head is supervised.

2. Gating quality. Correlation between the gating rank (Stage-1 max
   response) and the heads' ground-truth IoU ranks, plus the hit rate of
   the gating winner vs the GT-best head. A meaningless gating shows a
   near-zero correlation and ~25% hit rate.

Outputs (into --output_dir):
    head_grad_norms.csv / head_grad_norms.png
    gating_analysis.csv  / gating_analysis.png
    summary.txt

Usage:
    python diagnose_heads.py --preset sacm --data_root <test_dir> \
        --trained_weights best_model.pth --output_dir diagnosis/sacm_drive
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.sacm.configs import PRESETS, apply_preset, str2bool
from models.sam import build_sam_vit_l
from core.data import TestDataset
from core.io import get_prompt_embeddings, predict_all
from core.losses import DiceBCELoss, SoftclDiceLoss

STAGE1_KEY = 'output_hypernetworks_mlps'
STAGE2_KEY = 'output_hypernetworks_mlps_stage2'
IOU_HEAD_KEY = 'iou_prediction_head'


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def hard_iou(logits, gt):
    pred = (logits > 0).float()
    gt = (gt > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float(inter / (union + 1e-6))


def build_model(args, device):
    sam = build_sam_vit_l(
        checkpoint=None,
        use_adapter=True,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
    )
    sam.to(device)
    checkpoint = torch.load(args.trained_weights, map_location=device)
    sam.load_state_dict(checkpoint['model_state_dict'])
    sam.eval()
    return sam


def compute_head_grad_norms(sam, model_input, gt_mask, device, loss_mode):
    """Backward one loss variant and collect per-head gradient norms."""
    criterion = DiceBCELoss()
    soft_cl = SoftclDiceLoss(num_iter=5)

    model_input = sam.preprocess(model_input.to(device))
    gt_mask = gt_mask.to(device)

    sparse, dense = get_prompt_embeddings(sam, 1)
    m2, iou_pred, m1 = predict_all(sam, model_input, sparse, dense, return_stage1=True)
    m2 = F.interpolate(m2, size=(1024, 1024), mode='bilinear', align_corners=False)
    m1 = F.interpolate(m1, size=(1024, 1024), mode='bilinear', align_corners=False)
    gt = F.interpolate(gt_mask, size=(1024, 1024), mode='bilinear', align_corners=False)

    if loss_mode == 'sacm':
        # Original SACM training loss: only the gating winner is supervised
        loss = criterion(m2[:, 0:1], gt)
    else:  # 'full'
        loss_main = criterion(m2[:, 0:1], gt)
        loss_ds = sum(criterion(m1[:, k:k + 1], gt) for k in range(m1.shape[1])) / m1.shape[1]
        with torch.no_grad():
            m2_bin = (m2 > 0).float()
            gt_bin = (gt > 0.5).float()
            inter = (m2_bin * gt_bin).sum(dim=(2, 3))
            union = m2_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3)) - inter
            iou_gt = inter / (union + 1e-6)
        loss_iou = F.mse_loss(iou_pred, iou_gt)
        loss_cl = soft_cl(m2[:, 0:1], gt)
        loss = loss_main + 0.3 * loss_ds + 1.0 * loss_iou + 0.5 * loss_cl

    sam.zero_grad()
    loss.backward()

    # Zero gradient IS the finding here (unsupervised heads), so every head
    # key starts at 0.0 and only grows when the head actually receives a
    # gradient. This also keeps all rows key-complete for CSV/aggregation.
    num_heads = sam.mask_decoder.num_mask_tokens
    grads = {f'stage1_head_{h}': 0.0 for h in range(num_heads)}
    grads.update({f'stage2_head_{h}': 0.0 for h in range(num_heads)})
    grads['iou_head'] = 0.0
    for name, p in sam.named_parameters():
        if p.grad is None:
            continue
        if STAGE2_KEY in name:
            key = 'stage2_head_' + name.split('.')[2]
        elif STAGE1_KEY in name:
            key = 'stage1_head_' + name.split('.')[2]
        elif IOU_HEAD_KEY in name:
            key = 'iou_head'
        else:
            continue
        grads[key] = grads[key] + p.grad.norm().item()
    return grads, float(loss.item())


def run_grad_analysis(sam, loader, device, num_images, output_dir):
    rows = []
    for i, (model_input, original_mask, original_size, img_name) in enumerate(loader):
        if i >= num_images:
            break
        # Interpolate GT to 1024 (model-input space)
        gt = F.interpolate(original_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
        for mode in ('sacm', 'full'):
            grads, loss = compute_head_grad_norms(sam, model_input, gt, device, mode)
            row = {'image': img_name[0], 'loss_mode': mode, 'loss': loss}
            row.update(grads)
            rows.append(row)
            print(f"[grad] {img_name[0]} ({mode}): loss={loss:.4f}, "
                  f"zero-grad heads={[k for k, v in grads.items() if v == 0]}")
    if not rows:
        raise RuntimeError("No images processed for gradient analysis")

    keys = sorted({k for r in rows for k in r if k.startswith(('stage1', 'stage2', 'iou'))})
    with open(os.path.join(output_dir, 'head_grad_norms.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'loss_mode', 'loss'] + keys)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregated bar chart (log1p scale)
    modes = ('sacm', 'full')
    means = {k: [] for k in keys}
    for k in keys:
        for mode in modes:
            # .get(0.0) is defensive; rows are key-complete since every
            # head key is pre-initialized to 0 in compute_head_grad_norms
            vals = [r.get(k, 0.0) for r in rows if r['loss_mode'] == mode]
            means[k].append(np.mean(vals))
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, [np.log1p(means[k][0]) for k in keys], width, label='SACM loss (winner-take-all)')
    ax.bar(x + width / 2, [np.log1p(means[k][1]) for k in keys], width, label='Full loss (ours)')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_ylabel('log1p(mean gradient norm)')
    ax.set_title('Per-head supervision: gradient norms')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'head_grad_norms.png'), dpi=150)
    plt.close(fig)
    print(f"Gradient analysis saved to {output_dir}")


def run_gating_analysis(sam, loader, device, num_images, output_dir):
    rows = []
    with torch.no_grad():
        for i, (model_input, original_mask, original_size, img_name) in enumerate(loader):
            if i >= num_images:
                break
            model_input = sam.preprocess(model_input.to(device))
            gt = F.interpolate(
                original_mask.to(device), size=(1024, 1024),
                mode='bilinear', align_corners=False,
            )
            sparse, dense = get_prompt_embeddings(sam, 1)
            # Single forward pass: all Stage-2 heads, IoU predictions and
            # the Stage-1 masks (from which the gating scores derive).
            m2, iou_pred, m1 = predict_all(sam, model_input, sparse, dense, return_stage1=True)
            m2 = F.interpolate(m2, size=(1024, 1024), mode='bilinear', align_corners=False)
            gating_scores = m1.view(1, m1.shape[1], -1).max(dim=-1).values[0].cpu().numpy()

            # NOTE: predict_masks returns m1 already REORDERED by the gating
            # (position 0 = gating's #1 pick). Consequently the per-head
            # gating-rank columns below are degenerate ([1,2,3,4]) and the
            # informative quantity is the IoU of the head AT each position:
            # iou_by_gating_rank aggregates exactly that (and the hit rate
            # compares gating's #1 pick against the GT-best head).
            ious = np.array([hard_iou(m2[0, k], gt[0]) for k in range(m2.shape[1])])
            gating_rank = rankdata(-gating_scores)           # 1 = highest response
            iou_rank = rankdata(-ious)                       # 1 = best IoU
            gating_winner = int(np.argmax(gating_scores))
            gt_best = int(np.argmax(ious))
            iou_head_winner = int(iou_pred[0].argmax())

            rows.append({
                'image': img_name[0],
                'head_0_iou': ious[0], 'head_1_iou': ious[1],
                'head_2_iou': ious[2], 'head_3_iou': ious[3],
                'gating_rank_head0': gating_rank[0], 'gating_rank_head1': gating_rank[1],
                'gating_rank_head2': gating_rank[2], 'gating_rank_head3': gating_rank[3],
                'iou_rank_head0': iou_rank[0], 'iou_rank_head1': iou_rank[1],
                'iou_rank_head2': iou_rank[2], 'iou_rank_head3': iou_rank[3],
                'gating_winner': gating_winner,
                'gt_best_head': gt_best,
                'iou_head_winner': iou_head_winner,
                'spearman': spearman(gating_rank, iou_rank),
            })
            print(f"[gating] {img_name[0]}: gating winner=head{gating_winner}, "
                  f"GT best=head{gt_best}, IoU-head winner=head{iou_head_winner}, "
                  f"spearman={rows[-1]['spearman']:.3f}")

    if not rows:
        raise RuntimeError("No images processed for gating analysis")

    with open(os.path.join(output_dir, 'gating_analysis.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hit_gating = np.mean([r['gating_winner'] == r['gt_best_head'] for r in rows])
    hit_iou = np.mean([r['iou_head_winner'] == r['gt_best_head'] for r in rows])
    sp_mean = np.nanmean([r['spearman'] for r in rows])

    # Mean GT IoU of the head placed at each gating-rank position
    iou_by_gating_rank = [[] for _ in range(4)]
    for r in rows:
        for h in range(4):
            iou_by_gating_rank[int(r[f'gating_rank_head{h}']) - 1].append(r[f'head_{h}_iou'])
    rank_means = [np.mean(v) for v in iou_by_gating_rank]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(np.arange(1, 5), rank_means)
    ax.set_xlabel('Gating rank position')
    ax.set_ylabel('Mean GT IoU of the head at this rank')
    ax.set_title(
        f'Gating quality: mean spearman={sp_mean:.3f}, '
        f'hit rate={hit_gating:.2f} (IoU head: {hit_iou:.2f})'
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gating_analysis.png'), dpi=150)
    plt.close(fig)

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(
            f"Gating analysis: mean spearman={sp_mean:.4f}, "
            f"gating hit rate={hit_gating:.4f}, IoU-head hit rate={hit_iou:.4f}\n"
        )
        f.write(f"IoU by gating rank: {[f'{v:.4f}' for v in rank_means]}\n")
    print(f"Gating analysis saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Head supervision and gating diagnostics')
    parser.add_argument('--preset', type=str, default='full', choices=sorted(PRESETS))
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--trained_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1)
    parser.add_argument('--use_geo_i', type=str2bool, default=True)
    parser.add_argument('--use_geo_e', type=str2bool, default=True)
    parser.add_argument('--use_coarse_to_fine', type=str2bool, default=True)
    parser.add_argument('--use_fusion_v2', type=str2bool, default=True)
    parser.add_argument('--use_multi_depth', type=str2bool, default=True)
    parser.add_argument('--num_images_grad', type=int, default=8, help='Images for the gradient analysis')
    parser.add_argument('--num_images_gating', type=int, default=32, help='Images for the gating analysis')
    parser.add_argument('--skip_grad', action='store_true')
    parser.add_argument('--skip_gating', action='store_true')
    args = parser.parse_args()
    apply_preset(args, args.preset)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam = build_model(args, device)
    dataset = TestDataset(args.data_root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    if not args.skip_grad:
        run_grad_analysis(sam, loader, device, args.num_images_grad, args.output_dir)
    if not args.skip_gating:
        run_gating_analysis(sam, loader, device, args.num_images_gating, args.output_dir)


if __name__ == '__main__':
    main()
