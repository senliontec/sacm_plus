#!/usr/bin/env python3
"""Compare Stage-2 head selection/fusion on the validation split.

This is an inference-only ablation: no training.  It is intended for a
4-head full/fixed2 checkpoint.  If the checkpoint is stale, treat the
result as a relative ranking of fusion strategies, not as a final score.

Strategies:
  fixed2        old default: sort by gating and take rank 2
  mean_logit    mean of the 4 Stage-2 logits
  mean_prob     mean of the 4 sigmoid probabilities
  iou_logit     softmax(iou_pred) weighted logits
  iou_prob      softmax(iou_pred) weighted probabilities

Usage:
  python scripts/sacm_x/eval_head_fusion_val.py \
    --checkpoint /path/to/best_model.pth \
    --data-root data/sacm_3shot_seed0 --split val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.data import TestDataset
from core.io import get_prompt_embeddings, predict_all
from core.metrics import binary_metrics
from models.sacm_x.specs import sam_preprocess
from models.sam_network.build_sam import build_sam_vit_l

STRATEGIES = ('fixed2', 'mean_logit', 'mean_prob', 'iou_logit', 'iou_prob')


def _load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and isinstance(
            payload.get('model_state_dict'), dict):
        return payload['model_state_dict'], payload.get('config', {})
    return payload, {}


def _build(config):
    return build_sam_vit_l(
        checkpoint=None,
        use_adapter=True,
        adapter_dim_ratio=float(config.get('adapter_dim_ratio', 0.1)),
        use_geo_i=bool(config.get('use_geo_i', True)),
        use_geo_e=bool(config.get('use_geo_e', True)),
        geo_e_layers=config.get('geo_e_layers', 'all'),
        use_coarse_to_fine=bool(config.get('use_coarse_to_fine', True)),
        use_fusion_v2=bool(config.get('use_fusion_v2', True)),
        use_multi_depth=bool(config.get('use_multi_depth', True)),
        use_typed_heads=False,
        single_head=False,
        use_detail_stem=False,
        high_res_output=False,
    )


def _fuse(logits, iou, strategy):
    """logits [1,4,H,W], iou [1,4] -> binary probability [1,1,H,W]."""
    if strategy == 'fixed2':
        return torch.sigmoid(logits[:, 2:3])
    if strategy == 'mean_logit':
        return torch.sigmoid(logits.mean(dim=1, keepdim=True))
    if strategy == 'mean_prob':
        return torch.sigmoid(logits).mean(dim=1, keepdim=True)
    w = torch.softmax(iou, dim=1)[..., None, None]
    if strategy == 'iou_logit':
        return torch.sigmoid((logits * w).sum(dim=1, keepdim=True))
    if strategy == 'iou_prob':
        return (torch.sigmoid(logits) * w).sum(dim=1, keepdim=True)
    raise ValueError(strategy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/sacm_3shot_seed0')
    parser.add_argument('--split', default='val')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='')
    parser.add_argument('--save-dir', default='',
                        help='Optional directory for fixed2 predictions')
    parser.add_argument('--deep-closing-checkpoint', default='',
                        help='Optional Deep Closing shape prior; adds '
                             'fixed2_dc to the comparison')
    args = parser.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    refiner = None
    strategies = list(STRATEGIES)
    if args.deep_closing_checkpoint:
        from core.postprocess.deep_closing import DeepClosingRefiner
        refiner = DeepClosingRefiner.from_checkpoint(
            args.deep_closing_checkpoint, device=args.device)
        strategies.append('fixed2_dc')

    state, config = _load_checkpoint(args.checkpoint)
    model = _build(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'loaded {args.checkpoint}: missing={len(missing)} '
          f'unexpected={len(unexpected)}')
    model.eval().to(args.device)

    dataset = TestDataset(args.data_root, split=args.split,
                          input_mode='resize', image_size=1024)
    values = {s: defaultdict(list) for s in strategies}
    per_dataset = {s: defaultdict(lambda: defaultdict(list))
                   for s in strategies}
    n = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    with torch.no_grad():
        for i in range(n):
            image, gt, _, _, fov = dataset[i]
            original_hw = gt.shape[-2:]
            batch = sam_preprocess(model, image.unsqueeze(0).to(args.device))
            sparse, dense = get_prompt_embeddings(model, 1)
            masks, iou = predict_all(
                model, batch, sparse, dense, return_stage1=False)
            masks = F.interpolate(
                masks, size=original_hw, mode='bilinear',
                align_corners=False)
            iou = iou.to(args.device)
            fov_np = fov[0].numpy() > 0.5
            gt_np = (gt[0].numpy() > 0.5) & fov_np
            ds = dataset.dataset_for(dataset.image_files[i])
            fixed2_pred = (_fuse(masks, iou, 'fixed2')[0, 0].cpu().numpy()
                           > 0.5) & fov_np
            for strategy in strategies:
                if strategy == 'fixed2_dc':
                    pred = refiner.refine(
                        fixed2_pred, threshold=0.5) & fov_np
                else:
                    prob = _fuse(masks, iou, strategy)
                    pred = (prob[0, 0].cpu().numpy() > 0.5) & fov_np
                metric = binary_metrics(pred, gt_np)
                for key in ('dice', 'iou', 'cldice', 'hd95'):
                    values[strategy][key].append(metric[key])
                    per_dataset[strategy][ds][key].append(metric[key])
                if args.save_dir and strategy == 'fixed2':
                    stem = os.path.splitext(dataset.image_files[i])[0] + '.png'
                    Image.fromarray(
                        pred.astype(np.uint8) * 255, mode='L'
                    ).save(os.path.join(args.save_dir, stem))
            print(f'{i + 1}/{n} {dataset.image_files[i]}', flush=True)

    print('\nValidation head fusion (image-macro)')
    print('=' * 74)
    print(f"{'strategy':12s} {'dice':>8s} {'iou':>8s} {'cldice':>8s} {'hd95':>8s}")
    for strategy in strategies:
        row = {k: float(np.mean(values[strategy][k]))
               for k in ('dice', 'iou', 'cldice', 'hd95')}
        print(f'{strategy:12s} {row["dice"]*100:8.2f} {row["iou"]*100:8.2f} '
              f'{row["cldice"]*100:8.2f} {row["hd95"]:8.2f}')
    print('\nPer-dataset dice')
    datasets = sorted(per_dataset[strategies[0]])
    print(f"{'dataset':14s} " + ' '.join(f'{s:>10s}' for s in strategies))
    for ds in datasets:
        cells = []
        for strategy in strategies:
            vals = per_dataset[strategy][ds]['dice']
            cells.append(f'{float(np.mean(vals))*100:10.2f}' if vals else ' ' * 10)
        print(f'{ds:14s} ' + ' '.join(cells))

    if args.out:
        payload = {
            'checkpoint': args.checkpoint,
            'strategies': {
                s: {k: float(np.mean(v)) for k, v in values[s].items()}
                for s in strategies
            },
        }
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)


if __name__ == '__main__':
    main()
