"""Protocol evaluator for the independent CoSNA package."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline_zoo.data import Sample, load_mask, load_split_samples
from baseline_zoo.runtime import REPO_ROOT, binary_metrics, build_parser

from .adapter import CoSNAAdapter


def _load(checkpoint: Path, data_root: Path, device: str,
          extra: list[str]) -> tuple[CoSNAAdapter, torch.nn.Module]:
    adapter = CoSNAAdapter()
    cli = build_parser(adapter).parse_args([
        '--stage', 'predict', '--checkpoint', str(checkpoint),
        '--data-root', str(data_root), '--device', device,
        '--test-datasets', 'DUMMY:/tmp', '--use_wandb', '0',
        *extra,
    ])
    adapter.args = cli
    adapter.device = torch.device(device)
    adapter.prediction_mode = True
    payload = torch.load(checkpoint, map_location='cpu',
                         weights_only=False)
    adapter.checkpoint_config = payload.get('config', {}) or {}
    adapter.restore_config(
        cli, adapter.checkpoint_config.get('adapter_config', {}))
    model = adapter.build_model(cli)
    adapter.load_state_dict(model, payload['model_state_dict'])
    model.to(adapter.device).eval()
    return adapter, model


def _samples(protocol: Path, split: str, datasets: set[str] | None):
    if split == 'val':
        samples = load_split_samples(protocol.parent, 'val')
        if datasets:
            samples = [s for s in samples if s.dataset in datasets]
        return samples
    rows = list(csv.DictReader(protocol.open()))
    out = []
    for row in rows:
        if not row.get('split', '').endswith('_test'):
            continue
        if datasets and row['dataset'] not in datasets:
            continue
        out.append(Sample(
            REPO_ROOT / row['image_path'], REPO_ROOT / row['mask_path'],
            row['dataset'], Path(row['image_path']).stem))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rcd', type=int, default=1)
    parser.add_argument('--close-radius', type=int, default=2)
    parser.add_argument('--as-calibration', type=int, default=1)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=896)
    parser.add_argument('--routing-skel-ratio', type=float, default=1.0)
    parser.add_argument('--routing-area-ratio', type=float, default=0.7)
    parser.add_argument('--routing-mode', default='auto')
    parser.add_argument('--cosna-gate', type=float, default=0.10)
    parser.add_argument('--cosna-near-radius', type=int, default=4)
    args = parser.parse_args()
    protocol = args.protocol or (args.data_root / 'protocol_manifest.csv')
    out = args.out or (args.workdir / 'eval_summary.json')
    extra = [
        '--rcd', str(args.rcd), '--close-radius', str(args.close_radius),
        '--as-calibration', str(args.as_calibration), '--tile', str(args.tile),
        '--stride', str(args.stride),
        '--routing-skel-ratio', str(args.routing_skel_ratio),
        '--routing-area-ratio', str(args.routing_area_ratio),
        '--routing-mode', args.routing_mode,
        '--cosna-gate', str(args.cosna_gate),
        '--cosna-near-radius', str(args.cosna_near_radius),
    ]
    adapter, model = _load(
        args.checkpoint, args.data_root, args.device, extra)
    datasets = set(args.datasets) if args.datasets else None
    samples = _samples(protocol, args.split, datasets)
    pred_root = args.workdir / f'predictions_{args.split}' / 'cosna'
    per = defaultdict(list)
    missing = 0
    for index, sample in enumerate(samples, 1):
        prepared = adapter.prepare_eval_sample(sample, adapter.device)
        mask = np.asarray(adapter.predict_sample(model, prepared, sample))
        gt = load_mask(sample.mask_path)
        if mask.shape != gt.shape:
            missing += 1
            continue
        out_path = pred_root / sample.dataset / f'{sample.stem}.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)
        per[sample.dataset].append(binary_metrics(mask, gt))
        if index % 20 == 0:
            print(f'{index}/{len(samples)}', flush=True)
    means = {
        ds: {k: float(np.mean([m[k] for m in values]))
             for k in ('dice', 'iou', 'cldice', 'hd95')}
        for ds, values in per.items()}
    macro = {k: round(float(np.mean([m[k] for m in means.values()])), 4)
             for k in ('dice', 'iou', 'cldice', 'hd95')}
    payload = {
        'split': args.split,
        'images': sum(len(v) for v in per.values()),
        'missing': missing,
        'macro': macro,
        'datasets': {ds: {k: round(v, 4) for k, v in m.items()}
                     for ds, m in means.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(json.dumps(macro, indent=1))
    if missing:
        print('MISSING', missing)


if __name__ == '__main__':
    main()
