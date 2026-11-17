#!/usr/bin/env python3
"""Export per-candidate TopoStrand masks and simple head fusions.

This is a diagnostic / screening tool.  It runs one forward pass per view and
writes the four post-gating Stage-2 candidates plus three label-free fusion
variants (mean probability, intersection, union) in the canonical prediction
contract so the metric collector can score each one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'pipeline'))

from baseline_zoo.data import RawPairDataset, load_rgb  # noqa: E402
from methods.topostrand.evaluate_routed import _load_model  # noqa: E402
from protocol_specs import dataset_specs  # noqa: E402

HEADS = ('h0', 'h1', 'h2', 'h3', 'mean', 'and', 'or')


def _views(prepared, flip: bool):
    flips = (None, (3,), (2,), (2, 3)) if flip else (None,)
    for dims in flips:
        view = prepared if dims is None else torch.flip(prepared, dims=list(dims))
        yield view, dims


def candidate_probs(adapter, model, prepared, flip: bool):
    parts = []
    for view, dims in _views(prepared, flip):
        stage2, _ = adapter._forward(model, view, return_stage1=False)
        if dims is not None:
            stage2 = torch.flip(stage2, dims=list(dims))
        parts.append(stage2)
    return torch.sigmoid(torch.stack(parts).mean(0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/sacm_3shot_seed0'))
    parser.add_argument('--protocol', type=Path,
                        default=Path('data/sacm_3shot_seed0/protocol_manifest.csv'))
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tta-flip', action='store_true')
    parser.add_argument('--save-probs', action='store_true',
                        help='save float16 per-head sigmoid probabilities')
    parser.add_argument('--tag', default='_candidates')
    args = parser.parse_args(argv)

    adapter, model = _load_model(
        args.checkpoint, args.data_root, args.device)
    adapter.tta_flip = bool(args.tta_flip)
    selected = set(args.datasets) if args.datasets else None
    specs = dataset_specs(args.protocol, selected, REPO_ROOT)
    roots = {name: args.workdir / f'predictions{args.tag}' / name
             for name in HEADS}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    prob_root = args.workdir / f'probs{args.tag}'

    for spec in specs:
        dataset, _, root = spec.partition(':')
        ds = RawPairDataset(Path(root), dataset_name=dataset)
        for head_root in roots.values():
            (head_root / dataset).mkdir(parents=True, exist_ok=True)
        if args.save_probs:
            (prob_root / dataset).mkdir(parents=True, exist_ok=True)
        print(f'{dataset}: {len(ds.samples)} images', flush=True)
        for index, sample in enumerate(ds.samples, 1):
            prepared = adapter.prepare_eval_sample(sample, adapter.device)
            with Image.open(sample.mask_path) as mask:
                width, height = mask.size
            with torch.no_grad():
                probs = candidate_probs(
                    adapter, model, prepared, args.tta_flip)
            probs = F.interpolate(
                probs, size=(height, width), mode='bilinear',
                align_corners=False)[0].cpu().numpy()
            binary = probs > args.threshold
            outputs = {
                'h0': binary[0], 'h1': binary[1],
                'h2': binary[2], 'h3': binary[3],
                'mean': probs.mean(0) > args.threshold,
                'and': binary.all(0), 'or': binary.any(0),
            }
            for name, mask in outputs.items():
                Image.fromarray(
                    (mask.astype(np.uint8) * 255), mode='L').save(
                        roots[name] / dataset / f'{sample.stem}.png')
            if args.save_probs:
                np.save(
                    prob_root / dataset / f'{sample.stem}.npy',
                    probs.astype(np.float16))
            if index % 20 == 0:
                print(f'  {dataset}: {index}/{len(ds.samples)}', flush=True)
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
