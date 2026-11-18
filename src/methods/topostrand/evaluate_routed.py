"""Evaluate TopoStrand-SAM with label-free resolution routing.

For images whose native size exceeds the SAM input, run two views:
  * a full-image view (optionally 4-flip TTA), and
  * native-resolution overlapping tiles.

The routed output uses only prediction statistics:
  * tile skeleton/full skeleton <= skel_ratio  -> full view
  * tile area/full area < area_ratio          -> tile view
  * otherwise                                  -> agreement (AND) view

No target labels, supports or dataset-specific rules are used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import binary_closing, disk, skeletonize

from baseline_zoo.data import RawPairDataset, load_rgb
from baseline_zoo.runtime import REPO_ROOT, build_parser
from methods.topostrand.adapter import TopoStrandAdapter

PIPELINE = REPO_ROOT / 'scripts' / 'pipeline'
sys.path.insert(0, str(PIPELINE))
from protocol_specs import dataset_specs  # noqa: E402


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, stride))
    if out[-1] != length - tile:
        out.append(length - tile)
    return out


def _prep(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(
        np.asarray(image, dtype=np.float32).copy()
    ).permute(2, 0, 1).unsqueeze(0) / 255.0


@torch.no_grad()
def _tile_mask(adapter, model, image: Image.Image, tile: int, stride: int,
               flip: bool, rot90: bool, threshold: float) -> np.ndarray:
    width, height = image.size
    logits = torch.zeros((1, 1, height, width), dtype=torch.float32)
    weight = torch.zeros((1, 1, height, width), dtype=torch.float32)
    rotations = (0, 1, 2, 3) if rot90 else (0,)
    flips = (None, (3,), (2,), (2, 3)) if flip else (None,)
    for y in _starts(height, tile, stride):
        for x in _starts(width, tile, stride):
            crop = image.crop(
                (x, y, min(x + tile, width), min(y + tile, height)))
            cw, ch = crop.size
            view = _prep(crop, adapter.input_size).to(adapter.device)
            parts = []
            for rotation in rotations:
                rotated = view if rotation == 0 else torch.rot90(
                    view, rotation, dims=[2, 3])
                for dims in flips:
                    item = rotated if dims is None else torch.flip(
                        rotated, dims=list(dims))
                    part = adapter._deploy_logits(model, item)
                    if dims is not None:
                        part = torch.flip(part, dims=list(dims))
                    if rotation:
                        part = torch.rot90(part, -rotation, dims=[2, 3])
                    parts.append(part)
            part = torch.stack(parts).mean(0)
            part = F.interpolate(
                part, size=(ch, cw), mode='bilinear', align_corners=False)
            logits[0, 0, y:y + ch, x:x + cw] += part[0, 0].cpu()
            weight[0, 0, y:y + ch, x:x + cw] += 1.0
    logits = logits / weight.clamp_min(1.0)
    return (torch.sigmoid(logits)[0, 0].numpy() > threshold)


def _load_model(checkpoint: Path, data_root: Path, device: str):
    adapter = TopoStrandAdapter()
    cli = build_parser(adapter).parse_args([
        '--stage', 'predict', '--checkpoint', str(checkpoint),
        '--data-root', str(data_root), '--device', device,
        '--test-datasets', 'DUMMY:/tmp', '--use_wandb', '0',
    ])
    adapter.args = cli
    adapter.device = torch.device(device)
    adapter.prediction_mode = True
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    adapter.checkpoint_config = payload.get('config', {}) or {}
    adapter.restore_config(
        cli, adapter.checkpoint_config.get('adapter_config', {}))
    model = adapter.build_model(cli)
    adapter.load_state_dict(model, payload['model_state_dict'])
    model.to(adapter.device).eval()
    return adapter, model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--use-wandb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--tta-flip', action='store_true')
    parser.add_argument(
        '--tta-rot90', action='store_true',
        help='full-view 4-rotation TTA; with --tta-flip this becomes 8 views')
    parser.add_argument('--tile-flip', action='store_true')
    parser.add_argument(
        '--tile-rot90', action='store_true',
        help='apply 4-rotation TTA inside native-resolution tiles')
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=896)
    parser.add_argument('--routing-skel-ratio', type=float, default=1.0)
    parser.add_argument('--routing-area-ratio', type=float, default=0.7)
    parser.add_argument(
        '--routing-mode', choices=['auto', 'full', 'tile', 'and', 'union'],
        default='auto',
        help='high-resolution routing: auto uses the skeleton/area rule; '
             'full/tile/and/union force one branch for inference ablations')
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='override the checkpoint deployment threshold for inference')
    parser.add_argument(
        '--deploy-fusion', choices=['none', 'mean'], default='none',
        help='deploy the selected head (none) or the mean logit of all '
             'four Stage-2 candidates (mean)')
    parser.add_argument(
        '--deployment-head', type=int, default=None,
        help='override the fixed Stage-2 slot at inference (0-3)')
    parser.add_argument(
        '--postprocess', choices=['none', 'close1', 'close2', 'close3'],
        default='close2',
        help='label-free topology-preserving morphological closing')
    parser.add_argument('--tag', default='_routed')
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args(argv)

    adapter, model = _load_model(args.checkpoint, args.data_root, args.device)
    adapter.tta_flip = bool(args.tta_flip)
    adapter.tta_rot90 = bool(args.tta_rot90)
    adapter.deploy_fusion = str(args.deploy_fusion)
    if args.deployment_head is not None:
        if not 0 <= args.deployment_head < 4:
            raise SystemExit('--deployment-head must be in [0, 3]')
        adapter.deployment_head = int(args.deployment_head)
        adapter.args.deployment_head = int(args.deployment_head)
    selected = set(args.datasets) if args.datasets else None
    specs = dataset_specs(args.protocol, selected, REPO_ROOT)
    pred_root = args.workdir / f'predictions{args.tag}' / 'topostrand'
    threshold = float(getattr(adapter.args, 'deploy_threshold', 0.5))
    if args.threshold is not None:
        threshold = float(args.threshold)
        adapter.args.deploy_threshold = threshold
    routed = {'full': 0, 'tile': 0, 'and': 0, 'union': 0}
    print(f'routing_mode={args.routing_mode} threshold={threshold:.3f} '
          f'postprocess={args.postprocess}', flush=True)
    for spec in specs:
        dataset, _, root = spec.partition(':')
        root = Path(root)
        ds = RawPairDataset(root, dataset_name=dataset)
        out_dir = pred_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'{dataset}: {len(ds.samples)} images', flush=True)
        for index, sample in enumerate(ds.samples, 1):
            with Image.open(sample.mask_path) as mask:
                gt_w, gt_h = mask.size
            image = load_rgb(sample.image_path)
            prepared = adapter.prepare_eval_sample(sample, adapter.device)
            with torch.no_grad():
                full = adapter.predict_sample(model, prepared, sample)
            if (image.size[0] > args.tile or image.size[1] > args.tile):
                if args.routing_mode == 'full':
                    out = full
                    routed['full'] += 1
                else:
                    tiled = _tile_mask(
                        adapter, model, image, args.tile, args.stride,
                        args.tile_flip, args.tile_rot90, threshold)
                    if tiled.shape != full.shape:
                        raise RuntimeError(
                            f'{dataset}/{sample.image_path.name}: '
                            f'{tiled.shape} != {full.shape}')
                    if args.routing_mode == 'tile':
                        out = tiled
                        routed['tile'] += 1
                    elif args.routing_mode == 'and':
                        out = full & tiled
                        routed['and'] += 1
                    elif args.routing_mode == 'union':
                        out = full | tiled
                        routed['union'] += 1
                    else:
                        fs = float(skeletonize(full).sum())
                        ts = float(skeletonize(tiled).sum())
                        area_ratio = (float(tiled.sum()) /
                                      max(float(full.sum()), 1.0))
                        skel_ratio = ts / max(fs, 1.0)
                        if skel_ratio <= args.routing_skel_ratio:
                            out = full
                            routed['full'] += 1
                        elif area_ratio < args.routing_area_ratio:
                            out = tiled
                            routed['tile'] += 1
                        else:
                            out = full & tiled
                            routed['and'] += 1
            else:
                out = full
                routed['full'] += 1
            if out.shape != (gt_h, gt_w):
                raise RuntimeError(
                    f'{dataset}/{sample.image_path.name}: '
                    f'{out.shape} != {(gt_h, gt_w)}')
            if args.postprocess != 'none':
                radius = {'close1': 1, 'close2': 2, 'close3': 3}[
                    args.postprocess]
                out = binary_closing(out, disk(radius))
            Image.fromarray((out.astype(np.uint8) * 255), mode='L').save(
                out_dir / f'{sample.stem}.png')
            if index % 20 == 0:
                print(f'  {dataset}: {index}/{len(ds.samples)}', flush=True)
    print(f'routing counts: {routed}', flush=True)

    eval_root = args.workdir / f'eval{args.tag}'
    cmd = [sys.executable, REPO_ROOT / 'scripts' / 'pipeline' /
           'collect_paper_metrics.py',
           '--methods', 'topostrand',
           '--pred-root', args.workdir / f'predictions{args.tag}',
           '--out-root', eval_root,
           '--metrics', 'dice,iou,cldice,hd95',
           '--empty-metric-policy', 'penalty',
           '--protocol', args.protocol,
           '--layout', 'canonical',
           '--family', 'baseline',
           '--variant', 'topostrand',
           '--seed', str(args.seed)]
    if args.datasets:
        cmd += ['--datasets', *args.datasets]
    print('+', ' '.join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT,
                   check=True, env=os.environ.copy())
    print(f'eval tables -> {eval_root / "tables" / "all_metrics.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
