#!/usr/bin/env python3
"""Inference-only native-resolution tiling for SACM-X.

The standard evaluator resizes the whole image to 1024x1024, which loses
capillary-scale pixels on large OCTA/retinal images.  This script tiles the
ORIGINAL image into overlapping 1024x1024 patches, runs the existing SACM-X
checkpoint on each patch, averages the predicted probabilities, and writes
original-resolution ``*_result.png`` masks.  Metrics/component audits are
computed afterwards by ``scripts/pipeline/audit_predictions.py``.

No training, no checkpoint changes, no test-label-driven tile selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from baseline_zoo.data import list_pairs  # noqa: E402
from core.registry import get_model_spec  # noqa: E402
from models.sacm_x.configs import PRESETS, apply_preset, str2bool  # noqa: E402


def tile_positions(total: int, tile: int, stride: int) -> list[int]:
    """Cover [0, total) with tiles; the last tile is clamped to the edge."""
    if total <= tile:
        return [0]
    stride = max(1, int(stride))
    positions = list(range(0, total - tile + 1, stride))
    last = total - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _build_model(args, device):
    spec = get_model_spec(args.model)
    build_kwargs = dict(
        checkpoint=None,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        geo_e_layers=args.geo_e_layers,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
        use_typed_heads=getattr(args, 'use_typed_heads', False),
        use_tcsu=getattr(args, 'use_tcsu', False),
        tcsu_scale=getattr(args, 'tcsu_scale', 4),
        tcsu_mid_channels=getattr(args, 'tcsu_mid_channels', 32),
        tcsu_max_length=getattr(args, 'tcsu_max_length', 5),
        use_tadt=getattr(args, 'use_tadt', False),
        tadt_hidden=getattr(args, 'tadt_hidden', 128),
        tadt_t_min=getattr(args, 'tadt_t_min', -4.0),
        tadt_t_max=getattr(args, 'tadt_t_max', 4.0),
        use_ranker=getattr(args, 'use_ranker', False),
        ranker_hidden=getattr(args, 'ranker_hidden', 128),
        ranker_pair_hidden=getattr(args, 'ranker_pair_hidden', 128),
        use_detail_stem=getattr(args, 'use_detail_stem', False),
        detail_channels=getattr(args, 'detail_channels', 32),
        detail_out_channels=getattr(args, 'detail_out_channels', 32),
        high_res_output=getattr(args, 'high_res_output', False),
        high_res_size=getattr(args, 'high_res_size', 512),
        high_res_refine=getattr(args, 'high_res_refine', False),
        high_res_refine_size=getattr(args, 'high_res_refine_size', 1024),
        single_head=getattr(args, 'single_head', False),
    )
    # Keep this script runnable against checkouts that do not yet contain
    # the optional dynamic-strip model code.  For the default `full` preset
    # this flag is false and is not passed at all.
    if bool(getattr(args, 'use_dynamic_strip', False)):
        build_kwargs['use_dynamic_strip'] = True
    sam = spec.build(**build_kwargs)
    payload = torch.load(args.checkpoint, map_location='cpu',
                         weights_only=False)
    state = payload.get('model_state_dict') if isinstance(payload, dict) else None
    if not state:
        raise SystemExit(
            f'checkpoint has no model_state_dict: {args.checkpoint}')
    sam.load_state_dict(state)
    sam.to(device).eval()
    return spec, sam


@torch.inference_mode()
def predict_image(spec, sam, image_path: Path, device, args) -> np.ndarray:
    image = _load_rgb(image_path)
    orig_h, orig_w = int(image.shape[-2]), int(image.shape[-1])
    tile = int(args.tile_size)
    stride = int(args.tile_stride)

    if orig_h < tile or orig_w < tile:
        pad_h = max(0, tile - orig_h)
        pad_w = max(0, tile - orig_w)
        # ``reflect`` requires padding < input size, which fails for small
        # OCTA-like images.  Replicate is safe for any padding and does not
        # invent new content; tiling a small image is still only a proximity
        # check, not a resolution gain.
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
    img_h, img_w = int(image.shape[-2]), int(image.shape[-1])
    ys = tile_positions(img_h, tile, stride)
    xs = tile_positions(img_w, tile, stride)

    patches = []
    coords = []
    for top in ys:
        for left in xs:
            patches.append(image[:, top:top + tile, left:left + tile])
            coords.append((top, left))
    if not patches:
        raise RuntimeError(f'no tiles generated for {image_path}')

    prob = torch.zeros((img_h, img_w), dtype=torch.float32)
    count = torch.zeros((img_h, img_w), dtype=torch.float32)
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(patches), batch_size):
        batch = torch.stack(patches[start:start + batch_size]).to(device)
        processed = spec.preprocess(sam, batch)
        sparse, dense = spec.prompt_builder(sam, batch.shape[0])
        masks, _ = spec.forward(
            sam, processed, sparse, dense, return_stage1=False)
        if masks.ndim != 4 or masks.shape[1] <= int(args.fixed_head):
            raise RuntimeError(
                f'unexpected mask shape {tuple(masks.shape)} for fixed head '
                f'{args.fixed_head}')
        logits = masks[:, int(args.fixed_head):int(args.fixed_head) + 1]
        probs = torch.sigmoid(F.interpolate(
            logits, size=(tile, tile), mode='bilinear',
            align_corners=False))[:, 0].cpu()
        for (top, left), patch_prob in zip(
                coords[start:start + batch_size], probs):
            prob[top:top + tile, left:left + tile] += patch_prob
            count[top:top + tile, left:left + tile] += 1.0

    prob = prob / count.clamp_min(1.0)
    prob = prob[:orig_h, :orig_w]
    prob = F.interpolate(
        prob[None, None], size=(orig_h, orig_w),
        mode='bilinear', align_corners=False)[0, 0]
    threshold = torch.sigmoid(torch.tensor(
        float(args.pred_threshold), dtype=torch.float32)).item()
    return (prob.numpy() > threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-name', default='OCTA-500')
    parser.add_argument('--model', default='sam_l',
                        choices=['sam_l', 'sam_b', 'sam_h'])
    parser.add_argument('--preset', default='full', choices=sorted(PRESETS))
    parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
    parser.add_argument('--use-geo-i', type=str2bool, default=True)
    parser.add_argument('--use-geo-e', type=str2bool, default=True)
    parser.add_argument('--geo-e-layers', default='all',
                        choices=['all', 'shallow', 'deep'])
    parser.add_argument('--use-coarse-to-fine', type=str2bool, default=True)
    parser.add_argument('--use-fusion-v2', type=str2bool, default=True)
    parser.add_argument('--use-multi-depth', type=str2bool, default=True)
    parser.add_argument('--use-dynamic-strip', type=str2bool, default=False)
    parser.add_argument('--use-typed-heads', type=str2bool, default=False)
    parser.add_argument('--use-tcsu', type=str2bool, default=False)
    parser.add_argument('--use-tadt', type=str2bool, default=False)
    parser.add_argument('--use-ranker', type=str2bool, default=False)
    parser.add_argument('--use-detail-stem', type=str2bool, default=False)
    parser.add_argument('--high-res-output', type=str2bool, default=False)
    parser.add_argument('--high-res-refine', type=str2bool, default=False)
    parser.add_argument('--single-head', type=str2bool, default=False)
    parser.add_argument('--fixed-head', type=int, default=2,
                        choices=[0, 1, 2, 3])
    parser.add_argument('--pred-threshold', type=float, default=0.0,
                        help='logit threshold; 0.0 = probability 0.5')
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--tile-stride', type=int, default=896)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    apply_preset(args, args.preset)
    if args.tile_size != 1024:
        raise SystemExit('SACM-X expects tile-size 1024')
    if args.tile_stride <= 0 or args.tile_stride > args.tile_size:
        raise SystemExit('tile-stride must be in [1, tile-size]')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device else
        ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')

    pairs = list_pairs(args.data_root)
    if args.limit:
        pairs = pairs[:max(0, int(args.limit))]
    if not pairs:
        raise SystemExit(f'no image/mask pairs under {args.data_root}')

    spec, sam = _build_model(args, device)
    pred_dir = args.output_dir / args.dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f'device={device} images={len(pairs)}; '
          f'tile={args.tile_size} stride={args.tile_stride} '
          f'batch={args.batch_size}', flush=True)
    for index, (image_path, _mask_path) in enumerate(pairs, 1):
        pred = predict_image(spec, sam, image_path, device, args)
        out_path = pred_dir / f'{image_path.stem}_result.png'
        Image.fromarray((pred.astype(np.uint8) * 255)).save(out_path)
        if index % 20 == 0 or index == len(pairs):
            print(f'  {index}/{len(pairs)} {image_path.name}', flush=True)
    print(f'wrote: {pred_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
