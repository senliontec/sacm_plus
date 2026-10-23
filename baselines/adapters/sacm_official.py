#!/usr/bin/env python3
"""Prediction adapter for the official SACM baseline.

The official SACM model is SAM ViT-L with trainable adapters and a
prompt-free mask decoder.  Training is handled by
``scripts/run_sacm_official_train.sh`` (which calls
``baselines/sacm/train_sam.py``); this adapter only runs inference and writes
0/255 PNG masks in the project's prediction format.

Usage:
    python baselines/adapters/sacm_official.py --stage predict \
        --checkpoint workdirs/baselines/sacm_official/best_model.pth \
        --test-datasets CHASEDB1:data/sacm_3shot/test/CHASEDB1 \
                        WIRE:datasets/novel/WIRE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'sacm'))

from segment_anything import build_sam_vit_l  # noqa: E402

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def list_pairs(root: Path):
    image_dir, mask_dir = root / 'images', root / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f'expected images/ and masks/ under {root}')
    images = {p.stem: p for p in sorted(image_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in sorted(mask_dir.iterdir())
             if p.suffix.lower() in IMAGE_EXTS}
    common = sorted(set(images) & set(masks))
    if not common:
        raise SystemExit(f'no image/mask pairs under {root}')
    return [(images[stem], masks[stem]) for stem in common]


def build_model(checkpoint: Path, adapter_ratio: float, device):
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True,
                          adapter_dim_ratio=adapter_ratio)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sam.load_state_dict(payload['model_state_dict'])
    sam.to(device).eval()
    return sam, payload


@torch.no_grad()
def predict_logits(sam, image: torch.Tensor, device):
    """Official zero-prompt forward: no prompt encoder call."""
    sparse = torch.zeros((1, 0, 256), dtype=torch.float, device=device)
    dense = torch.zeros((1, 256, 64, 64), dtype=torch.float, device=device)
    encoder_output = sam.image_encoder(image, return_features=True)
    if len(encoder_output) == 3:
        image_embeddings, _, adapter_features = encoder_output
    else:
        image_embeddings, _ = encoder_output
        adapter_features = None
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        adapter_features=adapter_features,
    )
    return low_res_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['predict'], default='predict')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPO_ROOT / 'workdirs' / 'baselines' /
                        'sacm_official' / 'best_model.pth')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' /
                        'sacm_official')
    parser.add_argument('--test-datasets', nargs='+', default=[],
                        help='NAME:GT_ROOT entries for prediction')
    parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f'missing checkpoint: {args.checkpoint}')
    # Capture the hash before the prediction loop so a concurrent training
    # job that replaces the checkpoint cannot break metadata writing.
    checkpoint_sha256 = file_sha256(args.checkpoint)
    device = torch.device(args.device)
    sam, payload = build_model(args.checkpoint, args.adapter_dim_ratio, device)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    datasets = []
    for spec in args.test_datasets:
        name, separator, root = spec.partition(':')
        if not separator or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        pairs = list_pairs(Path(root))
        datasets.append((name, pairs))

    for name, pairs in datasets:
        out_dir = args.pred_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'{name}: predicting {len(pairs)} images on {device} '
              f'-> {out_dir}', flush=True)
        t0 = time.time()
        for done, (image_path, mask_path) in enumerate(pairs, 1):
            with Image.open(image_path) as image_file:
                image = transform(image_file.convert('RGB')).unsqueeze(0)
            with Image.open(mask_path) as mask_file:
                width, height = mask_file.size
            logits = predict_logits(sam, image.to(device), device)
            logits = F.interpolate(logits, size=(height, width),
                                   mode='bilinear', align_corners=False)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
            Image.fromarray((pred.astype(np.uint8) * 255), mode='L').save(
                out_dir / f'{image_path.stem}.png')
            if done % 50 == 0 and done != len(pairs):
                rate = done / max(time.time() - t0, 1e-6)
                print(f'  {name}: {done}/{len(pairs)}  {rate:.1f} img/s',
                      flush=True)
        print(f'{name}: done {len(pairs)} predictions in '
              f'{time.time() - t0:.0f}s', flush=True)

    meta = {
        'method': 'sacm_official',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'split_manifest_sha256': payload.get('split_manifest_sha256'),
        'adapter_dim_ratio': args.adapter_dim_ratio,
        'datasets': [name for name, _ in datasets],
    }
    with open(args.pred_root / '.prediction_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
