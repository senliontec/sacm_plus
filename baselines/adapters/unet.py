#!/usr/bin/env python3
"""U-Net baseline adapter for the frozen 3-shot protocol.

The upstream architecture in ``baselines/unet`` is used unchanged.  This
adapter supplies the project's own data loading, training loop, validation
checkpoint selection, and prediction format so that U-Net is comparable with
the other baselines.

Examples:
    # Train on the 18-image train split, select on the 6-image val split.
    python baselines/adapters/unet.py --stage train

    # Predict and evaluate two test sets.
    python baselines/adapters/unet.py --stage predict \
        --test-datasets DRIVE:data/sacm_3shot/test/DRIVE \
                        WIRE:datasets/novel/WIRE
    python scripts/evaluate_predictions.py \
        --pred-root results/predictions/unet \
        --datasets DRIVE:data/sacm_3shot/test/DRIVE \
                   WIRE:datasets/novel/WIRE \
        --out-root results/unet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'unet'))
sys.path.insert(0, str(REPO_ROOT / 'src'))
from unet import UNet  # noqa: E402  (upstream model, unchanged)
from core.metrics import cldice_score, hd95_score  # noqa: E402
from core.wandb_utils import (  # noqa: E402
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')
MASK_EXTS = IMAGE_EXTS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def list_pairs(root: Path):
    image_dir, mask_dir = root / 'images', root / 'masks'
    pairs = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        mask = next(
            (mask_dir / (image.stem + ext) for ext in MASK_EXTS
             if (mask_dir / (image.stem + ext)).is_file()),
            None)
        if mask is not None:
            pairs.append((image, mask))
    return pairs


class PairDataset(Dataset):
    def __init__(self, root: Path, input_size: int = 512, augment: bool = False):
        self.pairs = list_pairs(root)
        if not self.pairs:
            raise SystemExit(f'no image/mask pairs under {root}')
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert('RGB').resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = Image.open(mask_path).convert('L').resize(
            (self.input_size, self.input_size), Image.NEAREST)

        image = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(
            np.asarray(mask, dtype=np.float32).copy()).unsqueeze(0)
        mask = (mask > 127.0).float()

        if self.augment:
            if random.random() < 0.5:
                image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
            if random.random() < 0.5:
                image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
            k = random.randrange(4)
            if k:
                image = torch.rot90(image, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
        return image, mask


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def dice_coefficient(prob: torch.Tensor, target: torch.Tensor) -> float:
    pred = (prob > 0.5).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter + 1e-7) / (denom + 1e-7)


def iou_coefficient(prob: torch.Tensor, target: torch.Tensor) -> float:
    pred = (prob > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter + 1e-7) / (union + 1e-7)


def _mean(values):
    if not values:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(values))


@torch.no_grad()
def validate(model, pairs, device, input_size):
    """Validate at the ORIGINAL GT resolution, per dataset and overall."""
    model.eval()
    overall = {key: [] for key in ('dice', 'iou', 'cldice', 'hd95', 'empty')}
    per_dataset = {}
    for image_path, mask_path in pairs:
        dataset = image_path.stem.split('_', 1)[0]
        bucket = per_dataset.setdefault(
            dataset, {key: [] for key in ('dice', 'iou', 'cldice', 'hd95',
                                          'empty')})
        image = Image.open(image_path).convert('RGB').resize(
            (input_size, input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        tensor = (tensor / 255.0).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor))
        with Image.open(mask_path) as mask_image:
            width, height = mask_image.size
            gt = np.asarray(mask_image.convert('L')) > 127
        prob = F.interpolate(prob, size=(height, width),
                             mode='bilinear', align_corners=False)
        pred = prob[0, 0].cpu().numpy() > 0.5

        inter = np.logical_and(pred, gt).sum()
        denom = pred.sum() + gt.sum()
        union = denom - inter
        values = {
            'dice': float((2.0 * inter + 1e-7) / (denom + 1e-7)),
            'iou': float((inter + 1e-7) / (union + 1e-7)),
            'cldice': cldice_score(pred, gt),
            'hd95': hd95_score(pred, gt),
            'empty': float(pred.sum() == 0),
        }
        for key, value in values.items():
            overall[key].append(value)
            bucket[key].append(value)

    metrics = {key: _mean(values) for key, values in overall.items()}
    metrics['empty_prediction_rate'] = metrics.pop('empty')
    datasets = {}
    for dataset, values in sorted(per_dataset.items()):
        datasets[dataset] = {
            key: _mean(items) for key, items in values.items()
            if key != 'empty'
        }
        datasets[dataset]['empty_prediction_rate'] = _mean(values['empty'])
    return metrics, datasets


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    train_set = PairDataset(args.data_root / 'train', args.input_size, augment=True)
    val_set = PairDataset(args.data_root / 'val', args.input_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)

    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    split_manifest = args.data_root / 'split_manifest.csv'
    split_sha256 = (file_sha256(split_manifest)
                    if split_manifest.is_file() else None)

    run = setup_wandb(
        args,
        name=f'unet-seed{args.seed}',
        config={
            'method': 'unet',
            'data_root': str(args.data_root),
            'train_images': len(train_set),
            'val_images': len(val_set),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'input_size': args.input_size,
            'seed': args.seed,
            'optimizer': 'AdamW',
            'loss': 'BCE+Dice',
            'split_manifest_sha256': split_sha256,
        },
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.workdir / 'best_model.pth'
    if ckpt_path.is_file():
        backup = ckpt_path.with_name(
            f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        shutil.copy2(ckpt_path, backup)
        print(f'backup previous checkpoint -> {backup}')
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()
        val_metrics, val_per_dataset = validate(
            model, val_set.pairs, device, args.input_size)
        val_dice = val_metrics['dice']
        train_loss = running / len(train_set)
        wb = {
            'epoch': epoch,
            'train/loss': train_loss,
            'lr': optimizer.param_groups[0]['lr'],
        }
        wb.update({f'val/{key}': value
                   for key, value in val_metrics.items()})
        for dataset, values in val_per_dataset.items():
            wb.update({f'val/{dataset}/{key}': value
                       for key, value in values.items()})
        log_scalars(run, wb)
        print(f'epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  '
              f'val_dice={val_metrics["dice"]:.4f}  '
              f'val_iou={val_metrics["iou"]:.4f}  '
              f'val_cldice={val_metrics["cldice"]:.4f}  '
              f'val_hd95={val_metrics["hd95"]:.4f}  '
              f'empty={val_metrics["empty_prediction_rate"]:.2f}', flush=True)
        for dataset, values in val_per_dataset.items():
            print(f'  {dataset}: dice={values["dice"]:.4f} '
                  f'iou={values["iou"]:.4f} '
                  f'cldice={values["cldice"]:.4f} '
                  f'hd95={values["hd95"]:.4f} '
                  f'empty={values["empty_prediction_rate"]:.0f}', flush=True)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_dice': val_dice,
                'val_metrics': val_metrics,
                'val_per_dataset': val_per_dataset,
                'input_size': args.input_size,
                'split_manifest_sha256': split_sha256,
            }, ckpt_path)
    print(f'best val_dice={best_dice:.4f} -> {ckpt_path}')
    log_artifact(run, ckpt_path, name=f'unet-seed{args.seed}',
                 artifact_type='model')
    finish_run(run)
    return ckpt_path


@torch.no_grad()
def predict(args, ckpt_path: Path):
    device = torch.device(args.device)
    payload = torch.load(ckpt_path, map_location=device)
    input_size = payload.get('input_size', args.input_size)
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()

    for spec in args.test_datasets:
        name, sep, root = spec.partition(':')
        if not sep or not name or not root:
            raise SystemExit(f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        root = Path(root)
        out_dir = args.pred_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = list_pairs(root)
        if not pairs:
            raise SystemExit(f'no image/mask pairs under {root}')
        for image_path, mask_path in pairs:
            image = Image.open(image_path).convert('RGB').resize(
                (input_size, input_size), Image.BICUBIC)
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
            tensor = (tensor / 255.0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))
            with Image.open(mask_path) as mask_image:
                width, height = mask_image.size
            prob = F.interpolate(prob, size=(height, width),
                                 mode='bilinear', align_corners=False)
            pred = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            Image.fromarray(pred, mode='L').save(out_dir / f'{image_path.stem}.png')
        print(f'{name}: {len(pairs)} predictions -> {out_dir}')

    meta = {
        'method': 'unet',
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': file_sha256(ckpt_path),
        'split_manifest_sha256': payload.get('split_manifest_sha256'),
        'datasets': [spec.partition(':')[0] for spec in args.test_datasets],
    }
    with open(args.pred_root / '.prediction_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'predict', 'all'],
                        default='all')
    parser.add_argument('--data-root', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot')
    parser.add_argument('--workdir', type=Path,
                        default=REPO_ROOT / 'workdirs' / 'baselines' / 'unet')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='checkpoint used by --stage predict '
                             '(default: <workdir>/best_model.pth)')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' / 'unet')
    parser.add_argument('--test-datasets', nargs='+', default=[],
                        help='NAME:GT_ROOT entries for prediction')
    parser.add_argument('--epochs', type=int, default=200,
                        help='U-Net needs ~200 epochs to converge on 18 images')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true',
                        help='log training scalars and the best checkpoint to W&B')
    parser.add_argument('--wandb_project', default='sacm')
    parser.add_argument('--wandb_entity',
                        default=os.environ.get('WANDB_ENTITY', 'buaazqk'))
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_group', default='baselines')
    parser.add_argument('--wandb_job_type', default='train')
    parser.add_argument('--wandb_tags', default='unet,3shot')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'],
                        default=None)
    parser.add_argument('--wandb_host',
                        default=os.environ.get(
                            'WANDB_BASE_URL', 'http://172.16.1.7:8080'))
    parser.add_argument('--wandb_api_key', default=os.environ.get('WANDB_API_KEY'))
    parser.add_argument('--wandb_dir', default=None)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.stage in ('predict', 'all'):
        if not args.test_datasets:
            if args.stage == 'predict':
                raise SystemExit('--test-datasets is required for --stage predict')
            print('no --test-datasets; training only')
    ckpt_path = args.checkpoint or (args.workdir / 'best_model.pth')
    if args.stage in ('train', 'all') and args.checkpoint is not None:
        raise SystemExit('--checkpoint is a predict-time option; '
                         'use --workdir for training output')
    if args.stage == 'train':
        train(args)
    elif args.stage == 'predict':
        if not ckpt_path.is_file():
            raise SystemExit(f'missing checkpoint: {ckpt_path}')
        predict(args, ckpt_path)
    else:
        train(args)
        if args.test_datasets:
            predict(args, ckpt_path)


if __name__ == '__main__':
    main()
