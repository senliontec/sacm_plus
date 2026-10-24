#!/usr/bin/env python3
"""SAM2-UNet baseline adapter for the frozen 3-shot protocol.

Upstream architecture (``baselines/sam2-unet``) is used unchanged.  This
adapter adds the project's seed-aware data loading, validation/checkpoint
selection, W&B logging and the standard 0/255 prediction format.

Training:
    python baselines/adapters/sam2_unet.py --stage train \
        --data-root data/sacm_3shot_seed0 \
        --workdir results/paper/sam2_unet/seed_0 \
        --sam2-checkpoint checkpoints/sam2_hiera_large.pt \
        --epochs 200 --batch-size 4 --device cuda:0 --use_wandb

Prediction (normally called by scripts/run_baseline_test.py):
    python baselines/adapters/sam2_unet.py --stage predict \
        --checkpoint results/paper/sam2_unet/seed_0/best_model.pth \
        --test-datasets DRIVE:data/sacm_3shot_seed0/test/DRIVE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'sam2-unet'))

from SAM2UNet import SAM2UNet  # noqa: E402
from core.metrics import cldice_score, dice_score, hd95_score, iou_score  # noqa: E402
from core.wandb_utils import finish_run, log_scalars, setup_wandb  # noqa: E402

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
METRICS = ('dice', 'iou', 'cldice', 'hd95')


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


class PairDataset(Dataset):
    def __init__(self, root: Path, input_size: int, train: bool):
        self.pairs = list_pairs(root)
        self.input_size = input_size
        self.train = train
        self.image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        mask = Image.open(mask_path).convert('L').resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = torch.from_numpy(
            np.asarray(mask, dtype=np.float32).copy() / 255.0).unsqueeze(0)
        if self.train:
            if random.random() < 0.5:
                image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
            if random.random() < 0.5:
                image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])
        return image, mask


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def preprocess(image_path: Path, input_size: int, device):
    image = Image.open(image_path).convert('RGB').resize(
        (input_size, input_size), Image.BICUBIC)
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize(MEAN, STD)(tensor).unsqueeze(0)
    return tensor.to(device), image.size


@torch.no_grad()
def validate(model, pairs, device, input_size):
    model.eval()
    overall = {key: [] for key in METRICS}
    per_dataset = {}
    for image_path, mask_path in pairs:
        image, _ = preprocess(image_path, input_size, device)
        logits, _, _ = model(image)
        with Image.open(mask_path) as mask_image:
            width, height = mask_image.size
            gt = np.asarray(mask_image.convert('L')) > 127
        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, size=(height, width), mode='bilinear',
                             align_corners=False)
        pred = prob[0, 0].cpu().numpy() > 0.5
        values = {
            'dice': dice_score(pred, gt),
            'iou': iou_score(pred, gt),
            'cldice': cldice_score(pred, gt),
            'hd95': hd95_score(pred, gt),
        }
        dataset = image_path.stem.split('_', 1)[0]
        bucket = per_dataset.setdefault(dataset, {key: [] for key in METRICS})
        for key, value in values.items():
            overall[key].append(value)
            bucket[key].append(value)
    metrics = {key: float(np.nanmean(vals)) if vals else float('nan')
               for key, vals in overall.items()}
    datasets = {
        name: {key: float(np.nanmean(vals)) for key, vals in values.items()}
        for name, values in sorted(per_dataset.items())
    }
    return metrics, datasets


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.sam2_checkpoint.is_file():
        raise SystemExit(f'missing SAM2 checkpoint: {args.sam2_checkpoint}')
    device = torch.device(args.device)
    train_set = PairDataset(args.data_root / 'train', args.input_size, True)
    val_pairs = list_pairs(args.data_root / 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    model = SAM2UNet(str(args.sam2_checkpoint)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)

    split_path = args.data_root / 'split_manifest.csv'
    split_sha = file_sha256(split_path) if split_path.is_file() else None
    run_config = {
        'method': 'sam2_unet',
        'data_root': str(args.data_root),
        'train_images': len(train_set),
        'val_images': len(val_pairs),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': args.input_size,
        'seed': args.seed,
        'split_manifest_sha256': split_sha,
    }
    run = setup_wandb(args, name=f'sam2_unet-seed{args.seed}',
                      config=run_config)

    args.workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.workdir / 'best_model.pth'
    best_dice = -1.0
    best_epoch = 0
    print(f'sam2_unet: {len(train_set)} train / {len(val_pairs)} val '
          f'epochs={args.epochs} seed={args.seed} device={device}')
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            pred0, pred1, pred2 = model(images)
            loss = (structure_loss(pred0, masks)
                    + structure_loss(pred1, masks)
                    + structure_loss(pred2, masks))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            metrics, per_dataset = validate(
                model, val_pairs, device, args.input_size)
            val_dice = metrics['dice']
            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_dice': val_dice,
                    'val_metrics': metrics,
                    'val_per_dataset': per_dataset,
                    'input_size': args.input_size,
                    'split_manifest_sha256': split_sha,
                    'sam2_checkpoint_sha256': file_sha256(args.sam2_checkpoint),
                }, checkpoint)
            lr = optimizer.param_groups[0]['lr']
            wb = {'epoch': epoch, 'train/loss': running / len(train_set),
                  'lr': lr}
            wb.update({f'val/{key}': value for key, value in metrics.items()})
            log_scalars(run, wb)
            print(f'E{epoch:03d} loss={running / len(train_set):.4f} '
                  f'dice={val_dice:.4f} clDice={metrics["cldice"]:.4f} '
                  f'hd95={metrics["hd95"]:.2f} lr={lr:.2e} '
                  f'{time.time() - started:.0f}s'
                  f'{" *" if improved else ""}', flush=True)
    finish_run(run)
    print(f'best epoch={best_epoch} val_dice={best_dice:.4f} -> {checkpoint}')


@torch.no_grad()
def predict(args) -> None:
    if not args.checkpoint.is_file():
        raise SystemExit(f'missing checkpoint: {args.checkpoint}')
    if not args.sam2_checkpoint.is_file():
        raise SystemExit(f'missing SAM2 checkpoint: {args.sam2_checkpoint}')
    device = torch.device(args.device)
    model = SAM2UNet(str(args.sam2_checkpoint)).to(device)
    payload = torch.load(args.checkpoint, map_location='cpu',
                         weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    for spec in args.test_datasets:
        name, separator, root = spec.partition(':')
        if not separator or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        pairs = list_pairs(Path(root))
        out_dir = args.pred_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'{name}: predicting {len(pairs)} images on {device} -> '
              f'{out_dir}', flush=True)
        for image_path, mask_path in pairs:
            image, _ = preprocess(image_path, args.input_size, device)
            logits, _, _ = model(image)
            with Image.open(mask_path) as mask_image:
                width, height = mask_image.size
            prob = torch.sigmoid(logits)
            prob = F.interpolate(prob, size=(height, width), mode='bilinear',
                                 align_corners=False)
            pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            Image.fromarray(pred, mode='L').save(
                out_dir / f'{image_path.stem}.png')
    meta = {
        'method': 'sam2_unet',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'sam2_checkpoint': str(args.sam2_checkpoint),
        'sam2_checkpoint_sha256': file_sha256(args.sam2_checkpoint),
        'split_manifest_sha256': payload.get('split_manifest_sha256'),
        'datasets': [spec.partition(':')[0] for spec in args.test_datasets],
    }
    with open(args.pred_root / '.prediction_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'predict'],
                        default='train')
    parser.add_argument('--data-root', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot')
    parser.add_argument('--workdir', type=Path,
                        default=REPO_ROOT / 'workdirs' / 'baselines' /
                        'sam2_unet')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='checkpoint for --stage predict '
                             '(default: <workdir>/best_model.pth)')
    parser.add_argument('--sam2-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam2_hiera_large.pt')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' /
                        'sam2_unet')
    parser.add_argument('--test-datasets', nargs='+', default=[])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--input-size', type=int, default=352)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='sacm')
    parser.add_argument('--wandb_entity', default='buaazqk')
    parser.add_argument('--wandb_host', default='http://172.16.1.7:8080')
    parser.add_argument('--wandb_api_key', default=None)
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_group', default='baselines')
    parser.add_argument('--wandb_job_type', default='train')
    parser.add_argument('--wandb_tags', default='sam2_unet,3shot')
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--wandb_dir', default=None)
    parser.add_argument('--wandb_log_artifact', action='store_true')
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.workdir / 'best_model.pth'
    if args.stage == 'train':
        train(args)
    else:
        if not args.test_datasets:
            raise SystemExit('--test-datasets is required for predict')
        predict(args)


if __name__ == '__main__':
    main()
