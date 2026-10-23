#!/usr/bin/env python3
"""SAM ViT-L + U-Net decoder baseline adapter for the frozen 3-shot protocol.

The SAM ViT-L image encoder is frozen and loaded from
``checkpoints/sam_vit_l_0b3195.pth``.  A trainable U-Net-style decoder is
attached to the encoder:

* four intermediate ViT block features (blocks 0, depth/3, 2*depth/3,
  depth-1) plus the neck output are projected to 64 channels and fused;
* a stride-2 RGB stem supplies 1/2, 1/4 and 1/8 skip connections so the
  decoder keeps high-resolution detail for thin structures;
* the decoder upsamples back to the input resolution and predicts one logit
  channel.

Only the projections, stem and decoder are trained (the ViT-L encoder stays
frozen), which matches the 18-image fine-tuning budget used by the other
baselines.  Training caches the frozen encoder features once, so the decoder
can be trained quickly; flips/rot90 are applied consistently to the cached
features, the image and the mask.

Examples:
    # Train on the 18-image train split, select on the 6-image val split.
    python baselines/adapters/sam_vit_l_unet.py --stage train \
        --data-root data/sacm_3shot \
        --sam-checkpoint checkpoints/sam_vit_l_0b3195.pth \
        --workdir workdirs/baselines/sam_vit_l_unet \
        --epochs 200 --batch-size 4 --input-size 1024

    # Predict (the standard runner calls this stage automatically).
    python baselines/adapters/sam_vit_l_unet.py --stage predict \
        --checkpoint workdirs/baselines/sam_vit_l_unet/best_model.pth \
        --test-datasets CHASEDB1:data/sacm_3shot/test/CHASEDB1 \
                        WIRE:datasets/novel/WIRE
    python scripts/run_baseline_test.py --method sam_vit_l_unet
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.console import console  # noqa: E402
from core.metrics import cldice_score, dice_score, hd95_score, iou_score  # noqa: E402
from core.wandb_utils import (  # noqa: E402
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)
from models.sam.build_sam import build_sam_vit_l  # noqa: E402

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')
MASK_EXTS = IMAGE_EXTS

# SAM's official preprocessing constants (0-255 scale, converted to 0-1).
_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1) / 255.0
_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) / 255.0

# ECG-SAM style fixed-width line table, matching the U-Net adapter.
LOG_SEP = ' │ '
LOG_COLS = ('ep', 'tr_loss', 'Dice↑', 'IoU↑', 'clDice↑', 'HD95↓', 'lr', 'time')
LOG_WIDTHS = (5, 8, 7, 7, 8, 7, 8, 8)
LOG_RULE = '─┼─'.join('─' * width for width in LOG_WIDTHS)
LOG_FOOTER = '─' * (sum(LOG_WIDTHS) + 3 * (len(LOG_WIDTHS) - 1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def list_pairs(root: Path):
    """Pair images/ and masks/ by stem, accepting mixed extensions."""
    image_dir, mask_dir = root / 'images', root / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f'expected images/ and masks/ under {root}')
    images = {p.stem: p for p in sorted(image_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in sorted(mask_dir.iterdir())
             if p.suffix.lower() in MASK_EXTS}
    common = sorted(set(images) & set(masks))
    if not common:
        raise SystemExit(f'no image/mask pairs under {root}')
    return [(images[stem], masks[stem]) for stem in common]


def load_pair(image_path: Path, mask_path: Path, input_size: int):
    """Return model image (3,S,S), resized mask (1,S,S), original GT (H,W)."""
    with Image.open(image_path) as image_file:
        image = image_file.convert('RGB').resize(
            (input_size, input_size), Image.BICUBIC)
    with Image.open(mask_path) as mask_file:
        mask_l = mask_file.convert('L')
        gt = np.asarray(mask_l) > 127
        mask = mask_l.resize((input_size, input_size), Image.NEAREST)
    image = torch.from_numpy(
        np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mask = torch.from_numpy(
        np.asarray(mask, dtype=np.float32).copy()).unsqueeze(0)
    return image, (mask > 127.0).float(), gt


def normalize(image: torch.Tensor) -> torch.Tensor:
    return (image - _PIXEL_MEAN) / _PIXEL_STD


@torch.no_grad()
def extract_features(encoder, image: torch.Tensor, device):
    """Frozen ViT-L features: 4 intermediate maps + neck output, all 1/16."""
    output, features = encoder(
        normalize(image).unsqueeze(0).to(device), return_features=True)
    maps = [feature.permute(0, 3, 1, 2).contiguous() for feature in features]
    maps.append(output)
    return maps


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample, concatenate a skip, then double-conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SAMViTLUNetDecoder(nn.Module):
    """Trainable part of the SAM ViT-L + U-Net decoder."""

    def __init__(self):
        super().__init__()
        # 4 intermediate ViT features (1024 ch) + neck output (256 ch).
        self.proj = nn.ModuleList(
            [nn.Conv2d(1024, 64, 1) for _ in range(4)] +
            [nn.Conv2d(256, 64, 1)])
        self.proj_bn = nn.ModuleList(
            [nn.BatchNorm2d(64) for _ in range(5)])

        # High-resolution stem: 1/2, 1/4, 1/8, 1/16.
        self.stem_conv = nn.ModuleList([
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
        ])
        self.stem_bn = nn.ModuleList([
            nn.BatchNorm2d(32), nn.BatchNorm2d(64),
            nn.BatchNorm2d(64), nn.BatchNorm2d(64),
        ])

        self.bottleneck = DoubleConv(64, 64)
        self.up1 = Up(64, 64, 64)   # 1/16 -> 1/8
        self.up2 = Up(64, 64, 64)   # 1/8  -> 1/4
        self.up3 = Up(64, 32, 32)   # 1/4  -> 1/2
        self.up4 = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def run_stem(self, image):
        x = F.relu(self.stem_bn[0](self.stem_conv[0](image)), inplace=True)
        skip_12 = x
        x = F.relu(self.stem_bn[1](self.stem_conv[1](x)), inplace=True)
        skip_14 = x
        x = F.relu(self.stem_bn[2](self.stem_conv[2](x)), inplace=True)
        skip_18 = x
        x = F.relu(self.stem_bn[3](self.stem_conv[3](x)), inplace=True)
        return skip_12, skip_14, skip_18, x

    def forward(self, image, features):
        skip_12, skip_14, skip_18, stem_116 = self.run_stem(image)
        fused = 0
        for feature, projection, norm in zip(
                features, self.proj, self.proj_bn):
            fused = fused + F.relu(norm(projection(feature.float())),
                                   inplace=True)
        fused = fused + stem_116
        fused = self.bottleneck(fused)
        fused = self.up1(fused, skip_18)
        fused = self.up2(fused, skip_14)
        fused = self.up3(fused, skip_12)
        fused = F.interpolate(fused, scale_factor=2, mode='bilinear',
                              align_corners=False)
        return self.head(self.up4(fused))


def _metric_text(value, digits=4):
    return '—' if value != value else f'{value:.{digits}f}'


def _hd95_text(value):
    if value != value:
        return '—'
    value = abs(value)
    if value >= 1000:
        return f'{value:.0f}'
    if value >= 100:
        return f'{value:.1f}'
    return f'{value:.2f}'


def print_epoch_header(epochs, seed, device, num_params):
    console.print()
    console.print(
        f' [bold cyan]sam_vit_l_unet[/]  [dim]{num_params:,} params  '
        f'{epochs} epochs  seed {seed}  {device}[/]')
    console.print(
        LOG_SEP.join(f'{name:^{width}}'
                     for name, width in zip(LOG_COLS, LOG_WIDTHS)),
        style='bold cyan')
    console.print(LOG_RULE, style='dim')


def print_epoch_row(epoch, loss, metrics, lr, elapsed, improved):
    star = ' *' if improved else ''
    values = (
        f'E{epoch:03d}',
        f'{loss:.4f}',
        _metric_text(metrics['dice']),
        _metric_text(metrics['iou']),
        _metric_text(metrics['cldice']),
        _hd95_text(metrics['hd95']),
        f'{lr:.2e}',
        f'{elapsed:.0f}s{star}',
    )
    row = LOG_SEP.join(f'{value:^{width}}'
                       for value, width in zip(values, LOG_WIDTHS))
    console.print(row, style='bold green' if improved else None)


def print_epoch_footer():
    console.print(LOG_FOOTER, style='dim')


def _json_value(value):
    if value is None:
        return None
    value = float(value)
    return None if value != value else round(value, 6)


def _mean(values):
    if not values:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(values))


def write_history(path, config, history, best_epoch, best_dice):
    payload = {
        'config': config,
        'best_epoch': best_epoch,
        'best_val_dice': None if best_dice < 0 else round(float(best_dice), 6),
        'history': history,
    }
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        print(f'WARNING: failed to write {path}: {exc}')


def build_encoder(sam_checkpoint: Path, device):
    sam = build_sam_vit_l(checkpoint=str(sam_checkpoint), use_adapter=False)
    encoder = sam.image_encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    del sam.prompt_encoder, sam.mask_decoder
    return encoder


def cache_split(encoder, pairs, input_size, device):
    cache = []
    for image_path, mask_path in pairs:
        image, mask, gt = load_pair(image_path, mask_path, input_size)
        features = extract_features(encoder, image, device)
        cache.append({
            'features': [feature.half().cpu() for feature in features],
            'image': image.half().cpu(),
            'mask': mask.half().cpu(),
            'gt': gt,
            'stem': image_path.stem,
        })
    return cache


def augment(tensor, hflip, vflip, rot):
    if hflip:
        tensor = tensor.flip(-1)
    if vflip:
        tensor = tensor.flip(-2)
    if rot:
        tensor = torch.rot90(tensor, rot, [-2, -1])
    return tensor.contiguous()


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum()
    denominator = prob.sum() + target.sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def _batch(items, device):
    features = [torch.cat([item['features'][index] for item in items], dim=0)
                .to(device).float()
                for index in range(len(items[0]['features']))]
    images = torch.stack([item['image'] for item in items]).to(device).float()
    masks = torch.stack([item['mask'] for item in items]).to(device).float()
    return features, images, masks


@torch.no_grad()
def validate(model, cache, device):
    model.eval()
    values = {'dice': [], 'iou': [], 'cldice': [], 'hd95': []}
    for item in cache:
        features = [feature.to(device).float() for feature in item['features']]
        image = item['image'].to(device).float().unsqueeze(0)
        logits = model(image, features)
        gt = item['gt']
        logits = F.interpolate(logits, size=gt.shape, mode='bilinear',
                               align_corners=False)
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
        values['dice'].append(dice_score(pred, gt))
        values['iou'].append(iou_score(pred, gt))
        values['cldice'].append(cldice_score(pred, gt))
        values['hd95'].append(hd95_score(pred, gt))
    return {key: _mean(items) for key, items in values.items()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.input_size != 1024:
        raise SystemExit('SAM ViT-L expects --input-size 1024 '
                         '(the image encoder has a fixed 64x64 pos_embed).')
    if not args.sam_checkpoint.is_file():
        raise SystemExit(f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')

    device = torch.device(args.device)
    train_pairs = list_pairs(args.data_root / 'train')
    val_pairs = list_pairs(args.data_root / 'val')
    encoder = build_encoder(args.sam_checkpoint, device)

    split_manifest = args.data_root / 'split_manifest.csv'
    split_sha = (file_sha256(split_manifest)
                 if split_manifest.is_file() else None)
    run_config = {
        'method': 'sam_vit_l_unet',
        'data_root': str(args.data_root),
        'train_images': len(train_pairs),
        'val_images': len(val_pairs),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': args.input_size,
        'seed': args.seed,
        'optimizer': 'AdamW',
        'loss': 'BCE+Dice',
        'encoder': 'SAM ViT-L frozen',
        'sam_checkpoint': str(args.sam_checkpoint),
        'sam_checkpoint_sha256': file_sha256(args.sam_checkpoint),
        'split_manifest_sha256': split_sha,
    }
    run = setup_wandb(
        args, name=f'sam_vit_l_unet-seed{args.seed}', config=run_config)

    args.workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.workdir / 'best_model.pth'
    if ckpt_path.is_file():
        backup = ckpt_path.with_name(
            f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        shutil.copy2(ckpt_path, backup)
        print(f'backup previous checkpoint -> {backup}')
    history_path = args.workdir / 'metrics_history.json'
    history = []
    best_dice = -1.0
    best_epoch = 0

    print('[cache] extracting frozen SAM ViT-L features ...')
    cache_t0 = time.time()
    train_cache = cache_split(encoder, train_pairs, args.input_size, device)
    val_cache = cache_split(encoder, val_pairs, args.input_size, device)
    print(f'[cache] {len(train_cache) + len(val_cache)} images in '
          f'{time.time() - cache_t0:.1f}s -> {args.workdir}')

    model = SAMViTLUNetDecoder().to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    num_params = sum(p.numel() for p in model.parameters()
                     if p.requires_grad)
    print_epoch_header(args.epochs, args.seed, device, num_params)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        order = list(range(len(train_cache)))
        random.shuffle(order)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            items = []
            for index in indices:
                item = train_cache[index]
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                rot = random.randrange(4)
                items.append({
                    'features': [augment(feature, hflip, vflip, rot)
                                 for feature in item['features']],
                    'image': augment(item['image'], hflip, vflip, rot),
                    'mask': augment(item['mask'], hflip, vflip, rot),
                })
            features, images, masks = _batch(items, device)
            logits = model(images, features)
            loss = (F.binary_cross_entropy_with_logits(logits, masks) +
                    dice_loss(logits, masks))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(items)
        scheduler.step()

        metrics = validate(model, val_cache, device)
        elapsed = time.time() - epoch_t0
        improved = metrics['dice'] > best_dice
        if improved:
            best_dice = metrics['dice']
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_dice': metrics['dice'],
                'val_metrics': metrics,
                'input_size': args.input_size,
                'split_manifest_sha256': split_sha,
                'sam_checkpoint_sha256': run_config['sam_checkpoint_sha256'],
            }, ckpt_path)
        log_scalars(run, {
            'epoch': epoch,
            'train/loss': running / len(train_cache),
            'lr': optimizer.param_groups[0]['lr'],
            'val/dice': metrics['dice'],
            'val/iou': metrics['iou'],
            'val/cldice': metrics['cldice'],
            'val/hd95': metrics['hd95'],
        })
        print_epoch_row(epoch, running / len(train_cache), metrics,
                        optimizer.param_groups[0]['lr'], elapsed, improved)
        history.append({
            'epoch': epoch,
            'train_loss': round(running / len(train_cache), 6),
            'lr': optimizer.param_groups[0]['lr'],
            'is_best': improved,
            'overall': {key: _json_value(value)
                        for key, value in metrics.items()},
        })
        write_history(history_path, run_config, history, best_epoch,
                      best_dice)

    print_epoch_footer()
    console.print(
        f' [bold]Best[/]  E{best_epoch:03d}  '
        f'val_dice={best_dice:.4f}  -> {ckpt_path}')
    print(f'metrics history -> {history_path}')
    if run is not None and getattr(args, 'wandb_log_artifact', False):
        log_artifact(run, ckpt_path,
                     name=f'sam_vit_l_unet-seed{args.seed}',
                     artifact_type='model')
    finish_run(run)
    return ckpt_path


@torch.no_grad()
def predict(args, ckpt_path: Path):
    # Resolve and hash the checkpoints before the (potentially long)
    # prediction loop.  If another job moves/replaces a checkpoint while we
    # are predicting, the already-loaded model can still finish and the
    # metadata remains valid.
    ckpt_path = Path(ckpt_path).resolve()
    sam_checkpoint = Path(args.sam_checkpoint).resolve()
    if not sam_checkpoint.is_file():
        raise SystemExit(f'missing SAM ViT-L checkpoint: {sam_checkpoint}')
    if not ckpt_path.is_file():
        raise SystemExit(f'missing checkpoint: {ckpt_path}')
    checkpoint_sha256 = file_sha256(ckpt_path)
    sam_checkpoint_sha256 = file_sha256(sam_checkpoint)
    device = torch.device(args.device)
    encoder = build_encoder(sam_checkpoint, device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_size = payload.get('input_size', args.input_size)
    if input_size != 1024:
        raise SystemExit(f'checkpoint input_size={input_size}; '
                         'SAM ViT-L expects 1024')
    model = SAMViTLUNetDecoder().to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    datasets = []
    for spec in args.test_datasets:
        name, separator, root = spec.partition(':')
        if not separator or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        pairs = list_pairs(Path(root))
        datasets.append((name, Path(root), pairs))

    for name, _, pairs in datasets:
        out_dir = args.pred_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'{name}: predicting {len(pairs)} images on {device} '
              f'-> {out_dir}', flush=True)
        t0 = time.time()
        for done, (image_path, mask_path) in enumerate(pairs, 1):
            image, _, gt = load_pair(image_path, mask_path, input_size)
            features = extract_features(encoder, image, device)
            logits = model(image.to(device).unsqueeze(0), features)
            logits = F.interpolate(logits, size=gt.shape, mode='bilinear',
                                   align_corners=False)
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
        'method': 'sam_vit_l_unet',
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': checkpoint_sha256,
        'sam_checkpoint': str(sam_checkpoint),
        'sam_checkpoint_sha256': sam_checkpoint_sha256,
        'split_manifest_sha256': payload.get('split_manifest_sha256'),
        'datasets': [name for name, _, _ in datasets],
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
                        default=REPO_ROOT / 'workdirs' / 'baselines' /
                        'sam_vit_l_unet')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='trained decoder checkpoint used by predict '
                             '(default: <workdir>/best_model.pth)')
    parser.add_argument('--sam-checkpoint', type=Path,
                        default=REPO_ROOT / 'checkpoints' /
                        'sam_vit_l_0b3195.pth',
                        help='frozen SAM ViT-L image-encoder checkpoint')
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' /
                        'sam_vit_l_unet')
    parser.add_argument('--test-datasets', nargs='+', default=[],
                        help='NAME:GT_ROOT entries for prediction')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true',
                        help='log training scalars and the best checkpoint to W&B')
    parser.add_argument('--wandb_project', default='sacm')
    parser.add_argument('--wandb_entity',
                        default=os.environ.get('WANDB_ENTITY', 'buaazqk'))
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_group', default='baselines')
    parser.add_argument('--wandb_job_type', default='train')
    parser.add_argument('--wandb_tags', default='sam_vit_l_unet,3shot')
    parser.add_argument('--wandb_mode',
                        choices=['online', 'offline', 'disabled'],
                        default=None)
    parser.add_argument('--wandb_host',
                        default=os.environ.get(
                            'WANDB_BASE_URL', 'http://172.16.1.7:8080'))
    parser.add_argument('--wandb_api_key',
                        default=os.environ.get('WANDB_API_KEY'))
    parser.add_argument('--wandb_dir', default=None)
    parser.add_argument('--wandb_log_artifact', action='store_true',
                        default=os.environ.get(
                            'WANDB_LOG_ARTIFACT', '').lower()
                        in ('1', 'true', 'yes', 'on'),
                        help='upload the best checkpoint to W&B (default: off)')
    args = parser.parse_args()

    ckpt_path = args.checkpoint or (args.workdir / 'best_model.pth')
    if args.stage == 'train':
        train(args)
    elif args.stage == 'predict':
        if not ckpt_path.is_file():
            raise SystemExit(
                f'missing checkpoint: {ckpt_path}\n'
                'train first: python baselines/adapters/sam_vit_l_unet.py '
                '--stage train --epochs 200')
        predict(args, ckpt_path)
    else:
        train(args)
        if args.test_datasets:
            predict(args, ckpt_path)


if __name__ == '__main__':
    main()
