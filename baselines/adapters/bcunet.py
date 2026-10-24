#!/usr/bin/env python3
"""BCUNet baseline adapter for the frozen 3-shot protocol.

The vendored ``baselines/bcunet/code/core/models.py`` is used unchanged except
for the ``OutConv`` repair in ``core/unet_parts.py`` (upstream referenced an
undefined ``upconv2x2``).  This adapter supplies the project's data loading,
training loop, validation checkpoint selection, and prediction format.

Preprocessing note: upstream ``core/utils.get_data`` feeds raw BGR ``[0,255]``
to a ConvNeXt pretrained on ImageNet.  That mismatch makes the pretrained
weights unusable, so this adapter uses RGB ``[0,1]`` plus ImageNet
normalization, matching the backbone's pretraining.

Examples:
    python baselines/adapters/bcunet.py --stage train
    python baselines/adapters/bcunet.py --stage predict \
        --test-datasets DRIVE:data/sacm_3shot/test/DRIVE
"""

from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.console import console  # noqa: E402
from core.metrics import cldice_score, hd95_score  # noqa: E402
from core.run_status import RunStatus  # noqa: E402
from core.wandb_utils import (  # noqa: E402
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm')
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_bcunet_class():
    """Import ``BCUNet`` from the vendored upstream package.

    Upstream names its package ``core``, which collides with this project's
    ``src/core``.  Import it under a temporary path shadow, then restore the
    project modules so the framework imports above keep working.
    """
    code_dir = str(REPO_ROOT / 'baselines' / 'bcunet' / 'code')
    saved = {name: module for name, module in sys.modules.items()
             if name == 'core' or name.startswith('core.')}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, code_dir)
    try:
        from core.models import BCUNet  # noqa: PLC0415
    finally:
        sys.path.remove(code_dir)
        for name in [n for n in sys.modules
                     if n == 'core' or n.startswith('core.')]:
            del sys.modules[name]
        sys.modules.update(saved)
    return BCUNet


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
            (mask_dir / (image.stem + ext) for ext in IMAGE_EXTS
             if (mask_dir / (image.stem + ext)).is_file()),
            None)
        if mask is not None:
            pairs.append((image, mask))
    return pairs


class PairDataset(Dataset):
    def __init__(self, root: Path, input_size: int = 512,
                 augment: bool = False):
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
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        mask = torch.from_numpy(
            np.asarray(mask, dtype=np.float32).copy()).unsqueeze(0)
        mask = (mask > 127.0).long().squeeze(0)

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


class BinaryBCUNet(nn.Module):
    """Expose the fused BCUNet logits as a single output."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        _, _, fused = self.net(x)
        return fused


class FusedBCUNetLoss(nn.Module):
    """Upstream NLL loss on the two branch outputs and the fused output."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        out1, out2, fused = outputs
        return (self.ce(fused, target) + self.ce(out1, target) +
                self.ce(out2, target))


def _mean(values):
    if not values:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(values))


# Same fixed-width terminal table as the other baselines.
LOG_SEP = ' │ '
LOG_COLS = ('ep', 'tr_loss', 'Dice↑', 'IoU↑', 'clDice↑', 'HD95↓',
            'lr', 'time')
LOG_WIDTHS = (5, 8, 7, 7, 8, 7, 8, 8)
LOG_RULE = '─┼─'.join('─' * width for width in LOG_WIDTHS)
LOG_FOOTER = '─' * (sum(LOG_WIDTHS) + 3 * (len(LOG_WIDTHS) - 1))


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
        f' [bold cyan]bcunet[/]  [dim]{num_params:,} params  '
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


@torch.no_grad()
def validate(model, pairs, device, input_size):
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
        tensor = (tensor / 255.0)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = tensor.unsqueeze(0).to(device)
        prob = torch.softmax(model(tensor), dim=1)[:, 1:2]
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


def build_model(pretrained: bool) -> nn.Module:
    BCUNet = _load_bcunet_class()
    return BinaryBCUNet(
        BCUNet(n_channels=3, n_classes=2, bilinear=False,
               pretrained=pretrained, in_22k=True))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    train_set = PairDataset(args.data_root / 'train', args.input_size,
                            augment=True)
    val_set = PairDataset(args.data_root / 'val', args.input_size,
                          augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)

    model = build_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    criterion = FusedBCUNetLoss()
    split_manifest = args.data_root / 'split_manifest.csv'
    split_sha256 = (file_sha256(split_manifest)
                    if split_manifest.is_file() else None)

    run_config = {
        'method': 'bcunet',
        'data_root': str(args.data_root),
        'train_images': len(train_set),
        'val_images': len(val_set),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': args.input_size,
        'seed': args.seed,
        'pretrained': args.pretrained,
        'optimizer': 'AdamW',
        'loss': 'CE(fused)+CE(branch1)+CE(branch2)',
        'split_manifest_sha256': split_sha256,
    }
    run = setup_wandb(args, name=f'bcunet-seed{args.seed}', config=run_config)

    args.workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.workdir / 'best_model.pth'
    if ckpt_path.is_file():
        backup = ckpt_path.with_name(
            f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        shutil.copy2(ckpt_path, backup)
        print(f'backup previous checkpoint -> {backup}')
    best_dice = -1.0
    best_epoch = 0

    stamp = time.strftime('%Y%m%d_%H%M%S')
    history = []
    history_path = args.workdir / 'metrics_history.json'
    dataset_csv_path = args.workdir / 'val_metrics_by_dataset.csv'
    for artifact in (history_path, dataset_csv_path):
        if artifact.is_file():
            backup = artifact.with_name(
                f'{artifact.stem}.before_{stamp}{artifact.suffix}')
            shutil.copy2(artifact, backup)
            print(f'backup previous {artifact.name} -> {backup}')
    dataset_names = sorted({
        image_path.stem.split('_', 1)[0] for image_path, _ in val_set.pairs})
    metric_names = ('dice', 'iou', 'cldice', 'hd95')
    csv_fields = (['epoch', 'train_loss', 'lr', 'is_best'] +
                  [f'{dataset}_{metric}'
                   for dataset in dataset_names for metric in metric_names])
    with dataset_csv_path.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()
    write_history(history_path, run_config, history, best_epoch, best_dice)
    print(f'metrics history -> {history_path}')
    print(f'per-dataset metrics -> {dataset_csv_path}')
    num_params = sum(p.numel() for p in model.parameters())
    print_epoch_header(args.epochs, args.seed, device, num_params)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model.net(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()
        val_metrics, val_per_dataset = validate(
            model, val_set.pairs, device, args.input_size)
        elapsed = time.time() - epoch_t0
        val_dice = val_metrics['dice']
        train_loss = running / len(train_set)
        lr = optimizer.param_groups[0]['lr']
        wb = {'epoch': epoch, 'train/loss': train_loss, 'lr': lr}
        wb.update({f'val/{key}': value
                   for key, value in val_metrics.items()})
        for dataset, values in val_per_dataset.items():
            wb.update({f'val/{dataset}/{metric}': values[metric]
                       for metric in metric_names})
        log_scalars(run, wb)
        improved = val_dice > best_dice
        print_epoch_row(epoch, train_loss, val_metrics, lr, elapsed, improved)
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_dice': val_dice,
                'val_metrics': val_metrics,
                'val_per_dataset': val_per_dataset,
                'input_size': args.input_size,
                'num_classes': 2,
                'pretrained': args.pretrained,
                'split_manifest_sha256': split_sha256,
            }, ckpt_path)

        entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'lr': lr,
            'is_best': improved,
            'overall': {
                key: _json_value(value) for key, value in val_metrics.items()
            },
            'datasets': {
                dataset: {key: _json_value(value)
                          for key, value in values.items()}
                for dataset, values in val_per_dataset.items()
            },
        }
        history.append(entry)
        csv_row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'lr': f'{lr:.8g}',
            'is_best': int(improved),
        }
        for dataset in dataset_names:
            values = val_per_dataset.get(dataset, {})
            for metric in metric_names:
                value = values.get(metric)
                csv_row[f'{dataset}_{metric}'] = (
                    '' if value is None else f'{float(value):.6f}')
        try:
            with dataset_csv_path.open('a', newline='') as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(csv_row)
        except OSError as exc:
            print(f'WARNING: failed to append {dataset_csv_path}: {exc}')
        write_history(history_path, run_config, history,
                      best_epoch, best_dice)

    print_epoch_footer()
    console.print(
        f' [bold]Best[/]  E{best_epoch:03d}  val_dice={best_dice:.4f}  '
        f'-> {ckpt_path}')
    print(f'metrics history -> {history_path}')
    print(f'per-dataset metrics -> {dataset_csv_path}')
    if run is not None:
        if args.wandb_log_artifact:
            log_artifact(run, ckpt_path, name=f'bcunet-seed{args.seed}',
                         artifact_type='model')
        else:
            print('W&B artifact upload skipped '
                  '(set WANDB_LOG_ARTIFACT=true to enable)')
    finish_run(run)
    return ckpt_path


@torch.no_grad()
def predict(args, ckpt_path: Path):
    device = torch.device(args.device)
    payload = torch.load(ckpt_path, map_location=device)
    input_size = payload.get('input_size', args.input_size)
    model = build_model(payload.get('pretrained', args.pretrained))
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()

    datasets = []
    for spec in args.test_datasets:
        name, sep, root = spec.partition(':')
        if not sep or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        root = Path(root)
        pairs = list_pairs(root)
        if not pairs:
            raise SystemExit(f'no image/mask pairs under {root}')
        datasets.append((name, root, pairs))
    total_images = sum(len(pairs) for _, _, pairs in datasets)
    status = RunStatus(method='bcunet', stage='predict', heartbeat=10.0)
    status.update(datasets_total=len(datasets), overall_total=total_images)

    try:
        for dataset_index, (name, root, pairs) in enumerate(datasets, 1):
            out_dir = args.pred_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            total = len(pairs)
            status.start_dataset(name, index=dataset_index, total=total,
                                 overall_total=total_images)
            print(f'{name}: predicting {total} images on {device} -> {out_dir}',
                  flush=True)
            t0 = time.time()
            for done, (image_path, mask_path) in enumerate(pairs, 1):
                image = Image.open(image_path).convert('RGB').resize(
                    (input_size, input_size), Image.BICUBIC)
                tensor = torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
                tensor = (tensor / 255.0)
                tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
                tensor = tensor.unsqueeze(0).to(device)
                prob = torch.softmax(model(tensor), dim=1)[:, 1:2]
                with Image.open(mask_path) as mask_image:
                    width, height = mask_image.size
                prob = F.interpolate(prob, size=(height, width),
                                     mode='bilinear', align_corners=False)
                pred = (prob.squeeze().cpu().numpy() > 0.5).astype(
                    np.uint8) * 255
                Image.fromarray(pred, mode='L').save(
                    out_dir / f'{image_path.stem}.png')
                status.advance(1, write=(done % 10 == 0))
                if done % 50 == 0 and done != total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate > 0 else 0.0
                    print(f'  {name}: {done}/{total}  {rate:.1f} img/s  '
                          f'ETA {eta:.0f}s', flush=True)
            elapsed = time.time() - t0
            status.finish_dataset()
            print(f'{name}: done {total} predictions in {elapsed:.0f}s '
                  f'-> {out_dir}', flush=True)

        meta = {
            'method': 'bcunet',
            'checkpoint': str(ckpt_path),
            'checkpoint_sha256': file_sha256(ckpt_path),
            'split_manifest_sha256': payload.get('split_manifest_sha256'),
            'datasets': [name for name, _, _ in datasets],
        }
        with open(args.pred_root / '.prediction_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
            f.write('\n')
        status.finish('predictions complete')
    except BaseException as exc:
        status.fail(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'predict', 'all'],
                        default='all')
    parser.add_argument('--data-root', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot')
    parser.add_argument('--workdir', type=Path,
                        default=REPO_ROOT / 'workdirs' / 'baselines' /
                        'bcunet')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' /
                        'bcunet')
    parser.add_argument('--test-datasets', nargs='+', default=[],
                        help='NAME:GT_ROOT entries for prediction')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch 2 OOMs on an 11 GiB card at 512x512')
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pretrained', dest='pretrained',
                        action='store_false',
                        help='do not load ImageNet-22k ConvNeXt weights')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='sacm')
    parser.add_argument('--wandb_entity',
                        default=os.environ.get('WANDB_ENTITY', 'buaazqk'))
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_group', default='baselines')
    parser.add_argument('--wandb_job_type', default='train')
    parser.add_argument('--wandb_tags', default='bcunet,3shot')
    parser.add_argument('--wandb_mode',
                        choices=['online', 'offline', 'disabled'],
                        default=None)
    parser.add_argument('--wandb_host',
                        default=os.environ.get(
                            'WANDB_BASE_URL', 'http://172.16.1.7:8080'))
    parser.add_argument('--wandb_api_key',
                        default=os.environ.get('WANDB_API_KEY'))
    parser.add_argument('--wandb_dir', default=None)
    parser.add_argument(
        '--wandb_log_artifact', action='store_true',
        default=os.environ.get('WANDB_LOG_ARTIFACT', '').lower()
        in ('1', 'true', 'yes', 'on'))
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.stage in ('predict', 'all'):
        if not args.test_datasets:
            if args.stage == 'predict':
                raise SystemExit(
                    '--test-datasets is required for --stage predict')
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
