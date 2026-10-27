#!/usr/bin/env python
"""Train the Deep Closing MIM shape prior from protocol masks only.

The shape prior is trained on the 18 training masks of a split root.  It
never sees validation or test masks.  The resulting checkpoint is consumed
by ``evaluate.py --deep_closing_checkpoint`` or by the standalone
``apply_deep_closing.py`` utility.

Example:
    PYTHONPATH=src python scripts/sacm_x/train_deep_closing.py \
        --data_root data/sacm_3shot_seed0 \
        --output results/deep_closing/seed0_shape_prior.pth \
        --epochs 300 --resolution 512
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from core.postprocess.deep_closing import (  # noqa: E402
    SimplePointErosion,
    UNetMIM,
    mim_loss,
    random_block_mask,
    recurrent_tube_scale,
    save_checkpoint,
)

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


class MaskDataset(Dataset):
    """Masks from ``<root>/<split>/masks``; images are not loaded."""

    def __init__(self, root: Path, split: str, resolution: int,
                 limit: int | None = None):
        mask_dir = root / split / 'masks'
        if not mask_dir.is_dir():
            raise FileNotFoundError(f'missing mask directory: {mask_dir}')
        files = sorted(
            path for path in mask_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit is not None:
            files = files[:max(0, int(limit))]
        if not files:
            raise ValueError(f'no masks found in {mask_dir}')
        self.files = files
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            array = np.asarray(image.convert('L')) > 127
        resized = Image.fromarray(
            array.astype(np.uint8) * 255,
        ).resize((self.resolution, self.resolution), Image.NEAREST)
        return torch.from_numpy(
            np.asarray(resized, dtype=np.float32).copy()
        ).unsqueeze(0) / 255.0


def augment_mask(mask: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Geometric augmentation for the shape prior (no image needed)."""
    if rng.random() < 0.5:
        mask = torch.flip(mask, dims=[-1])
    if rng.random() < 0.5:
        mask = torch.flip(mask, dims=[-2])
    rotations = rng.randrange(4)
    if rotations:
        mask = torch.rot90(mask, k=rotations, dims=[-2, -1])
    return mask.contiguous()


def make_loss(model: UNetMIM, masks: torch.Tensor,
              spe: SimplePointErosion, alpha: float,
              rng: random.Random) -> torch.Tensor:
    scaled = recurrent_tube_scale(masks, spe, rng, max_k=2)
    masked = random_block_mask(scaled, rng)
    prediction = model(scaled * (1.0 - masked))
    return mim_loss(
        prediction, scaled, masked, spe, topo_weight=alpha)


def evaluate(model: UNetMIM, loader: DataLoader,
             spe: SimplePointErosion, alpha: float,
             device: torch.device, seed: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    rng = random.Random(seed)
    with torch.no_grad():
        for masks in loader:
            masks = masks.to(device, non_blocking=True)
            loss = make_loss(model, masks, spe, alpha, rng)
            total += float(loss.item()) * masks.shape[0]
            count += masks.shape[0]
    return total / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the Deep Closing MIM shape prior on train masks')
    parser.add_argument('--data_root', required=True,
                        help='Protocol root with train/val splits')
    parser.add_argument('--output', required=True,
                        help='Output shape-prior checkpoint (.pth)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--resolution', type=int, default=512,
                        help='Square working resolution (default: 512)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Topology-loss weight in Eq. 4')
    parser.add_argument('--channels', default='16,32,64,128,256',
                        help='Comma-separated U-Net channels')
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--limit_train', type=int, default=None)
    parser.add_argument('--limit_val', type=int, default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None,
                        help='cuda/cpu; default = cuda when available')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    if args.epochs <= 0 or args.resolution <= 0:
        raise SystemExit('epochs and resolution must be positive')
    channels = tuple(
        int(value) for value in args.channels.split(',') if value.strip())
    if len(channels) < 2:
        raise SystemExit('--channels needs at least two levels')

    root = Path(args.data_root)
    train_set = MaskDataset(
        root, 'train', args.resolution, limit=args.limit_train)
    try:
        val_set = MaskDataset(
            root, 'val', args.resolution, limit=args.limit_val)
    except (FileNotFoundError, ValueError) as exc:
        logging.warning(
            'No validation masks (%s); checkpoint selection uses the last '
            'epoch. Do not use test masks here.', exc)
        val_set = None

    device = torch.device(
        args.device if args.device else
        ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = UNetMIM(
        in_channels=1, out_channels=1, channels=channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4)
    spe = SimplePointErosion(device)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=False)
    val_loader = (
        DataLoader(val_set, batch_size=1, shuffle=False,
                   num_workers=args.workers, drop_last=False)
        if val_set is not None else None)

    config = {
        'model': {
            'in_channels': 1,
            'out_channels': 1,
            'channels': list(channels),
        },
        'resolution': int(args.resolution),
        'dilation_threshold': 0.2,
        'max_iters': 1000,
        'alpha': float(args.alpha),
        'patch_range': [2, 8],
        'mask_ratio_range': [0.1, 0.2],
        'train_root': str(root),
        'train_masks': len(train_set),
        'val_masks': len(val_set) if val_set is not None else 0,
        'seed': int(args.seed),
        'source': 'DeepClosing (TMI 2024), MIM shape prior',
    }
    output = Path(args.output)
    logging.info(
        'Deep Closing shape prior: train=%d val=%d resolution=%d device=%s',
        len(train_set), len(val_set) if val_set is not None else 0,
        args.resolution, device)
    logging.info('Channels: %s; output=%s', list(channels), output)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch)
        running = 0.0
        seen = 0
        for masks in train_loader:
            masks = masks.to(device, non_blocking=True)
            masks = augment_mask(masks, rng)
            loss = make_loss(model, masks, spe, args.alpha, rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().item()) * masks.shape[0]
            seen += masks.shape[0]
        scheduler.step()
        train_loss = running / max(seen, 1)

        if epoch == 1 or epoch % max(args.val_interval, 1) == 0:
            if val_loader is not None:
                val_loss = evaluate(
                    model, val_loader, spe, args.alpha, device,
                    args.seed + 10_000 + epoch)
            else:
                val_loss = train_loss
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    output, model, config, epoch, val_loss)
            logging.info(
                'E%03d train=%.6f val=%.6f best=%.6f',
                epoch, train_loss, val_loss, best_loss)
        else:
            logging.info('E%03d train=%.6f', epoch, train_loss)

    if not output.exists():
        save_checkpoint(output, model, config, args.epochs, best_loss)
    logging.info('Wrote %s', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
