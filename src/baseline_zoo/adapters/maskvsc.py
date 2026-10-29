"""MaskVSC baseline: masked-vessel segmentation plus ViG completion.

Official reference: https://github.com/Zhouyi-Zura/MaskVSC.
Stage 1 follows ``train.py`` (UNet, masked-vessel corruption, BCE +
Connect_Loss).  Stage 2 follows ``train_ViG.py``: the frozen stage-1 model
produces probability maps and the ViG completion head is trained on them.
The official inference path is ``ViG(Seg(image))``.
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from baseline_zoo.data import SampleListDataset, load_rgb, load_split_samples
from baseline_zoo.models.maskvsc import (
    Connect_Loss,
    MaskVSCFull,
    UNet,
    ViGSeg_Fundus,
    initialize_weights,
    mask_vessel_corrupt,
)
from baseline_zoo.runtime import (
    BaselineAdapter,
    evaluate,
    file_sha256,
    print_epoch_footer,
    print_epoch_header,
    print_epoch_row,
    seed_everything,
    write_history,
)
from core.console import console
from core.wandb_utils import finish_run, log_scalars, setup_wandb


def mask_ratio(epoch: int, epochs: int) -> float:
    """Upstream epoch-0-based schedule, mapped to our 1-based epoch."""
    ratio = (epoch - 1) / max(epochs, 1)
    if ratio < 0.1 or ratio > 0.9:
        return 0.0
    if ratio <= 0.5:
        return 0.4 * (ratio - 0.1) / 0.4
    return 0.4 * (0.9 - ratio) / 0.4


def _load_pair(sample):
    image = load_rgb(sample.image_path)
    with Image.open(sample.mask_path) as mask:
        label = mask.convert('L').copy()
    return image, label


class MaskVSCAdapter(BaselineAdapter):
    id = 'maskvsc'
    display_name = 'MaskVSC'
    official_source = 'Zhouyi-Zura/MaskVSC (self-contained copy)'
    official_entrypoint = 'https://github.com/Zhouyi-Zura/MaskVSC'
    input_size = 512
    official_lr = 2e-4
    official_weight_decay = 0.0
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'Full two-stage MaskVSC: stage-1 UNet (masked-vessel corruption + '
        'Connect_Loss) followed by the official ViG completion head '
        '(ViG(Seg(image))), matching the upstream train_ViG.py/test_ViG.py '
        'inference path. The previous stage-1-only port is superseded.',
        'The released ViG code cannot run as shipped: it reshapes the token '
        'vector instead of the (img/patch)^2 token grid, and its Fundus call '
        'passes patch_size=16 while the hardcoded predictor expects 1024 '
        '(32^2) channels. The port repairs the token-grid reshape and uses '
        'patch_size=32 for the 512 input, preserving the predictor width.',
        'Stage 2 is a single pooled ViG across all protocol domains. Upstream '
        'trains separate Fundus/OCTA/2PFM completion models per dataset; '
        'unseen/novel domains have no per-domain training data, so a pooled '
        'model is the only protocol-consistent choice.',
        'Stage 2 trains on clean images and the frozen stage-1 probability '
        'maps from the 18 train images, with BCE only. Stage 1 is forced to '
        'eval() and requires_grad=False; the release does not call eval() '
        'and would update its BatchNorm statistics and accumulate unused '
        'gradients.',
        'Stage-2 checkpoint selection uses the 6-image validation split by '
        'Dice. Upstream evaluates the test split every epoch and selects the '
        'best ViG by test Dice, which violates the protocol.',
        'The shared 800-epoch budget is split 600 stage-1 + 200 stage-2 by '
        'default; the masked-vessel corruption ramp spans those 600 stage-1 '
        'epochs. Upstream uses 200 + 200; use --maskvsc-stage1-epochs 800 '
        'for the ViG-isolation A/B (1000 total).',
        'Data protocol: the 18/6-shot split replaces the original Train/Test '
        'folders; grayscale domains are converted to RGB because the official '
        'Fundus UNet expects three channels. Masks are binarized after resize '
        'for both stages; the released GCN dataset passes soft PIL-resized '
        'labels to BCE.',
        'Official test.py writes 512x512 masks; predictions are restored to '
        'the original GT resolution so every baseline shares one evaluator.',
        'The protocol seeds 0-4 replace upstream seed 1234 and deterministic '
        'cuDNN. The official corruption sampler consumes RNG when ratio=0; '
        'this port early-returns before that draw and uses workers=0.',
        'The custom two-stage pipeline does not support --resume, --accum-steps '
        'or gradient clipping; a rerun retrains from scratch. Upstream stage-1 '
        'evaluates the test split every epoch; this port uses validation Dice.',
        'Checkpoints from the previous stage-1-only MaskVSC port contain no '
        'ViG weights and must be retrained for the full two-stage baseline.',
    )

    def __init__(self):
        super().__init__()
        self._vig_blocks = 16
        self._vig_edges = 9
        self._vig_heads = 1
        self._vig_patch_size = 32
        self._vig_batch_size = None
        self._vig_lr = None
        self._stage1_epochs = None
        self._stage2_epochs = 200
        self._stage1_epochs_used = 0
        self._stage2_epochs_used = 0

    def add_args(self, parser):
        parser.add_argument(
            '--maskvsc-stage1-epochs', type=int, default=None,
            help='stage-1 epochs; default = --epochs - stage-2 epochs')
        parser.add_argument(
            '--maskvsc-stage2-epochs', type=int, default=200,
            help='ViG completion epochs (official 200)')
        parser.add_argument(
            '--maskvsc-vig-blocks', type=int, default=16,
            help='number of ViG blocks (official 16)')
        parser.add_argument(
            '--maskvsc-vig-edges', type=int, default=9,
            help='ViG graph edges (official 9)')
        parser.add_argument(
            '--maskvsc-vig-heads', type=int, default=1,
            help='ViG multi-head width (official 1)')
        parser.add_argument(
            '--maskvsc-vig-patch-size', type=int, default=32,
            help='ViG patch size; 32 matches the released Fundus predictor')
        parser.add_argument(
            '--maskvsc-vig-batch-size', type=int, default=None,
            help='stage-2 batch size; default = --batch-size')
        parser.add_argument(
            '--maskvsc-vig-lr', type=float, default=None,
            help='stage-2 Adam LR; default = --lr')

    def _resolve_stage_epochs(self, args) -> tuple[int, int]:
        stage2 = int(getattr(args, 'maskvsc_stage2_epochs', 200))
        raw = getattr(args, 'maskvsc_stage1_epochs', None)
        stage1 = (int(args.epochs) - stage2) if raw is None else int(raw)
        if stage1 < 1 or stage2 < 1:
            raise SystemExit(
                'MaskVSC needs both stages positive; got '
                f'stage1={stage1}, stage2={stage2}')
        if stage1 + stage2 != int(args.epochs):
            print(
                f'[maskvsc] stage budget {stage1}+{stage2}='
                f'{stage1 + stage2}, requested --epochs={args.epochs}',
                flush=True)
        return stage1, stage2

    def _apply_arch(self, args):
        self._vig_blocks = int(getattr(args, 'maskvsc_vig_blocks', 16))
        self._vig_edges = int(getattr(args, 'maskvsc_vig_edges', 9))
        self._vig_heads = int(getattr(args, 'maskvsc_vig_heads', 1))
        self._vig_patch_size = int(
            getattr(args, 'maskvsc_vig_patch_size', 32))
        self._vig_batch_size = getattr(args, 'maskvsc_vig_batch_size', None)
        self._vig_lr = getattr(args, 'maskvsc_vig_lr', None)
        if self.input_size % self._vig_patch_size:
            raise SystemExit(
                '--maskvsc-vig-patch-size must divide the 512 input')
        self._stage1_epochs = getattr(args, 'maskvsc_stage1_epochs', None)
        self._stage2_epochs = int(
            getattr(args, 'maskvsc_stage2_epochs', 200))

    def build_model(self, args) -> nn.Module:
        self._apply_arch(args)
        seg = UNet(n_channels=3, n_classes=1)
        initialize_weights(seg)
        vig = ViGSeg_Fundus(
            num_channels=1,
            img_size=self.input_size,
            patch_size=self._vig_patch_size,
            num_ViGBlocks=self._vig_blocks,
            num_edges=self._vig_edges,
            head_num=self._vig_heads,
        )
        initialize_weights(vig)
        model = MaskVSCFull(seg, vig)
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict')
        model.set_stage('stage2' if predicting else 'stage1')
        return model

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.seg.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.5, 0.999),
            weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return None

    def prepare_train_batch(self, samples, rng: random.Random):
        if self.args is None:
            raise RuntimeError('MaskVSC requires configured args before training')
        epochs = self._stage1_epochs_used or int(self.args.epochs)
        ratio = mask_ratio(getattr(self, '_epoch', 1), epochs)
        images, labels = [], []
        for sample in samples:
            image, label = _load_pair(sample)
            if rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            image_np = np.array(image.resize([self.input_size,
                                              self.input_size]))
            label_np = np.array(label.resize([self.input_size,
                                              self.input_size]))
            if ratio > 0:
                image_np = mask_vessel_corrupt(image_np, label_np, ratio)

            if label_np.max() == 255:
                label_np = (label_np >= 128).astype(np.uint8) * 255
            elif label_np.max() == 1:
                label_np = (label_np >= 0.5).astype(np.uint8)

            image_t = torch.from_numpy(
                image_np.astype(np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            label_t = torch.from_numpy(
                label_np.astype(np.float32).copy()
            ).unsqueeze(0) / 255.0
            images.append(image_t)
            labels.append(label_t)
        return (torch.stack(images).to(self.device),
                torch.stack(labels).to(self.device))

    def _prepare_clean_batch(self, samples, device):
        images, labels = [], []
        for sample in samples:
            image, label = _load_pair(sample)
            image_np = np.array(image.resize([self.input_size,
                                              self.input_size]))
            label_np = np.array(label.resize([self.input_size,
                                              self.input_size]))
            label_np = (label_np >= 0.5).astype(np.float32) \
                if label_np.max() <= 1 else \
                (label_np >= 128).astype(np.float32)
            image_t = torch.from_numpy(
                image_np.astype(np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            label_t = torch.from_numpy(label_np).unsqueeze(0)
            images.append(image_t)
            labels.append(label_t)
        return (torch.stack(images).to(device),
                torch.stack(labels).to(device))

    def train_step(self, model, batch):
        images, labels = batch
        pred = model(images)
        loss = nn.BCELoss()(pred, labels)
        if getattr(model, 'stage', 'stage1') == 'stage1':
            loss = loss + Connect_Loss()(pred, labels)
        return loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size))
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        with torch.no_grad():
            pred = model(prepared)[0, 0].cpu().numpy()
        value_range = pred.max() - pred.min()
        pred = (pred - pred.min()) / (value_range + 1e-10)
        binary = pred >= 0.5
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        binary = Image.fromarray((binary.astype(np.uint8) * 255)).resize(
            (width, height), Image.NEAREST)
        return np.asarray(binary) > 127

    # ---- custom two-stage training -----------------------------------
    def _train_stage1_epoch(self, model, loader, optimizer, args):
        model.set_stage('stage1')
        model.train()
        rng = random.Random(args.seed + int(self._epoch))
        running = 0.0
        seen = 0
        for samples in loader:
            images, labels = self.prepare_train_batch(samples, rng)
            pred = model(images)
            loss = (nn.BCELoss()(pred, labels)
                    + Connect_Loss()(pred, labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        return running / max(seen, 1)

    def _train_stage2_epoch(self, model, loader, optimizer, args):
        model.set_stage('stage2')
        model.train()
        running = 0.0
        seen = 0
        for samples in loader:
            images, labels = self._prepare_clean_batch(samples, self.device)
            pred = model(images)
            loss = nn.BCELoss()(pred, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        return running / max(seen, 1)

    @staticmethod
    def _json_value(value):
        if value is None:
            return None
        value = float(value)
        return None if math.isnan(value) else round(value, 6)

    def _history_row(self, epoch, stage, train_loss, lr, improved,
                     metrics, per_dataset):
        return {
            'epoch': int(epoch),
            'stage': stage,
            'train_loss': round(float(train_loss), 6),
            'lr': round(float(lr), 8),
            'is_best': bool(improved),
            'selection': self._json_value(
                metrics.get(self.args.selection)),
            'overall': {
                key: self._json_value(metrics.get(key))
                for key in ('dice', 'iou', 'cldice', 'hd95',
                            'empty_prediction_rate')},
            'datasets': {
                name: {
                    key: self._json_value(values.get(key))
                    for key in ('dice', 'iou', 'cldice', 'hd95')}
                for name, values in per_dataset.items()},
        }

    @staticmethod
    def _write_dataset_csv(path, history):
        fields = ['epoch', 'stage', 'dataset', 'dice', 'iou', 'cldice',
                  'hd95']
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in history:
                for dataset, values in row['datasets'].items():
                    writer.writerow({
                        'epoch': row['epoch'],
                        'stage': row['stage'],
                        'dataset': dataset,
                        **{key: values.get(key)
                           for key in ('dice', 'iou', 'cldice', 'hd95')},
                    })

    def _save_checkpoint(self, path, model, args, run_config, split_sha,
                         epoch, score, metrics, per_dataset):
        payload = {
            'adapter': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'scheduler_state_dict': None,
            'epoch': int(epoch),
            'best_score': float(score),
            'selection': args.selection,
            'val_dice': metrics['dice'],
            'val_metrics': metrics,
            'val_per_dataset': per_dataset,
            'input_size': self.input_size,
            'split_manifest_sha256': split_sha,
            'config': run_config,
        }
        tmp = path.with_name(path.name + '.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def run_training(self, args):
        if args.resume:
            raise SystemExit(
                'MaskVSC does not support intra-run --resume; the custom '
                'two-stage pipeline retrains from scratch')
        unsupported = []
        if int(args.accum_steps) != 1:
            unsupported.append('--accum-steps')
        if float(args.grad_clip) != 0:
            unsupported.append('--grad-clip')
        if getattr(args, 'scheduler', 'official') not in ('official', None):
            unsupported.append('--scheduler (no scheduler in MaskVSC)')
        if getattr(args, 'warmup_epochs', None) is not None:
            unsupported.append('--warmup-epochs (no warmup in MaskVSC)')
        if unsupported:
            raise SystemExit(
                'MaskVSC custom pipeline does not implement: '
                + ', '.join(unsupported))
        seed_everything(args.seed)
        device = torch.device(args.device)
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device.index)
        self.args = args
        self.device = device
        self._apply_arch(args)
        self._stage1_epochs_used, self._stage2_epochs_used = (
            self._resolve_stage_epochs(args))

        train_samples = load_split_samples(Path(args.data_root), 'train')
        val_samples = load_split_samples(Path(args.data_root), 'val')
        if args.limit_train:
            train_samples = train_samples[:args.limit_train]
        if args.limit_val:
            val_samples = val_samples[:args.limit_val]
        if not train_samples or not val_samples:
            raise SystemExit('MaskVSC needs non-empty train and val splits')

        model = self.build_model(args).to(device)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        ckpt_path = workdir / 'best_model.pth'
        if ckpt_path.is_file():
            backup = ckpt_path.with_name(
                f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
            shutil.copy2(ckpt_path, backup)
            print(f'backup previous checkpoint -> {backup}', flush=True)

        split_manifest = Path(args.data_root) / 'split_manifest.csv'
        split_sha = (file_sha256(split_manifest)
                     if split_manifest.is_file() else None)
        vig_batch = int(self._vig_batch_size or args.batch_size)
        vig_lr = float(self._vig_lr if self._vig_lr is not None else args.lr)
        num_params = sum(p.numel() for p in model.parameters())
        trainable_stage1 = sum(
            p.numel() for p in model.seg.parameters() if p.requires_grad)
        trainable_stage2 = sum(p.numel() for p in model.vig.parameters())
        run_config = {
            'method': self.id,
            'display_name': self.display_name,
            'official_source': self.official_source,
            'official_entrypoint': self.official_entrypoint,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'train_images': len(train_samples),
            'val_images': len(val_samples),
            'epochs': self._stage1_epochs_used + self._stage2_epochs_used,
            'requested_epochs': int(args.epochs),
            'stage1_epochs': self._stage1_epochs_used,
            'stage2_epochs': self._stage2_epochs_used,
            'batch_size': int(args.batch_size),
            'vig_batch_size': vig_batch,
            'lr': float(args.lr),
            'vig_lr': vig_lr,
            'weight_decay': float(args.weight_decay),
            'input_size': self.input_size,
            'adapter_config': self.save_config(args),
            'selection': args.selection,
            'seed': int(args.seed),
            'optimizer': 'Adam(betas=(0.5,0.999))',
            'scheduler': None,
            'stage1_frozen_in_stage2': True,
            'vig_arch': {
                'blocks': self._vig_blocks,
                'edges': self._vig_edges,
                'heads': self._vig_heads,
                'patch_size': self._vig_patch_size,
            },
            'split_manifest_sha256': split_sha,
            'num_params': num_params,
            'trainable_params_stage1': trainable_stage1,
            'trainable_params_stage2': trainable_stage2,
        }
        run = setup_wandb(args, name=f'{self.id}-seed{args.seed}',
                          config=run_config)
        for deviation in self.deviations:
            console.print(f' [yellow]deviation:[/] {deviation}')
        console.print(
            f'[cyan]MaskVSC stage 1[/] {self._stage1_epochs_used} epochs '
            f'(UNet); [cyan]stage 2[/] {self._stage2_epochs_used} epochs '
            f'(ViG, batch {vig_batch})')
        print_epoch_header(
            self, self._stage1_epochs_used + self._stage2_epochs_used,
            args.seed, device, num_params, trainable_stage1)

        train_loader = DataLoader(
            SampleListDataset(train_samples), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=lambda batch: batch)
        vig_loader = DataLoader(
            SampleListDataset(train_samples), batch_size=vig_batch,
            shuffle=True, num_workers=args.num_workers,
            collate_fn=lambda batch: batch)

        history = []
        best1 = -float('inf')
        best1_state = None
        best1_epoch = 0
        stage1_optimizer = torch.optim.Adam(
            [p for p in model.seg.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.5, 0.999),
            weight_decay=args.weight_decay)
        if not stage1_optimizer.param_groups[0]['params']:
            raise RuntimeError('MaskVSC stage 1 has no trainable parameters')

        try:
            for epoch in range(1, self._stage1_epochs_used + 1):
                started = time.time()
                self._epoch = epoch
                train_loss = self._train_stage1_epoch(
                    model, train_loader, stage1_optimizer, args)
                model.set_stage('stage1')
                metrics, per_dataset = evaluate(
                    self, model, val_samples, device)
                score = self.selection_score(metrics, per_dataset)
                improved = score > best1
                if improved:
                    best1 = score
                    best1_epoch = epoch
                    best1_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.seg.state_dict().items()}
                elapsed = time.time() - started
                print_epoch_row(
                    epoch, train_loss, metrics, args.lr, elapsed, improved)
                history.append(self._history_row(
                    epoch, 'stage1', train_loss, args.lr, improved,
                    metrics, per_dataset))
                if run is not None:
                    scalars = {
                        'epoch': epoch,
                        'stage': 1,
                        'train/loss': train_loss,
                        'lr': args.lr,
                        'val/selection': score,
                    }
                    scalars.update(
                        {f'val/{key}': value
                         for key, value in metrics.items()})
                    log_scalars(run, scalars)

            if best1_state is None:
                best1_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.seg.state_dict().items()}
                best1_epoch = self._stage1_epochs_used
                best1 = 0.0
            model.seg.load_state_dict(best1_state)
            run_config['stage1_best_epoch'] = best1_epoch
            run_config['stage1_best_score'] = self._json_value(best1)
            console.print(
                f'[cyan]MaskVSC stage 1 best[/] epoch {best1_epoch} '
                f'-> load frozen stage-1 weights; starting ViG')

            # Switch to stage 2 before collecting trainable ViG parameters;
            # stage 1 leaves them frozen and would yield an empty optimizer.
            model.set_stage('stage2')
            stage2_optimizer = torch.optim.Adam(
                [p for p in model.vig.parameters() if p.requires_grad],
                lr=vig_lr, betas=(0.5, 0.999),
                weight_decay=args.weight_decay)
            if not stage2_optimizer.param_groups[0]['params']:
                raise RuntimeError(
                    'MaskVSC stage 2 has no trainable ViG parameters')
            best2 = -float('inf')
            best2_epoch = 0
            for offset in range(1, self._stage2_epochs_used + 1):
                epoch = self._stage1_epochs_used + offset
                started = time.time()
                self._epoch = epoch
                train_loss = self._train_stage2_epoch(
                    model, vig_loader, stage2_optimizer, args)
                model.set_stage('stage2')
                metrics, per_dataset = evaluate(
                    self, model, val_samples, device)
                score = self.selection_score(metrics, per_dataset)
                improved = score > best2
                if improved:
                    best2 = score
                    best2_epoch = epoch
                    run_config['stage2_best_epoch'] = best2_epoch
                    run_config['stage2_best_score'] = self._json_value(best2)
                    self._save_checkpoint(
                        ckpt_path, model, args, run_config, split_sha,
                        epoch, best2, metrics, per_dataset)
                elapsed = time.time() - started
                print_epoch_row(
                    epoch, train_loss, metrics, vig_lr, elapsed, improved)
                history.append(self._history_row(
                    epoch, 'stage2', train_loss, vig_lr, improved,
                    metrics, per_dataset))
                if run is not None:
                    scalars = {
                        'epoch': epoch,
                        'stage': 2,
                        'train/loss': train_loss,
                        'lr': vig_lr,
                        'val/selection': score,
                    }
                    scalars.update(
                        {f'val/{key}': value
                         for key, value in metrics.items()})
                    log_scalars(run, scalars)

            if best2_epoch == 0:
                best2 = 0.0
                self._save_checkpoint(
                    ckpt_path, model, args, run_config, split_sha,
                    self._stage1_epochs_used + self._stage2_epochs_used,
                    best2, metrics, per_dataset)
            print_epoch_footer()
            best_for_history = best2_epoch or (
                self._stage1_epochs_used + self._stage2_epochs_used)
            write_history(
                workdir / 'metrics_history.json', run_config, history,
                best_for_history, best2)
            self._write_dataset_csv(
                workdir / 'val_metrics_by_dataset.csv', history)
            if run is not None:
                log_scalars(run, {
                    'final/stage1_best_score': best1 if best1 != -float('inf')
                    else None,
                    'final/stage2_best_score': best2,
                    'final/stage2_best_epoch': best2_epoch,
                })
            finish_run(run)
            return ckpt_path
        except BaseException:
            finish_run(run)
            raise

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'use_vig': True,
            'stage1_epochs': int(self._stage1_epochs_used or 0),
            'stage2_epochs': int(self._stage2_epochs_used or 0),
            'vig_blocks': int(self._vig_blocks),
            'vig_edges': int(self._vig_edges),
            'vig_heads': int(self._vig_heads),
            'vig_patch_size': int(self._vig_patch_size),
            'vig_batch_size': self._vig_batch_size,
            'vig_lr': self._vig_lr,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        for key, attribute in (
                ('vig_blocks', '_vig_blocks'),
                ('vig_edges', '_vig_edges'),
                ('vig_heads', '_vig_heads'),
                ('vig_patch_size', '_vig_patch_size'),
                ('vig_batch_size', '_vig_batch_size'),
                ('vig_lr', '_vig_lr'),
                ('stage1_epochs', '_stage1_epochs'),
                ('stage2_epochs', '_stage2_epochs')):
            if key in config:
                setattr(self, attribute, config[key])
        for key, attribute in (
                ('vig_blocks', 'maskvsc_vig_blocks'),
                ('vig_edges', 'maskvsc_vig_edges'),
                ('vig_heads', 'maskvsc_vig_heads'),
                ('vig_patch_size', 'maskvsc_vig_patch_size'),
                ('vig_batch_size', 'maskvsc_vig_batch_size'),
                ('vig_lr', 'maskvsc_vig_lr')):
            if key in config:
                setattr(args, attribute, config[key])
