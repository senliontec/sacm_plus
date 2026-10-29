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
from baseline_zoo.runtime import (
    run_prediction as run_shared_prediction,
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


def _augment_pair(image: Image.Image, label: Image.Image,
                  rng: random.Random, rotation_degrees: float):
    """Paper training augmentation: random rotation + horizontal flip.

    The paper gives ±30° but not probabilities or interpolation.  Rotation is
    applied to every sample with a uniform angle; images use bilinear
    interpolation and masks use nearest so binary labels stay binary.
    ``rotation_degrees=0`` reproduces the release (flip only) path.
    """
    if rotation_degrees > 0:
        angle = rng.uniform(-rotation_degrees, rotation_degrees)
        image = image.rotate(
            angle, resample=Image.BILINEAR, expand=False,
            fillcolor=(0, 0, 0))
        label = label.rotate(
            angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
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
    custom_pipeline = True
    deviations = (
        'Full two-stage MaskVSC: stage-1 UNet (masked-vessel corruption + '
        'Connect_Loss) followed by the official ViG completion head '
        '(ViG(Seg(image))), matching the upstream train_ViG.py/test_ViG.py '
        'inference path. The previous stage-1-only port is superseded.',
        'Stage 2 uses the released ViG decode exactly: patch_size=16 for the '
        '512 input, N=(512/16)^2=1024 token maps as predictor channels, and '
        'the official features.unsqueeze(2).view(B,N,patch,patch) fold. The '
        'official Fundus ViG has 23.84M parameters; the full model is '
        '31.61M. --maskvsc-vig-decode coherent keeps the earlier '
        'transpose-token-grid/patch-32 reconstruction (371.6M ViG) as an '
        'explicit ablation only; it is not the released architecture.',
        'Main baseline uses the official release stage-1 behavior '
        '(count_nonzero L_C, flip-only augmentation) and the released '
        'dot-product ViG graph. Paper-mode L_C, ±30° rotation and Euclidean '
        'KNN remain explicit ablations via --maskvsc-connect-loss paper, '
        '--maskvsc-rotation-degrees 30 and --maskvsc-vig-graph paper. The '
        'paper-mode L_C scale |S(Y)-S(Y_G)|*N/#foreground is not the default '
        'because sparse masks receive 10-100x the BCE weight in the pooled '
        '3-shot protocol; it also skips the released 3x3 opening, which erased '
        'thin vessels and made empty predictions free on CrackTree/CREMI/CORN. '
        'Use --maskvsc-connect-cap to bound paper-mode L_C during '
        'recalibration. Stage 2 has '
        'no augmentation, matching the released GCN loader; the paper does '
        'not specify augmentation for step 2.',
        'Stage 2 is a single pooled ViG across all protocol domains. Upstream '
        'trains separate Fundus/OCTA/2PFM completion models per dataset; '
        'unseen/novel domains have no per-domain training data, so a pooled '
        'model is the only protocol-consistent choice.',
        'Stage 2 trains on clean images and the frozen stage-1 probability '
        'maps from the 18 train images, with BCE only. The paper explicitly '
        'freezes step 1; this port forces eval() and requires_grad=False. '
        'The release does not call eval() and would update its BatchNorm '
        'statistics and accumulate unused gradients.',
        'Checkpoint selection uses the 6-image validation split by Dice and '
        'compares the stage-1 and stage-2 best scores. Upstream evaluates the '
        'test split every epoch and selects the best ViG by test Dice, which '
        'violates the protocol. If stage 2 does not beat stage 1, the final '
        'checkpoint records use_vig=false and inference uses the UNet alone.',
        'The shared 800-epoch budget is split 600 stage-1 + 200 stage-2 by '
        'default; the masked-vessel corruption ramp spans those 600 stage-1 '
        'epochs. Upstream uses 200 + 200; use --maskvsc-stage1-epochs 800 '
        'for the ViG-isolation A/B (1000 total).',
        'Data protocol: the 18/6-shot split replaces the original Train/Test '
        'folders; grayscale domains are converted to RGB because the official '
        'Fundus UNet expects three channels. Stage-1 masks are thresholded '
        'after resize as in ImageDataset_Fundus; stage 2 uses the official '
        'soft PIL-resized labels from ImageDataset_GCN_Fundus for BCE.',
        'Benchmark-size protocol: the paper uses per-dataset input sizes '
        '(512 DRIVE/RITE, 576 STARE, 1024 LES-AV/2PFM, 384/304 OCTA-500, 304 '
        'ROSE) and per-modality corruption (Fundus SD20/7x7, OCTA SD100/3x3, '
        '2PFM SD20/11x11). The pooled 20-domain port uses 512, SD20 and 7x7 '
        'for every domain so all baselines share one input size.',
        'Official test.py writes 512x512 masks; predictions are restored to '
        'the original GT resolution so every baseline shares one evaluator.',
        'The protocol seeds 0-4 replace upstream seed 1234 and deterministic '
        'cuDNN. Rotation/flip draw from random.Random(seed+epoch); the '
        'corruption sampler keeps the upstream global random/numpy draws, '
        'while the release also draws the flip from numpy. Exact per-step '
        'RNG sequences therefore differ. The official corruption sampler '
        'consumes RNG when ratio=0; this port early-returns before that draw '
        'and uses workers=0.',
        'The custom two-stage pipeline does not support --resume, --accum-steps '
        'or gradient clipping; a rerun retrains from scratch. Upstream stage-1 '
        'evaluates the test split every epoch; this port uses validation Dice.',
        'Stage-1-only checkpoints from the first MaskVSC port contain no ViG '
        'weights and must be retrained. Full two-stage checkpoints from the '
        'earlier port remain loadable: a missing vig_decode is inferred as '
        'coherent, and missing vig_graph/connect_loss/rotation_degrees are '
        'inferred as the release graph, release L_C and flip-only '
        'augmentation. New runs default to the release graph/L_C/flip-only '
        'with the official release ViG decode; paper graph/L_C/rotation are '
        'explicit ablations.',
    )

    def __init__(self):
        super().__init__()
        self._vig_blocks = 16
        self._vig_edges = 9
        self._vig_heads = 1
        self._vig_patch_size = 16
        self._vig_decode = 'release'
        self._vig_graph = 'release'
        self._connect_loss_mode = 'release'
        self._connect_cap = 0.0
        self._rotation_degrees = 0.0
        self._vig_batch_size = None
        self._vig_lr = None
        self._stage1_epochs = None
        self._stage2_epochs = 200
        self._stage1_epochs_used = 0
        self._stage2_epochs_used = 0
        self._use_vig = True

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
            '--maskvsc-vig-patch-size', type=int, default=16,
            help='ViG patch size (official 16 for the 512 Fundus input)')
        parser.add_argument(
            '--maskvsc-vig-decode', choices=['release', 'coherent'],
            default='release',
            help='release: official token-fold/predictor=N decode; coherent: '
                 'non-official patch-32 token-grid ablation')
        parser.add_argument(
            '--maskvsc-vig-graph', choices=['paper', 'release'],
            default='release',
            help='release: dot-product top-k from the released gcn.py '
                 '(main baseline); paper: Euclidean KNN ablation')
        parser.add_argument(
            '--maskvsc-connect-loss', choices=['paper', 'release'],
            default='release',
            help='release: official count_nonzero implementation (main '
                 'baseline); paper: foreground/N-normalized L_C ablation')
        parser.add_argument(
            '--maskvsc-connect-cap', type=float, default=0.0,
            help='paper-mode L_C cap; 0 disables. Use a bounded value for '
                 'the sparse-domain recalibration A/B')
        parser.add_argument(
            '--maskvsc-rotation-degrees', type=float, default=0.0,
            help='stage-1 random rotation range; 0 is the release flip-only '
                 'path (main baseline), 30 is the paper augmentation ablation')
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
            getattr(args, 'maskvsc_vig_patch_size', 16))
        self._vig_decode = str(
            getattr(args, 'maskvsc_vig_decode', 'release'))
        if self._vig_decode not in ('release', 'coherent'):
            raise SystemExit(
                '--maskvsc-vig-decode must be release or coherent')
        self._vig_graph = str(
            getattr(args, 'maskvsc_vig_graph', 'release'))
        if self._vig_graph not in ('paper', 'release'):
            raise SystemExit(
                '--maskvsc-vig-graph must be paper or release')
        self._connect_loss_mode = str(
            getattr(args, 'maskvsc_connect_loss', 'release'))
        if self._connect_loss_mode not in ('paper', 'release'):
            raise SystemExit(
                '--maskvsc-connect-loss must be paper or release')
        self._connect_cap = float(
            getattr(args, 'maskvsc_connect_cap', 0.0))
        if self._connect_cap < 0:
            raise SystemExit(
                '--maskvsc-connect-cap must be >= 0')
        self._rotation_degrees = float(
            getattr(args, 'maskvsc_rotation_degrees', 0.0))
        if not 0 <= self._rotation_degrees <= 180:
            raise SystemExit(
                '--maskvsc-rotation-degrees must be between 0 and 180')
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
            decode=self._vig_decode,
            graph_metric=self._vig_graph,
        )
        initialize_weights(vig)
        model = MaskVSCFull(seg, vig)
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict')
        use_vig = bool(getattr(self, '_use_vig', True))
        model.set_stage('stage2' if predicting and use_vig else 'stage1')
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
            image, label = _augment_pair(
                image, label, rng, self._rotation_degrees)
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
            # Official ImageDataset_GCN_Fundus keeps the soft, PIL-resized
            # grayscale label and ToTensor simply divides by 255.
            label_np = np.asarray(
                label.resize([self.input_size, self.input_size]),
                dtype=np.float32) / 255.0
            image_t = torch.from_numpy(
                image_np.astype(np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            label_t = torch.from_numpy(label_np.copy()).unsqueeze(0)
            images.append(image_t)
            labels.append(label_t)
        return (torch.stack(images).to(device),
                torch.stack(labels).to(device))

    def train_step(self, model, batch):
        images, labels = batch
        pred = model(images)
        loss = nn.BCELoss()(pred, labels)
        if getattr(model, 'stage', 'stage1') == 'stage1':
            loss = loss + Connect_Loss(mode=self._connect_loss_mode,
                                    cap=self._connect_cap)(pred, labels)
        return loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size))
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def run_prediction(self, args, checkpoint):
        """Use the shared prediction path after the custom two-stage train."""
        run_shared_prediction(self, args, Path(checkpoint))

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
                    + Connect_Loss(mode=self._connect_loss_mode,
                                    cap=self._connect_cap)(pred, labels))
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

    @staticmethod
    def _add_validation_scalars(scalars, metrics, per_dataset):
        """Match the shared runtime's overall and per-dataset W&B keys."""
        scalars.update(
            {f'val/{key}': value for key, value in metrics.items()})
        for dataset, values in per_dataset.items():
            for metric in ('dice', 'iou', 'cldice', 'hd95'):
                scalars[f'val/{dataset}/{metric}'] = values[metric]
        return scalars

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
        # ``use_vig`` is chosen across stages, so refresh the nested adapter
        # config at every save instead of trusting the initial run config.
        config = dict(run_config)
        config['adapter_config'] = self.save_config(args)
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
            'config': config,
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
        stage1_ckpt_path = workdir / 'stage1_best.pth'
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
            'connect_loss': self._connect_loss_mode,
            'connect_cap': self._connect_cap,
            'rotation_degrees': self._rotation_degrees,
            'stage2_augmentation': False,
            'vig_graph': self._vig_graph,
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
                'decode': self._vig_decode,
                'graph_metric': self._vig_graph,
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
            f'(ViG, batch {vig_batch}, patch {self._vig_patch_size}, '
            f'{self._vig_decode} decode, {self._vig_graph} graph); '
            f'L_C={self._connect_loss_mode}, rotation='
            f'±{self._rotation_degrees:g}°')
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
        best1_metrics = None
        best1_per_dataset = None
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
                    best1_metrics = metrics
                    best1_per_dataset = per_dataset
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
                    self._add_validation_scalars(
                        scalars, metrics, per_dataset)
                    log_scalars(run, scalars)

            if best1_state is None:
                best1_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.seg.state_dict().items()}
                best1_epoch = self._stage1_epochs_used
                best1 = 0.0
            if best1_metrics is None:
                best1_metrics = metrics
                best1_per_dataset = per_dataset
            model.seg.load_state_dict(best1_state)
            run_config['stage1_best_epoch'] = best1_epoch
            run_config['stage1_best_score'] = self._json_value(best1)
            # Keep a usable stage-1 checkpoint even if stage 2 is worse or
            # the process dies before stage 2 finishes.
            self._use_vig = False
            run_config['selected_stage'] = 1
            run_config['selected_epoch'] = best1_epoch
            run_config['selected_score'] = self._json_value(best1)
            self._save_checkpoint(
                ckpt_path, model, args, run_config, split_sha,
                best1_epoch, best1, best1_metrics, best1_per_dataset)
            # Preserve the stage-1 result even when stage 2 later wins, so
            # the two stages can be compared after the run.
            shutil.copy2(ckpt_path, stage1_ckpt_path)
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
            best2_metrics = None
            best2_per_dataset = None
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
                    best2_metrics = metrics
                    best2_per_dataset = per_dataset
                    run_config['stage2_best_epoch'] = best2_epoch
                    run_config['stage2_best_score'] = self._json_value(best2)
                    # Stage-2 "improvement" is only stage-local.  Do not
                    # overwrite the stage-1 checkpoint unless the ViG also
                    # beats the stage-1 best on the shared validation split.
                    if score > best1:
                        self._use_vig = True
                        run_config['selected_stage'] = 2
                        run_config['selected_epoch'] = epoch
                        run_config['selected_score'] = self._json_value(
                            score)
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
                    self._add_validation_scalars(
                        scalars, metrics, per_dataset)
                    log_scalars(run, scalars)

            selected_stage = (
                2 if best2_epoch and best2 > best1 else 1)
            self._use_vig = selected_stage == 2
            if selected_stage == 2:
                # The on-disk checkpoint is the best ViG epoch; reload it so
                # the final save also carries the final run config.
                payload = torch.load(
                    ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(payload['model_state_dict'])
                model.set_stage('stage2')
                selected_epoch = best2_epoch
                selected_score = best2
                selected_metrics = best2_metrics
                selected_per_dataset = best2_per_dataset
            else:
                model.seg.load_state_dict(best1_state)
                model.set_stage('stage1')
                selected_epoch = best1_epoch
                selected_score = best1
                selected_metrics = best1_metrics
                selected_per_dataset = best1_per_dataset
            run_config['selected_stage'] = selected_stage
            run_config['selected_epoch'] = selected_epoch
            run_config['selected_score'] = self._json_value(selected_score)
            run_config['use_vig'] = self._use_vig
            run_config['adapter_config'] = self.save_config(args)
            self._save_checkpoint(
                ckpt_path, model, args, run_config, split_sha,
                selected_epoch, selected_score, selected_metrics,
                selected_per_dataset)
            print_epoch_footer()
            write_history(
                workdir / 'metrics_history.json', run_config, history,
                selected_epoch, selected_score)
            self._write_dataset_csv(
                workdir / 'val_metrics_by_dataset.csv', history)
            if run is not None:
                log_scalars(run, {
                    'final/stage1_best_score': best1 if best1 != -float('inf')
                    else None,
                    'final/stage2_best_score': best2 if best2_epoch else None,
                    'final/stage2_best_epoch': best2_epoch,
                    'final/selected_stage': selected_stage,
                    'final/selected_epoch': selected_epoch,
                    'final/selected_score': selected_score,
                })
            finish_run(run)
            return ckpt_path
        except BaseException:
            finish_run(run)
            raise

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'use_vig': bool(self._use_vig),
            'stage1_epochs': int(self._stage1_epochs_used or 0),
            'stage2_epochs': int(self._stage2_epochs_used or 0),
            'vig_blocks': int(self._vig_blocks),
            'vig_edges': int(self._vig_edges),
            'vig_heads': int(self._vig_heads),
            'vig_patch_size': int(self._vig_patch_size),
            'vig_decode': str(self._vig_decode),
            'vig_graph': str(self._vig_graph),
            'connect_loss': str(self._connect_loss_mode),
            'connect_cap': float(self._connect_cap),
            'rotation_degrees': float(self._rotation_degrees),
            'vig_batch_size': self._vig_batch_size,
            'vig_lr': self._vig_lr,
        }

    def restore_config(self, args, config):
        # Older checkpoints always used the ViG completion head.
        self._use_vig = bool(config.get('use_vig', True))
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if 'vig_decode' in config:
            self._vig_decode = str(config['vig_decode'])
        elif 'vig_patch_size' in config:
            # Checkpoints from the first full two-stage port used the
            # non-official transpose/patch-32 decode and carried no
            # vig_decode key. Rebuild that architecture so they still load.
            self._vig_decode = 'coherent'
            print('[maskvsc] checkpoint has no vig_decode; using coherent '
                  'decode for backward compatibility', flush=True)
        else:
            self._vig_decode = 'release'
        setattr(args, 'maskvsc_vig_decode', self._vig_decode)
        if 'vig_graph' in config:
            self._vig_graph = str(config['vig_graph'])
        else:
            # Old checkpoints were trained/evaluated with the released
            # dot-product graph.
            self._vig_graph = 'release'
        if 'connect_loss' in config:
            self._connect_loss_mode = str(config['connect_loss'])
        else:
            self._connect_loss_mode = 'release'
        if 'connect_cap' in config:
            self._connect_cap = float(config['connect_cap'])
        else:
            self._connect_cap = 0.0
        if 'rotation_degrees' in config:
            self._rotation_degrees = float(config['rotation_degrees'])
        else:
            self._rotation_degrees = 0.0
        setattr(args, 'maskvsc_vig_graph', self._vig_graph)
        setattr(args, 'maskvsc_connect_loss', self._connect_loss_mode)
        setattr(args, 'maskvsc_connect_cap', self._connect_cap)
        setattr(args, 'maskvsc_rotation_degrees', self._rotation_degrees)
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
