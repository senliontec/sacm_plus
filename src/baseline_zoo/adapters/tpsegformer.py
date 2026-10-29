"""TPSegformer adapter (CVPR 2026).

Swin-T + Topology-Preserving Decoder.  The release is an MMSegmentation
overlay with an undefined classifier, a 19-class auxiliary head and a
missing external Hilbert-curve package.  This port uses timm Swin-T, a
vendored pure-PyTorch decoder with the classifier fixed, a binary main
head, and the release's auxiliary Dice 0.5 + Sobel edge-aware 1.0 losses.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.tpsegformer import (
    build_tpsegformer,
    edge_aware_loss,
    soft_dice_loss,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

MEAN = (0.511, 0.539, 0.556)
STD = (0.084, 0.082, 0.083)


class TPSegformerAdapter(BaselineAdapter):
    id = 'tpsegformer'
    display_name = 'TPSegformer'
    family = 'transformer'
    official_source = 'TPSegformer (CVPR 2026)'
    official_entrypoint = 'https://github.com/mumu-k/TPSegformer'
    input_size = 512
    official_lr = 6e-5
    official_weight_decay = 0.01
    train_batch_size = 4
    accumulation_steps = 1
    default_grad_clip = 0.0
    selection = 'dice'
    deviations = (
        'The released repository is an MMSegmentation overlay; this port '
        'uses timm Swin-T and a pure-PyTorch vendored decoder, so mmcv and '
        'mmseg are not required.',
        'The release calls the non-existent self.cls_seg and configures the '
        'head with in_channels=3 although the LCCM emits five class-specific '
        'channels. This port defines conv_seg explicitly (5 -> 1) and fixes '
        'the auxiliary head from 19 to one binary channel.',
        'The release imports the external hilbertcurve/reference hbtrans '
        'modules; this port implements the same Hilbert traversal locally '
        'and keeps the LCCM attention math unchanged.',
        'The official OpenMMLab Swin-T checkpoint is auto-detected under '
        'checkpoints/*swin*tiny* and MMSeg keys are mapped to timm; random '
        'initialization is used only when no checkpoint is present.',
        'The official 512 random-resize/random-crop/pad pipeline and '
        'PhotoMetricDistortion are approximated by a 512-square bilinear '
        'resize (nearest GT), horizontal flip and a light MMSeg-style '
        'brightness/contrast/saturation/hue jitter.',
        'The official main cross-entropy is replaced by binary BCE for the '
        'one-channel head; the auxiliary Dice (0.5) and Sobel edge-aware '
        '(1.0) losses, five-conv FCN head and dropout are preserved.',
        'AdamW lr 6e-5, wd 0.01, head lr x10 and zero weight decay for '
        'norm/position parameters are preserved. The official per-iteration '
        'LinearLR(1500) + PolyLR(160k) schedule is approximated per epoch '
        '(5-epoch warmup starting at 1e-6 of the base LR, then linear poly '
        'decay over the shared budget) because the runtime steps the '
        'scheduler once per epoch.',
        'The official 20k/100-epoch budget becomes the shared 800 epochs '
        'with validation-Dice selection; the Dacl10k sliding-window test '
        '(512/341) and TTA are not used, and predictions are exported at '
        'the exact GT resolution.',
        'The source paper targets UAV building-defect segmentation; this '
        'port applies its topology-preserving decoder to curvilinear '
        'structures as the task adaptation.',
    )

    def __init__(self):
        super().__init__()
        self._backbone_ckpt: str | None = None
        self._backbone_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument(
            '--tpsegformer-backbone',
            default='swin_tiny_patch4_window7_224')
        parser.add_argument(
            '--tpsegformer-drop-path', type=float, default=0.3)
        parser.add_argument(
            '--tpsegformer-lccm-mode', default='hilbert',
            choices=['hilbert', 'row-first', 'z-curve'])
        parser.add_argument(
            '--tpsegformer-backbone-ckpt', type=Path, default=None)
        parser.add_argument(
            '--tpsegformer-head-lr-mult', type=float, default=10.0)
        parser.add_argument(
            '--tpsegformer-warmup-epochs', type=int, default=5)
        parser.add_argument('--tpsegformer-no-augment', action='store_true')
        parser.add_argument(
            '--tpsegformer-no-photometric', action='store_true')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 512))
        if self.input_size != 512:
            raise SystemExit('TPSegformer uses the official 512x512 input')
        return build_tpsegformer(
            image_size=512,
            num_classes=1,
            lccm_mode=getattr(args, 'tpsegformer_lccm_mode', 'hilbert'),
            drop_path_rate=float(getattr(
                args, 'tpsegformer_drop_path', 0.3)),
            backbone=getattr(
                args, 'tpsegformer_backbone',
                'swin_tiny_patch4_window7_224'),
        )

    @staticmethod
    def _find_local_backbone() -> Path | None:
        root = REPO_ROOT / 'checkpoints'
        if not root.is_dir():
            return None
        candidates = sorted(
            path for path in root.rglob('*')
            if path.is_file()
            and path.suffix.lower() in {'.pth', '.pt', '.bin', '.safetensors'}
            and 'swin' in path.name.lower() and 'tiny' in path.name.lower())
        if len(candidates) > 1:
            names = ', '.join(str(path) for path in candidates)
            raise SystemExit(
                f'multiple Swin-T checkpoints under {root}: {names}; '
                'pass --tpsegformer-backbone-ckpt explicitly')
        return candidates[0] if candidates else None

    def load_pretrained(self, model, args) -> None:
        if self.prediction_mode:
            return
        checkpoint = getattr(args, 'tpsegformer_backbone_ckpt', None)
        if checkpoint is None:
            checkpoint = self._find_local_backbone()
            if checkpoint is not None:
                args.tpsegformer_backbone_ckpt = checkpoint
        if checkpoint is None:
            print('TPSegformer: random Swin-T initialization '
                  '(no local ImageNet checkpoint)')
            return
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing Swin-T backbone checkpoint: {checkpoint}')
        counts = model.load_backbone_checkpoint(checkpoint)
        self._backbone_ckpt = str(checkpoint)
        self._backbone_sha = file_sha256(checkpoint)
        print(f'TPSegformer: loaded Swin-T backbone {counts}')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        print(f'TPSegformer parameters: {total:,} total, {total:,} trainable')

    def make_optimizer(self, model, args):
        head_mult = float(getattr(
            args, 'tpsegformer_head_lr_mult', 10.0))
        groups = {
            'backbone_decay': [], 'backbone_nodecay': [],
            'head_decay': [], 'head_nodecay': [],
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_head = (name.startswith('head.')
                       or name.startswith('aux_head.'))
            no_decay = (
                'norm' in name or 'pos_embed' in name
                or 'relative_position' in name)
            key = ('head' if is_head else 'backbone') + (
                '_nodecay' if no_decay else '_decay')
            groups[key].append(parameter)
        param_groups = []
        for key, parameters in groups.items():
            if not parameters:
                continue
            is_head = key.startswith('head')
            param_groups.append({
                'params': parameters,
                'lr': float(args.lr) * (head_mult if is_head else 1.0),
                'weight_decay': (0.0 if key.endswith('nodecay')
                                 else float(args.weight_decay)),
            })
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def make_scheduler(self, optimizer, args):
        epochs = max(1, int(args.epochs))
        warmup = max(0, min(
            int(getattr(args, 'tpsegformer_warmup_epochs', 5)),
            epochs - 1))

        def factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                # Official PolyLrUpdater starts at 1e-6 of the base LR.
                return 1e-6 + (1.0 - 1e-6) * epoch / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_pair(self, sample: Sample,
                   rng: random.Random | None = None,
                   train: bool = False):
        image = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path)
        image = TF.resize(
            image, [512, 512],
            interpolation=TF.InterpolationMode.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        mask_img = TF.resize(
            mask_img, [512, 512],
            interpolation=TF.InterpolationMode.NEAREST)
        if train and rng is not None and not bool(getattr(
                self.args, 'tpsegformer_no_augment', False)):
            if rng.random() < 0.5:
                image = TF.hflip(image)
                mask_img = TF.hflip(mask_img)
            if not bool(getattr(
                    self.args, 'tpsegformer_no_photometric', False)):
                if rng.random() < 0.5:
                    image = TF.adjust_brightness(
                        image, 1 + rng.uniform(-0.125, 0.125))
                if rng.random() < 0.5:
                    image = TF.adjust_contrast(
                        image, rng.uniform(0.5, 1.5))
                if rng.random() < 0.5:
                    image = TF.adjust_saturation(
                        image, rng.uniform(0.5, 1.5))
                if rng.random() < 0.5:
                    image = TF.adjust_hue(
                        image, rng.uniform(-0.07, 0.07))
        image_t = TF.normalize(TF.to_tensor(image), MEAN, STD)
        mask_t = TF.to_tensor(mask_img)
        return image_t, mask_t

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._load_pair(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
        return {
            'image': torch.stack(images).to(self.device),
            'gt': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        main, aux = model(batch['image'])
        return (F.binary_cross_entropy_with_logits(main, batch['gt'])
                + 0.5 * soft_dice_loss(aux, batch['gt'])
                + 1.0 * edge_aware_loss(aux, batch['gt']))

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        image, _ = self._load_pair(sample, None, train=False)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.unsqueeze(0).to(device),
            'size': (int(height), int(width)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        main, _ = model(prepared['image'])
        height, width = prepared['size']
        logits = F.interpolate(
            main, size=(height, width), mode='bilinear',
            align_corners=False)
        return logits.sigmoid()[0, 0].cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'backbone': getattr(
                args, 'tpsegformer_backbone',
                'swin_tiny_patch4_window7_224'),
            'drop_path': float(getattr(
                args, 'tpsegformer_drop_path', 0.3)),
            'lccm_mode': getattr(
                args, 'tpsegformer_lccm_mode', 'hilbert'),
            'head_lr_mult': float(getattr(
                args, 'tpsegformer_head_lr_mult', 10.0)),
            'warmup_epochs': int(getattr(
                args, 'tpsegformer_warmup_epochs', 5)),
            'backbone_ckpt': self._backbone_ckpt,
            'backbone_sha256': self._backbone_sha,
            'augment': not bool(getattr(
                args, 'tpsegformer_no_augment', False)),
            'photometric': not bool(getattr(
                args, 'tpsegformer_no_photometric', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        mappings = {
            'input_size': 'input_size',
            'backbone': 'tpsegformer_backbone',
            'drop_path': 'tpsegformer_drop_path',
            'lccm_mode': 'tpsegformer_lccm_mode',
            'head_lr_mult': 'tpsegformer_head_lr_mult',
            'warmup_epochs': 'tpsegformer_warmup_epochs',
        }
        for key, attribute in mappings.items():
            if key in config:
                setattr(args, attribute, config[key])
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if config.get('backbone_ckpt'):
            self._backbone_ckpt = config['backbone_ckpt']
            args.tpsegformer_backbone_ckpt = Path(
                config['backbone_ckpt'])
        if 'backbone_sha256' in config:
            self._backbone_sha = config['backbone_sha256']
        if 'augment' in config:
            args.tpsegformer_no_augment = not config['augment']
        if 'photometric' in config:
            args.tpsegformer_no_photometric = not config['photometric']
