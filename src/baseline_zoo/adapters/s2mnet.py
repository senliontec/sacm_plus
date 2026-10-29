"""S2M-Net adapter (ICML 2026), PyTorch rewrite of the Keras release.

The architecture and Morphology-Aware Adaptive Loss are reimplemented in
``baseline_zoo.models.s2mnet`` without TensorFlow.  Training uses 352 random
crops, RGB [0,1] inputs, flips/rot90 and light colour jitter; validation and
prediction use overlapping 352 patches (stride 176) with probability
averaging.  The five MAL weights are clipped to [0.1, 10] after each
optimizer step.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.s2mnet import S2MNet
from baseline_zoo.runtime import BaselineAdapter


class S2MNetAdapter(BaselineAdapter):
    id = 's2mnet'
    display_name = 'S2M-Net'
    family = 'cnn'
    official_source = 'S2M-Net (ICML 2026)'
    official_entrypoint = (
        'https://github.com/sanaullah-ashfat/'
        'S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation')
    input_size = 352
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 8
    accumulation_steps = 1
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'The official implementation is TensorFlow/Keras; this port is a '
        'PyTorch rewrite of the stem, five MRF-SE/SSTM stages, BFP decoder '
        'and MAL loss, so TensorFlow is not required.',
        'The architecture math is preserved: 16-channel stem, filters '
        '(24,32,64,80,128), MRF-SE with 1x1 expansion x6 and depthwise '
        '3/5/7 + SE-16, SSTM with per-stage FFT truncation K=min(32,H), '
        'spectral-only stages 1-2, gated-projection stages 3-5, and BFP '
        'soft region/boundary routing.',
        'Keras defaults that affect behavior are matched: Conv2D biases, '
        'ELU alpha 1, BatchNorm eps 1e-3 / momentum 0.01, elementwise '
        'dropout and LayerNorm over channels.',
        'The MAL five components, morphology modulation and learnable '
        'weights initialized at (1,1,1,.5,.5) with clipping to [0.1,10] are '
        'preserved; the weights are clamped after each optimizer step. '
        'TensorFlow kernel L2 1e-4 is implemented as Adam weight decay.',
        'The official retinal CLAHE/green-channel/FOV-mask/256-patch '
        'pipeline is not applied to the mixed protocol: images are RGB '
        '[0,1], training uses 352 random crops with flips, rot90 and light '
        'brightness/contrast jitter, and inference uses overlapping 352 '
        'patches (stride 176) with probability averaging.',
        'The official effective batch 32 and 150 epochs become batch 8 '
        '(effective 8) on the shared 18-image split and the shared '
        '800-epoch budget; validation-Dice selection replaces the release '
        'training loop.',
        'The official per-iteration cosine schedule with 10 warmup epochs '
        'is approximated per epoch (10-epoch warmup then cosine to zero) '
        'because the runtime steps the scheduler once per epoch.',
        'The release uses retinal patch stride 32; this port uses stride '
        '176 for validation/prediction to keep the shared every-epoch '
        'validation affordable. Predictions are exported at the exact GT '
        'resolution.',
        'The released code labels its gated dense projection as an SSM but '
        'it is not a selective scan; the port preserves the released math. '
        'The head is binary sigmoid; multi-class output is not used.',
        'No official S2M-Net weights are provided, so training is from '
        'random initialization.',
    )

    def __init__(self):
        super().__init__()
        self._criterion_param_count = 0

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=352)
        parser.add_argument('--s2mnet-dropout', type=float, default=0.1)
        parser.add_argument('--s2mnet-sstm-k', type=int, default=32)
        parser.add_argument(
            '--s2mnet-warmup-epochs', type=int, default=10)
        parser.add_argument('--s2mnet-tile-stride', type=int, default=176)
        parser.add_argument('--s2mnet-no-augment', action='store_true')
        parser.add_argument(
            '--s2mnet-random-init', action='store_true',
            help='compatibility flag; S2M-Net always starts from random')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 352))
        if self.input_size != 352:
            raise SystemExit('S2M-Net uses the official 352x352 input')
        return S2MNet(
            input_size=352,
            filters=(24, 32, 64, 80, 128),
            use_mrfse=True,
            mrfse_kernels=(3, 5, 7),
            se_reduction=16,
            expand_ratio=6,
            use_sstm=True,
            sstm_k=int(getattr(args, 's2mnet_sstm_k', 32)),
            sstm_stages=(True, True, True, True, True),
            sstm_use_spectral=(True, True, True, True, True),
            sstm_use_ssm=(False, False, True, True, True),
            sstm_dropout=float(getattr(args, 's2mnet_dropout', 0.1)),
            use_bfp=True,
            bfp_routing='soft',
            dropout=float(getattr(args, 's2mnet_dropout', 0.1)),
        )

    def load_pretrained(self, model, args) -> None:
        print('S2M-Net: random initialization (no official weights)')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        self._criterion_param_count = sum(
            p.numel() for p in model.criterion.parameters())
        print(f'S2M-Net parameters: {total:,} total, {total:,} trainable, '
              f'{self._criterion_param_count} MAL scalars')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        epochs = max(1, int(args.epochs))
        warmup = max(0, min(
            int(getattr(args, 's2mnet_warmup_epochs', 10)), epochs - 1))

        floor = 1e-6 / max(float(args.lr), 1e-12)

        def factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return max(floor, (epoch + 1) / warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return max(floor, 0.5 * (1.0 + np.cos(
                np.pi * min(max(progress, 0.0), 1.0))))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _prepare_image(self, sample: Sample, tile: int,
                       rng: random.Random | None = None,
                       train: bool = False):
        image = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path)
        width, height = image.size
        if height < tile or width < tile:
            scale = max(tile / height, tile / width)
            new_h = max(tile, int(round(height * scale)))
            new_w = max(tile, int(round(width * scale)))
            image = TF.resize(
                image, [new_h, new_w],
                interpolation=TF.InterpolationMode.BILINEAR)
            mask_img = Image.fromarray(
                mask.astype(np.uint8) * 255, mode='L')
            mask_img = TF.resize(
                mask_img, [new_h, new_w],
                interpolation=TF.InterpolationMode.NEAREST)
            mask = np.asarray(mask_img) > 127
            height, width = new_h, new_w

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        top = rng.randrange(0, height - tile + 1) if train else 0
        left = rng.randrange(0, width - tile + 1) if train else 0
        image_np = image_np[top:top + tile, left:left + tile]
        mask = mask[top:top + tile, left:left + tile]
        image_t = torch.from_numpy(
            np.ascontiguousarray(image_np)).permute(2, 0, 1)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(mask.astype(np.float32))).unsqueeze(0)

        if train and rng is not None and not bool(getattr(
                self.args, 's2mnet_no_augment', False)):
            if rng.random() < 0.5:
                image_t = torch.flip(image_t, dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if rng.random() < 0.5:
                image_t = torch.flip(image_t, dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
            turns = rng.randrange(4)
            if turns:
                image_t = torch.rot90(image_t, turns, dims=[-2, -1])
                mask_t = torch.rot90(mask_t, turns, dims=[-2, -1])
            if rng.random() < 0.5:
                brightness = 1.0 + rng.uniform(-0.1, 0.1)
                contrast = 1.0 + rng.uniform(-0.1, 0.1)
                mean = image_t.mean()
                image_t = ((image_t - mean) * contrast + mean) * brightness
                image_t = image_t.clamp(0.0, 1.0)
        return image_t.contiguous(), mask_t.contiguous()

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._prepare_image(
                sample, self.input_size, rng, train=True)
            images.append(image)
            masks.append(mask)
        return {
            'image': torch.stack(images).to(self.device),
            'gt': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        probability = model(batch['image'])
        return model.criterion(batch['gt'], probability)

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        model.train()
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        accum = max(int(args.accum_steps), 1)
        pending = 0
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for samples in loader:
            batch = self.prepare_train_batch(samples, rng)
            loss = self.train_step(model, batch)
            (loss / accum).backward()
            pending += 1
            if pending == accum:
                self._optimizer_step(model, optimizer, args)
                pending = 0
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        if pending:
            if pending < accum:
                scale = accum / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            self._optimizer_step(model, optimizer, args)
        if scheduler is not None:
            scheduler.step()
        return running / max(seen, 1)

    def _optimizer_step(self, model, optimizer, args) -> None:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.criterion.clip_weights()

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        image = load_rgb(sample.image_path)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.permute(2, 0, 1)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': tensor.to(device),
            'size': (int(height), int(width)),
        }

    @staticmethod
    def _starts(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        if starts[-1] != length - tile:
            starts.append(length - tile)
        return starts

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        model.eval()
        image = prepared['image']
        original = prepared['size']
        tile = int(getattr(model, 'image_size', self.input_size))
        height, width = image.shape[-2:]
        resized = False
        if height < tile or width < tile:
            scale = max(tile / height, tile / width)
            new_h = max(tile, int(round(height * scale)))
            new_w = max(tile, int(round(width * scale)))
            image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                align_corners=False)[0]
            height, width = new_h, new_w
            resized = True
        stride = min(
            int(getattr(self.args, 's2mnet_tile_stride', tile)), tile)
        accumulator = torch.zeros(
            (height, width), device=image.device, dtype=torch.float32)
        counts = torch.zeros(
            (height, width), device=image.device, dtype=torch.float32)
        for top in self._starts(height, tile, stride):
            for left in self._starts(width, tile, stride):
                patch = image[:, top:top + tile, left:left + tile]
                probability = model(patch.unsqueeze(0))[0, 0]
                accumulator[top:top + tile, left:left + tile] += probability
                counts[top:top + tile, left:left + tile] += 1
        probability = accumulator / counts.clamp_min(1)
        if resized:
            probability = F.interpolate(
                probability[None, None], size=original, mode='bilinear',
                align_corners=False)[0, 0]
        return (probability > 0.5).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'dropout': float(getattr(args, 's2mnet_dropout', 0.1)),
            'sstm_k': int(getattr(args, 's2mnet_sstm_k', 32)),
            'warmup_epochs': int(getattr(
                args, 's2mnet_warmup_epochs', 10)),
            'tile_stride': int(getattr(
                args, 's2mnet_tile_stride', 176)),
            'augment': not bool(getattr(
                args, 's2mnet_no_augment', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        mappings = {
            'input_size': 'input_size',
            'dropout': 's2mnet_dropout',
            'sstm_k': 's2mnet_sstm_k',
            'warmup_epochs': 's2mnet_warmup_epochs',
            'tile_stride': 's2mnet_tile_stride',
        }
        for key, attribute in mappings.items():
            if key in config:
                setattr(args, attribute, config[key])
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        if 'augment' in config:
            args.s2mnet_no_augment = not config['augment']
