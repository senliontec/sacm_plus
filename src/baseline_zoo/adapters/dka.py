"""DKA baseline adapter (ICLR 2026).

The released DKA code provides the dual-kernel adapter and classification
training.  Its segmentation experiments use Segmenter-B in MMSegmentation,
which is not part of the release.  This port applies DKA to a frozen
torchvision ViT-B/16 backbone with Segmenter's linear decoder, following
the paper's segmentation recipe: DKA in all transformer blocks, middle
dimension 192, head lr 1e-4, DKA lr 1e-3, batch 1.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.dka import build_dka
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DKAAdapter(BaselineAdapter):
    id = 'dka'
    display_name = 'DKA'
    official_source = 'misswayguy/DKA (DKA + Segmenter-B port)'
    official_entrypoint = 'https://github.com/misswayguy/DKA'
    input_size = 512
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 1
    accumulation_steps = 1
    selection = 'dice'
    deviations = (
        'The released DKA code includes the dual-kernel adapter and '
        'classification training only. The paper\'s segmentation experiments '
        'use Segmenter-B in MMSegmentation, which is not released. This port '
        'keeps the official DKA module (51x51 + 5x5 depthwise branches, '
        'middle dimension 192, zero-initialized down/up projections) in all '
        '12 ViT-B/16 blocks and uses Segmenter\'s linear decoder.',
        'The backbone is torchvision ViT-B/16 pretrained on ImageNet-1k. '
        'The MMSegmentation Segmenter-B checkpoint/pretraining recipe and '
        'the exact segmentation input pipeline are not specified in the '
        'release; the port uses the standard 512-square input, ImageNet '
        'normalization, horizontal-flip augmentation only, and bilinear '
        'upsampling of the linear decoder.',
        'The paper uses head lr 1e-4 and DKA lr 1e-3; both are kept. The '
        'released classification code instead uses a 1e-5 adapter lr and a '
        'StepLR; the paper\'s segmentation values are used here. No '
        'segmentation loss is stated, so this port uses cross-entropy plus '
        'foreground Dice, a task adaptation for the 0.25-3% foreground '
        'threshold domains.',
        'The official 300-epoch segmentation budget is extended to the '
        'shared 800-epoch budget with validation-Dice selection; the '
        'official batch size 1 is kept. DKA and the decoder are trainable; '
        'the ViT backbone is frozen.',
        'No official DKA weights exist; training starts from the local '
        'torchvision ViT-B/16 ImageNet checkpoint. --dka-random-init is a '
        'code-smoke option only.',
    )

    def __init__(self):
        super().__init__()
        self._vit_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument(
            '--dka-vit-weights', type=Path,
            default=REPO_ROOT / 'checkpoints' /
            'vit_b_16_imagenet1k_v1.pth')
        parser.add_argument('--dka-middle-dim', type=int, default=192)
        parser.add_argument('--dka-adapter-lr', type=float, default=1e-3)
        parser.add_argument('--dka-random-init', action='store_true',
                            help='smoke only: random ViT-B/16 weights')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 512))
        if self.input_size != 512:
            raise SystemExit('DKA uses the official 512x512 input')
        random_init = bool(getattr(args, 'dka_random_init', False))
        weights = Path(getattr(
            args, 'dka_vit_weights',
            REPO_ROOT / 'checkpoints' / 'vit_b_16_imagenet1k_v1.pth'))
        if not random_init:
            if not weights.is_file():
                raise SystemExit(
                    f'missing ViT-B/16 checkpoint: {weights}\n'
                    'download: https://download.pytorch.org/models/'
                    'vit_b_16-c867db91.pth\n'
                    'and place it at checkpoints/vit_b_16_imagenet1k_v1.pth')
            self._vit_sha = file_sha256(weights)
        return build_dka(
            None if random_init else weights,
            input_size=self.input_size,
            middle_dim=int(getattr(args, 'dka_middle_dim', 192)),
        )

    def load_pretrained(self, model, args) -> None:
        print('loaded frozen ViT-B/16 and inserted DKA adapters')

    def configure_trainable(self, model) -> None:
        adapters = [
            p for module in model.vit.encoder.layers
            for p in module.adapter.parameters() if p.requires_grad]
        if not adapters:
            raise RuntimeError('DKA adapters are not trainable')
        if any(p.requires_grad for p in model.vit.conv_proj.parameters()):
            raise RuntimeError('DKA ViT backbone must stay frozen')

    def make_optimizer(self, model, args):
        adapter_params, head_params = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if '.adapter.' in name:
                adapter_params.append(parameter)
            else:
                head_params.append(parameter)
        return torch.optim.Adam([
            {'params': adapter_params,
             'lr': float(getattr(args, 'dka_adapter_lr', 1e-3))},
            {'params': head_params, 'lr': float(args.lr)},
        ], weight_decay=float(args.weight_decay))

    def make_scheduler(self, optimizer, args):
        # MMSegmentation's default poly schedule (power 1.0), stepped once
        # per epoch by the shared runtime.
        epochs = max(1, int(args.epochs))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: max(0.0, 1.0 - epoch / epochs))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _pair(self, sample, rng: random.Random | None = None):
        image = np.asarray(load_rgb(sample.image_path), dtype=np.float32)
        mask = load_mask(sample.mask_path).astype(np.float32)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_NEAREST)
        if rng is not None and rng.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        image = (image / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        image_t = torch.from_numpy(
            image.transpose(2, 0, 1).copy()).float()
        return image_t, torch.from_numpy(mask.copy()).long()

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._pair(sample, rng)
            images.append(image)
            masks.append(mask)
        return {
            'images': torch.stack(images).to(self.device),
            'masks': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        logits = model(batch['images'])
        target = batch['masks']
        ce = F.cross_entropy(logits, target)
        prob = logits.softmax(1)[:, 1]
        fg = (target == 1).float()
        intersection = (prob * fg).sum(dim=(1, 2))
        denominator = prob.sum(dim=(1, 2)) + fg.sum(dim=(1, 2))
        dice = 1 - (2 * intersection + 1.0) / (denominator + 1.0)
        return ce + dice.mean()

    def prepare_eval_sample(self, sample, device):
        image, _ = self._pair(sample)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits = model(prepared['image'])
        width, height = prepared['size']
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy() > 0

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------
    def save_config(self, args):
        weights = getattr(args, 'dka_vit_weights', None)
        sha = self._vit_sha
        if sha is None and weights is not None and Path(weights).is_file():
            sha = file_sha256(Path(weights))
        return {
            'input_size': int(self.input_size),
            'vit_weights': str(weights) if weights else None,
            'vit_weights_sha256': sha,
            'middle_dim': int(getattr(args, 'dka_middle_dim', 192)),
            'adapter_lr': float(getattr(args, 'dka_adapter_lr', 1e-3)),
            'random_init': bool(getattr(args, 'dka_random_init', False)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if config.get('vit_weights'):
            args.dka_vit_weights = Path(config['vit_weights'])
        if 'middle_dim' in config:
            args.dka_middle_dim = int(config['middle_dim'])
        if 'adapter_lr' in config:
            args.dka_adapter_lr = float(config['adapter_lr'])
        if 'random_init' in config:
            args.dka_random_init = bool(config['random_init'])
