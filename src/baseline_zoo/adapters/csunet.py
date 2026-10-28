"""CS-UNet adapter (ICASSP 2026), PyTorch rewrite of the TF notebook.

The released notebook is a Keras U-Net with a placeholder "dynamic snake"
convolution and Gegenbauer FSF skip fusion.  This adapter uses the PyTorch
translation, 512-square RGB [0,1] inputs without augmentation, SGD 0.01 /
momentum 0.9, the notebook's per-epoch ``lr - 1e-6`` decay and Dice loss.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.csunet import CSUNet, dice_loss
from baseline_zoo.runtime import BaselineAdapter


class CSUNetAdapter(BaselineAdapter):
    id = 'csunet'
    display_name = 'CS-UNet'
    family = 'cnn'
    official_source = 'CS-UNet (ICASSP 2026)'
    official_entrypoint = 'https://github.com/Jayku88/CSUNet'
    input_size = 512
    official_lr = 0.01
    official_weight_decay = 0.0
    train_batch_size = 4
    accumulation_steps = 1
    default_grad_clip = 0.0
    selection = 'dice'
    deviations = (
        'The official code is a TensorFlow/Keras notebook; this port is a '
        'PyTorch rewrite of the U-Net, GFR/FSF fusion and Dice loss, so '
        'TensorFlow is not required.',
        'The notebook DynamicSnakeConv2D is explicitly a placeholder: a plain '
        '3x3 Conv-GroupNorm(16)-ReLU followed by the fixed-alpha Snake '
        'activation, with no deformable offsets. The port preserves that '
        'released behavior; the paper describes a PACB with batch '
        'normalization, while the code uses GroupNorm+ReLU.',
        'GFR/FSF follows the notebook: global-max channel squeeze, reduction '
        '8, degree-5 Gegenbauer activations with learnable softplus alpha '
        'and sigmoid-bounded inputs, 1x1 compression to half channels, a '
        'global-max sigmoid gate on the skip, and skip*gate + decoder.',
        'The official batch 32 with the 18-image split gives '
        'steps_per_epoch=0; this port uses batch 4 and SGD 0.01 with '
        'momentum 0.9 and no gradient clipping (official settings), '
        'documented as a necessary reduction.',
        'The official 100-epoch budget and val-loss early stopping become '
        'the shared 800-epoch budget with validation-Dice selection.',
        'The official constant LR minus 1e-6 per epoch is applied per epoch '
        'by a LambdaLR (the shared runtime steps once per epoch).',
        'No augmentation, RGB /255 normalization and 512-square resize are '
        'used; predictions are bilinearly resized to the exact GT '
        'resolution and thresholded at 0.5.',
        'The notebook Dice loss uses segmentation_models; this port uses the '
        'same squared-free Dice form with smooth=1.0 and per-image mean, '
        'which may differ by a small constant from the package reduction.',
        'There are no official CS-UNet weights; training starts from random '
        'initialization. The paper targets road extraction; this is the '
        'curvilinear-structure task adaptation.',
    )

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument('--csunet-filters', type=int, default=32)
        parser.add_argument(
            '--csunet-lr-decay', type=float, default=1e-6,
            help='absolute learning-rate decrement per epoch (notebook)')
        parser.add_argument('--csunet-random-init', action='store_true',
                            help='compatibility flag; always random')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 512))
        if self.input_size != 512:
            raise SystemExit('CS-UNet uses the official 512x512 input')
        return CSUNet(
            input_size=512,
            filters=int(getattr(args, 'csunet_filters', 32)),
            num_classes=1,
        )

    def load_pretrained(self, model, args) -> None:
        print('CS-UNet: random initialization (no official weights)')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        print(f'CS-UNet parameters: {total:,} total, {total:,} trainable')

    def make_optimizer(self, model, args):
        return torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr), momentum=0.9)

    def make_scheduler(self, optimizer, args):
        base = max(float(args.lr), 1e-12)
        decay = float(getattr(args, 'csunet_lr_decay', 1e-6))

        def factor(epoch: int) -> float:
            return max(0.0, (base - decay * epoch) / base)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_pair(self, sample: Sample):
        image = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path)
        image = TF.resize(
            image, [512, 512],
            interpolation=TF.InterpolationMode.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        mask_img = TF.resize(
            mask_img, [512, 512],
            interpolation=TF.InterpolationMode.NEAREST)
        image_t = TF.to_tensor(image)
        mask_t = TF.to_tensor(mask_img)
        return image_t, mask_t

    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng) -> dict:
        images, masks = [], []
        for sample in samples:
            image, mask = self._load_pair(sample)
            images.append(image)
            masks.append(mask)
        return {
            'image': torch.stack(images).to(self.device),
            'gt': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        probability = model(batch['image'])
        return dice_loss(probability, batch['gt'])

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

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        model.eval()
        image = prepared['image']
        tile = int(getattr(model, 'image_size', self.input_size))
        if image.shape[-2:] != (tile, tile):
            image = F.interpolate(
                image.unsqueeze(0), size=(tile, tile),
                mode='bilinear', align_corners=False)
        else:
            image = image.unsqueeze(0)
        probability = model(image)[0, 0]
        height, width = prepared['size']
        probability = F.interpolate(
            probability[None, None], size=(height, width),
            mode='bilinear', align_corners=False)[0, 0]
        return (probability > 0.5).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'filters': int(getattr(args, 'csunet_filters', 32)),
            'lr_decay': float(getattr(args, 'csunet_lr_decay', 1e-6)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'filters' in config:
            args.csunet_filters = int(config['filters'])
        if 'lr_decay' in config:
            args.csunet_lr_decay = float(config['lr_decay'])
