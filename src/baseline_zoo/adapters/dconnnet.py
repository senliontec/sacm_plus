"""DconnNet adapter (CVPR 2023).

Official CHASE recipe: 960x960 BGR inputs normalized as
``[0,1] * 3.2 - 1.6``, an 8-direction connectivity head, the CHASE
``BCE + connectivity BCE + edge + Dice`` objective, and sigmoid ->
0.5 -> bilateral-voting inference.  The auxiliary 8-channel map receives
the same objective with weight 0.3 during training, as in the release.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import Sample, load_mask, load_rgb
from baseline_zoo.models.dconnnet import (
    Bilateral_voting,
    DconnNet,
    connect_loss,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class DconnNetAdapter(BaselineAdapter):
    id = 'dconnnet'
    display_name = 'DconnNet'
    family = 'cnn'
    official_source = 'DconnNet (CVPR 2023)'
    official_entrypoint = 'https://github.com/Zyun-Y/DconnNet'
    input_size = 960
    official_lr = 3.8e-3
    official_weight_decay = 0.0
    train_batch_size = 4
    accumulation_steps = 1
    default_grad_clip = 0.0
    selection = 'dice'
    deviations = (
        'The official CHASE script uses 960x960, batch 4, Adam lr 3.8e-3, '
        'polynomial LR power 0.9 and 130 epochs.  This port keeps the input, '
        'batch, learning rate and polynomial shape but uses the shared '
        '800-epoch budget and validation-Dice selection instead of the '
        'official test-Dice selection.',
        'The release uses apex O2 mixed precision and torchsummary; this '
        'port is plain fp32 and does not require either package.  Batch 4 '
        'is retained for the 24 GiB target GPUs.',
        'The release reads images with OpenCV BGR and normalizes '
        '[0,1]*3.2-1.6; the port converts the RGB protocol loader to BGR and '
        'keeps that normalization.  Masks are resized and thresholded at 0.5.',
        'The official HSV shift, perspective shift/scale/rotation, flips '
        'and rot90 are reimplemented for original-resolution protocol pairs; '
        'no crop is used.',
        'Inference follows the official CHASE path: sigmoid -> 0.5 -> '
        'reshape to [B,1,8,H,W] -> Bilateral_voting -> max over directions, '
        'then nearest resize to the exact GT resolution.',
        'The ResNet-34 ImageNet checkpoint is optional and loaded only from '
        'a local file (default checkpoints/resnet34-333f7ec4.pth).  If it '
        'is absent, training starts from random initialization, which is '
        'recorded as a deviation; runtime never downloads weights.',
        'The release has no LICENSE file; treat this as research code and '
        'verify redistribution rights before publishing.',
    )

    def __init__(self):
        super().__init__()
        self._criterion = None
        self._hori_translation = None
        self._verti_translation = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=960)
        parser.add_argument(
            '--dconnnet-backbone-ckpt',
            default=str(REPO_ROOT / 'checkpoints' / 'resnet34-333f7ec4.pth'),
            help='local ImageNet ResNet-34 state dict; never downloaded')
        parser.add_argument(
            '--dconnnet-no-augment', action='store_true',
            help='disable the release augmentation (smoke/tests only)')
        parser.add_argument(
            '--dconnnet-random-init', action='store_true',
            help='compatibility flag; missing backbone weights already '
                 'fall back to random initialization')

    def _build_translations(self) -> None:
        size = int(self.input_size)
        self._hori_translation = torch.zeros(
            [1, 1, size, size], device=self.device)
        self._verti_translation = torch.zeros(
            [1, 1, size, size], device=self.device)
        for index in range(size - 1):
            self._hori_translation[:, :, index, index + 1] = 1.0
            self._verti_translation[:, :, index, index + 1] = 1.0

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 960))
        if self.input_size < 32 or self.input_size % 32:
            raise SystemExit(
                'DconnNet input size must be a multiple of 32 (official 960)')
        model = DconnNet(num_class=1)
        self._build_translations()
        self._criterion = connect_loss(
            1, self._hori_translation, self._verti_translation,
            dataset='chase')
        return model

    def load_pretrained(self, model, args) -> None:
        path = Path(getattr(
            args, 'dconnnet_backbone_ckpt',
            REPO_ROOT / 'checkpoints' / 'resnet34-333f7ec4.pth'))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            print(
                'DconnNet: local ResNet-34 checkpoint not found at '
                f'{path}; training from random initialization (deviation)')
            return
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, dict):
            for key in ('state_dict', 'model', 'backbone'):
                if isinstance(payload.get(key), dict):
                    payload = payload[key]
                    break
        state = {}
        for key, value in payload.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('backbone.'):
                key = key[len('backbone.'):]
            state[key] = value
        try:
            model.backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            backbone_state = model.backbone.state_dict()
            filtered = {
                key: value for key, value in state.items()
                if key in backbone_state
                and hasattr(value, 'shape')
                and value.shape == backbone_state[key].shape
            }
            model.backbone.load_state_dict(filtered, strict=False)
            print(
                'WARNING: DconnNet ResNet-34 checkpoint did not match '
                f'strictly ({exc}); loaded {len(filtered)} matching keys')
        else:
            print(f'DconnNet: loaded ResNet-34 backbone from {path}')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        total = sum(p.numel() for p in model.parameters())
        print(f'DconnNet parameters: {total:,} total, {total:,} trainable')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
        )

    def make_scheduler(self, optimizer, args):
        epochs = max(1, int(getattr(args, 'epochs', 800)))

        def factor(epoch: int) -> float:
            return max(0.0, 1.0 - epoch / epochs) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ------------------------------------------------------------------
    # Official CHASE preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _resize(image: np.ndarray, mask: np.ndarray | None, size: int):
        image = cv2.resize(
            image, (size, size), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(
            mask, (size, size), interpolation=cv2.INTER_LINEAR)
        return image, (mask > 127).astype(np.uint8) * 255

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray,
                 rng: random.Random):
        # randomHueSaturationValue, u=0.5
        if rng.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            hue = np.mod(
                hue.astype(np.int16) + rng.randint(-30, 30), 180).astype(
                    np.uint8)
            saturation = np.clip(
                saturation.astype(np.int16) + rng.uniform(-5, 5),
                0, 255).astype(np.uint8)
            value = np.clip(
                value.astype(np.int16) + rng.uniform(-15, 15),
                0, 255).astype(np.uint8)
            image = cv2.cvtColor(
                cv2.merge((hue, saturation, value)),
                cv2.COLOR_HSV2BGR)

        # randomShiftScaleRotate, u=0.5 (official angle limit is 0)
        if rng.random() < 0.5:
            height, width = image.shape[:2]
            angle = 0.0
            scale = rng.uniform(0.9, 1.1)
            aspect = rng.uniform(0.9, 1.1)
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(rng.uniform(-0.1, 0.1) * width)
            dy = round(rng.uniform(-0.1, 0.1) * height)
            cc = np.cos(np.deg2rad(angle)) * sx
            ss = np.sin(np.deg2rad(angle)) * sy
            rotation = np.array([[cc, -ss], [ss, cc]])
            box0 = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]],
                dtype=np.float32)
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotation.T) + np.array(
                [width / 2 + dx, height / 2 + dy])
            matrix = cv2.getPerspectiveTransform(
                box0, box1.astype(np.float32))
            image = cv2.warpPerspective(
                image, matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(
                mask, matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if rng.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if rng.random() < 0.5:
            image = np.rot90(image)
            mask = np.rot90(mask)
        return image, mask

    def _load_pair(self, sample: Sample, train: bool,
                   rng: random.Random | None = None):
        # The release uses cv2.imread (BGR); load_rgb returns RGB.
        image = np.asarray(load_rgb(sample.image_path))[:, :, ::-1].copy()
        mask = load_mask(sample.mask_path).astype(np.uint8) * 255
        image, mask = self._resize(image, mask, self.input_size)
        if train and not bool(getattr(
                self.args, 'dconnnet_no_augment', False)):
            image, mask = self._augment(image, mask, rng or random.Random(0))
        mask = (mask > 127).astype(np.uint8) * 255
        image_t = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
                * 3.2 - 1.6))
        mask_t = torch.from_numpy(
            np.ascontiguousarray(mask.astype(np.float32)[None] / 255.0))
        return image_t, mask_t

    def prepare_train_batch(self, samples: Sequence[Sample], rng):
        rng = rng or random.Random(0)
        images, masks = [], []
        for sample in samples:
            image, mask = self._load_pair(sample, True, rng)
            images.append(image)
            masks.append(mask)
        return {
            'image': torch.stack(images).to(self.device),
            'gt': torch.stack(masks).to(self.device),
        }

    def train_step(self, model, batch):
        output, auxiliary = model(batch['image'])
        loss_main = self._criterion(output, batch['gt'])
        loss_aux = self._criterion(auxiliary, batch['gt'])
        return loss_main + 0.3 * loss_aux

    # ------------------------------------------------------------------
    # Evaluation / prediction
    # ------------------------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        image = np.asarray(load_rgb(sample.image_path))[:, :, ::-1].copy()
        image = cv2.resize(
            image, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR)
        image_t = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
                * 3.2 - 1.6))
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': image_t.to(device),
            'size': (int(height), int(width)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        model.eval()
        image = prepared['image'].unsqueeze(0)
        output, _ = model(image)
        probability = torch.sigmoid(output)
        batch, _, row, column = probability.shape
        class_pred = probability.view(batch, 1, 8, row, column)
        hard = torch.where(
            class_pred > 0.5,
            torch.ones_like(class_pred), torch.zeros_like(class_pred))
        hori = self._hori_translation.repeat(batch, 1, 1, 1)
        verti = self._verti_translation.repeat(batch, 1, 1, 1)
        vote, _ = Bilateral_voting(hard, hori, verti)
        height, width = prepared['size']
        vote = F.interpolate(
            vote.float(), size=(height, width), mode='nearest')[0, 0]
        return vote.cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint / config
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'backbone_ckpt': str(getattr(
                args, 'dconnnet_backbone_ckpt', '')),
            'augment': not bool(getattr(
                args, 'dconnnet_no_augment', False)),
            'epochs': int(getattr(args, 'epochs', 800)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'backbone_ckpt' in config:
            args.dconnnet_backbone_ckpt = str(config['backbone_ckpt'])
        if config.get('augment') is False:
            args.dconnnet_no_augment = True
