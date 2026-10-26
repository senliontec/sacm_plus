"""BCUNet baseline with the official ConvNeXt+U-Net training logic.

Official reference: vendored ``baselines/bcunet`` (upstream URL is not
recorded there).  The model is copied into ``baseline_zoo.models.bcunet``
and the undefined ``OutConv.upconv2x2`` is repaired as a 1x1 logits
projection; everything else follows the official training script.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.models.bcunet import BCUNet
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class BCUNetAdapter(BaselineAdapter):
    id = 'bcunet'
    display_name = 'BCUNet'
    official_source = 'BCU-Net (self-contained copy)'
    official_entrypoint = 'upstream code/main.py (vendored snapshot)'
    input_size = 512
    official_epochs = 50
    official_lr = 1.5e-4
    official_weight_decay = 0.0
    train_batch_size = 1
    accumulation_steps = 2
    selection = 'dice'
    deviations = (
        'Upstream batch size 2 does not fit an 11 GiB card at 512x512; the '
        'adapter uses batch 1 with 2-step gradient accumulation. This '
        'preserves gradient sums but not batch-2 BatchNorm statistics. '
        'On >=24 GiB pass --batch-size 2 --accum-steps 1.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream has no checkpoint selection (it saves every 10 epochs); '
        'the port adds val-Dice selection for a common protocol.',
        'Upstream preprocessing is BGR [0,255] with no normalization '
        '(cv2.imread + INTER_AREA); preserved exactly.',
        'Upstream OutConv referenced an undefined upconv2x2; repaired as a '
        '1x1 logits projection at full resolution. Upstream main.py also '
        'imports a missing warmlearnrate.py and cannot run; this adapter '
        'supplies the official loss/optimizer loop instead.',
        'The official Adam 1.5e-4, no scheduler and no augmentation are '
        'preserved. The official 50-epoch budget is replaced by 200 epochs '
        'for the 18-image 3-shot protocol.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--no-pretrained', dest='pretrained', action='store_false',
            help='do not load ImageNet-22k ConvNeXt weights')
        parser.add_argument(
            '--convnext-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'convnext_base_22k_224.pth',
            help='local ImageNet-22k ConvNeXt initialization')

    def build_model(self, args) -> nn.Module:
        # Prediction loads a complete self-contained checkpoint next; do
        # not load ImageNet-22k backbone weights only to overwrite them.
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict'
            or getattr(args, 'checkpoint', None) is not None)
        load_pretrained = (
            getattr(args, 'pretrained', True) and not predicting)
        # Build without torch.hub: the official ConvNeXt URL/cache is not
        # used.  Training loads the local checkpoint below; prediction uses
        # the full baseline checkpoint and must not touch the backbone cache.
        model = BCUNet(n_channels=3, n_classes=2, bilinear=False,
                       pretrained=False, in_22k=True)
        if load_pretrained:
            self._load_local_convnext(model, args)
        return model

    def _load_local_convnext(self, model: nn.Module, args) -> None:
        checkpoint_path = Path(getattr(
            args, 'convnext_checkpoint',
            REPO_ROOT / 'checkpoints' / 'convnext_base_22k_224.pth'))
        if not checkpoint_path.is_file():
            raise SystemExit(
                f'missing ConvNeXt checkpoint: {checkpoint_path}\n'
                'download: https://dl.fbaipublicfiles.com/convnext/'
                'convnext_base_22k_224.pth\n'
                'put it in checkpoints/ or pass --convnext-checkpoint')
        payload = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        state = payload.get('model', payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise SystemExit(
                f'unexpected ConvNeXt checkpoint format: {checkpoint_path}')
        target = model.backbone.state_dict()
        matched = {
            key: value for key, value in state.items()
            if key in target and getattr(value, 'shape', None) == target[key].shape
        }
        if not matched:
            raise SystemExit(
                f'ConvNeXt checkpoint has no matching backbone tensors: '
                f'{checkpoint_path}')
        missing = [key for key in target if key not in matched]
        target.update(matched)
        # Match the official loader: it filters to shape-compatible keys and
        # leaves the small number of absent norm layers at their initialization.
        model.backbone.load_state_dict(target, strict=True)
        print(f'loaded ConvNeXt initialization: {checkpoint_path} '
              f'({len(matched)} matched, {len(missing)} left initialized)')

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def _load_image(self, sample):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise SystemExit(f'cv2 could not read {sample.image_path}')
        return cv2.resize(image, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_AREA)

    def _load_mask(self, sample):
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f'cv2 could not read {sample.mask_path}')
        # Upstream thresholds at original resolution, then INTER_AREA
        # resizes the 0/1 label.
        mask = (mask >= 128).astype(np.uint8)
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_AREA)
        return mask.astype(np.int64)

    def prepare_train_batch(self, samples, rng: random.Random):
        images = []
        masks = []
        for sample in samples:
            image = self._load_image(sample)
            images.append(torch.from_numpy(
                np.transpose(image, [2, 0, 1]).copy()).float())
            masks.append(torch.from_numpy(self._load_mask(sample)))
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        out1, out2, fused = model(images)
        criterion = nn.CrossEntropyLoss()
        return (criterion(fused, masks) + criterion(out1, masks) +
                criterion(out2, masks))

    def prepare_eval_sample(self, sample, device):
        image = self._load_image(sample)
        tensor = torch.from_numpy(
            np.transpose(image, [2, 0, 1]).copy()).float()
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        _, _, fused = model(prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        # Upstream upsamples the soft logits to original size and takes
        # argmax there; nearest-resizing a 512 hard mask changes boundary
        # pixels.
        fused = F.interpolate(fused, size=(height, width),
                              mode='bilinear', align_corners=False)
        return fused.argmax(dim=1)[0].cpu().numpy().astype(bool)
