"""BCU-Net paper-faithful port: ConvNeXt + U-Net with MRL.

Official reference: https://github.com/ZX-ECJTU/BCU-Net (and the local
``3rd/BCU-Net`` snapshot).  The paper's main method uses the multilabel
recall loss (MRL) and average fusion of the two branches.  The released
``main.py`` instead uses CE and a 0.4/0.6 fusion; those remain available as
explicit ablations via ``--bcunet-loss ce`` and fusion-weight flags.
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

from baseline_zoo.models.bcunet import (
    BCUNet,
    BCUNetTrainDataset,
    RecallCrossEntropy,
)
from baseline_zoo.runtime import BaselineAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]


class BCUNetAdapter(BaselineAdapter):
    id = 'bcunet'
    display_name = 'BCUNet'
    official_source = 'ZX-ECJTU/BCU-Net (self-contained copy)'
    official_entrypoint = 'https://github.com/ZX-ECJTU/BCU-Net'
    input_size = 512
    official_lr = 1.5e-4
    official_weight_decay = 0.0
    train_batch_size = 2
    accumulation_steps = 1
    selection = 'dice'
    deviations = (
        'Paper profile: multilabel recall loss (MRL). L1 supervises the '
        'ConvNeXt output, L3 the U-Net output, and L2 the average-fused '
        'output; total loss is L1+L2+L3. Eq.2 and Fig.2 of the paper are '
        'not fully consistent; this port follows Fig.2/Fig.8: '
        'fused = 0.5*U-Net + 0.5*ConvNeXt and L2 = RecallCE(fused).',
        'The paper reports BCU-Net (CE) and BCU-Net (RE/MRL); the release '
        'README points to main.py, which uses CE and comments out the '
        'RecallCrossEntropy. This port uses the paper MRL by default; '
        '--bcunet-loss ce reproduces the paper/ release CE ablation.',
        'Paper augmentation is reconstructed: each training image is '
        'expanded into 18 deterministic offline variants including the '
        'identity (rotation, flips, brightness, Gaussian noise, Gaussian '
        'blur, and combinations). The paper does not state whether the '
        'original counts inside the 18; this port uses 18 total. The '
        'paper scopes the transforms to retinal vessel datasets; the '
        'default --bcunet-augment-scope retinal applies them only to '
        'CHASEDB1/DRIVE/STARE/DRHAGIS/HRF/LES-AV and repeats other domains '
        'with identity to preserve the pooled 18/6 balance. '
        '--bcunet-augment-scope all is a task adaptation. The paper does '
        'not state amplitudes or probabilities; the exact recipe is '
        'documented in models/bcunet/augmentation.py and is a '
        'reconstruction, not an official parameter set.',
        'Paper-code architecture conflicts: the paper text says the '
        'ConvNeXt branch has three upsampling blocks, but the release has '
        'four (the final one has no skip); the port follows the release '
        'so 32x32 recovers to 512x512. The paper text says one BN/ReLU in '
        'a U-Net block, while Fig.2 and the code have two; the port '
        'follows Fig.2/code. Algorithm 1 suggests training the branches '
        'before bridging, but main.py jointly optimizes L1/L2/L3 from the '
        'first batch and gives no stage schedule; the port follows the '
        'joint code path.',
        'U-Net upsampling is bilinear as described in the paper; the release '
        'main.py uses ConvTranspose2d (bilinear=False). Pass '
        '--bcunet-convtranspose for the release variant.',
        'Both branch outputs are raw 2-class logits; the release U-Net '
        'OutConv applied Sigmoid but is unrunnable (undefined upconv2x2 and '
        'missing warmlearnrate.py). The paper MRL uses softmax/RecallCE, so '
        'this port keeps logits and a 1x1 projection reconstruction.',
        'The paper uses 100 epochs, batch 2 and Adam lr 1.5e-4. The shared '
        '3-shot protocol still replaces the paper data split with 18 train / '
        '6 val images and selects the checkpoint by validation Dice; the '
        'paper reports five-fold cross-validation and test metrics.',
        'No official BCU-Net trained weights are released or required; only '
        'the ConvNeXt ImageNet-22k backbone is initialized locally. '
        'Preprocessing remains the official BGR [0,255] read + INTER_AREA '
        'resize with no normalization and no extra test-time preprocessing.',
        'Official main.py uses nn.DataParallel on four GPUs, workers=8 and an '
        'unseeded global RNG; this port uses one GPU per job, workers=0 and '
        'the shared protocol seed, so floating-point accumulation and data '
        'order are not bitwise identical.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--no-pretrained', dest='pretrained', action='store_false',
            help='do not load ImageNet-22k ConvNeXt weights')
        parser.add_argument(
            '--convnext-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'convnext_base_22k_224.pth',
            help='local ImageNet-22k ConvNeXt initialization')
        parser.add_argument(
            '--bcunet-loss', choices=['mrl', 'ce'], default='mrl',
            help='paper MRL (default) or the release/CE ablation')
        parser.add_argument(
            '--bcunet-augment-variants', type=int, default=18,
            help='offline variants per training image (paper 18; 1 disables)')
        parser.add_argument(
            '--bcunet-augment-scope', choices=['retinal', 'all'],
            default='retinal',
            help='paper scope is retinal vessel datasets; all is a task '
                 'adaptation for the pooled 20-domain protocol')
        parser.add_argument(
            '--bcunet-fusion-unet', type=float, default=0.5,
            help='fused-output weight for the U-Net branch (paper 0.5)')
        parser.add_argument(
            '--bcunet-fusion-convnext', type=float, default=0.5,
            help='fused-output weight for the ConvNeXt branch (paper 0.5)')
        parser.add_argument(
            '--bcunet-convtranspose', action='store_true',
            help='use the release ConvTranspose2d U-Net upsampling')

    def _fusion_weights(self, args):
        unet = float(getattr(args, 'bcunet_fusion_unet', 0.5))
        convnext = float(getattr(args, 'bcunet_fusion_convnext', 0.5))
        if unet < 0 or convnext < 0 or unet + convnext <= 0:
            raise SystemExit(
                'BCU-Net fusion weights must be non-negative and non-zero')
        return unet, convnext

    def build_model(self, args) -> nn.Module:
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict'
            or getattr(args, 'checkpoint', None) is not None)
        load_pretrained = (
            getattr(args, 'pretrained', True) and not predicting)
        unet_weight, convnext_weight = self._fusion_weights(args)
        model = BCUNet(
            n_channels=3, n_classes=2,
            bilinear=not bool(getattr(args, 'bcunet_convtranspose', False)),
            pretrained=False, in_22k=True,
            fusion_unet=unet_weight, fusion_convnext=convnext_weight,
        )
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

    def make_train_dataset(self, samples, args):
        variants = int(getattr(args, 'bcunet_augment_variants', 18))
        if variants <= 1:
            return None
        return BCUNetTrainDataset(
            samples, input_size=self.input_size, variants=variants,
            seed=int(getattr(args, 'seed', 0)),
            augment_scope=str(getattr(
                args, 'bcunet_augment_scope', 'retinal')))

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
        del rng
        if samples and isinstance(samples[0], tuple):
            images = torch.stack([item[0] for item in samples])
            masks = torch.stack([item[1] for item in samples])
            return images.to(self.device), masks.to(self.device)
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
        if getattr(self.args, 'bcunet_loss', 'mrl') == 'mrl':
            criterion = RecallCrossEntropy(n_classes=2).to(images.device)
            # Paper: L1 ConvNeXt, L3 U-Net, L2 average-fused.
            return (criterion(out2, masks) + criterion(fused, masks)
                    + criterion(out1, masks))
        criterion = nn.CrossEntropyLoss()
        return (criterion(out1, masks) + criterion(out2, masks)
                + criterion(fused, masks))

    def prepare_eval_sample(self, sample, device):
        image = self._load_image(sample)
        tensor = torch.from_numpy(
            np.transpose(image, [2, 0, 1]).copy()).float()
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        with torch.no_grad():
            _, _, fused = model(prepared)
        # Upstream upsamples soft logits to original size and takes argmax
        # there; nearest-resizing a 512 hard mask changes boundary pixels.
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        fused = F.interpolate(fused, size=(height, width),
                              mode='bilinear', align_corners=False)
        return fused.argmax(dim=1)[0].cpu().numpy().astype(bool)

    def save_config(self, args):
        unet_weight, convnext_weight = self._fusion_weights(args)
        return {
            'input_size': int(self.input_size),
            'loss': str(getattr(args, 'bcunet_loss', 'mrl')),
            'augment_variants': int(
                getattr(args, 'bcunet_augment_variants', 18)),
            'augment_scope': str(
                getattr(args, 'bcunet_augment_scope', 'retinal')),
            'fusion_unet': unet_weight,
            'fusion_convnext': convnext_weight,
            'convtranspose': bool(
                getattr(args, 'bcunet_convtranspose', False)),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
        for key, attribute in (
                ('loss', 'bcunet_loss'),
                ('augment_variants', 'bcunet_augment_variants'),
                ('augment_scope', 'bcunet_augment_scope'),
                ('fusion_unet', 'bcunet_fusion_unet'),
                ('fusion_convnext', 'bcunet_fusion_convnext'),
                ('convtranspose', 'bcunet_convtranspose')):
            if key in config:
                setattr(args, attribute, config[key])
