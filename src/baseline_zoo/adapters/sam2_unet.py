"""SAM2-UNet appendix baseline (self-contained SAM2 Hiera-L + U-Net decoder).

The architecture is vendored from ``WZH0120/SAM2-UNet`` under
``baseline_zoo.models.sam2_unet``.  Training uses the shared
``baseline_zoo.runtime`` loop; only the official preprocessing, loss,
optimizer recipe and inference postprocessing live here.

This model is an appendix/extended baseline, not one of the SACM paper's ten
baselines.  Real training requires ``checkpoints/sam2_hiera_large.pt``;
``--random-init`` exists only for code smoke tests.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from baseline_zoo.data import load_rgb
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
SAM2_URL = ("https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_large.pt")


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upstream weighted BCE + weighted IoU.

    ``train.py`` in the upstream snapshot used the removed ``reduce='none'``
    keyword; ``reduction='none'`` is the modern equivalent and is what the
    deleted legacy adapter used.
    """
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2, 3))
    union = ((prob + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class SAM2UNetAdapter(BaselineAdapter):
    id = 'sam2_unet'
    display_name = 'SAM2-UNet'
    official_source = 'WZH0120/SAM2-UNet (self-contained copy, Apache-2.0)'
    official_entrypoint = 'https://github.com/WZH0120/SAM2-UNet'
    input_size = 352
    official_epochs = 200  # project protocol budget; upstream uses 20
    official_lr = 1e-3
    official_weight_decay = 5e-4
    train_batch_size = 4  # upstream 12; Hiera-L memory
    selection = 'dice'
    deviations = (
        'Appendix extended baseline, not one of the paper ten and not in '
        'the main table.',
        'Upstream trains 20 epochs, batch 12, with no validation; this port '
        'uses the shared 3-shot protocol (200 epochs), batch 4 for Hiera-L '
        'memory, and selects the best checkpoint on the 6-image val split '
        'by Dice.',
        'Upstream train.py never calls model.train(); the shared runtime '
        'does. The vendored Hiera trunk has no BatchNorm, but the RFB/U-Net '
        'decoder BatchNorm follows the runtime train/eval mode.',
        'Official AdamW receives all parameters; this port optimizes only '
        'requires_grad parameters (prompt-learn MLPs, RFB, decoder). Frozen '
        'Hiera parameters receive no gradient either way.',
        'Images use bilinear 352 resize and ImageNet normalization, matching '
        'upstream. Masks use bicubic resize and remain soft in [0,1], also '
        'matching upstream; the evaluator still uses the original binary GT.',
        'Upstream inference writes per-image min-max normalized grayscale '
        'maps and does not binarize; this port bilinearly resizes the main '
        'logits to GT and thresholds sigmoid at 0.5 to satisfy the shared '
        'prediction contract.',
        'Checkpoints store the full SAM2UNet state dict including the frozen '
        'encoder. Prediction therefore does not need the SAM2 pretrained '
        'file; training does unless --random-init (code smoke only).',
        'Fixed 352x352 square input; the Hiera window/positional embedding '
        'and the decoder 16/8/4 upsampling factors follow the upstream 352 '
        'recipe. No aspect-ratio preservation.',
        'Official train.py uses workers=8 and has its seed code commented '
        'out; this port uses workers=0 and the shared protocol seed.',
        'No TTA or sliding-window inference.',
    )

    def __init__(self):
        super().__init__()
        self._sam2_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument(
            '--sam2-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt')
        parser.add_argument('--input-size', type=int, default=352)
        parser.add_argument(
            '--random-init', action='store_true',
            help='code smoke only: build Hiera-L without SAM2 weights')
        parser.add_argument(
            '--allow-random-init', action='store_true',
            help='predict-only: allow a checkpoint trained with --random-init')

    def build_model(self, args):
        from baseline_zoo.models.sam2_unet import SAM2UNet

        if int(args.input_size) != 352:
            raise SystemExit(
                'SAM2-UNet expects --input-size 352; the Hiera-L window/pos '
                'embedding and decoder scales follow the upstream 352 '
                'recipe.')
        self.input_size = int(args.input_size)

        stage = getattr(args, 'stage', 'train')
        training = stage in ('train', 'all')
        random_init = bool(getattr(args, 'random_init', False))
        checkpoint = getattr(args, 'sam2_checkpoint', None)
        checkpoint = Path(checkpoint) if checkpoint else None

        if training and not random_init and (
                checkpoint is None or not checkpoint.is_file()):
            raise SystemExit(
                f'missing SAM2 Hiera-L checkpoint: {checkpoint}\n'
                f'download: curl -L -o {checkpoint} {SAM2_URL}\n'
                'use --random-init only for a code smoke test')

        if random_init:
            print('[sam2_unet] WARNING: --random-init is smoke-only; do not '
                  'report these weights', flush=True)

        source = None
        if checkpoint is not None and checkpoint.is_file():
            source = str(checkpoint)
            self._sam2_sha = file_sha256(checkpoint)
        return SAM2UNet(source, device='cpu')

    def configure_trainable(self, model):
        # Freeze the original Hiera blocks, then re-enable the inserted
        # prompt-learn MLPs.  The RFB/U-Net decoder stays trainable.
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.encoder.named_parameters():
            if 'prompt_learn' in name:
                parameter.requires_grad_(True)
        for module in (model.rfb1, model.rfb2, model.rfb3, model.rfb4,
                       model.up1, model.up2, model.up3, model.up4,
                       model.side1, model.side2, model.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('SAM2-UNet has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)

    def _image_tensor(self, path) -> torch.Tensor:
        image = load_rgb(path)
        image = TF.resize(
            image, [self.input_size, self.input_size],
            interpolation=InterpolationMode.BILINEAR)
        return TF.normalize(TF.to_tensor(image), IMAGE_MEAN, IMAGE_STD)

    def _mask_tensor(self, path) -> torch.Tensor:
        with Image.open(path) as image:
            mask = image.convert('L')
        mask = TF.resize(
            mask, [self.input_size, self.input_size],
            interpolation=InterpolationMode.BICUBIC)
        return TF.to_tensor(mask)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = self._image_tensor(sample.image_path)
            mask = self._mask_tensor(sample.mask_path)
            if rng.random() < 0.5:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if rng.random() < 0.5:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        out, out1, out2 = model(images)
        return (structure_loss(out, masks)
                + structure_loss(out1, masks)
                + structure_loss(out2, masks))

    def prepare_eval_sample(self, sample, device):
        return self._image_tensor(sample.image_path).unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        out, _, _ = model(prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        out = F.interpolate(
            out, size=(height, width), mode='bilinear',
            align_corners=False)
        return (torch.sigmoid(out)[0, 0].cpu().numpy() > 0.5)

    def save_config(self, args):
        checkpoint = getattr(args, 'sam2_checkpoint', None)
        return {
            'input_size': int(self.input_size),
            'sam2_checkpoint': str(checkpoint) if checkpoint else None,
            'sam2_checkpoint_sha256': self._sam2_sha,
            'random_init': bool(getattr(args, 'random_init', False)),
        }

    def restore_config(self, args, config):
        size = config.get('input_size')
        if size is not None:
            args.input_size = int(size)
        args.random_init = bool(config.get('random_init', False))
        if args.random_init and not getattr(args, 'allow_random_init', False):
            raise SystemExit(
                'checkpoint was trained with --random-init (smoke only); '
                'pass --allow-random-init to evaluate it anyway')
        stored = config.get('sam2_checkpoint_sha256')
        current = getattr(args, 'sam2_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                raise SystemExit(
                    f'SAM2 checkpoint changed since training: {current}\n'
                    f'stored sha256={stored}, current sha256={actual}')
