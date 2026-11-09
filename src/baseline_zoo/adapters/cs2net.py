"""CS2Net-2D baseline with the official 2D CS-Net/CS2-Net logic.

Official reference: https://github.com/iMED-Lab/CS-Net, vendored from
``3rd/CS-Net``.  The 2D network is ``model/csnet.py`` and the official 2D
entrypoint is ``train.py``.  ``model/csnet_3d.py``/``train3d.py`` are the
separate 3D CS2-Net extension and cannot consume 2D protocol images.
The legacy model id ``cs2net`` is retained.

Protocol-only deviations are listed in ``deviations``; every model-side
detail (geometry, augmentation, loss, optimizer, LR schedule, Otsu output)
follows the upstream code.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.cs2net import CSNet
from baseline_zoo.runtime import BaselineAdapter

DICE_EPS = 1e-4


def _center_square(image: Image.Image, mask: Image.Image, size: int):
    """Upstream ``Data.rescale``: centre-crop the min-side square, resize."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    box = (left, top, left + side, top + side)
    return (image.crop(box).resize((size, size)),
            mask.crop(box).resize((size, size)))


def _restore_center_square(pred: np.ndarray, width: int, height: int):
    """Place a 512 prediction back into the original centre square."""
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    resized = Image.fromarray((pred.astype(np.uint8) * 255)).resize(
        (side, side), Image.NEAREST)
    out = np.zeros((height, width), dtype=bool)
    out[top:top + side, left:left + side] = np.asarray(resized) > 127
    return out


CS2NET_MULTIPLE = 16


def _full_meta(width: int, height: int, max_side: int):
    """Geometry metadata for full-resolution, multiple-of-16 padded input."""
    scale = 1.0
    if max_side and max(width, height) > max_side:
        scale = float(max_side) / float(max(width, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return {
        'width': int(width),
        'height': int(height),
        'new_w': new_w,
        'new_h': new_h,
        'pad_w': (-new_w) % CS2NET_MULTIPLE,
        'pad_h': (-new_h) % CS2NET_MULTIPLE,
        'scale': scale,
    }


def _full_transform(image: Image.Image, mask: Image.Image, max_side: int):
    """Keep native resolution (capped at max_side), pad to /16, no crop."""
    meta = _full_meta(image.size[0], image.size[1], max_side)
    image = image.resize((meta['new_w'], meta['new_h']), Image.BILINEAR)
    mask = mask.resize((meta['new_w'], meta['new_h']), Image.NEAREST)
    if meta['pad_w'] or meta['pad_h']:
        size = (meta['new_w'] + meta['pad_w'],
                meta['new_h'] + meta['pad_h'])
        canvas = Image.new('RGB', size, (0, 0, 0))
        canvas.paste(image, (0, 0))
        image = canvas
        mask_canvas = Image.new('L', size, 0)
        mask_canvas.paste(mask, (0, 0))
        mask = mask_canvas
    return image, mask, meta


def _restore_full(pred: np.ndarray, meta: dict):
    """Crop the /16 padding and restore the original image size."""
    cropped = pred[:meta['new_h'], :meta['new_w']]
    resized = Image.fromarray((cropped.astype(np.uint8) * 255)).resize(
        (meta['width'], meta['height']), Image.NEAREST)
    return np.asarray(resized) > 127


def _replace_batchnorm(module: nn.Module, kind: str) -> nn.Module:
    """Task adaptation: per-sample norm for pooled multi-domain 3-shot data."""
    if kind != 'instance':
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(
                child.num_features, affine=True,
                track_running_stats=False))
        else:
            _replace_batchnorm(child, kind)
    return module


def _pad_square(image: Image.Image, mask: Image.Image, size: int):
    """Aspect-preserving pad to a square, then resize to ``size``.

    Unlike the official centre crop, this keeps the whole field of view.
    The extra canvas is black/background and is removed after prediction.
    """
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(image, (left, top))
    mask_canvas = Image.new('L', (side, side), 0)
    mask_canvas.paste(mask, (left, top))
    return (canvas.resize((size, size)),
            mask_canvas.resize((size, size)))


def _restore_padded_square(pred: np.ndarray, width: int, height: int):
    """Remove the square-pad canvas and restore the original aspect ratio."""
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    resized = Image.fromarray((pred.astype(np.uint8) * 255)).resize(
        (side, side), Image.NEAREST)
    out = np.asarray(resized) > 127
    return out[top:top + height, left:left + width]


def _official_dice_loss(pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Upstream ``dice_coeff_loss`` (per-image Dice, eps 1e-4)."""
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + DICE_EPS
    dice = (2 * inter + DICE_EPS) / union
    return (1 - dice).mean()


class CS2NetAdapter(BaselineAdapter):
    id = 'cs2net'
    display_name = 'CS2Net-2D'
    official_source = 'iMED-Lab/CS-Net, model/csnet.py (2D)'
    official_entrypoint = 'https://github.com/iMED-Lab/CS-Net'
    input_size = 512
    official_lr = 1e-4
    official_weight_decay = 5e-4
    train_batch_size = 8
    selection = 'dice'
    deviations = (
        'This is the 2D CS2Net/CS-Net implementation from the official '
        'repository: model/csnet.py and train.py. The 3D extension '
        'model/csnet_3d.py/train3d.py uses Conv3d and is not applicable to '
        'the 2D protocol images; SACM\'s 2D datasets therefore require this '
        '2D branch unless a private depth-1 adaptation is documented.',
        'Data protocol: the 18/6-shot split replaces the original DRIVE '
        'training/test folders; all 20 mixed-modality domains are converted '
        'to RGB because the official loader assumes 3-channel fundus images.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream evaluates the test split and its best-checkpoint branch is '
        'unreachable, so it never writes a best model; using test here would '
        'violate the protocol.',
        'The official predict.py depends on an undefined center_crop and is '
        'not runnable. The default task adaptation uses aspect-preserving '
        'black padding to keep the full field of view (--cs2net-geometry '
        'padded); --cs2net-geometry center restores the official data-loader '
        'min-side centre crop, which zeroes everything outside the centre '
        'square.',
        'Official whole-model pickle checkpoints are replaced by the unified '
        'state_dict schema so every baseline shares one evaluator.',
        'Per-epoch validation runs once per epoch (upstream also tests every '
        'epoch but returns after the first image).',
        'Official train.py uses nn.DataParallel on two GPUs, DataLoader workers=2 '
        'and an unseeded global RNG; this port runs one GPU, workers=0 and a '
        'seeded runtime RNG. Training trajectories are not bitwise identical.',
        'Task adaptation (default with the shipped config): full native '
        'resolution capped at --cs2net-max-side, multiple-of-16 padding, '
        'InstanceNorm instead of BatchNorm, batch 1, mild flips/rot90/'
        'photometric augmentation, Adam 3e-4/1e-4 with cosine+warmup, and a '
        'fixed 0.5 threshold. These changes target the pooled 18-image '
        'six-domain protocol; the official 512 center-crop, BatchNorm, '
        'official augmentation and Otsu recipe remains selectable with '
        '--cs2net-geometry center --cs2net-norm batch '
        '--cs2net-augment official --cs2net-threshold otsu.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--cs2net-geometry', choices=['padded', 'center', 'full'],
            default='padded',
            help='padded preserves the full field of view with '
                 'aspect-preserving black padding; center reproduces the '
                 'official min-side centre crop; full keeps native resolution '
                 '(capped at --cs2net-max-side) and pads to a multiple of 16')
        parser.add_argument(
            '--cs2net-max-side', type=int, default=1024,
            help='long-side cap for --cs2net-geometry full (0 = native)')
        parser.add_argument(
            '--cs2net-norm', choices=['batch', 'instance'],
            default='batch',
            help='batch = official BatchNorm; instance = task-adapted '
                 'InstanceNorm for pooled multi-domain data')
        parser.add_argument(
            '--cs2net-augment', choices=['official', 'mild'],
            default='official',
            help='official = upstream +-40deg/ImageEnhance/crop/flip; mild = '
                 'flips, 90-degree rotations, mild brightness/contrast')
        parser.add_argument(
            '--cs2net-threshold', choices=['otsu', 'fixed'],
            default='otsu',
            help='otsu = official test threshold; fixed = probability 0.5')
        parser.add_argument(
            '--cs2net-threshold-value', type=float, default=0.5)

    def save_config(self, args):
        return {
            'cs2net_geometry': getattr(args, 'cs2net_geometry', 'padded'),
            'cs2net_max_side': int(getattr(args, 'cs2net_max_side', 1024)),
            'cs2net_norm': getattr(args, 'cs2net_norm', 'batch'),
            'cs2net_augment': getattr(args, 'cs2net_augment', 'official'),
            'cs2net_threshold': getattr(
                args, 'cs2net_threshold', 'otsu'),
            'cs2net_threshold_value': float(getattr(
                args, 'cs2net_threshold_value', 0.5)),
        }

    def restore_config(self, args, config):
        for key in ('cs2net_geometry', 'cs2net_max_side', 'cs2net_norm',
                    'cs2net_augment', 'cs2net_threshold',
                    'cs2net_threshold_value'):
            if key in config:
                setattr(args, key, config[key])

    def _geometry_mode(self) -> str:
        args = getattr(self, 'args', None)
        return str(getattr(args, 'cs2net_geometry', 'padded'))

    def _augment_mode(self) -> str:
        args = getattr(self, 'args', None)
        return str(getattr(args, 'cs2net_augment', 'official'))

    def _threshold_mode(self) -> str:
        args = getattr(self, 'args', None)
        return str(getattr(args, 'cs2net_threshold', 'otsu'))

    def _max_side(self) -> int:
        args = getattr(self, 'args', None)
        return int(getattr(args, 'cs2net_max_side', 1024))

    def build_model(self, args) -> nn.Module:
        model = CSNet(classes=1, channels=3)
        norm = str(getattr(args, 'cs2net_norm', 'batch'))
        return _replace_batchnorm(model, norm)

    def make_optimizer(self, model, args):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        max_epochs = max(args.epochs, 1)

        def lr_lambda(epoch):
            # Upstream calls adjust_lr(iter=epoch) *after* each 0-based epoch
            # and its first call (iter=0) is a no-op.  LambdaLR applies
            # lambda(0) at construction, so shift by one to avoid decaying
            # one epoch early.
            return (1.0 - max(epoch - 1, 0) / max_epochs) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

    @staticmethod
    def _enhance(image: Image.Image, rng: random.Random) -> Image.Image:
        value = rng.uniform(-2, 2)
        choice = rng.randint(1, 4)
        enhancer = {
            1: ImageEnhance.Brightness,
            2: ImageEnhance.Color,
            3: ImageEnhance.Contrast,
            4: ImageEnhance.Sharpness,
        }[choice]
        return enhancer(image).enhance(value)

    @staticmethod
    def _mild_augment(image: Image.Image, mask: Image.Image,
                      rng: random.Random):
        """Task-adapted augmentation: geometry-preserving, no negative gain."""
        if rng.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        rotation = rng.choice((0, 90, 180, 270))
        if rotation:
            op = {
                90: Image.ROTATE_90,
                180: Image.ROTATE_180,
                270: Image.ROTATE_270,
            }[rotation]
            image = image.transpose(op)
            mask = mask.transpose(op)
        if rng.random() > 0.5:
            factor = rng.uniform(0.8, 1.2)
            enhancer = rng.choice(
                (ImageEnhance.Brightness, ImageEnhance.Contrast))
            image = enhancer(image).enhance(factor)
        return image, mask

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        geometry = self._geometry_mode()
        for sample in samples:
            image = load_rgb(sample.image_path)
            mask = Image.open(sample.mask_path).convert('L')
            if geometry == 'full':
                image, mask, _ = _full_transform(
                    image, mask, self._max_side())
            elif geometry == 'padded':
                image, mask = _pad_square(image, mask, self.input_size)
            else:
                image, mask = _center_square(image, mask, self.input_size)

            if self._augment_mode() == 'official' and geometry == 'full':
                raise SystemExit(
                    'cs2net: official augmentation requires center/padded '
                    'geometry; use --cs2net-augment mild for full geometry')
            if self._augment_mode() == 'official' and geometry != 'full':
                angle = rng.randint(-40, 40)
                image = image.rotate(angle)
                mask = mask.rotate(angle)
                if rng.random() > 0.5:
                    image = self._enhance(image, rng)

                # Upstream RandomCrop(512,512) after the 512 resize: a no-op
                # kept for exact sequencing of the random stream.
                side = self.input_size
                width, height = image.size
                left = rng.randint(0, width - side)
                top = rng.randint(0, height - side)
                image = image.crop((left, top, left + side, top + side))
                mask = mask.crop((left, top, left + side, top + side))

                if rng.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image, mask = self._mild_augment(image, mask, rng)

            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask_np = np.asarray(mask, dtype=np.float32).copy()
            if mask_np.size and float(mask_np.max()) <= 1.0:
                mask_np *= 255.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0) / 255.0
            images.append(image_t)
            masks.append(mask_t)
        shapes = {tuple(item.shape) for item in images}
        if geometry == 'full' and len(shapes) > 1:
            raise SystemExit(
                'cs2net full geometry produces variable shapes; use '
                '--batch-size 1')
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        pred = model(images)
        return (nn.MSELoss()(pred, masks) +
                _official_dice_loss(pred, masks))

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path)
        mask = Image.open(sample.mask_path).convert('L')
        geometry = self._geometry_mode()
        if geometry == 'full':
            image, _, _ = _full_transform(image, mask, self._max_side())
        elif geometry == 'padded':
            image, _ = _pad_square(image, mask, self.input_size)
        else:
            image, _ = _center_square(image, mask, self.input_size)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        pred = model(prepared)[0, 0].detach().cpu().numpy()
        if self._threshold_mode() == 'fixed':
            value = float(getattr(self.args, 'cs2net_threshold_value', 0.5))
            binary = pred > value
        else:
            arr = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            if arr.min() == arr.max():
                threshold = 0.0  # cv2 THRESH_OTSU returns 0 for constants
            else:
                threshold = float(threshold_otsu(arr))
            binary = arr > threshold
        # Use the same image source as the forward transform; mask/image
        # sizes can differ in external test roots.
        with Image.open(sample.image_path) as image:
            width, height = image.size
        geometry = self._geometry_mode()
        if geometry == 'full':
            return _restore_full(
                binary, _full_meta(width, height, self._max_side()))
        if geometry == 'padded':
            return _restore_padded_square(binary, width, height)
        return _restore_center_square(binary, width, height)
