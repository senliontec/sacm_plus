"""Unified data loading.

- SegmentationDataset: train/val splits (root/{split}/images + masks),
  optional joint augmentation (verbatim from the original train_sam.py).
- TestDataset: a single images/ + masks/ root, returning items at the
  original resolution for evaluation (verbatim from test_sam.py).
"""

import csv
import logging
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# DIS5K 含超大原图(1 亿+ 像素),关闭 PIL 的"解压炸弹"保护以避免警告刷屏;
# 数据为本项目自管数据集,无恶意输入风险。
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff',
                        '.bmp', '.ppm', '.pgm')

INPUT_MODES = ('resize', 'aspect_pad', 'native_crop')


def letterbox_rect(width, height, size=1024):
    """Return ``(left, top, new_w, new_h)`` for aspect-preserving padding."""
    scale = float(size) / float(max(width, height))
    new_w = max(1, min(size, int(round(width * scale))))
    new_h = max(1, min(size, int(round(height * scale))))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    return left, top, new_w, new_h


def _resize_pad_pair(image, mask, size=1024):
    """Resize the long side to ``size`` and pad to a square.

    Unlike ``transforms.Resize((size, size))`` this preserves the aspect
    ratio, so a 3504x2336 fundus image is not horizontally squashed.  The
    mask uses nearest-neighbour interpolation to keep 1-pixel curvilinear
    structures from disappearing in the resize.
    """
    new_w, new_h = letterbox_rect(
        image.size[0], image.size[1], size)[2:]
    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)

    canvas_image = Image.new('RGB', (size, size), (0, 0, 0))
    canvas_mask = Image.new('L', (size, size), 0)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas_image.paste(image, (left, top))
    canvas_mask.paste(mask, (left, top))
    return canvas_image, canvas_mask


def _random_native_crop(image, mask, size=1024, foreground_prob=0.5,
                        rng=random):
    """Random native-resolution crop, optionally centred on foreground.

    For an image larger than ``size`` this keeps the original pixel scale
    instead of shrinking a 3504px fundus image to 1024px.  The crop is
    resized/padded only when the source is smaller than ``size``.
    """
    width, height = image.size
    if width < size or height < size:
        return _resize_pad_pair(image, mask, size)

    if foreground_prob > 0 and rng.random() < foreground_prob:
        fg = np.asarray(mask) > 0
        ys, xs = np.nonzero(fg)
        if xs.size:
            centre_x = int(xs[rng.randrange(xs.size)])
            centre_y = int(ys[rng.randrange(ys.size)])
            left = int(np.clip(centre_x - size // 2, 0, width - size))
            top = int(np.clip(centre_y - size // 2, 0, height - size))
        else:
            left = rng.randrange(0, width - size + 1)
            top = rng.randrange(0, height - size + 1)
    else:
        left = rng.randrange(0, width - size + 1)
        top = rng.randrange(0, height - size + 1)

    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box)


def _to_tensor_pair(image, mask):
    """Convert PIL image/mask to float tensors in [0, 1]."""
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    return to_tensor(image), to_tensor(mask)


def _prepare_pair(image, mask, input_mode, image_size, foreground_prob,
                  rng=random):
    if input_mode == 'resize':
        return image, mask
    if input_mode == 'aspect_pad':
        return _resize_pad_pair(image, mask, image_size)
    if input_mode == 'native_crop':
        return _random_native_crop(
            image, mask, image_size, foreground_prob, rng)
    raise ValueError(
        f"Unknown input_mode={input_mode!r}; expected one of {INPUT_MODES}")


def _find_pairs(image_dir, mask_dir):
    pairs = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            base_name = os.path.splitext(filename)[0]
            for ext in SUPPORTED_EXTENSIONS:
                if os.path.exists(os.path.join(mask_dir, base_name + ext)):
                    pairs.append(filename)
                    break
    # os.listdir order is filesystem-dependent.  Sorting makes the
    # index -> image mapping identical across machines/copies and is
    # required for reproducible DataLoader shuffling.
    return sorted(pairs)


class SegmentationDataset(Dataset):
    """Train/val split dataset (root/<split>/images + root/<split>/masks).

    ``input_mode`` controls the geometric preprocessing:

    - ``resize``: legacy behaviour; ``transform`` must resize to 1024.
    - ``aspect_pad``: resize the long side to ``image_size`` and pad.
    - ``native_crop``: random native-resolution crop (foreground-biased
      with probability ``foreground_crop_prob``) for large images;
      small images fall back to aspect-preserving resize+pad.

    For the two new modes ``transform`` should be ``ToTensor`` only; the
    dataset performs the spatial operation itself.
    """

    def __init__(self, root_dir, split='train', transform=None, augment=None,
                 input_mode='resize', image_size=1024,
                 foreground_crop_prob=0.5):
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f"Unknown input_mode={input_mode!r}; expected {INPUT_MODES}")
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.foreground_crop_prob = float(foreground_crop_prob)
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        if self.input_mode != 'native_crop' and self.foreground_crop_prob > 0:
            logging.warning(
                "foreground_crop_prob=%s is ignored for input_mode=%s; "
                "use input_mode='native_crop' to enable foreground cropping",
                self.foreground_crop_prob, self.input_mode)

        self.image_files = _find_pairs(self.image_dir, self.mask_dir)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")
        logging.info(
            f"Found {len(self.image_files)} image-mask pairs in {split} set "
            f"(input_mode={self.input_mode}, size={self.image_size})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in SUPPORTED_EXTENSIONS:
            potential = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential):
                mask_path = potential
                break
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.input_mode == 'resize':
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
        else:
            image, mask = _prepare_pair(
                image, mask, self.input_mode, self.image_size,
                self.foreground_crop_prob)
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            else:
                image, mask = _to_tensor_pair(image, mask)

        # Joint augmentation (train only): identical geometric transform
        # applied to image and mask so the annotation stays aligned.
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return image, mask


class TestDataset(Dataset):
    """Evaluation dataset (root/images + root/masks): the model input is
    prepared at 1024x1024, the GT stays at the ORIGINAL resolution for
    metric computation.

    ``split='val'`` reads root/val/images + root/val/masks, which lets
    training validation use the same original-resolution metric path as
    test evaluation instead of 1024-space metrics.

    ``input_mode='aspect_pad'`` preserves the source aspect ratio by
    resizing the long side to ``image_size`` and padding; the legacy
    ``resize`` mode stretches to a square.
    """

    EVAL_INPUT_MODES = ('resize', 'aspect_pad')

    def __init__(self, root_dir, split=None, input_mode='resize',
                 image_size=1024):
        if input_mode not in self.EVAL_INPUT_MODES:
            raise ValueError(
                f"Unknown eval input_mode={input_mode!r}; expected "
                f"{self.EVAL_INPUT_MODES}")
        self.root_dir = root_dir
        self.input_mode = input_mode
        self.image_size = int(image_size)
        base = root_dir if split is None else os.path.join(root_dir, split)
        self.image_dir = os.path.join(base, 'images')
        self.mask_dir = os.path.join(base, 'masks')
        self.fov_dir = os.path.join(base, 'fov')
        self.has_fov = os.path.isdir(self.fov_dir)

        self.image_files = _find_pairs(self.image_dir, self.mask_dir)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")

        # Dataset name per output stem, used by Skein validation to log
        # the same per-dataset W&B keys as the baseline runner.  The
        # split manifest lives at the root (root/split_manifest.csv).
        self.dataset_by_output = {}
        manifest_path = os.path.join(root_dir, 'split_manifest.csv')
        if os.path.isfile(manifest_path):
            with open(manifest_path, newline='') as handle:
                for row in csv.DictReader(handle):
                    if split is not None and row.get('split') != split:
                        continue
                    output = row.get('output', '')
                    dataset = row.get('dataset', '')
                    if output and dataset:
                        self.dataset_by_output[os.path.splitext(output)[0]] = dataset

    def __len__(self):
        return len(self.image_files)

    def dataset_for(self, img_name):
        """Return the dataset that produced an image filename."""
        stem = os.path.splitext(os.path.basename(str(img_name)))[0]
        dataset = self.dataset_by_output.get(stem)
        if dataset:
            return dataset
        # Single-dataset test roots have no manifest; the protocol's
        # filenames are prefixed with the dataset name.
        return stem.split('_')[0] or 'val'

    def __getitem__(self, idx):
        from torchvision import transforms

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in SUPPORTED_EXTENSIONS:
            potential = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential):
                mask_path = potential
                break
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        original_size = image.size

        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.input_mode == 'aspect_pad':
            model_image, _ = _resize_pad_pair(
                image, mask, self.image_size)
            model_input = mask_transform(model_image)
        else:
            model_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
            model_input = model_transform(image)
        original_mask = mask_transform(mask)

        fov_mask = None
        if self.has_fov:
            for ext in SUPPORTED_EXTENSIONS:
                p = os.path.join(self.fov_dir, base_name + ext)
                if os.path.exists(p):
                    fov_mask = (mask_transform(Image.open(p).convert('L')) > 0.5).float()
                    break
        if fov_mask is None:
            fov_mask = original_mask.new_ones(original_mask.shape)

        return model_input, original_mask, original_size, img_name, fov_mask
