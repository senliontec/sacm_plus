"""SAM-OCTA baseline with the official LoRA/prompt training logic.

Official reference: vendored ``baselines/sam-octa``.  The SAM modeling code,
LoRA wrapper, Dice/clDice losses and GT point-prompt generator are copied
into ``baseline_zoo.models.sam_octa``.
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sam_octa import (
    DiceLoss,
    build_sam_octa,
    clDiceLoss,
    label_to_point_prompt_global,
)
from baseline_zoo.models.sam_octa.segment_anything.utils.transforms import (
    ResizeLongestSide,
)
from baseline_zoo.runtime import (
    REPO_ROOT,
    BaselineAdapter,
    file_sha256,
    seed_everything,
)

PROB = 0.3


class SAMOCTAAdapter(BaselineAdapter):
    id = 'sam_octa'
    display_name = 'SAM-OCTA'
    official_source = 'SAM-OCTA (self-contained copy)'
    official_entrypoint = 'upstream train_sam_octa.py'
    input_size = 1024
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Official input is three OCTA projection layers. For the 20-dataset '
        'protocol, RGB datasets use their three channels and OCTA-500 '
        'single-channel images are replicated to three channels.',
        'The official training path assumes square OCTA images and does not '
        'pad after the longest-side resize. Non-square protocol images use '
        "SAM's standard pipeline: longest-side resize, normalize, then "
        'zero-pad inside Sam.preprocess. The pre-pad shape is recorded as '
        'input_size so postprocess crops the padding before GT resize.',
        'ViT-L is used as the shared SAM backbone (paper requirement); the '
        'upstream default is ViT-H.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream saves every 10 epochs and selects manually. The official '
        '100-epoch budget is replaced by 800 epochs for the 18-image 3-shot '
        'protocol.',
        'Per-block activation checkpointing is enabled for 11 GiB training; '
        'it changes memory, not the computation.',
        'GT-derived point prompts (one positive per component plus one '
        'negative in the 8-pixel ring) are preserved. Official loss applies '
        'clDice only to Artery/Vein/LargeVessel classes; the binary '
        '20-dataset port applies 0.8*Dice+0.2*clDice uniformly. Official '
        'train_sam_octa.py uses an unseeded global RNG and DataParallel on one '
        'visible GPU; this port uses the shared protocol seed and a single '
        'module, so data/augmentation order is not bitwise identical. Final '
        'prediction prompt sampling is seeded from --seed before the first '
        'eval sample; upstream leaves it unseeded. Official metrics are '
        'Dice/Jaccard/Hausdorff; the shared evaluator reports '
        'Dice/IoU/clDice/HD95.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-octa-model-type', default='vit_l',
            choices=['vit_b', 'vit_l', 'vit_h'],
            help='official default is vit_h; vit_l is the 24 GiB default')
        parser.add_argument(
            '--sam-checkpoint', type=Path, default=None,
            help='defaults to the local SAM checkpoint matching '
                 '--sam-octa-model-type')

    def _default_checkpoint(self, model_type: str) -> Path:
        names = {
            'vit_b': 'sam_vit_b_01ec64.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_h': 'sam_vit_h_4b8939.pth',
        }
        return REPO_ROOT / 'checkpoints' / names[model_type]

    def build_model(self, args):
        self.model_type = str(getattr(
            args, 'sam_octa_model_type', 'vit_l'))
        checkpoint = getattr(args, 'sam_checkpoint', None)
        if checkpoint is None:
            checkpoint = self._default_checkpoint(self.model_type)
            args.sam_checkpoint = checkpoint
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM {self.model_type} checkpoint: {checkpoint}')
        return build_sam_octa(checkpoint, model_type=self.model_type,
                              rank=4)

    def configure_trainable(self, model):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                'linear_a_' in name or 'linear_b_' in name)
        model.sam.image_encoder.grad_checkpointing = True

    def make_optimizer(self, model, args):
        # Upstream uses AdamW(lr=1) with a LambdaLR that starts at zero.
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        warmup = max(args.epochs // 5, 1)
        peak = args.lr

        def lr_lambda(epoch):
            if epoch <= warmup:
                return max(1e-5, peak * epoch / warmup)
            return max(1e-5, peak * 0.98 ** (epoch - warmup))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        sha = None
        if checkpoint is not None and Path(checkpoint).is_file():
            sha = file_sha256(Path(checkpoint))
        return {
            'input_size': int(self.input_size),
            'model_type': str(getattr(
                self, 'model_type',
                getattr(args, 'sam_octa_model_type', 'vit_l'))),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            args.input_size = int(config['input_size'])
            self.input_size = args.input_size
        if config.get('model_type'):
            args.sam_octa_model_type = str(config['model_type'])
            self.model_type = args.sam_octa_model_type
        if config.get('sam_checkpoint'):
            args.sam_checkpoint = Path(config['sam_checkpoint'])

    def _augment(self, image: np.ndarray, label: np.ndarray):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=PROB),
            A.CLAHE(p=PROB),
            A.VerticalFlip(p=PROB),
            A.HorizontalFlip(p=PROB),
            A.AdvancedBlur(p=PROB),
        ])
        out = transform(image=image, mask=label)
        return out['image'], (out['mask'] > 0).astype(np.uint8)

    def _load_image(self, sample):
        image = load_rgb(sample.image_path)
        return np.asarray(image, dtype=np.uint8)

    def _load_label(self, sample):
        with Image.open(sample.mask_path) as mask:
            label = np.asarray(mask.convert('L'))
        return (label > 127).astype(np.uint8)

    def _prepare(self, sample, augment_rng: random.Random | None, device):
        image = self._load_image(sample)
        label = self._load_label(sample)
        if augment_rng is not None:
            image, label = self._augment(image, label)
        height, width = label.shape

        tensor = torch.from_numpy(
            image.astype(np.float32).copy()).permute(2, 0, 1).unsqueeze(0)
        resize = ResizeLongestSide(self.input_size)
        # Do not pad here: Sam.preprocess normalizes then pads, records the
        # pre-pad input_size, and postprocess crops to it.  Padding before
        # the model makes that crop a no-op and shifts predictions.
        tensor = resize.apply_image_torch(tensor)

        _, pos, neg = label_to_point_prompt_global(label, 1, 1)
        points = np.array(pos + neg, dtype=np.float32).reshape(-1, 2)
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
        coords = torch.from_numpy(points).unsqueeze(0)
        coords = resize.apply_coords_torch(coords, (height, width))
        point_labels = torch.from_numpy(labels).unsqueeze(0)

        target = torch.from_numpy(
            label.astype(np.float32).copy()).unsqueeze(0).unsqueeze(0)
        return {
            'image': tensor.to(device),
            'original_size': (height, width),
            'point_coords': coords.to(device),
            'point_labels': point_labels.to(device),
            'target': target.to(device),
        }

    def _prediction_rng_active(self) -> bool:
        return bool(getattr(self, 'prediction_mode', False)
                    or getattr(self.args, 'stage', None) == 'predict')

    def _seed_prediction_rng(self):
        """Seed before the first eval prompt draw, not after it."""
        if self._prediction_seeded or not self._prediction_rng_active():
            return
        seed_everything(int(getattr(self.args, 'seed', 0) or 0))
        self._prediction_seeded = True

    def prepare_train_batch(self, samples, rng: random.Random):
        # The official prompt encoder handles one variable-length prompt list;
        # silently dropping extra samples would under-train the batch.
        if len(samples) != 1:
            raise RuntimeError(
                'SAM-OCTA requires --batch-size 1 (official prompt encoder)')
        return self._prepare(samples[0], rng, self.device)

    def train_step(self, model, batch):
        pred = model(batch['image'], batch['original_size'],
                     batch['point_coords'], batch['point_labels'])
        target = batch['target']
        return 0.8 * DiceLoss()(pred, target) + 0.2 * clDiceLoss()(pred, target)

    def prepare_eval_sample(self, sample, device):
        self._seed_prediction_rng()
        return self._prepare(sample, None, device)

    def predict_sample(self, model, prepared, sample):
        self._seed_prediction_rng()
        pred = model(prepared['image'], prepared['original_size'],
                     prepared['point_coords'], prepared['point_labels'])
        pred = pred[0, 0].detach().cpu().numpy()
        return pred > 0.8
