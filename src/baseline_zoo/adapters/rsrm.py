"""RSRM baseline adapter (ECCV 2026).

Training-free cross-domain few-shot segmentation with a frozen DINOv3
ViT-B/16: SAFR re-fuses semantic attention layers, ASE calibrates support
features against robust query FG/BG regions, and HPM fuses global, regional
and pixel-level prototype matching.  Support-based; no target GT/prompts.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.rsrm import build_rsrm
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256
from baseline_zoo.support import load_support_groups, select_supports

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RSRMAdapter(BaselineAdapter):
    id = 'rsrm'
    display_name = 'RSRM'
    official_source = 'Sparkling-Water/RSRM (self-contained copy)'
    official_entrypoint = 'https://github.com/Sparkling-Water/RSRM'
    input_size = 480
    official_lr = 0.0
    train_batch_size = 1
    selection = 'dice'
    training_free = True
    support_based = True
    deviations = (
        'Training-free RSRM with a frozen DINOv3 backbone; no RSRM-specific '
        'checkpoint exists, only the DINOv3 backbone is loaded. The paper '
        'uses ViT-B/16; the local checkpoints/dinov3/pytorch_model.bin is a '
        'timm ViT-S/16 export and is auto-detected (official Meta ViT-B/16 '
        'state dicts are also accepted).',
        'The official code uses fast_pytorch_kmeans for regional prototypes; '
        'this port uses scikit-learn k-means++ on CPU with a deterministic '
        'per-call seed, keeping the same region-similarity computation.',
        'The shared 3-shot supports are used (own training shots for seen '
        'domains, a structurally matched donor domain for unseen/novel), '
        'with support masks driving SAFR/ASE/HPM. This is support-based, '
        'not prompt-free; no target GT or target prompts are used.',
        'Official images/masks are resized to 480 square with ImageNet '
        'normalization. Predictions are argmax at 480 (as in the official '
        'test loop) and nearest-resized to the original GT resolution.',
        'The official benchmark samples 1200/2400 episodes from '
        'Deepglobe/ISIC/Chest/FSS; the shared protocol instead runs the '
        'frozen 20-dataset test split. No test-time augmentation.',
        'The official evaluation resizes query masks to 480; this port '
        'exports GT-resolution masks for the shared evaluation contract.',
    )

    def __init__(self):
        super().__init__()
        self._support_groups = None
        self._support_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._support_root: Path | None = None
        self._dinov3_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=480)
        parser.add_argument(
            '--rsrm-dinov3-weights', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'dinov3' /
            'pytorch_model.bin')
        parser.add_argument('--support-shots', type=int, default=3)
        parser.add_argument('--support-dataset', default='auto')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 480))
        if self.input_size != 480:
            raise SystemExit('RSRM uses the official 480x480 input')
        weights = Path(getattr(
            args, 'rsrm_dinov3_weights',
            REPO_ROOT / 'checkpoints' / 'dinov3' / 'pytorch_model.bin'))
        if not weights.is_file():
            raise SystemExit(
                f'missing DINOv3 ViT-B/16 checkpoint: {weights}\n'
                'Upload the official DINOv3 weights under checkpoints/.')
        self._dinov3_sha = file_sha256(weights)
        if self._support_root is None:
            self._support_root = Path(getattr(args, 'data_root', ''))
        return build_rsrm(weights, seed=int(getattr(args, 'seed', 0) or 0))

    def load_pretrained(self, model, args) -> None:
        print('loaded frozen DINOv3 backbone for RSRM')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def prepare_train_batch(self, samples, rng: random.Random):
        raise RuntimeError('RSRM is training-free')

    def train_step(self, model, batch):
        raise RuntimeError('RSRM is training-free')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _groups(self):
        if self._support_groups is None:
            if self._support_root is None:
                raise RuntimeError('RSRM data_root is not configured')
            self._support_groups = load_support_groups(self._support_root)
        return self._support_groups

    def _pair(self, sample):
        image = np.asarray(load_rgb(sample.image_path), dtype=np.float32)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
        image = (image / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        mask = load_mask(sample.mask_path).astype(np.float32)
        mask = cv2.resize(mask, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_NEAREST)
        image_t = torch.from_numpy(
            image.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask.copy()).float()
        return image_t, mask_t

    def _supports(self, dataset: str, device):
        if dataset in self._support_cache:
            return self._support_cache[dataset]
        shots = int(getattr(self.args, 'support_shots', 3))
        samples, _ = select_supports(
            dataset, self._groups(), shots=shots,
            override=str(getattr(self.args, 'support_dataset', 'auto')))
        pairs = [self._pair(sample) for sample in samples]
        images = torch.stack([item[0] for item in pairs]).to(device)
        masks = torch.stack([item[1] for item in pairs]).to(device)
        self._support_cache[dataset] = (images, masks)
        return images, masks

    def prepare_eval_sample(self, sample, device):
        query, _ = self._pair(sample)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'query': query.unsqueeze(0).to(device),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        device = prepared['query'].device
        support_images, support_masks = self._supports(
            sample.dataset, device)
        img_s_list = [support_images[k:k + 1] for k in range(
            support_images.shape[0])]
        mask_s_list = [support_masks[k:k + 1] for k in range(
            support_masks.shape[0])]
        outputs = model(img_s_list, mask_s_list, prepared['query'])
        pred = torch.argmax(outputs[0], dim=1)
        width, height = prepared['size']
        pred = F.interpolate(
            pred.unsqueeze(1).float(), size=(height, width),
            mode='nearest')[0, 0]
        return pred.cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------
    def save_config(self, args):
        weights = getattr(args, 'rsrm_dinov3_weights', None)
        sha = self._dinov3_sha
        if sha is None and weights is not None and Path(weights).is_file():
            sha = file_sha256(Path(weights))
        return {
            'input_size': int(self.input_size),
            'dinov3_weights': str(weights) if weights else None,
            'dinov3_weights_sha256': sha,
            'support_shots': int(getattr(args, 'support_shots', 3)),
            'support_dataset': str(getattr(args, 'support_dataset', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if config.get('dinov3_weights'):
            args.rsrm_dinov3_weights = Path(config['dinov3_weights'])
        if 'support_shots' in config:
            args.support_shots = int(config['support_shots'])
        if 'support_dataset' in config:
            args.support_dataset = str(config['support_dataset'])
        if config.get('data_root'):
            self._support_root = Path(config['data_root'])
        stored = config.get('dinov3_weights_sha256')
        current = getattr(args, 'rsrm_dinov3_weights', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(
                    f'WARNING: dinov3_weights differs from training: '
                    f'{current}; prediction uses this weight file')
