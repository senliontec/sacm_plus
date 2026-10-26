"""Matcher baseline: training-free DINOv2 + SAM support matching.

Official reference: ``3rd/Matcher`` (2-Clause BSD).  Matcher natively accepts
multiple support images: DINOv2 patch matching produces target points, and
the modified SAM generator turns them into masks.  The port uses the shared
3-shot supports (or a structurally matched donor for unseen/novel domains)
and exports masks at original resolution.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256
from baseline_zoo.support import load_support_groups, select_supports


class MatcherAdapter(BaselineAdapter):
    id = 'matcher'
    display_name = 'Matcher'
    official_source = 'aim-uofa/Matcher (self-contained copy)'
    official_entrypoint = 'https://github.com/aim-uofa/Matcher'
    input_size = 518
    official_epochs = 0
    official_lr = 0.0
    train_batch_size = 1
    selection = 'dice'
    training_free = True
    deviations = (
        'Training-free Matcher with DINOv2 ViT-L/14 and SAM ViT-H. No '
        'Matcher-specific checkpoint exists; only the two backbone weights '
        'are loaded.',
        'The official pipeline square-resizes all inputs to 518 and returns '
        '518x518 masks; this port resizes the target/supports to 518 and '
        'upsamples the binary result to the original GT resolution.',
        'Matcher natively consumes n-shot supports; this port passes the '
        'three training shots for seen datasets and a structurally matched '
        'donor domain for unseen/novel datasets.',
        'The official COCO/FSS datasets and CLI are replaced by the shared '
        'Sample protocol. The official FSS score recipe is kept: '
        'alpha=0.8, beta=0.2, exp=1.0; EMD, purity and coverage filters '
        'off, top-k score threshold 0.7, 10 merging masks, 8 k-means '
        'centers, sample range (4,6). Prediction seeds Python/NumPy/Torch '
        'from --seed before stochastic matching.',
        'Support DINOv2 features are cached after set_reference so each '
        'target image is encoded once; this is a throughput change only.',
        'DINOv2 ViT-L weights are not bundled. --matcher-random-init is a '
        'code-smoke-only option; reported results require the official '
        'DINOv2 ViT-L/14 checkpoint.',
        'No target GT or target prompts are used; the matched points are '
        'generated from support similarity. This is support-based, not '
        'prompt-free.',
        'No test-time augmentation.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._dinov2_sha: str | None = None
        self._support_groups = None
        self._support_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._current_dataset: str | None = None
        self._support_root: Path | None = None
        self._prediction_seeded = False

    def add_args(self, parser):
        parser.add_argument(
            '--sam-weights', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth')
        parser.add_argument(
            '--dinov2-weights', type=Path,
            default=REPO_ROOT / 'checkpoints' /
            'dinov2_vitl14_pretrain.pth')
        parser.add_argument('--matcher-random-init', action='store_true',
                            help='smoke only: random DINOv2 weights')
        parser.add_argument('--support-shots', type=int, default=3)
        parser.add_argument('--support-dataset', default='auto')

    def build_model(self, args) -> torch.nn.Module:
        try:
            from baseline_zoo.models.matcher import build_matcher
        except ImportError as exc:
            raise SystemExit(
                f'Matcher import failed: {exc}') from exc
        sam_weights = Path(args.sam_weights)
        if not sam_weights.is_file():
            raise SystemExit(
                f'missing SAM ViT-H checkpoint: {sam_weights}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_h_4b8939.pth')
        dinov2_weights = Path(args.dinov2_weights)
        random_init = bool(getattr(args, 'matcher_random_init', False))
        if not random_init and not dinov2_weights.is_file():
            raise SystemExit(
                f'missing DINOv2 ViT-L/14 checkpoint: {dinov2_weights}\n'
                'download: https://dl.fbaipublicfiles.com/dinov2/'
                'dinov2_vitl14/dinov2_vitl14_pretrain.pth')
        self._sam_sha = file_sha256(sam_weights)
        if dinov2_weights.is_file():
            self._dinov2_sha = file_sha256(dinov2_weights)
        if self._support_root is None:
            self._support_root = Path(getattr(
                args, 'data_root', REPO_ROOT / 'data'))
        return build_matcher(
            sam_weights=sam_weights,
            dinov2_weights=dinov2_weights,
            device='cpu',
            random_init=random_init,
        )

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def prepare_train_batch(self, samples, rng: random.Random):
        raise RuntimeError('Matcher is training-free')

    def train_step(self, model, batch):
        raise RuntimeError('Matcher is training-free')

    def _support_tensors(self, dataset: str, device: torch.device):
        if dataset in self._support_cache:
            return self._support_cache[dataset]
        if self._support_groups is None:
            root = self._support_root
            if root is None:
                root = Path(self.args.data_root)
            self._support_groups = load_support_groups(root)
        shots = int(getattr(self.args, 'support_shots', 3))
        override = getattr(self.args, 'support_dataset', 'auto')
        samples, _ = select_supports(
            dataset, self._support_groups, shots=shots, override=override)
        images, masks = [], []
        for support in samples:
            image = load_rgb(support.image_path).resize(
                (self.input_size, self.input_size), Image.BILINEAR)
            mask = load_mask(support.mask_path)
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.input_size, self.input_size), Image.NEAREST))
            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()).permute(
                    2, 0, 1) / 255.0
            images.append(image_t)
            masks.append(torch.from_numpy(mask.astype(np.float32).copy()))
        support_images = torch.stack(images).unsqueeze(0).to(device)
        support_masks = torch.stack(masks).unsqueeze(0).to(device)
        self._support_cache[dataset] = (support_images, support_masks)
        return support_images, support_masks

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path)
        height, width = image.height, image.width
        image = image.resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()).permute(
                2, 0, 1).unsqueeze(0) / 255.0
        return {'image': tensor.to(device),
                'original_size': (height, width)}

    def predict_sample(self, model, prepared, sample):
        if not self._prediction_seeded:
            seed = int(getattr(self.args, 'seed', 0) or 0)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self._prediction_seeded = True
        device = next(model.parameters()).device
        matcher = model.matcher
        if self._current_dataset != sample.dataset:
            support_images, support_masks = self._support_tensors(
                sample.dataset, device)
            matcher.set_reference(support_images, support_masks)
            self._current_dataset = sample.dataset
        matcher.set_target(prepared['image'])
        pred = matcher.predict()
        height, width = prepared['original_size']
        pred = F.interpolate(
            pred[None], size=(height, width), mode='bilinear',
            align_corners=False)[0, 0]
        return pred.detach().cpu().numpy() > 0.5

    def save_config(self, args):
        sam_weights = getattr(args, 'sam_weights', None)
        dinov2_weights = getattr(args, 'dinov2_weights', None)
        return {
            'input_size': int(self.input_size),
            'sam_weights': str(sam_weights) if sam_weights else None,
            'dinov2_weights': str(dinov2_weights) if dinov2_weights else None,
            'sam_weights_sha256': self._sam_sha,
            'dinov2_weights_sha256': self._dinov2_sha,
            'matcher_random_init': bool(
                getattr(args, 'matcher_random_init', False)),
            'support_shots': int(getattr(args, 'support_shots', 3)),
            'support_dataset': str(getattr(args, 'support_dataset', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'support_shots'):
            if key in config:
                setattr(args, key, config[key])
        if 'support_dataset' in config:
            args.support_dataset = config['support_dataset']
        if config.get('sam_weights'):
            args.sam_weights = Path(config['sam_weights'])
        if config.get('dinov2_weights'):
            args.dinov2_weights = Path(config['dinov2_weights'])
        if 'matcher_random_init' in config:
            args.matcher_random_init = bool(config['matcher_random_init'])
        if config.get('data_root'):
            self._support_root = Path(config['data_root'])
        self.input_size = int(getattr(args, 'input_size', 518))
