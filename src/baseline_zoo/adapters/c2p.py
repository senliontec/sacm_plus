"""C2P baseline adapter (ECCV 2026).

Concept-to-Pixel is a prompt-free universal segmentation model built on a
ConvNeXtV2-B backbone, geometric tokens (explicitly supervised physical
attributes), MLLM-distilled semantic tokens, a token-image transformer, a
token-guided dynamic head, and geometry-aware inference consensus.

The released repository does not include a runnable trainer/evaluator
(``train.py`` imports a non-existent ``SRICL`` class), but it does provide
the full ``C2P`` network and an official HuggingFace checkpoint trained on
eight medical datasets.  This port evaluates that official checkpoint
zero-shot: C2P is the zoo's prompt-free universal foundation baseline and
does not consume the shared 18-image training split.  No target GT or
prompts are used at inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_rgb
from baseline_zoo.models.c2p import build_c2p
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class C2PAdapter(BaselineAdapter):
    id = 'c2p'
    display_name = 'C2P'
    official_source = 'Yundi218/Concept-to-Pixel (self-contained copy)'
    official_entrypoint = 'https://github.com/Yundi218/Concept-to-Pixel'
    input_size = 384
    official_lr = 0.0
    train_batch_size = 1
    selection = 'dice'
    training_free = True
    deviations = (
        'C2P is a prompt-free universal foundation baseline, evaluated '
        'zero-shot with the official HuggingFace checkpoint trained on '
        'eight medical datasets. It does not consume the shared 18-image '
        'training split; no target GT, support images or prompts are used. '
        'Applying it to non-medical curvilinear domains is a zero-shot '
        'generalization test, analogous to the paper\'s unseen-modality '
        'setting.',
        'The released repository has no runnable trainer/evaluator: train.py '
        'imports a non-existent SRICL class and evaluate_sem.py is absent. '
        'This port vendors the released C2P network, loads the official '
        'checkpoint, and reimplements the official geometry-aware inference '
        'consensus (three views, area/centroid consistency weighting) from '
        'predict_with_token_tta_fast.',
        'The official C2P checkpoint contains the ConvNeXtV2-B backbone, so '
        'the port builds the timm backbone with pretrained=False and never '
        'downloads weights. --c2p-random-init is a code-smoke option only.',
        'Inputs are resized to 384 square with ImageNet normalization and '
        'predictions are resized to the original GT resolution. Inference '
        'is fp32 (the official trainer uses AMP).',
        'The paper also supervises 9 geometric attributes and 9 MLLM-'
        'distilled semantic tokens during its 8-dataset training. Inference '
        'needs neither MLLM text nor PubMedBERT embeddings: the semantic '
        'and geometric tokens are parameters of the released checkpoint. '
        'No fine-tuning on the shared split is performed.',
        'No test-time labels are used; TTA is the official blind-view '
        'geometry-consistency mechanism, not a tuned ensemble.',
    )

    def __init__(self):
        super().__init__()
        self._checkpoint_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=384)
        parser.add_argument(
            '--c2p-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'c2p' / '1.pth')
        parser.add_argument('--c2p-random-init', action='store_true',
                            help='smoke only: random C2P weights')
        parser.add_argument('--c2p-tta', type=int, default=1,
                            help='1 = official geometry-aware 3-view TTA')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 384))
        if self.input_size != 384:
            raise SystemExit('C2P uses the official 384x384 input')
        random_init = bool(getattr(args, 'c2p_random_init', False))
        checkpoint = Path(getattr(
            args, 'c2p_checkpoint',
            REPO_ROOT / 'checkpoints' / 'c2p' / '1.pth'))
        if not random_init:
            if not checkpoint.is_file():
                raise SystemExit(
                    f'missing C2P checkpoint: {checkpoint}\n'
                    'download from https://huggingface.co/yunyundi/c2p:\n'
                    '  sem-OL-3dice-v2-mutil-deep_only-norfb-DS-onlyD5-'
                    'weightdecay-0.3-0.05/1.pth\n'
                    'and place it at checkpoints/c2p/1.pth')
            self._checkpoint_sha = file_sha256(checkpoint)
        model = build_c2p(
            checkpoint=checkpoint, random_init=random_init)
        model.eval()
        return model

    def load_pretrained(self, model, args) -> None:
        print('loaded the official C2P checkpoint (zero-shot evaluation)')

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def prepare_train_batch(self, samples, rng):
        raise RuntimeError('C2P is evaluated zero-shot; no training')

    def train_step(self, model, batch):
        raise RuntimeError('C2P is evaluated zero-shot; no training')

    def prepare_eval_sample(self, sample, device):
        image = np.asarray(load_rgb(sample.image_path), dtype=np.float32)
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
        image = (image / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(
            image.transpose(2, 0, 1).copy()).float().unsqueeze(0)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'image': tensor.to(device),
            'size': (int(width), int(height)),
        }

    @staticmethod
    def _mask_props(mask_prob):
        height, width = mask_prob.shape
        area = float(mask_prob.sum()) / (height * width)
        ys = torch.arange(
            height, device=mask_prob.device, dtype=torch.float32
        ).unsqueeze(1).expand(height, width)
        xs = torch.arange(
            width, device=mask_prob.device, dtype=torch.float32
        ).unsqueeze(0).expand(height, width)
        denom = mask_prob.sum() + 1e-6
        cy = (ys * mask_prob).sum() / denom / height
        cx = (xs * mask_prob).sum() / denom / width
        return area, torch.stack([cx, cy])

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image = prepared['image']
        width, height = prepared['size']
        use_tta = bool(int(getattr(self.args, 'c2p_tta', 1)))
        if not use_tta:
            logits, _, _, _ = model(image)
            logits = F.interpolate(
                logits, size=(height, width), mode='bilinear',
                align_corners=True)
            return torch.sigmoid(logits[0, 0]).cpu().numpy() > 0.5

        batch = torch.cat([
            image, torch.flip(image, [3]), torch.flip(image, [2])], dim=0)
        logits, _, geo_pred, _ = model(batch)
        logits = F.interpolate(
            logits.float(), size=(height, width), mode='bilinear',
            align_corners=True)
        probs = torch.sigmoid(logits)

        inverse = [
            lambda value: value,
            lambda value: torch.flip(value, [3]),
            lambda value: torch.flip(value, [2]),
        ]
        accumulated = torch.zeros(
            (1, 1, height, width), device=image.device)
        total_weight = 0.0
        token_areas = []
        for i in range(3):
            prob = probs[i:i + 1]
            area_token = (geo_pred['area'][i].item()
                          if geo_pred is not None and 'area' in geo_pred
                          else None)
            token_areas.append(1.0 if area_token is None else area_token)
            weight = 1.0
            if area_token is not None:
                mask_area, mask_centroid = self._mask_props(
                    prob.squeeze(0).squeeze(0))
                delta_area = abs(mask_area - area_token)
                delta_centroid = (
                    torch.norm(
                        mask_centroid - geo_pred['centroid'][i]).item()
                    if geo_pred is not None and 'centroid' in geo_pred
                    else 0.0)
                inconsistency = 10.0 * delta_area + delta_centroid
                weight = math.exp(-inconsistency * 5.0)
            accumulated += inverse[i](prob) * weight
            total_weight += weight

        final = accumulated / (total_weight + 1e-8)
        average_area = sum(token_areas) / len(token_areas)
        if average_area < 0.001 and final.mean() < 0.005:
            return torch.zeros_like(final[0, 0]).cpu().numpy() > 0.5
        return final[0, 0].cpu().numpy() > 0.5

    def save_config(self, args):
        checkpoint = getattr(args, 'c2p_checkpoint', None)
        sha = self._checkpoint_sha
        if (sha is None and checkpoint is not None
                and Path(checkpoint).is_file()):
            sha = file_sha256(Path(checkpoint))
        return {
            'input_size': int(self.input_size),
            'c2p_checkpoint': str(checkpoint) if checkpoint else None,
            'c2p_checkpoint_sha256': sha,
            'c2p_random_init': bool(getattr(
                args, 'c2p_random_init', False)),
            'c2p_tta': int(getattr(args, 'c2p_tta', 1)),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if config.get('c2p_checkpoint'):
            args.c2p_checkpoint = Path(config['c2p_checkpoint'])
        if 'c2p_random_init' in config:
            args.c2p_random_init = bool(config['c2p_random_init'])
        if 'c2p_tta' in config:
            args.c2p_tta = int(config['c2p_tta'])
