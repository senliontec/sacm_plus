"""PerSAM baseline: training-free one-shot personalization of SAM.

Official reference: ``3rd/Personalize-SAM`` (MIT).  PerSAM encodes one
reference image/mask, derives target points from feature similarity, and
prompts a frozen SAM ViT-H.  The project's 3-shot protocol is handled by
running each support independently and majority-voting the masks.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.persam import (
    build_persam,
    majority_vote,
    predict_with_embedding,
    prepare_reference_embedding,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256
from baseline_zoo.support import load_support_groups, select_supports


class PerSAMAdapter(BaselineAdapter):
    id = 'persam'
    display_name = 'PerSAM'
    official_source = 'ZrrSkywalker/Personalize-SAM (PerSAM)'
    official_entrypoint = 'https://github.com/ZrrSkywalker/Personalize-SAM'
    input_size = 1024
    official_lr = 0.0
    train_batch_size = 1
    selection = 'dice'
    training_free = True
    deviations = (
        'PerSAM is one-shot. This port runs each of the three support images '
        'independently and majority-votes the binary masks (3-shot '
        'extension); --support-shots 1 reproduces the official one-shot '
        'prediction.',
        'Seen datasets use their three training shots; unseen/novel datasets '
        'use a structurally matched donor domain from the training split.',
        'The official driver expects fixed 00.jpg/00.png files and a data '
        'directory per object; the port consumes the shared Sample protocol.',
        'The support mask is converted to RGB 0/255 before SAM '
        'normalization, as required by the official predictor.',
        'Empty first-pass masks skip refinement and contribute an empty '
        'vote; the official code would fail on x.min().',
        'The PerSAM-F two-parameter fine-tuned variant is not used; this is '
        'the training-free PerSAM method.',
        'No PerSAM method checkpoint exists; only the SAM ViT-H base is '
        'loaded. Predictions use original target resolution and no TTA.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._support_groups = None
        self._support_embeddings: dict[str, list[torch.Tensor]] = {}
        self._support_root: Path | None = None

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth')
        parser.add_argument('--sam-type', default='vit_h')
        parser.add_argument('--support-shots', type=int, default=3)
        parser.add_argument('--support-dataset', default='auto')

    def build_model(self, args) -> torch.nn.Module:
        checkpoint = Path(args.sam_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-H checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_h_4b8939.pth')
        self._sam_sha = file_sha256(checkpoint)
        if self._support_root is None:
            self._support_root = Path(getattr(
                args, 'data_root', REPO_ROOT / 'data'))
        return build_persam(str(checkpoint), args.sam_type)

    def configure_trainable(self, model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def prepare_train_batch(self, samples, rng: random.Random):
        raise RuntimeError('PerSAM is training-free')

    def train_step(self, model, batch):
        raise RuntimeError('PerSAM is training-free')

    def _ensure_support_embeddings(self, model, dataset: str):
        if dataset in self._support_embeddings:
            return self._support_embeddings[dataset]
        if self._support_groups is None:
            root = self._support_root
            if root is None:
                root = Path(self.args.data_root)
            self._support_groups = load_support_groups(root)
        shots = int(getattr(self.args, 'support_shots', 3))
        override = getattr(self.args, 'support_dataset', 'auto')
        samples, _ = select_supports(
            dataset, self._support_groups, shots=shots, override=override)
        embeddings = []
        for support in samples:
            image = np.asarray(load_rgb(support.image_path))
            mask = load_mask(support.mask_path)
            mask_rgb = np.repeat(
                (mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
            similarity, target_embedding = prepare_reference_embedding(
                model.predictor, image, mask_rgb)
            embeddings.append((similarity.detach(), target_embedding.detach()))
        self._support_embeddings[dataset] = embeddings
        return embeddings

    def prepare_eval_sample(self, sample, device):
        return np.asarray(load_rgb(sample.image_path))

    def predict_sample(self, model, prepared, sample):
        embeddings = self._ensure_support_embeddings(model, sample.dataset)
        predictor = model.predictor
        predictor.set_image(prepared)
        features = predictor.features
        original_size = predictor.original_size
        input_size = predictor.input_size
        masks = []
        for shot, (similarity, target_embedding) in enumerate(embeddings):
            try:
                masks.append(predict_with_embedding(
                    predictor, features, original_size, input_size,
                    similarity.to(features.device),
                    target_embedding.to(features.device)))
            except (RuntimeError, ValueError, IndexError) as exc:
                # Do not silently turn a failed support into an empty vote:
                # that would hide a broken prompt/refinement path behind a
                # plausible-looking prediction.
                raise RuntimeError(
                    f'PerSAM support {shot} failed for '
                    f'{sample.dataset}/{sample.stem}: {exc}') from exc
        return majority_vote(masks)

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        sha = self._sam_sha
        if sha is None and checkpoint is not None and Path(checkpoint).is_file():
            sha = file_sha256(Path(checkpoint))
        return {
            'input_size': int(self.input_size),
            'sam_type': str(getattr(args, 'sam_type', 'vit_h')),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': sha,
            'support_shots': int(getattr(args, 'support_shots', 3)),
            'support_dataset': str(getattr(args, 'support_dataset', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'support_shots'):
            if key in config:
                setattr(args, key, config[key])
        for key in ('sam_type', 'support_dataset'):
            if key in config:
                setattr(args, key, config[key])
        if config.get('sam_checkpoint'):
            args.sam_checkpoint = Path(config['sam_checkpoint'])
        if config.get('data_root'):
            self._support_root = Path(config['data_root'])
        self.input_size = int(getattr(args, 'input_size', 1024))
        stored = config.get('sam_checkpoint_sha256')
        current = getattr(args, 'sam_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(f'WARNING: base SAM checkpoint differs from training: '
                      f'{current}; prediction uses the checkpoint weights')
