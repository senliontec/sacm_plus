"""CWSAM baseline with the official ClassWise-SAM-Adapter logic.

Official reference: vendored ``baselines/cwsam``.  The frequency-prompt
encoder, per-block adapters and mask decoder are copied into
``baseline_zoo.models.cwsam``; the upstream mmseg/mmcv import chain is gone.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.cwsam import build_cwsam
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CWSAMAdapter(BaselineAdapter):
    id = 'cwsam'
    display_name = 'CWSAM'
    official_source = 'ClassWise-SAM-Adapter (self-contained copy)'
    official_entrypoint = 'upstream train.py with ViT-L encoder'
    input_size = 1024
    official_lr = 2e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'The upstream SAR config is 5-class; this is a 2-class '
        '(background/foreground) port. The official per-class weights '
        '[1.5, 1, 0.5, 1.9, 0.1] have no binary mapping, so neutral CE is '
        'used.',
        'Default ViT-B matches the released ClassWise-SAM-Adapter configs '
        'and checkpoint. The paper states that all SAM-based baselines share '
        'a ViT-L encoder, but that claim could not be verified locally; '
        '--cwsam-encoder vit_l exposes that variant as an ablation.',
        'Checkpoint selection uses the 6-image validation split by Dice. '
        'Upstream selects by S-measure on its own val loader.',
        'Per-block activation checkpointing is enabled to fit 1024x1024 '
        'ViT-L training on 11 GiB; it changes memory, not the computation.',
        'Official ImageNet normalization and the absence of augmentation '
        'are preserved; predictions are restored to GT resolution.',
        'The shared 3-shot project budget is 800 epochs instead of the '
        'official 200.',
        'Official training uses 2-GPU DDP with per-GPU batch 1 (effective '
        'batch 2), workers=8 and DistributedSampler; this port uses one '
        'GPU, batch 1, workers=0 and the shared seed. Optimizer steps per '
        'epoch and the RNG stream therefore differ.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--cwsam-encoder', default='vit_b',
            choices=['vit_b', 'vit_l'],
            help='official config is ViT-B; ViT-L is the paper-stated '
                 'shared-backbone variant')
        parser.add_argument(
            '--sam-checkpoint', type=Path, default=None,
            help='base SAM checkpoint; defaults to the local ViT-B/ViT-L '
                 'checkpoint matching --cwsam-encoder')

    def _default_checkpoint(self, encoder: str) -> Path:
        name = ('sam_vit_b_01ec64.pth' if encoder == 'vit_b'
                else 'sam_vit_l_0b3195.pth')
        return REPO_ROOT / 'checkpoints' / name

    def build_model(self, args) -> nn.Module:
        self.encoder = str(getattr(args, 'cwsam_encoder', 'vit_b'))
        return build_cwsam(
            num_classes=2, inp_size=self.input_size, encoder=self.encoder)

    def load_pretrained(self, model, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        if checkpoint is None:
            checkpoint = self._default_checkpoint(self.encoder)
            args.sam_checkpoint = checkpoint
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM {self.encoder} checkpoint: {checkpoint}; '
                'download the matching SAM base checkpoint first')
        payload = torch.load(checkpoint, map_location='cpu',
                             weights_only=False)
        state = payload.get('model', payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        def allowed_missing(key):
            if key.startswith('image_encoder.prompt_generator.'):
                return True
            if key.startswith('mask_decoder.cls_upscaling.'):
                return True
            if key in ('pe_layer.positional_encoding_gaussian_matrix',
                       'no_mask_embed.weight'):
                return True
            return (key.startswith('image_encoder.blocks.')
                    and any(tag in key for tag in (
                        '.MLP_Adapter.', '.Space_Adapter.',
                        '.Depth_Adapter.')))
        bad_missing = [key for key in missing if not allowed_missing(key)]
        bad_unexpected = [
            key for key in unexpected
            if not key.startswith('prompt_encoder.')]
        if bad_missing or bad_unexpected:
            raise SystemExit(
                f'SAM checkpoint is incompatible with CWSAM: '
                f'{checkpoint}; missing={bad_missing[:3]}, '
                f'unexpected={bad_unexpected[:3]}')
        print(f'loaded SAM weights: {len(missing)} missing '
              f'(adapters/prompt generator), {len(unexpected)} unexpected '
              f'(prompt encoder)')

    def configure_trainable(self, model):
        for name, parameter in model.named_parameters():
            if ('image_encoder' in name and 'prompt_generator' not in name
                    and 'Adapter' not in name):
                parameter.requires_grad_(False)
        model.image_encoder.grad_checkpointing = True

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        sha = None
        if checkpoint is not None and Path(checkpoint).is_file():
            sha = file_sha256(Path(checkpoint))
        return {
            'input_size': int(self.input_size),
            'encoder': str(getattr(self, 'encoder', 'vit_b')),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            args.input_size = int(config['input_size'])
            self.input_size = args.input_size
        if config.get('encoder'):
            args.cwsam_encoder = str(config['encoder'])
            self.encoder = args.cwsam_encoder
        if config.get('sam_checkpoint'):
            args.sam_checkpoint = Path(config['sam_checkpoint'])

    def _preprocess(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
        return tensor.unsqueeze(0).to(device)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            images.append(self._preprocess(sample, self.device)[0])
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.NEAREST)
            masks.append(torch.from_numpy(
                (np.asarray(mask) > 127).astype(np.int64).copy()))
        return torch.stack(images), torch.stack(masks).to(self.device)

    def train_step(self, model, batch):
        images, masks = batch
        if images.shape[0] != 1:
            raise RuntimeError('CWSAM.infer only supports batch size 1')
        logits = model.infer(images)
        return nn.CrossEntropyLoss()(logits, masks)

    def prepare_eval_sample(self, sample, device):
        return self._preprocess(sample, device)

    def predict_sample(self, model, prepared, sample):
        logits = model.infer(prepared)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        prob = F.interpolate(prob, size=(height, width),
                             mode='bilinear', align_corners=False)
        return (prob[0, 0].cpu().numpy() > 0.5)
