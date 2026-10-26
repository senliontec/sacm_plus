"""Official SACM baseline (SAM ViT-L + adapters, prompt-free decoder).

The official ``segment_anything`` package is copied into
``baseline_zoo.models.sacm_official``.  Training follows the released
``train_sam.py`` loss/optimizer/forward; selection and export follow the
unified protocol.
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
from baseline_zoo.models.sacm_official import build_sam_vit_l
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class SACMOfficialAdapter(BaselineAdapter):
    id = 'sacm_official'
    display_name = 'SACM (official)'
    official_source = 'kylechuuuuu/SACM (self-contained copy)'
    official_entrypoint = 'upstream train_sam.py'
    input_size = 1024
    official_epochs = 60
    official_lr = 3e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Checkpoint selection uses the 6-image validation split by Dice; '
        'upstream selects by validation F1/loss.',
        'Upstream test_sam.py resizes predictions back to the original image '
        'size (test_sam.py:260-261); this port exports GT-resolution masks '
        'for the shared evaluator.',
        'Upstream test_sam.py thresholds raw logits at 0.5; this port '
        'applies sigmoid first (logit > 0). The model is logits-in, so the '
        'official threshold is an inference bug; this correction is '
        'deliberate and gives the standard 0.5-probability decision.',
        'The official AdamW 3e-4 with weight decay 0.01, cosine to 1e-6, '
        'gradient clipping 1.0, BCE 0.7 + Dice 0.3, frozen encoder except '
        'adapters, and the zero-prompt forward are preserved; the official '
        '60-epoch budget is replaced by 800 epochs for the 18-image 3-shot '
        'protocol.',
        'Per-block activation checkpointing is enabled for 11 GiB training; '
        'it changes memory, not the computation.',
        'Official train_sam.py uses DataLoader workers=4 and no explicit seed; '
        'this port uses workers=0 and the shared seeded runtime. Official '
        'test_sam.py reads its own SegmentationDataset(root_dir) over '
        'images/ and masks/ with no split argument; this port evaluates '
        'held-out 20-dataset test roots.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--bce-weight', type=float, default=0.7)
        parser.add_argument('--dice-weight', type=float, default=0.3)
        # Official train_sam.py clips by 1.0.  The shared runtime reads
        # ``grad_clip``; this alias keeps the upstream option name and sets
        # the effective default instead of being a silent no-op.
        parser.add_argument(
            '--clip-grad-norm', dest='grad_clip', type=float,
            help='gradient clipping norm (official default: 1.0)')
        # argparse does not override the generic --grad-clip default when a
        # second action shares the destination, so set it explicitly.
        parser.set_defaults(grad_clip=1.0)

    def build_model(self, args):
        # Prediction loads a full SACM state dict.  Upstream test_sam.py
        # constructs with checkpoint=None, so a base SAM file is not
        # required (or consumed) in that phase.
        predicting = bool(
            getattr(self, 'prediction_mode', False)
            or getattr(args, 'stage', None) == 'predict'
            or getattr(args, 'checkpoint', None) is not None)
        checkpoint = None
        if not predicting:
            if not args.sam_checkpoint.is_file():
                raise SystemExit(
                    f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')
            checkpoint = str(args.sam_checkpoint)
        return build_sam_vit_l(
            checkpoint=checkpoint, use_adapter=True,
            adapter_dim_ratio=args.adapter_dim_ratio)

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'adapter_dim_ratio': float(
                getattr(args, 'adapter_dim_ratio', 0.1)),
            'bce_weight': float(getattr(args, 'bce_weight', 0.7)),
            'dice_weight': float(getattr(args, 'dice_weight', 0.3)),
        }

    def restore_config(self, args, config):
        for key in ('adapter_dim_ratio', 'bce_weight', 'dice_weight'):
            if key in config:
                setattr(args, key, config[key])

    def configure_trainable(self, model):
        for name, parameter in model.image_encoder.named_parameters():
            parameter.requires_grad_('adapter' in name)
        for parameter in model.prompt_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad_(True)
        model.image_encoder.grad_checkpointing = True

    def make_optimizer(self, model, args):
        adapter_params = [p for n, p in model.image_encoder.named_parameters()
                          if 'adapter' in n and p.requires_grad]
        decoder_params = [p for p in model.mask_decoder.parameters()
                          if p.requires_grad]
        return torch.optim.AdamW([
            {'params': adapter_params, 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': decoder_params, 'lr': args.lr,
             'weight_decay': args.weight_decay},
        ])

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    def _forward(self, model, images):
        batch = images.shape[0]
        sparse = torch.zeros((batch, 0, 256), dtype=torch.float,
                             device=self.device)
        dense = torch.zeros((batch, 256, 64, 64), dtype=torch.float,
                            device=self.device)
        output = model.image_encoder(images, return_features=True)
        if len(output) == 3:
            embeddings, _, adapter_features = output
        else:
            embeddings, _ = output
            adapter_features = None
        low_res, _ = model.mask_decoder(
            image_embeddings=embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            adapter_features=adapter_features)
        return low_res

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BILINEAR)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.BILINEAR)
            images.append(torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0)
            masks.append(torch.from_numpy(
                np.asarray(mask, dtype=np.float32).copy()
            ).unsqueeze(0) / 255.0)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        images, masks = batch
        logits = self._forward(model, images)
        logits = F.interpolate(logits, (1024, 1024), mode='bilinear',
                               align_corners=False)
        masks = F.interpolate(masks, (1024, 1024), mode='bilinear',
                              align_corners=False)
        bce = nn.BCEWithLogitsLoss()(logits, masks)
        prob = torch.sigmoid(logits)
        inter = (prob * masks).sum()
        union = prob.sum() + masks.sum()
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return (self.args.bce_weight * bce + self.args.dice_weight * dice)

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        logits = self._forward(model, prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
