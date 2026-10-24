"""SAM ViT-L + U-Net decoder (project extension baseline)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sam_vit_l_unet import SAMViTLUNet
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


class SAMViTLUNetAdapter(BaselineAdapter):
    id = 'sam_vit_l_unet'
    display_name = 'SAM ViT-L + U-Net'
    official_source = 'project extension (not a paper baseline)'
    official_entrypoint = 'previous project adapter, self-contained now'
    input_size = 1024
    official_epochs = 200
    official_lr = 1e-3
    official_weight_decay = 1e-4
    train_batch_size = 4
    selection = 'dice'
    deviations = (
        'This is the project SAM+U-Net probe, not one of the paper ten '
        'baselines; the decoder is unchanged from the previous adapter.',
        'Frozen ViT features are recomputed per batch instead of cached once '
        'per split. Features are quantized to float16 and back to float32, '
        'and flips/rot90 are applied to the encoded feature maps (and the '
        'image/mask) exactly as in the previous adapter, so the training '
        'semantics match; only the cache/precision storage differs.',
        'Best checkpoints store the decoder state dict only, matching the '
        'previous adapter. Prediction rebuilds the frozen encoder from the '
        'SAM checkpoint.',
        'Checkpoint selection uses the unified val Dice; the previous '
        'adapter also selected by validation Dice.',
        'Predictions are restored to the original GT resolution.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--input-size', type=int, default=1024)

    def build_model(self, args):
        if args.input_size != 1024:
            raise SystemExit(
                'SAM ViT-L expects --input-size 1024 (the frozen image '
                'encoder has a fixed 64x64 positional embedding).')
        if not args.sam_checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-L checkpoint: {args.sam_checkpoint}')
        self.input_size = args.input_size
        return SAMViTLUNet(str(args.sam_checkpoint))

    def configure_trainable(self, model):
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.decoder.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image = load_rgb(sample.image_path).resize(
                (self.input_size, self.input_size), Image.BICUBIC)
            mask = Image.open(sample.mask_path).convert('L').resize(
                (self.input_size, self.input_size), Image.NEAREST)
            image_t = torch.from_numpy(
                np.asarray(image, dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(
                (np.asarray(mask) > 127).astype(np.float32).copy()
            ).unsqueeze(0)
            images.append(image_t.contiguous())
            masks.append(mask_t.contiguous())
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    @staticmethod
    def _augment(tensor, hflip, vflip, rot):
        if hflip:
            tensor = tensor.flip(-1)
        if vflip:
            tensor = tensor.flip(-2)
        if rot:
            tensor = torch.rot90(tensor, rot, [-2, -1])
        return tensor.contiguous()

    def train_step(self, model, batch):
        images, masks = batch
        # Encode the *unaugmented* image, then apply the same flips/rot90
        # to the cached-style feature maps and to the image/mask.  This is
        # the previous adapter's semantics: encoding a flipped image is not
        # equivalent for a frozen ViT with absolute positional embeddings.
        feature_maps = [feature.half().float()
                        for feature in model.encode_features(images)]
        aug_images, aug_masks = [], []
        aug_maps = [[] for _ in feature_maps]
        for index in range(images.shape[0]):
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot = random.randrange(4)
            aug_images.append(
                self._augment(images[index], hflip, vflip, rot))
            aug_masks.append(
                self._augment(masks[index], hflip, vflip, rot))
            for map_index, feature_map in enumerate(feature_maps):
                aug_maps[map_index].append(
                    self._augment(feature_map[index], hflip, vflip, rot))
        images = torch.stack(aug_images)
        masks = torch.stack(aug_masks)
        feature_maps = [torch.stack(values) for values in aug_maps]
        logits = model.decoder(images, feature_maps)
        return (nn.BCEWithLogitsLoss()(logits, masks) +
                dice_loss(logits, masks))

    def state_dict(self, model):
        # Match the previous adapter: only the trainable decoder is saved.
        return model.decoder.state_dict()

    def load_state_dict(self, model, state):
        target = model.decoder.state_dict()
        if all(key in target for key in state):
            decoder_state = state
        elif all(f'decoder.{key}' in state for key in target):
            decoder_state = {key: state[f'decoder.{key}'] for key in target}
        else:
            # Accept a full SAMViTLUNet state dict by selecting decoder.*.
            decoder_state = {
                key: state[f'decoder.{key}'] for key in target
                if f'decoder.{key}' in state}
        model.decoder.load_state_dict(decoder_state)

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        logits = model(prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
