"""BLO-SAM baseline: bilevel-optimized LoRA SAM for limited data.

Official reference: ``3rd/BLO-SAM`` (ICML 2024, MIT claimed in README; no
license file in the checkout).  The official trainer splits the training
set in half for the two bilevel levels.  This port keeps the two-level
update but uses the previous training batch as the meta-validation batch,
because the shared runtime does not expose the validation loader to the
adapter.  This is recorded as a deviation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.models.blo_sam import DiceLoss, build_blo_sam
from baseline_zoo.models.blo_sam.prompt import Prompt
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256


class BLOSAMAdapter(BaselineAdapter):
    id = 'blo_sam'
    display_name = 'BLO-SAM'
    official_source = 'BLO-SAM (self-contained copy)'
    official_entrypoint = 'https://github.com/importZL/BLO-SAM'
    input_size = 256
    official_lr = 5e-3
    official_weight_decay = 1e-1
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'BLO-SAM is a limited-data SAM finetuning method, not a 3-shot '
        'segmenter; this port trains it on the shared 18-image split and '
        'evaluates all protocol datasets.',
        'The official trainer splits the training directory into two halves '
        'for bilevel levels. The shared runtime does not expose the val '
        'loader to adapters, so the meta-validation level uses the previous '
        'training batch (the first batch reuses itself).',
        'Official datasets are preprocessed folders of images/masks with '
        'five foreground classes; this port supports the binary protocol '
        'head (background + one foreground token) and converts GT masks to '
        'long labels at 256 and 64 resolution.',
        'Official optimizer is AdamW lr 5e-3, prompt lr 5e-3, weight decay '
        '0.1, rank 4, batch 1 and OneCycleLR over all iterations. This port '
        'uses the shared epoch budget and scheduler override; prompt '
        'optimizer settings follow the official defaults.',
        'The official image preprocessing uses the model pixel_mean=0 / '
        'pixel_std=1 with RGB images; this port follows that setting and '
        'adds flips/90-degree rotations for the tiny 3-shot split.',
        'No official BLO-SAM weights are present. Training starts from the '
        'official SAM ViT-B checkpoint; prediction rebuilds the same LoRA '
        'architecture and loads the zoo checkpoint.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._ce = nn.CrossEntropyLoss()
        self._dice = DiceLoss(2)
        self._prompt: Prompt | None = None
        self._prev_batch = None
        self._iter = 0

    def add_args(self, parser):
        parser.add_argument(
            '--blo-sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth')
        parser.add_argument('--input-size', type=int, default=256)
        parser.add_argument('--blo-sam-rank', type=int, default=4)
        parser.add_argument('--blo-sam-dice-weight', type=float, default=0.8)
        parser.add_argument('--blo-sam-prompt-lr', type=float, default=5e-3)
        parser.add_argument('--blo-sam-prompt-wd', type=float, default=0.1)
        parser.add_argument('--blo-sam-unrolled', type=int, choices=[0, 1],
                            default=0)

    def build_model(self, args) -> nn.Module:
        self.input_size = int(getattr(args, 'input_size', 256))
        checkpoint = Path(args.blo_sam_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-B checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_b_01ec64.pth')
        self._sam_sha = file_sha256(checkpoint)
        self._prompt = None
        self._prev_batch = None
        self._iter = 0
        return build_blo_sam(
            checkpoint=str(checkpoint),
            image_size=self.input_size,
            num_classes=1,
            rank=int(getattr(args, 'blo_sam_rank', 4)),
        )

    def load_pretrained(self, model, args) -> None:
        # SAM ViT-B is loaded by build_blo_sam; LoRA and the trainable
        # decoder/prompt weights start from their official initialization.
        print('loaded SAM ViT-B base weights before BLO-SAM LoRA injection')

    def configure_trainable(self, model) -> None:
        # LoRA_Sam already froze the encoder and decoder transformer and left
        # no_mask_embed/mask-decoder outputs trainable.  Verify the policy
        # instead of silently changing it.
        trainable = [name for name, p in model.named_parameters()
                     if p.requires_grad]
        if not trainable:
            raise RuntimeError('BLO-SAM has no trainable parameters')
        if not any('no_mask_embed' in name for name in trainable):
            raise RuntimeError('BLO-SAM prompt embedding is not trainable')

    def make_optimizer(self, model, args):
        params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and 'no_mask_embed' not in name
        ]
        return torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.lr),
            total_steps=max(1, int(args.epochs)),
        )

    def _pair(self, sample, rng: random.Random):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = Image.open(sample.mask_path).convert('L').resize(
            (self.input_size, self.input_size), Image.NEAREST)
        image_np = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_np = (np.asarray(mask) > 127).astype(np.int64)
        if rng.random() < 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=1).copy()
        if rng.random() < 0.5:
            image_np = np.flip(image_np, axis=0).copy()
            mask_np = np.flip(mask_np, axis=0).copy()
        k = rng.randrange(4)
        if k:
            image_np = np.rot90(image_np, k).copy()
            mask_np = np.rot90(mask_np, k).copy()
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_np)
        low = F.interpolate(
            mask_t[None, None].float(),
            size=(max(1, self.input_size // 4), max(1, self.input_size // 4)),
            mode='nearest',
        )[0, 0].long()
        return image_t, mask_t, low

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks, lows = [], [], []
        for sample in samples:
            image, mask, low = self._pair(sample, rng)
            images.append(image)
            masks.append(mask)
            lows.append(low)
        return {
            'image': torch.stack(images).to(self.device),
            'label': torch.stack(masks).to(self.device),
            'low_res_label': torch.stack(lows).to(self.device),
        }

    def _loss(self, outputs, low_res_label):
        logits = outputs['low_res_logits']
        ce = self._ce(logits, low_res_label.long())
        dice = self._dice(logits, low_res_label, softmax=True)
        weight = float(getattr(self.args, 'blo_sam_dice_weight', 0.8))
        return (1.0 - weight) * ce + weight * dice

    def train_step(self, model, batch):
        outputs = model(
            batch['image'], multimask_output=True, image_size=self.input_size)
        return self._loss(outputs, batch['low_res_label'])

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        model.train()
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        if self._prompt is None:
            args.num_classes = 1
            args.img_size = self.input_size
            args.dice_param = float(
                getattr(args, 'blo_sam_dice_weight', 0.8))
            args.prompt_base_lr = float(
                getattr(args, 'blo_sam_prompt_lr', 5e-3))
            args.prompt_weight_decay = float(
                getattr(args, 'blo_sam_prompt_wd', 0.1))
            args.base_lr = float(args.lr)
            args.weight_decay = float(args.weight_decay)
            max_iterations = max(1, int(args.epochs) * max(len(loader), 1))
            self._prompt = Prompt(model, args, max_iterations)
        running = 0.0
        seen = 0
        for samples in loader:
            batch = self.prepare_train_batch(samples, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = self.train_step(model, batch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            optimizer.step()
            valid_batch = self._prev_batch or batch
            self._prompt.step(
                batch,
                valid_batch,
                float(optimizer.param_groups[0]['lr']),
                optimizer,
                bool(int(getattr(args, 'blo_sam_unrolled', 0))),
                self._iter,
            )
            self._prev_batch = {
                key: value.detach() for key, value in batch.items()
            }
            self._iter += 1
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        if scheduler is not None:
            scheduler.step()
        return running / max(seen, 1)

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1).unsqueeze(0) / 255.0
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return tensor.to(device), (int(width), int(height))

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        image, (width, height) = prepared
        outputs = model(
            image, multimask_output=True, image_size=self.input_size)
        pred = outputs['masks'].argmax(dim=1)[0].float()
        pred = F.interpolate(
            pred[None, None], size=(height, width), mode='nearest')[0, 0]
        return pred.cpu().numpy() > 0.5

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'rank': int(getattr(args, 'blo_sam_rank', 4)),
            'num_classes': 1,
            'dice_weight': float(
                getattr(args, 'blo_sam_dice_weight', 0.8)),
            'blo_sam_checkpoint': str(getattr(args, 'blo_sam_checkpoint', '')),
            'sam_checkpoint_sha256': self._sam_sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'rank' in config:
            args.blo_sam_rank = int(config['rank'])
        if 'dice_weight' in config:
            args.blo_sam_dice_weight = float(config['dice_weight'])
        if 'blo_sam_checkpoint' in config:
            args.blo_sam_checkpoint = Path(config['blo_sam_checkpoint'])
