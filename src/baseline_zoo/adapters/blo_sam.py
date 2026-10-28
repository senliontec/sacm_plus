"""BLO-SAM baseline: bilevel-optimized LoRA SAM for limited data.

Official reference: ``3rd/BLO-SAM`` (ICML 2024, MIT claimed in README; no
license file in the checkout).  The official trainer splits the training
set once into disjoint D1/D2 halves: the network is updated on D1 and the
learnable no-mask prompt on D2.  This port keeps that split and uses D2
for checkpoint selection, matching the official trainer.
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
        'The official bilevel trainer splits the training set once into '
        'disjoint D1/D2 halves. This port uses a fixed seeded 9/9 split '
        'across epochs, stratified by source domain so all six pooled '
        'domains appear in both levels; the upper-level prompt update runs '
        'on D2 and checkpoint selection uses D2 Dice. Unlike the released '
        'loop, which repeatedly reads only the first D2 batch, the port '
        'cycles through all D2 batches.',
        'Official datasets are preprocessed folders of images/masks with '
        'five foreground classes; this port supports the binary protocol '
        'head (background + one foreground token) and converts GT masks to '
        'long labels at 256 and 64 resolution.',
        'Official optimizer is AdamW lr 5e-3, prompt lr 5e-3, weight decay '
        '0.1, rank 4, batch 1 and lr_i = lr_0 (1 - i / Iter_max)^0.9. This '
        'port follows that per-step polynomial schedule; the epoch budget '
        'remains configurable (the model config uses the official 100).',
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
        self._iter = 0
        self._d2_dice: float | None = None
        self._split_key: tuple | None = None
        self._d1_samples: list | None = None
        self._d2_samples: list | None = None
        self._multimask_output = False  # official binary setting

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
        self._iter = 0
        self._d2_dice = None
        self._split_key = None
        self._d1_samples = None
        self._d2_samples = None
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
        # The official trainer applies lr_i = lr_0 (1 - i / Iter_max)^0.9
        # once per global step; train_epoch owns that schedule.
        return None

    def _pair(self, sample, rng: random.Random, augment: bool = True):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BICUBIC)
        mask = Image.open(sample.mask_path).convert('L').resize(
            (self.input_size, self.input_size), Image.NEAREST)
        image_np = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_np = (np.asarray(mask) > 127).astype(np.int64)
        if augment and rng.random() < 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=1).copy()
        if augment and rng.random() < 0.5:
            image_np = np.flip(image_np, axis=0).copy()
            mask_np = np.flip(mask_np, axis=0).copy()
        k = rng.randrange(4) if augment else 0
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

    def prepare_train_batch(self, samples, rng: random.Random,
                            augment: bool = True):
        images, masks, lows = [], [], []
        for sample in samples:
            image, mask, low = self._pair(sample, rng, augment=augment)
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
            batch['image'], multimask_output=self._multimask_output,
            image_size=self.input_size)
        return self._loss(outputs, batch['low_res_label'])

    @torch.no_grad()
    def _meta_dice(self, model, batches, rng: random.Random) -> float | None:
        """Dice on the upper-level D2 subset, used for checkpoint selection."""
        if not batches:
            return None
        was_training = model.training
        model.eval()
        scores = []
        for samples in batches:
            # Deterministic selection, as the official validate transform.
            batch = self.prepare_train_batch(samples, rng, augment=False)
            outputs = model(
                batch['image'], multimask_output=self._multimask_output,
                image_size=self.input_size)
            # Official validate: foreground Dice from the low-resolution
            # logits and low-resolution target.
            pred = outputs['low_res_logits'].argmax(dim=1)
            target = batch['low_res_label'].long()
            true_positive = ((pred == 1) & (target == 1)).sum().float()
            false_positive = ((pred == 1) & (target != 1)).sum().float()
            false_negative = ((pred != 1) & (target == 1)).sum().float()
            denom = 2 * true_positive + false_positive + false_negative
            scores.append(
                float(2 * true_positive / denom) if denom > 0 else 0.0)
        model.train(was_training)
        return float(np.mean(scores)) if scores else None

    def selection_score(self, metrics, per_dataset):
        # The paper selects the checkpoint on the upper-level D2 subset.
        if (self._d2_dice is not None
                and getattr(self.args, 'selection', 'dice') == 'dice'):
            return float(self._d2_dice)
        return super().selection_score(metrics, per_dataset)

    def _bilevel_split(self, samples, seed: int):
        """Fixed, domain-stratified D1/D2 membership.

        The official trainer splits one dataset in half.  This protocol
        pools six domains, so an unstratified split can leave a domain out
        of D1 entirely; the port keeps fixed disjoint halves but allocates
        each domain proportionally to both levels.
        """
        key = (int(seed), tuple(sorted(
            (str(s.image_path), str(s.mask_path)) for s in samples)))
        if self._split_key != key:
            by_domain: dict[str, list] = {}
            for sample in sorted(
                    samples,
                    key=lambda s: (str(s.image_path), str(s.mask_path))):
                by_domain.setdefault(sample.dataset, []).append(sample)
            rng = random.Random(int(seed))
            domains = sorted(by_domain)
            rng.shuffle(domains)
            for domain in domains:
                rng.shuffle(by_domain[domain])

            total = len(samples)
            half = max(1, total // 2)
            exact = {
                domain: len(by_domain[domain]) * half / total
                for domain in domains
            }
            alloc = {domain: int(exact[domain]) for domain in domains}
            remainder = half - sum(alloc.values())
            for domain in sorted(
                    domains, key=lambda d: exact[d] - alloc[d],
                    reverse=True)[:max(0, remainder)]:
                alloc[domain] += 1

            self._d1_samples = []
            self._d2_samples = []
            for domain in domains:
                group = by_domain[domain]
                self._d1_samples.extend(group[:alloc[domain]])
                self._d2_samples.extend(group[alloc[domain]:])
            if not self._d1_samples:
                self._d1_samples = list(self._d2_samples[:1])
            if not self._d2_samples:
                self._d2_samples = list(self._d1_samples)
            self._split_key = key
        return list(self._d1_samples), list(self._d2_samples)

    def train_epoch(self, model, loader, optimizer, scheduler, args) -> float:
        model.train()
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        samples = list(getattr(getattr(loader, 'dataset', None),
                               'samples', []))
        if not samples:
            raise RuntimeError('BLO-SAM requires a sample list for D1/D2')
        d1_samples, d2_samples = self._bilevel_split(samples, args.seed)
        # Membership is fixed; only the order within each half changes.
        order_rng = random.Random(
            int(args.seed) * 100003 + int(getattr(self, '_epoch', 0)))
        order_rng.shuffle(d1_samples)
        order_rng.shuffle(d2_samples)
        batch_size = max(1, int(getattr(args, 'batch_size', 1)))
        d1_batches = [d1_samples[i:i + batch_size]
                      for i in range(0, len(d1_samples), batch_size)]
        d2_batches = [d2_samples[i:i + batch_size]
                      for i in range(0, len(d2_samples), batch_size)]
        steps = max(len(d1_batches), len(d2_batches))
        max_iterations = max(1, int(args.epochs) * steps)

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
        if self._prompt is None:
            self._prompt = Prompt(model, args, max_iterations)

        running = 0.0
        seen = 0
        for step in range(steps):
            d1 = self.prepare_train_batch(
                d1_batches[step % len(d1_batches)], rng)
            # D2 is the official meta-validation half: keep it resize-only,
            # like the released validloader.  D1 alone gets the task
            # adaptation flips/rotations.
            d2 = self.prepare_train_batch(
                d2_batches[step % len(d2_batches)], rng, augment=False)
            optimizer.zero_grad(set_to_none=True)
            loss = self.train_step(model, d1)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            optimizer.step()
            # Official order: the prompt step sees the LR used by the
            # network step, then the polynomial LR is advanced for the next
            # iteration.
            current_lr = float(optimizer.param_groups[0]['lr'])
            self._prompt.step(
                d1,
                d2,
                current_lr,
                optimizer,
                bool(int(getattr(args, 'blo_sam_unrolled', 0))),
                self._iter,
            )
            lr = args.base_lr * max(
                0.0, 1.0 - self._iter / max_iterations) ** 0.9
            for group in optimizer.param_groups:
                group['lr'] = lr
            self._iter += 1
            running += float(loss.detach()) * len(d1_batches[step % len(
                d1_batches)])
            seen += len(d1_batches[step % len(d1_batches)])
        self._d2_dice = self._meta_dice(model, d2_batches, rng)
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
            image, multimask_output=self._multimask_output,
            image_size=self.input_size)
        # outputs['masks'] are logits at the 256 input resolution.  Upsample
        # the foreground logit before thresholding so thin structures are not
        # block-quantized by a nearest resize from 256 to the GT resolution.
        foreground = outputs['masks'][0, 1][None, None]
        prob = torch.sigmoid(F.interpolate(
            foreground, size=(height, width), mode='bilinear',
            align_corners=False))[0, 0]
        return prob.cpu().numpy() > 0.5

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
