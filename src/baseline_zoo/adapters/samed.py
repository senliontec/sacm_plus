"""SAMed baseline: prompt-free LoRA SAM for binary segmentation.

Official reference: ``3rd/SAMed`` (MICCAI 2023, MIT).  SAMed uses SAM ViT-B
with rank-4 LoRA in the image encoder, a trainable prompt encoder and the
class-wise mask decoder.  No points, boxes or masks are passed to the prompt
encoder, so inference is prompt-free and oracle-free.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from torch import nn
from torch.utils.data import WeightedRandomSampler

from baseline_zoo.data import load_mask, load_rgb, load_split_samples
from baseline_zoo.models.samed import (
    ForegroundDiceLoss,
    SoftClDiceLoss,
    build_samed,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256


class SAMedAdapter(BaselineAdapter):
    id = 'samed'
    display_name = 'SAMed'
    official_source = 'hitachinsk/SAMed (self-contained copy)'
    official_entrypoint = 'https://github.com/hitachinsk/SAMed'
    input_size = 512
    # The paper's 5e-3 is for an effective batch of 24 (2 x 12 GPUs).
    # The project runs one GPU at batch 4, so the default is the linearly
    # scaled 1e-3; ``paper_lr`` keeps the upstream value for provenance.
    paper_lr = 5e-3
    official_lr = 1e-3
    official_weight_decay = 0.1
    train_batch_size = 4
    default_grad_clip = 1.0
    selection = 'dice'
    deviations = (
        'SAMed is a fully supervised per-dataset segmenter, not a few-shot '
        'matcher; this port trains one binary model on the shared 18-image '
        'train split and evaluates it on all 20 datasets.',
        'Official Synapse training uses batch 24 and 200 epochs (stop at '
        '159); the 3-shot split has 18 images, so this port uses batch 4 and '
        'the full 800-epoch project budget. The paper lr 5e-3 was tuned for '
        'batch 24; the port default is 1e-3 (~linear scaling to batch 4), '
        'weight decay 0.1 is retained, and HPO searches 1e-4-5e-3.',
        'The official 8-class head is replaced by a binary head '
        '(background + foreground); the released 8-class SAMed LoRA '
        'checkpoints are absent and not shape-compatible.',
        'The binary objective is adapted to the tiny curvilinear foreground: '
        'class-weighted CE (background 0.1, foreground 1.0) plus '
        'foreground-only Dice on the 128x128 low-res logits, a soft-clDice '
        'topology term (weight 0.1), and an auxiliary full-resolution CE+Dice '
        'term (weight 0.5). The official equal-weight class-wise CE/Dice is '
        'kept only as the SAMed reference recipe.',
        'A zero-initialized full-resolution residual refinement head reads '
        'the input image and corrects the bilinearly upsampled 128x128 SAMed '
        'logits; it is a task adaptation for thin structures, not part of '
        'the released SAMed architecture. It is supervised by the auxiliary '
        'term and can be disabled with --samed-high-res-head 0.',
        'Training oversamples images by inverse sqrt foreground ratio '
        '(WeightedRandomSampler) so the 0.17%-8% foreground images get more '
        'updates; validation remains the untouched 6-image split.',
        'Official training uses 224x224 training data upsampled to 512; '
        'this port resizes native 2D images directly to 512. The official '
        '2D test branch is also direct 512; the 224->512 resampling belongs '
        'to the 3D volume path. Gradient clipping 1.0 is the adapter default '
        'for stability at batch 4.',
        'The official CT intensity window is not applicable to the natural '
        'and medical RGB/grayscale protocol images; grayscale is replicated '
        'to RGB and pixels are scaled to [0,1] with SAM pixel_mean=0 / '
        'pixel_std=1.',
        'Checkpoint selection uses the shared 6-image validation Dice; the '
        'official trainer has no best-checkpoint rule and tests epoch 159.',
        'Official train.py defaults to SGD unless --AdamW is passed; the '
        'README training command uses --AdamW. This port follows the README '
        'AdamW recipe and discloses the source-default ambiguity. Official '
        '2-GPU DataParallel/batch24 and NumPy worker RNG are replaced by '
        'single-GPU/batch4 and the shared seeded runtime.',
        'The official 250-iteration warmup is preserved in iteration units: '
        'with the 18-image loader it spans about 50 epochs at batch 4 '
        '(250 / ceil(18 / 4)). The polynomial decay is stepped per epoch; '
        'the poly horizon follows the configured epochs (800 for the shared '
        'project budget), so the final epoch is not assigned lr=0.',
        'Predictions use argmax of the two-class full-resolution logits at '
        'the configured input size (512 by default) followed by '
        'nearest-neighbour resize to the original GT resolution, matching '
        'the official 2D evaluator. The official 3D volume path resamples '
        '224->512; that path is not used by this 2D protocol.',
        'No mixed precision and no test-time augmentation.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._fg_dice = ForegroundDiceLoss()
        self._cldice = SoftClDiceLoss()

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth')
        parser.add_argument('--sam-type', default='vit_b')
        parser.add_argument('--input-size', type=int, default=512)
        parser.add_argument('--lora-rank', type=int, default=4)
        parser.add_argument(
            '--samed-bg-weight', type=float, default=0.1,
            help='CE weight for the background class (foreground is 1.0)')
        parser.add_argument(
            '--samed-fg-weight', type=float, default=1.0,
            help='CE weight for the foreground class')
        parser.add_argument(
            '--samed-aux-weight', type=float, default=0.5,
            help='weight of the full-resolution auxiliary CE+Dice term; '
                 '0 disables it')
        parser.add_argument(
            '--samed-cldice-weight', type=float, default=0.1,
            help='soft-clDice topology loss weight on low-res logits; '
                 '0 disables it')
        parser.add_argument(
            '--samed-oversample', type=int, choices=[0, 1], default=1,
            help='oversample low-foreground training images')
        parser.add_argument(
            '--samed-high-res-head', type=int, choices=[0, 1], default=1,
            help='add the full-resolution residual refinement head')

    def build_model(self, args) -> nn.Module:
        self.input_size = int(args.input_size)
        training = getattr(args, 'stage', 'train') in ('train', 'all')
        checkpoint = Path(args.sam_checkpoint)
        if training and not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-B checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_b_01ec64.pth')
        source = str(checkpoint) if training else None
        return build_samed(
            image_size=self.input_size,
            num_classes=1,
            sam_checkpoint=source,
            rank=int(args.lora_rank),
            sam_type=args.sam_type,
            use_high_res_head=bool(
                int(getattr(args, 'samed_high_res_head', 1))),
        )

    def load_pretrained(self, model, args) -> None:
        checkpoint = Path(args.sam_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f'missing SAM ViT-B checkpoint: {checkpoint}')
        self._sam_sha = file_sha256(checkpoint)
        print('loaded SAM ViT-B base weights before SAMed LoRA injection')

    def configure_trainable(self, model) -> None:
        for name, parameter in model.named_parameters():
            if ('image_encoder' in name and 'linear_a' not in name
                    and 'linear_b' not in name):
                parameter.requires_grad_(False)
            elif 'prompt_encoder' in name or 'mask_decoder' in name:
                parameter.requires_grad_(True)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('SAMed has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    @staticmethod
    def _warmup_epochs(args) -> int:
        """Convert the official 250-iteration warmup to epoch units."""
        train_images = 18
        data_root = getattr(args, 'data_root', None)
        if data_root is not None and (
                Path(data_root) / 'split_manifest.csv').is_file():
            train_images = len(load_split_samples(Path(data_root), 'train'))
        batch_size = max(
            1, int(getattr(args, 'batch_size', SAMedAdapter.train_batch_size)))
        batches_per_epoch = max(1, math.ceil(train_images / batch_size))
        return max(1, math.ceil(250 / batches_per_epoch))

    def make_scheduler(self, optimizer, args):
        warmup_epochs = self._warmup_epochs(args)
        # Follow the configured/CLI epoch budget.  The extra step keeps the
        # last training epoch nonzero; no epoch count is hardcoded here.
        total = max(int(args.epochs), 1) + 1

        def lr_lambda(epoch: int) -> float:
            warmup = min(1.0, (epoch + 1) / warmup_epochs)
            poly = max(0.0, 1.0 - epoch / total) ** 0.9
            return warmup * poly

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _rotate_pair(image: np.ndarray, mask: np.ndarray,
                     rng: random.Random):
        if rng.random() > 0.5:
            k = rng.randrange(4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
            axis = rng.randrange(2)
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
        elif rng.random() > 0.5:
            angle = rng.randint(-20, 20)
            image = nd_rotate(image, angle, order=0, reshape=False)
            mask = nd_rotate(mask, angle, order=0, reshape=False)
        return image, mask

    def _preprocess(self, sample, rng: random.Random, train: bool):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.uint8)
        image = np.asarray(
            Image.fromarray(image).resize(
                (self.input_size, self.input_size), Image.BICUBIC))
        mask = np.asarray(
            Image.fromarray(mask).resize(
                (self.input_size, self.input_size), Image.NEAREST))
        if train:
            image, mask = self._rotate_pair(image, mask, rng)

        image_t = torch.from_numpy(
            image.astype(np.float32).copy()).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask.astype(np.int64).copy())
        low_size = max(1, self.input_size // 4)
        low_res = F.interpolate(
            mask_t[None, None].float(), size=(low_size, low_size),
            mode='nearest').squeeze(1).squeeze(0).long()
        return image_t, mask_t, low_res

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks, low_res = [], [], []
        for sample in samples:
            image, mask, low = self._preprocess(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
            low_res.append(low)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device),
                torch.stack(low_res).to(self.device))

    def _loss_weights(self, logits):
        args = getattr(self, 'args', None)
        bg = float(getattr(args, 'samed_bg_weight', 0.1))
        fg = float(getattr(args, 'samed_fg_weight', 1.0))
        return torch.tensor([bg, fg], dtype=logits.dtype,
                            device=logits.device)

    def make_train_sampler(self, samples, args):
        if not int(getattr(args, 'samed_oversample', 1)):
            return None
        if not samples:
            return None
        weights = []
        for sample in samples:
            foreground = float(load_mask(sample.mask_path).mean())
            weights.append(1.0 / math.sqrt(max(foreground, 1e-4)))
        generator = torch.Generator()
        generator.manual_seed(int(getattr(args, 'seed', 0)))
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
            generator=generator)

    def train_step(self, model, batch):
        images, targets, low_res = batch
        low_logits, full_logits = model(images)
        weights = self._loss_weights(low_logits)
        ce = F.cross_entropy(low_logits, low_res, weight=weights)
        dice = self._fg_dice(low_logits, low_res, softmax=True)
        loss = 0.2 * ce + 0.8 * dice
        args = getattr(self, 'args', None)
        cldice_weight = float(getattr(args, 'samed_cldice_weight', 0.1))
        if cldice_weight > 0:
            loss = loss + cldice_weight * self._cldice(
                low_logits, low_res, softmax=True)
        aux_weight = float(getattr(args, 'samed_aux_weight', 0.5))
        if aux_weight > 0:
            aux_ce = F.cross_entropy(
                full_logits, targets, weight=weights)
            aux_dice = self._fg_dice(
                full_logits, targets, softmax=True)
            loss = loss + aux_weight * (0.2 * aux_ce + 0.8 * aux_dice)
        return loss

    def prepare_eval_sample(self, sample, device):
        image, _, _ = self._preprocess(
            sample, random.Random(0), train=False)
        return image.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        _, full_logits = model(prepared)
        pred = torch.argmax(full_logits, dim=1)[0].float()
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        pred = F.interpolate(
            pred[None, None], size=(height, width), mode='nearest')[0, 0]
        return pred.cpu().numpy() > 0.5

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        return {
            'input_size': int(self.input_size),
            'sam_type': str(getattr(args, 'sam_type', 'vit_b')),
            'lora_rank': int(getattr(args, 'lora_rank', 4)),
            'num_classes': 1,
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': self._sam_sha,
            'samed_bg_weight': float(
                getattr(args, 'samed_bg_weight', 0.1)),
            'samed_fg_weight': float(
                getattr(args, 'samed_fg_weight', 1.0)),
            'samed_aux_weight': float(
                getattr(args, 'samed_aux_weight', 0.5)),
            'samed_cldice_weight': float(
                getattr(args, 'samed_cldice_weight', 0.1)),
            'samed_oversample': int(
                getattr(args, 'samed_oversample', 1)),
            'samed_high_res_head': int(
                getattr(args, 'samed_high_res_head', 1)),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'lora_rank'):
            if key in config:
                setattr(args, key, config[key])
        for key in ('samed_bg_weight', 'samed_fg_weight',
                    'samed_aux_weight', 'samed_cldice_weight'):
            if key in config:
                setattr(args, key, float(config[key]))
        if 'samed_oversample' in config:
            setattr(args, 'samed_oversample', int(config['samed_oversample']))
        if 'samed_high_res_head' in config:
            setattr(args, 'samed_high_res_head',
                    int(config['samed_high_res_head']))
        else:
            # Checkpoints written before the refinement head existed must
            # not be loaded into a model that has extra state_dict keys.
            setattr(args, 'samed_high_res_head', 0)
        if 'sam_type' in config:
            args.sam_type = config['sam_type']
        self.input_size = int(getattr(args, 'input_size', 512))
        stored = config.get('sam_checkpoint_sha256')
        current = getattr(args, 'sam_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(f'WARNING: base SAM checkpoint differs from training: '
                      f'{current}; prediction uses the checkpoint weights')
