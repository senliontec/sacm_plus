"""SANSA baseline adapter (NeurIPS 2025).

SANSA is support-based: reference images and masks are concatenated with the
target as a pseudo-video, and the frozen SAM 2 memory pipeline propagates the
reference concept to the target.  The released training loop uses one
reference plus one target; evaluation uses the protocol's three references
(own domain for seen sets, donor for unseen/novel).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.models.sansa import build_sansa, focal_dice_loss
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256
from baseline_zoo.support import load_support_groups, select_supports

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SANSAAdapter(BaselineAdapter):
    id = 'sansa'
    display_name = 'SANSA'
    official_source = 'SANSA (self-contained SAM 2 + AdaptFormer port)'
    official_entrypoint = 'https://github.com/ClaudiaCuttano/SANSA'
    input_size = 1024
    official_lr = 1e-4
    official_weight_decay = 0.0
    default_grad_clip = 1.0
    train_batch_size = 2
    accumulation_steps = 1
    selection = 'dice'
    support_based = True
    deviations = (
        'SANSA is support-based, not prompt-free: the protocol reference '
        'images and masks drive SAM 2 memory propagation to the target '
        '(own shots for seen test domains, donor shots otherwise). The '
        'released training loop uses one reference + one target, even '
        'though the paper describes a J=3 multi-target objective; this '
        'port follows the released engine. Evaluation uses the native '
        'k-shot path with the protocol three references.',
        'Only the AdaptFormer adapters in the last two Hiera-Large stages '
        'are trained (~9.9M parameters); SAM 2 is frozen. The official '
        'effective batch 32 and 5 strict-few-shot epochs are replaced by '
        'batch 2 (18-image split, 24 GiB target GPUs) and the shared '
        '800-epoch budget, with validation-Dice checkpoint selection.',
        'The released engine uses sigmoid focal loss (alpha 0.25, gamma '
        '2) + Dice, while the paper states BCE + Dice; this port follows '
        'the released code. The optimizer is AdamW lr 1e-4, wd 0, betas '
        '(0.9, 0.999), gradient clipping 1.0 (paper value; the CLI default '
        'is 0.1). The official per-iteration cosine schedule is stepped '
        'once per epoch by the shared runtime.',
        'The released datasets first resize to 518 square (ISIC, '
        'PASCAL-part) or 640 square (LVIS), apply ImageNet normalization, '
        'and then SANSA _preprocess bilinearly resizes to 1024 without '
        'normalizing. This port skips that lossy intermediate and applies '
        'one direct bilinear 1024-square resize followed by the same '
        'ImageNet normalization. Final resolution and normalization match; '
        'the two-step resampling is not reproduced. There is no training '
        'augmentation beyond the resize/normalization.',
        'Support prompts are masks only; the paper also supports points, '
        'boxes and scribbles, but the shared protocol provides support '
        'masks. No official trained adapters are required; training starts '
        'from the official SAM 2 Hiera-Large checkpoint and never '
        'downloads weights.',
    )

    def __init__(self):
        super().__init__()
        self._support_groups = None
        self._support_root: Path | None = None
        self._sam2_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument(
            '--sansa-sam2-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt')
        parser.add_argument(
            '--sansa-sam2-config', default='sam2_hiera_l.yaml',
            choices=['sam2_hiera_l.yaml', 'sam2_hiera_b+.yaml',
                     'sam2_hiera_t.yaml'])
        parser.add_argument('--sansa-adaptformer-stages', type=int,
                            nargs='+', default=[2, 3])
        parser.add_argument('--sansa-channel-factor', type=float, default=0.3)
        parser.add_argument('--sansa-shots', type=int, default=3)
        parser.add_argument('--sansa-support-dataset', default='auto')

    def build_model(self, args):
        self.input_size = int(getattr(args, 'input_size', 1024))
        if self.input_size != 1024:
            raise SystemExit('SANSA uses SAM 2\'s 1024x1024 input')
        checkpoint = Path(getattr(
            args, 'sansa_sam2_checkpoint',
            REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt'))
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM 2 checkpoint: {checkpoint}\n'
                'Upload the official weights under checkpoints/.')
        self._sam2_sha = file_sha256(checkpoint)
        self._support_root = Path(getattr(args, 'data_root', ''))
        return build_sansa(
            sam2_checkpoint=checkpoint,
            sam2_config=str(getattr(
                args, 'sansa_sam2_config', 'sam2_hiera_l.yaml')),
            adaptformer_stages=tuple(int(x) for x in getattr(
                args, 'sansa_adaptformer_stages', [2, 3])),
            channel_factor=float(getattr(
                args, 'sansa_channel_factor', 0.3)),
        )

    def load_pretrained(self, model, args) -> None:
        print('loaded frozen SAM 2 Hiera-Large and inserted AdaptFormer')

    def configure_trainable(self, model) -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('SANSA has no trainable adapters')
        if any(p.requires_grad for p in model.sam.sam_mask_decoder.parameters()):
            raise RuntimeError('SANSA SAM 2 decoder must stay frozen')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.999))

    def make_scheduler(self, optimizer, args):
        # Released schedule: CosineAnnealingLR over all iterations.  The
        # shared runtime steps once per epoch.
        epochs = max(1, int(args.epochs))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _groups(self):
        if self._support_groups is None:
            if self._support_root is None:
                raise RuntimeError('SANSA data_root is not configured')
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

    def _train_support(self, sample, rng: random.Random):
        groups = self._groups()
        candidates = [item for item in groups.get(sample.dataset, [])
                      if item.stem != sample.stem]
        if not candidates:
            candidates = [sample]
        return rng.choice(candidates)

    def _eval_supports(self, dataset: str, shots: int):
        groups = self._groups()
        samples, donor = select_supports(
            dataset, groups, shots=shots,
            override=str(getattr(
                self.args, 'sansa_support_dataset', 'auto')))
        pairs = [self._pair(sample) for sample in samples]
        images = torch.stack([item[0] for item in pairs])
        masks = torch.stack([item[1] for item in pairs])
        return images, masks, donor

    @staticmethod
    def _prompt_dict(support_masks: torch.Tensor, shots: int):
        prompt = {'shots': int(shots), 'prompt_type': 'mask'}
        for b in range(support_masks.shape[0]):
            prompt[b] = {}
            for i in range(shots):
                prompt[b][i] = {
                    'prompt': support_masks[b, i].unsqueeze(0).unsqueeze(0),
                    'prompt_type': 'mask',
                }
        return prompt

    def prepare_train_batch(self, samples, rng: random.Random):
        frames, masks = [], []
        for sample in samples:
            support = self._train_support(sample, rng)
            support_image, support_mask = self._pair(support)
            query_image, query_mask = self._pair(sample)
            frames.append(torch.stack([support_image, query_image]))
            masks.append(torch.stack([support_mask, query_mask]))
        samples_t = torch.stack(frames).to(self.device)
        masks_t = torch.stack(masks).to(self.device)
        return {
            'samples': samples_t,
            'masks': masks_t,
            'prompt': self._prompt_dict(masks_t, 1),
        }

    def train_step(self, model, batch):
        logits = model(batch['samples'], batch['prompt'])
        return focal_dice_loss(logits, batch['masks'], num_frames=1)

    def prepare_eval_sample(self, sample, device):
        query, _ = self._pair(sample)
        shots = int(getattr(self.args, 'sansa_shots', 3))
        supports, support_masks, _ = self._eval_supports(
            sample.dataset, shots)
        frames = torch.cat([supports, query.unsqueeze(0)], dim=0)
        support_masks = support_masks.unsqueeze(0).to(device)
        from PIL import Image
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        return {
            'samples': frames.unsqueeze(0).to(device),
            'prompt': self._prompt_dict(support_masks, shots),
            'size': (int(width), int(height)),
        }

    @torch.no_grad()
    def predict_sample(self, model, prepared, sample):
        logits = model(prepared['samples'], prepared['prompt'])
        # The real SANSA forward returns (B*T, H, W); select the query
        # frame and restore the NCHW channel dimension before interpolating.
        query_logits = logits[-1][None, None]
        width, height = prepared['size']
        query_logits = F.interpolate(
            query_logits, size=(height, width), mode='bilinear',
            align_corners=False)
        return torch.sigmoid(query_logits[0, 0]).cpu().numpy() > 0.5

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------
    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'sam2_checkpoint': str(
                getattr(args, 'sansa_sam2_checkpoint', '')),
            'sam2_config': str(getattr(
                args, 'sansa_sam2_config', 'sam2_hiera_l.yaml')),
            'adaptformer_stages': [
                int(x) for x in getattr(
                    args, 'sansa_adaptformer_stages', [2, 3])],
            'channel_factor': float(getattr(
                args, 'sansa_channel_factor', 0.3)),
            'shots': int(getattr(args, 'sansa_shots', 3)),
            'support_dataset': str(getattr(
                args, 'sansa_support_dataset', 'auto')),
            'data_root': str(getattr(args, 'data_root', '')),
            'sam2_sha256': self._sam2_sha,
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            self.input_size = int(config['input_size'])
            args.input_size = self.input_size
        if 'sam2_checkpoint' in config:
            args.sansa_sam2_checkpoint = Path(config['sam2_checkpoint'])
        for key, attribute in (
                ('sam2_config', 'sansa_sam2_config'),
                ('support_dataset', 'sansa_support_dataset')):
            if key in config:
                setattr(args, attribute, str(config[key]))
        if 'adaptformer_stages' in config:
            args.sansa_adaptformer_stages = [
                int(x) for x in config['adaptformer_stages']]
        if 'channel_factor' in config:
            args.sansa_channel_factor = float(config['channel_factor'])
        if 'shots' in config:
            args.sansa_shots = int(config['shots'])
        if 'data_root' in config:
            self._support_root = Path(config['data_root'])
