"""TopoLoRA-SAM baseline: prompt-free SAM ViT-B + LoRA + conv adapter.

Official reference: ``3rd/Seglab`` (TopoLoRA-SAM, MIT).  The vendored model
keeps the official LoRA injection, depthwise-separable conv adapter, frozen
SAM, trainable mask decoder, empty-prompt decode, BCE+Dice+clDice loss and
the 50-epoch AdamW/cosine recipe.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from baseline_zoo.data import load_mask, load_rgb
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

AUG = {
    'scale': [0.8, 1.2],
    'hflip': 0.5,
    'vflip': 0.5,
    'color_jitter_p': 0.2,
    'gauss_noise_p': 0.0,
}


class TopoLoRAAdapter(BaselineAdapter):
    id = 'topolora'
    display_name = 'TopoLoRA-SAM'
    official_source = 'Seglab / TopoLoRA-SAM (self-contained copy)'
    official_entrypoint = 'https://github.com/salimkhazem/Seglab'
    input_size = 384
    official_epochs = 50
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 1
    accumulation_steps = 4
    selection = 'dice'
    deviations = (
        'Uses the official ViT-B backbone, not the ViT-L used by several '
        'other SAM baselines in this zoo; the table records the backbone.',
        'The official AdamW 1e-4/1e-4, cosine, batch-1 x4 accumulation and '
        'gradient-clip 1.0 recipe is preserved; the official 50-epoch '
        'budget is replaced by 200 epochs for the 18-image 3-shot protocol.',
        'No mixed precision: upstream trains with 16-mixed. This changes '
        'speed and numerical details, not the model graph.',
        'Uses the shared 18/6 split and 20-dataset test protocol instead of '
        'Seglab\'s HuggingFace train/validation/test split; both select the '
        'checkpoint by validation Dice.',
        'Predictions are exported at original GT resolution; upstream '
        'evaluates at 384x384.',
        'No official TopoLoRA-SAM weights are present in the Seglab '
        'repository; training starts from the official SAM ViT-B '
        'checkpoint.',
        'The upstream PyTorch Lightning trainer, CSV/W&B loggers and '
        'artifact figures are replaced by the shared baseline runtime; the '
        'model, loss, optimizer and schedule are preserved, and accumulation '
        'uses the same batch-1 x4 granularity (partial-group handling differs; '
        'see below).',
        'Albumentations 1.3.1 has no per-Compose seed; the adapter seeds the '
        'global Python/NumPy RNG from the shared per-epoch RNG before each '
        'train batch.',
        'The prompt path is prompt-free: points/boxes/masks are all None and '
        'the learned no_mask_embed dense prompt is used, matching upstream.',
        'The shared runtime rescales the final partial accumulation group by '
        'accum/pending and filters optimizer parameters to requires_grad; '
        'upstream Lightning accumulation has no equivalent partial-group '
        'rescale. The 18-image batch-1 x4 run ends with a 2-step group.',
        'No test-time augmentation.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None
        self._train_tf = None
        self._eval_tf = None

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth')
        parser.add_argument('--sam-type', default='vit_b')
        parser.add_argument('--input-size', type=int, default=384)
        parser.add_argument('--lora-r', type=int, default=8)
        parser.add_argument('--lora-alpha', type=float, default=16.0)
        parser.add_argument('--lora-dropout', type=float, default=0.0)
        parser.add_argument('--adapter-hidden', type=int, default=256)
        parser.add_argument('--adapter-kernel', type=int, default=3)

    def build_model(self, args) -> nn.Module:
        from baseline_zoo.models.topolora import build_topolora, build_transforms

        self.input_size = int(args.input_size)
        self._train_tf = build_transforms(
            self.input_size, train=True, sar=False, aug=AUG)
        self._eval_tf = build_transforms(
            self.input_size, train=False, sar=False)

        training = getattr(args, 'stage', 'train') in ('train', 'all')
        checkpoint = Path(args.sam_checkpoint)
        if training and not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-B checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_b_01ec64.pth')
        source = str(checkpoint) if training else None
        return build_topolora(
            sam_type=args.sam_type,
            checkpoint=source,
            lora_enabled=True,
            lora_r=int(args.lora_r),
            lora_alpha=float(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            adapter_enabled=True,
            adapter_hidden=int(args.adapter_hidden),
            adapter_kernel=int(args.adapter_kernel),
            train_mask_decoder=True,
        )

    def load_pretrained(self, model, args) -> None:
        checkpoint = Path(args.sam_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f'missing SAM ViT-B checkpoint: {checkpoint}')
        self._sam_sha = file_sha256(checkpoint)
        print('loaded SAM ViT-B base weights before LoRA injection')

    def configure_trainable(self, model) -> None:
        for name, parameter in model.sam.named_parameters():
            if '.lora_A' in name or '.lora_B' in name:
                parameter.requires_grad_(True)
            elif name.startswith('mask_decoder.'):
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        if model.adapter is not None:
            for parameter in model.adapter.parameters():
                parameter.requires_grad_(True)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('TopoLoRA-SAM has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(args.epochs), 1))

    def _preprocess(self, sample, rng: random.Random, train: bool):
        image = np.asarray(load_rgb(sample.image_path))
        mask = load_mask(sample.mask_path).astype(np.uint8)
        if train:
            seed = rng.randrange(0, 2 ** 32)
            random.seed(seed)
            np.random.seed(seed)
            transformed = self._train_tf(image=image, mask=mask)
        else:
            transformed = self._eval_tf(image=image, mask=mask)
        return transformed['image'].float(), transformed['mask'].long()

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._preprocess(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def train_step(self, model, batch):
        from baseline_zoo.models.topolora.topo_losses import cldice_loss, dice_loss

        images, masks = batch
        logits = model(images)
        target = masks.unsqueeze(1).float()
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return bce + dice_loss(prob, target) + 0.5 * cldice_loss(
            prob, target)

    def prepare_eval_sample(self, sample, device):
        image, _ = self._preprocess(sample, random.Random(0), train=False)
        return image.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        logits = model(prepared)
        prob = torch.sigmoid(logits)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        prob = F.interpolate(
            prob, size=(height, width), mode='bilinear',
            align_corners=False)
        return prob[0, 0].cpu().numpy() > 0.5

    def save_config(self, args):
        checkpoint = getattr(args, 'sam_checkpoint', None)
        return {
            'input_size': int(self.input_size),
            'sam_type': str(getattr(args, 'sam_type', 'vit_b')),
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': self._sam_sha,
            'lora_r': int(getattr(args, 'lora_r', 8)),
            'lora_alpha': float(getattr(args, 'lora_alpha', 16.0)),
            'lora_dropout': float(getattr(args, 'lora_dropout', 0.0)),
            'adapter_hidden': int(getattr(args, 'adapter_hidden', 256)),
            'adapter_kernel': int(getattr(args, 'adapter_kernel', 3)),
        }

    def restore_config(self, args, config):
        for key in ('input_size', 'lora_r', 'lora_alpha', 'lora_dropout',
                    'adapter_hidden', 'adapter_kernel'):
            if key in config:
                setattr(args, key, config[key])
        if 'sam_type' in config:
            args.sam_type = config['sam_type']
        self.input_size = int(getattr(args, 'input_size', 384))
        stored = config.get('sam_checkpoint_sha256')
        current = getattr(args, 'sam_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(f'WARNING: base SAM checkpoint differs from training: '
                      f'{current}; prediction uses the checkpoint weights')
