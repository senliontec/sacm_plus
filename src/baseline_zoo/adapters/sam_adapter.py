"""SAM-Adapter baseline: prompt-free SAM ViT-L with FFT prompt injection.

Official reference: ``3rd/SAM-Adapter-PyTorch`` (ICCV 2023).  The vendored
model in ``baseline_zoo.models.sam_adapter`` keeps the official ViT-L image
encoder, FFT high-pass prompt generator, per-block prompt injection and
prompt-free mask decoder.  This adapter keeps the official binary loss,
freeze policy, optimizer/schedule, preprocessing and inference path.
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
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter, file_sha256

PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SAMAdapter(BaselineAdapter):
    id = 'sam_adapter'
    display_name = 'SAM-Adapter'
    official_source = 'SAM-Adapter-PyTorch (self-contained copy)'
    official_entrypoint = 'https://github.com/tianrun-chen/SAM-Adapter-PyTorch'
    input_size = 1024
    official_lr = 2e-4
    # Upstream constructs AdamW with only lr, so PyTorch's default
    # weight_decay=0.01 is the executable recipe.
    official_weight_decay = 0.01
    train_batch_size = 1
    accumulation_steps = 2
    selection = 'dice'
    deviations = (
        'Retrained from the official SAM ViT-L checkpoint. The authors\' '
        'SAM-Adapter Google-Drive weights are not part of the released tree '
        'and are not used.',
        'Uses the official AdamW 2e-4, cosine eta_min 1e-7 recipe; the '
        'official 20-epoch budget is replaced by 800 epochs for the 18-image '
        '3-shot protocol. The upstream optimizer passes only lr, so the '
        'executable weight decay is PyTorch AdamW default 0.01; this port '
        'uses 0.01.',
        'Checkpoint selection uses the shared 6-image validation Dice, not '
        'the official COD S-measure/E-measure/weighted-F/MAE validation.',
        'Predictions are exported at original GT resolution; the upstream '
        'COD evaluator compares at the resized 1024x1024 resolution.',
        'The executable upstream TrainDataset applies a random horizontal '
        'flip (p=0.5) even though its YAML says augment: false; the port '
        'follows the code. No other augmentation is added.',
        'The upstream data_norm block is unused by the executable training '
        'and eval loops; the port uses the dataset wrapper\'s ImageNet '
        'normalization.',
        'Official training uses 4 GPUs x batch 2 (effective 8), workers=8 '
        'and an unseeded global RNG; this port runs one GPU with batch 1 + '
        '2-step accumulation (effective batch 2) and the shared seeded '
        'runtime.',
        'The official batch size is 2. This port defaults to batch 1 + '
        '2-step accumulation and gradient checkpointing because batch 2 at '
        '1024x1024 OOMs on a 24 GiB card; effective batch remains 2.',
        'Gradient checkpointing is on by default for the ViT blocks; it '
        'changes memory only, not the computation.',
        'The upstream prompt path hardcodes bs=1 and broadcasts one prompt '
        'set over the image batch; the port preserves that behaviour.',
        'The dead evp branch and the upstream DDP/training plumbing are '
        'removed; no model computation or parameter names changed.',
        'No test-time augmentation beyond the prompt-free forward pass.',
    )

    def __init__(self):
        super().__init__()
        self._sam_sha: str | None = None

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--input-size', type=int, default=1024)
        parser.add_argument(
            '--grad-checkpointing', action='store_true', default=True,
            help='checkpoint ViT blocks to reduce activation memory '
                 '(default: on; memory-only)')

    def build_model(self, args) -> nn.Module:
        from baseline_zoo.models.sam_adapter import build_sam_adapter

        if int(args.input_size) != 1024:
            raise SystemExit('SAM-Adapter expects --input-size 1024')
        self.input_size = int(args.input_size)
        model = build_sam_adapter(inp_size=self.input_size)
        model.image_encoder.grad_checkpointing = bool(
            getattr(args, 'grad_checkpointing', True))
        return model

    def load_pretrained(self, model, args) -> None:
        checkpoint = Path(args.sam_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f'missing SAM ViT-L checkpoint: {checkpoint}\n'
                'download: https://dl.fbaipublicfiles.com/segment_anything/'
                'sam_vit_l_0b3195.pth')
        payload = torch.load(
            checkpoint, map_location='cpu', weights_only=False)
        state = payload
        if isinstance(payload, dict):
            for key in ('model', 'state_dict'):
                value = payload.get(key)
                if isinstance(value, dict):
                    state = value
                    break
        if not isinstance(state, dict):
            raise SystemExit(f'unrecognised SAM checkpoint: {checkpoint}')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if 'image_encoder.patch_embed.proj.weight' not in state:
            raise SystemExit(
                f'{checkpoint} does not contain SAM image-encoder weights')
        allowed_missing = (
            'image_encoder.prompt_generator.',
            'pe_layer.positional_encoding_gaussian_matrix',
            'no_mask_embed.weight',
        )
        bad_missing = [
            key for key in missing
            if not any(key.startswith(prefix) or key == prefix
                       for prefix in allowed_missing)]
        bad_unexpected = [
            key for key in unexpected
            if not key.startswith('prompt_encoder.')]
        if bad_missing or bad_unexpected:
            raise SystemExit(
                f'SAM checkpoint is incompatible with SAM-Adapter: '
                f'{checkpoint}; missing={bad_missing[:3]}, '
                f'unexpected={bad_unexpected[:3]}')
        self._sam_sha = file_sha256(checkpoint)
        print('loaded SAM ViT-L weights: '
              f'{len(state)} keys, {len(missing)} missing '
              f'(prompt generator / positional encoding), '
              f'{len(unexpected)} unexpected (prompt encoder)')

    def configure_trainable(self, model) -> None:
        for name, parameter in model.named_parameters():
            if 'image_encoder' in name and 'prompt_generator' not in name:
                parameter.requires_grad_(False)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('SAM-Adapter has no trainable parameters')

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(args.epochs), 1), eta_min=1e-7)

    def _preprocess(self, sample, rng: random.Random, train: bool):
        image = load_rgb(sample.image_path)
        with Image.open(sample.mask_path) as handle:
            mask = handle.convert('L').copy()

        if train and rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = image.resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        tensor = (tensor - PIXEL_MEAN) / PIXEL_STD

        raw_mask = np.asarray(mask)
        if raw_mask.max() <= 1:
            mask_np = raw_mask.astype(np.float32)
        else:
            mask_np = (raw_mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)
        return tensor, mask_tensor

    def prepare_train_batch(self, samples, rng: random.Random):
        images, masks = [], []
        for sample in samples:
            image, mask = self._preprocess(sample, rng, train=True)
            images.append(image)
            masks.append(mask)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    @staticmethod
    def _iou_loss(logits, targets):
        pred = torch.sigmoid(logits)
        inter = (pred * targets).sum(dim=(2, 3))
        union = (pred + targets).sum(dim=(2, 3)) - inter
        return (1 - inter / union).mean()

    def train_step(self, model, batch):
        images, masks = batch
        logits = model.infer(images)
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        return bce + self._iou_loss(logits, masks)

    def prepare_eval_sample(self, sample, device):
        image, _ = self._preprocess(sample, random.Random(0), train=False)
        return image.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        logits = model.infer(prepared)
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
            'sam_checkpoint': str(checkpoint) if checkpoint else None,
            'sam_checkpoint_sha256': self._sam_sha,
            'grad_checkpointing': bool(
                getattr(args, 'grad_checkpointing', True)),
        }

    def restore_config(self, args, config):
        if 'input_size' in config:
            args.input_size = int(config['input_size'])
            self.input_size = args.input_size
        args.grad_checkpointing = bool(
            config.get('grad_checkpointing', True))
        stored = config.get('sam_checkpoint_sha256')
        current = getattr(args, 'sam_checkpoint', None)
        if stored and current is not None and Path(current).is_file():
            actual = file_sha256(Path(current))
            if actual != stored:
                print(f'WARNING: base SAM checkpoint differs from training: '
                      f'{current}; prediction uses the checkpoint weights')
