"""SACM+ baseline: the official SACM implementation with correctness fixes.

This is a separate, self-contained variant of ``sacm_official``.  The
faithful port remains the main-table SACM entry; this adapter applies the
fixes documented in ``baseline_zoo/models/sacm_plus/UPSTREAM.md``:

* SAM normalization and the learned ``no_mask_embed`` dense anchor;
* per-block Adapter-E LayerNorm;
* paper Eq. 6 fusion weights (no Sigmoid saturation);
* Eq. 10 Stage-2 descriptor reordering and per-head IoU prediction;
* all Stage-1/Stage-2 heads and the IoU heads are supervised; inference
  selects by predicted IoU;
* batch-repeat, predictor, dead-code and weight-decay fixes.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sacm_plus import build_sam_vit_l
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class SACMPlusAdapter(BaselineAdapter):
    id = 'sacm_plus'
    display_name = 'SACM+ (fixed official)'
    official_source = 'kylechuuuuu/SACM (fixed derivative)'
    official_entrypoint = 'upstream train_sam.py + documented fixes'
    input_size = 1024
    official_lr = 3e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'SAM normalization is applied before the image encoder '
        '(scale [0,1] -> [0,255], then Sam.preprocess); upstream passed '
        '[0,1] directly into the frozen encoder.',
        'Prompt-free decoding uses the learned no_mask_embed dense anchor '
        'from the frozen prompt encoder instead of the upstream zero map.',
        'Adapter-E uses one LayerNorm per block (paper Eq. 3) instead of '
        'one shared LayerNorm.',
        'Adapter-Fusion implements Eq. 6 as Softmax(FFN(Concat(z_l))) with '
        'per-layer 32-d projections; the upstream Sigmoid-before-softmax '
        'compressed the layer weights into roughly [0.039, 0.106].',
        'Stage-2 consumes the Stage-1-ranked descriptors (paper Eq. 10) '
        'and is not gathered back; per-head IoU predictions are computed '
        'from the ranked descriptors so output slot j is bound to Stage-2 '
        'head j.',
        'Stage-2 all four heads and Stage-1 all four heads (deep supervision '
        'weight 0.3) receive mask loss; the IoU head is trained with MSE '
        '(weight 1.0) and selects the mask at inference. Upstream trained '
        'only the gating-champion Stage-2 head and never trained Stage-1 or '
        'IoU.',
        'Ground-truth masks are binarized at 0.5 after the bilinear resize; '
        'upstream trained on soft resized masks.',
        'AdamW excludes LayerNorm and bias parameters from weight decay; '
        'upstream applied 0.01 to every parameter.',
        'The 1024x1024 resize, no augmentation, 3e-4 learning rate, cosine '
        'to 1e-6, gradient clipping 1.0, BCE 0.7 + Dice 0.3, frozen encoder '
        'except adapters, and the shared 800-epoch budget are preserved.',
        'Checkpoint selection uses validation Dice on the selected mask; '
        'upstream selected by validation F1 on raw logits at 0.5.',
        'Predictions are exported at GT resolution and thresholded after '
        'sigmoid (> 0.5); upstream thresholded raw logits at 0.5.',
    )

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--bce-weight', type=float, default=0.7)
        parser.add_argument('--dice-weight', type=float, default=0.3)
        parser.add_argument(
            '--deep-sup-weight', type=float, default=0.3,
            help='weight of the Stage-1 all-head mask loss')
        parser.add_argument(
            '--iou-loss-weight', type=float, default=1.0,
            help='weight of the per-head IoU MSE')
        parser.add_argument(
            '--head-select', choices=['auto', 'iou', 'gating'], default=None,
            help='mask-head selection: auto -> iou when iou-loss-weight > 0, '
                 'otherwise gating')
        # Official train_sam.py clips by 1.0.  The shared runtime reads
        # ``grad_clip``; this alias keeps the upstream option name and sets
        # the effective default instead of being a silent no-op.
        parser.add_argument(
            '--clip-grad-norm', dest='grad_clip', type=float,
            help='gradient clipping norm (official default: 1.0)')
        parser.set_defaults(grad_clip=1.0)

    @staticmethod
    def _resolved_head_select(args):
        requested = getattr(args, 'head_select', None) or 'auto'
        if requested == 'auto':
            return ('iou'
                    if float(getattr(args, 'iou_loss_weight', 0.0)) > 0
                    else 'gating')
        return requested

    def build_model(self, args):
        # The prediction phase loads a full SACM+ state dict.  Upstream
        # test_sam.py constructs with checkpoint=None, so a base SAM file
        # is not required (or consumed) in that phase.
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

        self.head_select = self._resolved_head_select(args)
        if (self.head_select == 'iou'
                and float(getattr(args, 'iou_loss_weight', 0.0)) <= 0):
            print('[sacm_plus] iou-loss-weight <= 0: falling back to '
                  'gating head 0 for mask selection', flush=True)
            self.head_select = 'gating'
        if float(getattr(args, 'deep_sup_weight', 0.0)) <= 0:
            print('[sacm_plus] deep-sup-weight <= 0: Stage-1 heads receive '
                  'no gradient in this fixed port (no refiner path)',
                  flush=True)

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
            'deep_sup_weight': float(getattr(args, 'deep_sup_weight', 0.3)),
            'iou_loss_weight': float(getattr(args, 'iou_loss_weight', 1.0)),
            'head_select': getattr(
                self, 'head_select', self._resolved_head_select(args)),
        }

    def restore_config(self, args, config):
        for key in ('adapter_dim_ratio', 'bce_weight', 'dice_weight',
                    'deep_sup_weight', 'iou_loss_weight'):
            if key in config:
                setattr(args, key, config[key])
        saved_size = config.get('input_size')
        if saved_size is not None:
            self.input_size = int(saved_size)
            args.input_size = self.input_size
        # Only override the CLI selection when it was left at "auto".
        if (getattr(args, 'head_select', None) is None
                and config.get('head_select') in ('iou', 'gating')):
            args.head_select = config['head_select']

    def configure_trainable(self, model):
        for name, parameter in model.image_encoder.named_parameters():
            parameter.requires_grad_('adapter' in name)
        for parameter in model.prompt_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad_(True)
        model.image_encoder.grad_checkpointing = True

    def make_optimizer(self, model, args):
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith('.bias'):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [
            {'params': decay, 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW([g for g in groups if g['params']])

    def make_scheduler(self, optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    @staticmethod
    def _sam_preprocess(model, images):
        """Apply SAM normalization to [0,1] inputs (upstream omitted it)."""
        lo = images.amin()
        hi = images.amax()
        if bool(lo >= 0.0) and bool(hi <= 1.0 + 1e-6):
            images = images * 255.0
        return model.preprocess(images)

    def _forward(self, model, images, return_stage1=False):
        images = self._sam_preprocess(model, images)
        batch = images.shape[0]
        sparse, dense = model.prompt_encoder(
            points=None, boxes=None, masks=None)
        sparse = sparse.expand(batch, -1, -1)
        dense = dense.expand(batch, -1, -1, -1)
        encoder_output = model.image_encoder(images, return_features=True)
        if len(encoder_output) == 3:
            embeddings, _, adapter_features = encoder_output
        else:
            embeddings, _ = encoder_output
            adapter_features = None
        return model.mask_decoder.predict_masks(
            image_embeddings=embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            adapter_features=adapter_features,
            return_stage1=return_stage1,
        )

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
            mask_tensor = torch.from_numpy(
                np.asarray(mask, dtype=np.float32).copy()
            ).unsqueeze(0) / 255.0
            masks.append((mask_tensor > 0.5).float())
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    @staticmethod
    def _mask_criterion(logits, target, bce_weight, dice_weight):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return bce_weight * bce + dice_weight * dice.mean()

    def train_step(self, model, batch):
        images, masks = batch
        stage2, iou_pred, stage1 = self._forward(
            model, images, return_stage1=True)
        masks = F.interpolate(masks, (1024, 1024), mode='bilinear',
                              align_corners=False)
        stage2 = F.interpolate(stage2, (1024, 1024), mode='bilinear',
                               align_corners=False)
        stage1 = F.interpolate(stage1, (1024, 1024), mode='bilinear',
                               align_corners=False)

        bce_weight = float(self.args.bce_weight)
        dice_weight = float(self.args.dice_weight)
        head_losses = [
            self._mask_criterion(stage2[:, k:k + 1], masks,
                                 bce_weight, dice_weight)
            for k in range(stage2.shape[1])
        ]
        loss_main = torch.stack(head_losses).mean()

        deep_sup = float(getattr(self.args, 'deep_sup_weight', 0.0))
        loss_ds = stage2.new_zeros(())
        if deep_sup > 0:
            ds_losses = [
                self._mask_criterion(stage1[:, k:k + 1], masks,
                                     bce_weight, dice_weight)
                for k in range(stage1.shape[1])
            ]
            loss_ds = torch.stack(ds_losses).mean()

        iou_weight = float(getattr(self.args, 'iou_loss_weight', 0.0))
        loss_iou = stage2.new_zeros(())
        if iou_weight > 0:
            with torch.no_grad():
                pred_bin = (stage2 > 0).float()
                gt_bin = (masks > 0.5).float()
                inter = (pred_bin * gt_bin).sum(dim=(2, 3))
                union = (pred_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3))
                         - inter)
                iou_gt = inter / (union + 1e-6)
            loss_iou = F.mse_loss(iou_pred, iou_gt)

        return loss_main + deep_sup * loss_ds + iou_weight * loss_iou

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        stage2, iou_pred = self._forward(
            model, prepared, return_stage1=False)
        if self.head_select == 'iou':
            index = int(iou_pred.argmax(dim=1)[0].item())
        else:
            index = 0
        logits = stage2[:, index:index + 1]
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
