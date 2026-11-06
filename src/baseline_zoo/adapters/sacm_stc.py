"""SACM-STC: structure-task composition on the SACM prompt-free decoder.

The four native Stage-2 heads have fixed roles:

* head 0 -> dense mask
* head 1 -> centerline
* head 2 -> boundary
* head 3 -> normalized distance/width

Stage-1 keeps the four dense coarse heads and deep supervision.  Stage-2 uses
a stable head order (no gating reorder), the four task losses are applied
directly, and a lightweight composer turns the four outputs into one mask.
Inference uses the composed mask only: no head selector, no fixed deployment
head.  ``sacm_official`` remains the baseline and is not modified.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sacm_stc import build_sam_vit_l
from baseline_zoo.models.sacm_stc.stc import STCHead, structure_targets
from baseline_zoo.models.sacm_stc.topology_losses import (
    SkeletonRecallLoss,
    SoftclDiceLoss,
    TverskyLoss,
)
from baseline_zoo.models.sacm_stc.typed import (
    WIDTH_MAX_PX,
    WIDTH_MIN_PX,
    TypedComposer,
    typed_targets,
    width_to_logits,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class SACMSTCAdapter(BaselineAdapter):
    id = 'sacm_stc'
    display_name = 'SACM-STC'
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
        'SACM-STC fixes the four native Stage-2 heads to dense, centerline, '
        'boundary and normalized distance/width tasks, with a stable head order '
        '(no gating reorder). Stage-1 keeps four dense coarse heads with deep '
        'supervision. A lightweight composer produces the final mask without a '
        'test-time selector or fixed deployment head.',
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
        'The typed Stage-2 path is the SACM-STC method. The official SACM '
        'remains an independent baseline row and is not modified.',
    )

    def __init__(self):
        super().__init__()
        self._cl_loss = None
        self._tversky_loss = None
        self._skeleton_recall_loss = None
        self.deployment_head = 2

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
            '--head-select', choices=['auto', 'iou', 'gating', 'fixed'],
            default=None,
            help='mask-head selection: auto -> iou when iou-loss-weight > 0, '
                 'otherwise gating; fixed -> deployment-head')
        parser.add_argument(
            '--deployment-head', type=int, default=2,
            help='Stage-2 head used by fixed deployment and topology losses')
        parser.add_argument(
            '--cl-dice-weight', type=float, default=0.0,
            help='binary soft-clDice weight on the topology head (0 disables)')
        parser.add_argument(
            '--tversky-weight', type=float, default=0.0,
            help='binary Tversky weight on the topology head (0 disables)')
        parser.add_argument(
            '--skeleton-recall-weight', type=float, default=0.0,
            help='skeleton-recall weight on the topology head (0 disables)')
        parser.add_argument(
            '--topology-resolution', type=int, default=512,
            help='resolution used to compute optional topology losses')
        parser.add_argument(
            '--topology-head', choices=['deployment', 'all'],
            default='deployment',
            help='apply topology losses to the deployment head or all heads')
        parser.add_argument(
            '--topology-warmup-epochs', type=int, default=5,
            help='linearly ramp topology losses in over this many epochs')
        parser.add_argument('--tversky-alpha', type=float, default=0.3)
        parser.add_argument('--tversky-beta', type=float, default=0.7)
        parser.add_argument(
            '--use-stc', type=int, choices=[0, 1], default=0,
            help='enable the structural task-composition branch')
        parser.add_argument('--stc-resolution', type=int, default=512)
        parser.add_argument('--stc-width', type=int, default=32)
        parser.add_argument('--stc-dense-weight', type=float, default=1.0)
        parser.add_argument('--stc-centerline-weight', type=float, default=0.2)
        parser.add_argument('--stc-cl-dice-weight', type=float, default=0.2)
        parser.add_argument('--stc-boundary-weight', type=float, default=0.1)
        parser.add_argument('--stc-width-weight', type=float, default=0.05)
        parser.add_argument('--stc-compose-weight', type=float, default=0.5)
        parser.add_argument('--stc-gate-weight', type=float, default=0.05)
        parser.add_argument(
            '--use-typed-stage2', type=int, choices=[0, 1], default=1,
            help='use stable Stage-2 task heads + typed composition '
                 '(SACM-STC default)')
        parser.add_argument('--typed-resolution', type=int, default=512)
        parser.add_argument('--typed-centerline-weight', type=float, default=0.2)
        parser.add_argument('--typed-cl-dice-weight', type=float, default=0.2)
        parser.add_argument('--typed-boundary-weight', type=float, default=0.1)
        parser.add_argument('--typed-width-weight', type=float, default=0.05)
        parser.add_argument('--typed-compose-weight', type=float, default=0.5)
        parser.add_argument('--typed-gate-weight', type=float, default=0.05)
        parser.add_argument(
            '--typed-structure-warmup-epochs', type=int, default=10,
            help='linearly ramp structural losses after this many epochs')
        parser.add_argument(
            '--typed-composer-lr', type=float, default=3e-3,
            help='learning rate for the tiny gate/edge composer; the official '
                 '3e-4 budget is too small for the gate to leave its neutral '
                 'state within 800 epochs')
        parser.add_argument(
            '--typed-gate-temperature', type=float, default=0.02,
            help='temperature of the soft gate target derived from the '
                 'dense/structure error difference')
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
        # The prediction phase loads a full SACM-STC state dict.  Upstream
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
        self.deployment_head = int(getattr(args, 'deployment_head', 2))
        if (self.head_select == 'iou'
                and float(getattr(args, 'iou_loss_weight', 0.0)) <= 0):
            print('[sacm_stc] iou-loss-weight <= 0: falling back to '
                  'gating head 0 for mask selection', flush=True)
            self.head_select = 'gating'
        if float(getattr(args, 'deep_sup_weight', 0.0)) <= 0:
            print('[sacm_stc] deep-sup-weight <= 0: Stage-1 heads receive '
                  'no gradient in this fixed port (no refiner path)',
                  flush=True)

        model = build_sam_vit_l(
            checkpoint=checkpoint, use_adapter=True,
            adapter_dim_ratio=args.adapter_dim_ratio,
            stable_stage2=bool(getattr(args, 'use_typed_stage2', 0)))
        decoder = getattr(model, 'mask_decoder', None)
        if (decoder is not None
                and not 0 <= self.deployment_head
                < decoder.num_mask_tokens):
            raise SystemExit(
                f'deployment-head must be in [0, '
                f'{decoder.num_mask_tokens}), got '
                f'{self.deployment_head}')
        if int(getattr(args, 'use_stc', 0)) == 1:
            resolution = int(getattr(args, 'stc_resolution', 512))
            if resolution <= 0:
                raise SystemExit('stc-resolution must be positive')
            embed_dim = int(getattr(
                getattr(model, 'image_encoder', None), 'embed_dim', 1024))
            model.stc_head = STCHead(
                feature_ch=2 * embed_dim,
                width=int(getattr(args, 'stc_width', 32)))
        if int(getattr(args, 'use_typed_stage2', 0)) == 1:
            if int(getattr(args, 'use_stc', 0)) == 1:
                raise SystemExit('use-stc and use-typed-stage2 are mutually '
                                 'exclusive')
            model.typed_composer = TypedComposer()
        return model

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
            'deployment_head': int(getattr(args, 'deployment_head', 2)),
            'cl_dice_weight': float(getattr(args, 'cl_dice_weight', 0.0)),
            'tversky_weight': float(getattr(args, 'tversky_weight', 0.0)),
            'skeleton_recall_weight': float(
                getattr(args, 'skeleton_recall_weight', 0.0)),
            'topology_resolution': int(
                getattr(args, 'topology_resolution', 512)),
            'topology_head': getattr(args, 'topology_head', 'deployment'),
            'topology_warmup_epochs': int(
                getattr(args, 'topology_warmup_epochs', 5)),
            'tversky_alpha': float(getattr(args, 'tversky_alpha', 0.3)),
            'tversky_beta': float(getattr(args, 'tversky_beta', 0.7)),
            'use_stc': int(getattr(args, 'use_stc', 0)),
            'stc_resolution': int(getattr(args, 'stc_resolution', 512)),
            'stc_width': int(getattr(args, 'stc_width', 32)),
            'stc_dense_weight': float(
                getattr(args, 'stc_dense_weight', 1.0)),
            'stc_centerline_weight': float(
                getattr(args, 'stc_centerline_weight', 0.2)),
            'stc_cl_dice_weight': float(
                getattr(args, 'stc_cl_dice_weight', 0.2)),
            'stc_boundary_weight': float(
                getattr(args, 'stc_boundary_weight', 0.1)),
            'stc_width_weight': float(
                getattr(args, 'stc_width_weight', 0.05)),
            'stc_compose_weight': float(
                getattr(args, 'stc_compose_weight', 0.5)),
            'stc_gate_weight': float(
                getattr(args, 'stc_gate_weight', 0.05)),
            'use_typed_stage2': int(
                getattr(args, 'use_typed_stage2', 0)),
            'typed_resolution': int(getattr(args, 'typed_resolution', 512)),
            'typed_centerline_weight': float(
                getattr(args, 'typed_centerline_weight', 0.2)),
            'typed_cl_dice_weight': float(
                getattr(args, 'typed_cl_dice_weight', 0.2)),
            'typed_boundary_weight': float(
                getattr(args, 'typed_boundary_weight', 0.1)),
            'typed_width_weight': float(
                getattr(args, 'typed_width_weight', 0.05)),
            'typed_compose_weight': float(
                getattr(args, 'typed_compose_weight', 0.5)),
            'typed_gate_weight': float(
                getattr(args, 'typed_gate_weight', 0.05)),
            'typed_structure_warmup_epochs': int(
                getattr(args, 'typed_structure_warmup_epochs', 10)),
            'typed_composer_lr': float(
                getattr(args, 'typed_composer_lr', 3e-3)),
            'typed_gate_temperature': float(
                getattr(args, 'typed_gate_temperature', 0.02)),
        }

    def restore_config(self, args, config):
        for key in ('adapter_dim_ratio', 'bce_weight', 'dice_weight',
                    'deep_sup_weight', 'iou_loss_weight',
                    'deployment_head', 'cl_dice_weight', 'tversky_weight',
                    'skeleton_recall_weight', 'topology_resolution',
                    'topology_head', 'topology_warmup_epochs',
                    'tversky_alpha', 'tversky_beta', 'use_stc',
                    'stc_resolution', 'stc_width', 'stc_dense_weight',
                    'stc_centerline_weight', 'stc_cl_dice_weight',
                    'stc_boundary_weight', 'stc_width_weight',
                    'stc_compose_weight', 'stc_gate_weight',
                    'use_typed_stage2', 'typed_resolution',
                    'typed_centerline_weight', 'typed_cl_dice_weight',
                    'typed_boundary_weight', 'typed_width_weight',
                    'typed_compose_weight', 'typed_gate_weight',
                    'typed_structure_warmup_epochs', 'typed_composer_lr',
                    'typed_gate_temperature'):
            if key in config:
                setattr(args, key, config[key])
        saved_size = config.get('input_size')
        if saved_size is not None:
            self.input_size = int(saved_size)
            args.input_size = self.input_size
        # Only override the CLI selection when it was left at "auto".
        if (getattr(args, 'head_select', None) is None
                and config.get('head_select') in ('iou', 'gating', 'fixed')):
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
        decay, no_decay, composer = [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('typed_composer.'):
                composer.append(parameter)
            elif parameter.ndim <= 1 or name.endswith('.bias'):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        composer_lr = float(getattr(args, 'typed_composer_lr', 3e-3))
        scheduler = getattr(args, 'scheduler', 'official') or 'official'
        if composer and scheduler != 'official' and composer_lr != args.lr:
            # The shared runtime's generic scheduler overrides reset every
            # parameter group to args.lr.  That silently erases the composer
            # LR and makes the gate learn at the base rate again, so fail
            # loudly instead of producing an invalid experiment.
            raise SystemExit(
                'SACM-STC composer LR requires --scheduler official: generic '
                'scheduler overrides reset all parameter groups to --lr and '
                f'would drop --typed-composer-lr={composer_lr:g}. Use '
                '--scheduler official, or pass --typed-composer-lr '
                f'{float(args.lr):g} explicitly.')
        groups = [
            {'params': decay, 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        ]
        if composer:
            # The gate/edge composer has only a few thousand parameters and
            # has to move from its neutral initialisation during training;
            # the official 3e-4 budget cannot do that within 800 epochs.
            groups.append({'params': composer, 'lr': composer_lr,
                           'weight_decay': 0.0})
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
            embeddings, intermediate_features, adapter_features = encoder_output
        else:
            embeddings, intermediate_features = encoder_output
            adapter_features = None
        # STC v2 consumes high-resolution SAM features, not the dense anchor.
        self._last_sam_features = None
        if intermediate_features:
            last_inter = intermediate_features[-1]
            if adapter_features:
                last_adapter = adapter_features[-1]
                fused = torch.cat([last_adapter, last_inter], dim=-1)
            else:
                fused = last_inter
            self._last_sam_features = fused.permute(0, 3, 1, 2)
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

    @staticmethod
    def _gate_loss(gate_logits, dense_prob, structure, target,
                   temperature=0.02):
        """Gate BCE restricted to foreground and dense/structure disagreements.

        Background pixels where both branches already predict background
        carry no gate signal.  Averaging over them lets the closed class
        dominate and trains the gate to stay shut.  The target is a soft
        function of the error advantage, so pixels where the two branches
        are equally good contribute 0.5 instead of a brittle hard label.
        """
        with torch.no_grad():
            structure_err = (structure - target).pow(2)
            dense_err = (dense_prob - target).pow(2)
            advantage = dense_err - structure_err
            gate_target = torch.sigmoid(
                advantage / max(float(temperature), 1e-6))
            disagreement = (dense_prob - structure).abs()
            gate_pixels = (
                (target > 0.5) | (disagreement > 0.05)).float()
        gate_bce = F.binary_cross_entropy_with_logits(
            gate_logits, gate_target, reduction='none')
        return (gate_bce * gate_pixels).sum() / (
            gate_pixels.sum().clamp_min(1.0))

    def _typed_loss(self, model, masks, stage2, stage1):
        """Stage-2 typed task losses + structural composition."""
        args = self.args
        if int(getattr(args, 'use_typed_stage2', 0)) != 1:
            return stage2.new_zeros(())
        composer = getattr(model, 'typed_composer', None)
        if composer is None:
            raise RuntimeError('typed Stage-2 requires model.typed_composer')

        resolution = int(getattr(args, 'typed_resolution', 512))
        if resolution <= 0:
            raise ValueError('typed-resolution must be positive')
        target = F.interpolate(
            masks, (resolution, resolution), mode='nearest')
        heads = [
            F.interpolate(
                stage2[:, k:k + 1], (resolution, resolution),
                mode='bilinear', align_corners=False)
            for k in range(4)
        ]
        dense, centerline, boundary, width = heads

        bce_w = float(getattr(args, 'bce_weight', 0.7))
        dice_w = float(getattr(args, 'dice_weight', 0.3))
        loss = self._mask_criterion(dense, target, bce_w, dice_w)

        deep_sup = float(getattr(args, 'deep_sup_weight', 0.0))
        if deep_sup > 0:
            ds = sum(
                self._mask_criterion(
                    stage1[:, k:k + 1], masks, bce_w, dice_w)
                for k in range(stage1.shape[1])) / stage1.shape[1]
            loss = loss + deep_sup * ds

        # Structural losses start after the dense heads have learned a usable
        # mask, then ramp in linearly.  This keeps early optimisation from
        # being dominated by noisy skeleton/boundary/width targets.
        epoch = int(getattr(self, '_epoch', 1))
        warmup = max(0, int(getattr(
            args, 'typed_structure_warmup_epochs', 10)))
        ramp = 1.0 if warmup <= 0 else min(
            1.0, max(0.0, (epoch - warmup) / float(warmup)))
        if ramp <= 0.0:
            return loss

        skeleton, boundary_gt, width_gt = typed_targets(masks, resolution)
        skeleton = skeleton.to(stage2.device)
        boundary_gt = boundary_gt.to(stage2.device)
        width_gt = width_gt.to(stage2.device)

        center_w = float(getattr(args, 'typed_centerline_weight', 0.2))
        if center_w > 0:
            center_bce = F.binary_cross_entropy_with_logits(
                centerline, skeleton,
                pos_weight=centerline.new_tensor(5.0))
            if self._cl_loss is None:
                self._cl_loss = SoftclDiceLoss()
            center_cl = self._cl_loss(centerline, skeleton)
            loss = loss + ramp * center_w * center_bce
            loss = loss + ramp * float(
                getattr(args, 'typed_cl_dice_weight', 0.2)) * center_cl

        boundary_w = float(getattr(args, 'typed_boundary_weight', 0.1))
        if boundary_w > 0:
            boundary_bce = F.binary_cross_entropy_with_logits(
                boundary, boundary_gt,
                pos_weight=boundary.new_tensor(5.0))
            # One-pixel boundaries are extremely sparse; BCE alone lets the
            # boundary head settle on a near-zero map.  A Dice term keeps the
            # predicted boundary spatially aligned with the GT contour.
            boundary_prob = torch.sigmoid(boundary)
            dims = tuple(range(1, boundary_prob.ndim))
            inter = (boundary_prob * boundary_gt).sum(dim=dims)
            denom = (boundary_prob.sum(dim=dims)
                     + boundary_gt.sum(dim=dims))
            boundary_dice = (
                1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
            loss = loss + ramp * boundary_w * (
                boundary_bce + boundary_dice)

        width_w = float(getattr(args, 'typed_width_weight', 0.05))
        if width_w > 0:
            fg = target[:, 0] > 0.5
            width_px_gt = (width_gt * resolution).clamp(
                WIDTH_MIN_PX, WIDTH_MAX_PX)
            width_target = width_to_logits(width_px_gt)
            width_l1 = (width - width_target).abs()
            denom = fg.sum().clamp_min(1)
            loss = loss + ramp * width_w * (
                width_l1[:, 0] * fg).sum() / denom

        # The task heads are trained by their own losses.  Detaching them
        # here stops the composer from collapsing a head into a degenerate
        # scale (the old differentiable width path did exactly that).
        outputs = composer(
            dense.detach(), centerline.detach(), boundary.detach(),
            width.detach(), resolution)
        compose_w = float(getattr(args, 'typed_compose_weight', 0.5))
        if compose_w > 0:
            compose = self._mask_criterion(
                outputs['final'], target, bce_w, dice_w)
            loss = loss + ramp * compose_w * compose

        gate_w = float(getattr(args, 'typed_gate_weight', 0.05))
        if gate_w > 0:
            gate_loss = self._gate_loss(
                outputs['gate_logits'], torch.sigmoid(dense),
                outputs['structure'], target,
                float(getattr(args, 'typed_gate_temperature', 0.02)))
            loss = loss + ramp * gate_w * gate_loss
        return loss

    def _stc_loss(self, model, images, masks, stage2, iou_pred):
        """Structural task + gated composition loss for the STC branch."""
        args = self.args
        if int(getattr(args, 'use_stc', 0)) != 1:
            return stage2.new_zeros(())
        head = getattr(model, 'stc_head', None)
        if head is None:
            raise RuntimeError('use_stc=1 but model.stc_head is missing')

        # STC fuses all four Stage-2 candidates; no fixed head.
        anchor = stage2

        resolution = int(getattr(args, 'stc_resolution', 512))
        sam_features = getattr(self, '_last_sam_features', None)
        if sam_features is None:
            raise RuntimeError('STC requires SAM adapter/intermediate features')
        image = F.interpolate(
            images, size=(resolution, resolution),
            mode='bilinear', align_corners=False)
        anchor = F.interpolate(
            anchor, size=(resolution, resolution),
            mode='bilinear', align_corners=False)
        target = F.interpolate(
            masks, size=(resolution, resolution), mode='nearest')
        outputs = head(anchor, image, sam_features)

        bce_weight = float(getattr(args, 'bce_weight', 0.7))
        dice_weight = float(getattr(args, 'dice_weight', 0.3))
        dense = self._mask_criterion(
            outputs['refined'], target, bce_weight, dice_weight)
        loss = float(getattr(args, 'stc_dense_weight', 1.0)) * dense

        skeleton, boundary, width_bins = structure_targets(
            masks, resolution)
        skeleton = skeleton.to(stage2.device)
        boundary = boundary.to(stage2.device)
        width_bins = width_bins.to(stage2.device)

        center_w = float(getattr(args, 'stc_centerline_weight', 0.2))
        if center_w > 0:
            center_bce = F.binary_cross_entropy_with_logits(
                outputs['centerline'], skeleton,
                pos_weight=outputs['centerline'].new_tensor(5.0))
            if self._cl_loss is None:
                self._cl_loss = SoftclDiceLoss()
            center_cl = self._cl_loss(outputs['centerline'], skeleton)
            loss = loss + center_w * center_bce
            loss = loss + float(
                getattr(args, 'stc_cl_dice_weight', 0.2)) * center_cl

        boundary_w = float(getattr(args, 'stc_boundary_weight', 0.1))
        if boundary_w > 0:
            boundary_bce = F.binary_cross_entropy_with_logits(
                outputs['boundary'], boundary,
                pos_weight=outputs['boundary'].new_tensor(5.0))
            loss = loss + boundary_w * boundary_bce

        width_w = float(getattr(args, 'stc_width_weight', 0.05))
        if width_w > 0:
            fg = target[:, 0] > 0.5
            ce = F.cross_entropy(
                outputs['width_bins'], width_bins, reduction='none')
            denom = fg.sum().clamp_min(1)
            loss = loss + width_w * (ce * fg).sum() / denom

        gate_w = float(getattr(args, 'stc_gate_weight', 0.05))
        if gate_w > 0:
            with torch.no_grad():
                structure_err = (outputs['structure'] - target).pow(2)
                refined_err = (
                    torch.sigmoid(outputs['refined']) - target).pow(2)
                gate_target = (
                    structure_err + 0.01 < refined_err).float()
            gate_loss = F.binary_cross_entropy_with_logits(
                outputs['gate_logits'], gate_target)
            loss = loss + gate_w * gate_loss

        compose_w = float(getattr(args, 'stc_compose_weight', 0.5))
        if compose_w > 0:
            compose = self._mask_criterion(
                outputs['final'], target, bce_weight, dice_weight)
            loss = loss + compose_w * compose
        return loss

    def _topology_loss(self, stage2, masks):
        """Optional binary topology terms on the deployment/all candidates."""
        args = self.args
        cl_w = float(getattr(args, 'cl_dice_weight', 0.0))
        tv_w = float(getattr(args, 'tversky_weight', 0.0))
        sk_w = float(getattr(args, 'skeleton_recall_weight', 0.0))
        if max(cl_w, tv_w, sk_w) <= 0:
            return stage2.new_zeros(())

        if self._cl_loss is None:
            self._cl_loss = SoftclDiceLoss()
        if self._tversky_loss is None:
            self._tversky_loss = TverskyLoss(
                float(getattr(args, 'tversky_alpha', 0.3)),
                float(getattr(args, 'tversky_beta', 0.7)))
        if sk_w > 0 and self._skeleton_recall_loss is None:
            self._skeleton_recall_loss = SkeletonRecallLoss()

        resolution = int(getattr(args, 'topology_resolution', 512))
        if resolution <= 0:
            raise ValueError('topology-resolution must be positive')
        target = F.interpolate(
            masks, size=(resolution, resolution), mode='nearest')
        if getattr(args, 'topology_head', 'deployment') == 'all':
            indices = list(range(stage2.shape[1]))
        else:
            indices = [self.deployment_head]

        total = stage2.new_zeros(())
        for index in indices:
            logits = F.interpolate(
                stage2[:, index:index + 1],
                size=(resolution, resolution),
                mode='bilinear', align_corners=False)
            if cl_w > 0:
                total = total + cl_w * self._cl_loss(logits, target)
            if tv_w > 0:
                total = total + tv_w * self._tversky_loss(logits, target)
            if sk_w > 0:
                total = total + sk_w * self._skeleton_recall_loss(
                    logits, target)
        total = total / float(len(indices))

        epoch = int(getattr(self, '_epoch', 1))
        warmup = max(0, int(getattr(args, 'topology_warmup_epochs', 5)))
        ramp = 1.0 if warmup <= 0 else min(
            1.0, max(0.0, (epoch - warmup) / float(warmup)))
        return total * ramp

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

        if int(getattr(self.args, 'use_typed_stage2', 0)) == 1:
            return self._typed_loss(model, masks, stage2, stage1)

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
                # Match the evaluator: two empty masks count as IoU 1, not 0.
                iou_gt = torch.where(
                    union > 0,
                    inter / union.clamp_min(1e-6),
                    torch.ones_like(inter))
            loss_iou = F.mse_loss(iou_pred, iou_gt)

        loss = loss_main + deep_sup * loss_ds + iou_weight * loss_iou
        loss = loss + self._topology_loss(stage2, masks)
        if int(getattr(self.args, 'use_stc', 0)) == 1:
            loss = loss + self._stc_loss(
                model, images, masks, stage2, iou_pred)
        return loss

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
        if int(getattr(self.args, 'use_typed_stage2', 0)) == 1:
            composer = getattr(model, 'typed_composer', None)
            if composer is None:
                raise RuntimeError(
                    'typed Stage-2 requires model.typed_composer')
            resolution = int(getattr(self.args, 'typed_resolution', 512))
            heads = [
                F.interpolate(
                    stage2[:, k:k + 1], (resolution, resolution),
                    mode='bilinear', align_corners=False)
                for k in range(4)
            ]
            logits = composer(*heads, resolution)['final']
        elif int(getattr(self.args, 'use_stc', 0)) == 1:
            head = getattr(model, 'stc_head', None)
            if head is None:
                raise RuntimeError('use_stc=1 but model.stc_head is missing')
            resolution = int(getattr(self.args, 'stc_resolution', 512))
            anchor = F.interpolate(
                stage2, size=(resolution, resolution),
                mode='bilinear', align_corners=False)
            image = F.interpolate(
                prepared, size=(resolution, resolution),
                mode='bilinear', align_corners=False)
            sam_features = getattr(self, '_last_sam_features', None)
            if sam_features is None:
                raise RuntimeError(
                    'STC requires SAM adapter/intermediate features')
            logits = head(anchor, image, sam_features)['final']
        else:
            if self.head_select == 'iou':
                index = int(iou_pred.argmax(dim=1)[0].item())
            elif self.head_select == 'fixed':
                index = self.deployment_head
            else:
                index = 0
            logits = stage2[:, index:index + 1]
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
