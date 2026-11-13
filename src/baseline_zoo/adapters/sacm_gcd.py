"""SACM-GCD: geometry-consistent multi-candidate SACM decoder.

Canonical recipe (the current best model):

* all four native Stage-2 candidates receive the dense BCE+Dice loss;
* all four Stage-1 coarse heads receive dense deep supervision;
* soft-clDice and Skeleton Recall are applied to all four Stage-2 heads;
* the three auxiliary Stage-2 heads receive role losses
  (topology / boundary / recall-heavy Tversky);
* a zero-initialised linear fusion is trained as an auxiliary output;
* training uses the Stage-1 gating order, deployment uses a fixed candidate
  (default slot 3) with fusion disabled.

The kept ablations are exactly the ones attached to this model:
``--specialized-heads 0`` (native four-head control),
``--head-fusion 0`` (no-fusion training),
``--cl-dice-weight 0 --skeleton-recall-weight 0
--specialized-topology-weight 0`` (no-topology control), and
``--stage2-loss-scope`` / ``--topology-head`` for the dense/structure scope
factorial.  Everything else (STC, typed Stage-2, TopoQA, TPD/POMP,
quality-based selection, IoU selection) has been removed.

``sacm_official`` remains an independent baseline and is not modified.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.data import load_rgb
from baseline_zoo.models.sacm_gcd import build_sam_vit_l
from baseline_zoo.models.sacm_gcd.fusion import HeadFusion
from baseline_zoo.models.sacm_gcd.topology_losses import (
    SkeletonRecallLoss,
    SoftclDiceLoss,
    TverskyLoss,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter


class SACMGCDAdapter(BaselineAdapter):
    id = 'sacm_gcd'
    display_name = 'SACM-GCD'
    official_source = 'kylechuuuuu/SACM (fixed derivative)'
    official_entrypoint = 'upstream train_sam.py + documented fixes'
    input_size = 1024
    official_lr = 3e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Deployment-aligned Stage-2 aggregation: the deployed head keeps '
        'full loss weight and the three auxiliary heads contribute a '
        '0.25-weight mean term.  The legacy equal-weight mean is removed.',
        'Topology losses consume the native Stage-2 resolution before '
        'resampling, avoiding the 256->1024->512 blur.',
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
        'All four dense Stage-2 candidates and all four Stage-1 coarse '
        'heads are supervised.  Skeleton Recall and soft-clDice are '
        'consistency terms on the evaluated masks, not auxiliary heads.',
        'Training uses the official soft bilinear targets; foreground-'
        'weighted BCE (pos_weight 2) compensates the low foreground ratio.',
        'AdamW excludes LayerNorm and bias parameters from weight decay; '
        'upstream applied 0.01 to every parameter.',
        'The 1024x1024 resize, no augmentation, 3e-4 learning rate, cosine '
        'to 1e-6, gradient clipping 1.0, BCE 0.7 + Dice 0.3, frozen encoder '
        'except adapters, and the validation-Dice checkpoint selection are '
        'preserved.',
        'Predictions are exported at GT resolution and thresholded after '
        'sigmoid (> 0.5); upstream thresholded raw logits at 0.5.',
    )

    def __init__(self):
        super().__init__()
        self._cl_loss = None
        self._tversky_loss = None
        self._skeleton_recall_loss = None
        self._specialized_cl_loss = None
        self._specialized_skeleton_loss = None
        self._specialized_tversky = None
        self.deployment_head = 3
        self.specialized_heads = True
        self.specialized_head_weight = 0.1
        self.specialized_topology_weight = 0.1
        self.head_fusion = False
        self.fusion_loss_weight = 1.0
        self.head_fusion_mode = 'linear'
        self.head_fusion_warmup_epochs = 0
        self.head_select = 'gating'
        self.stage2_loss_scope = 'all'
        self.topology_head = 'all'
        self.topology_source = 'native'
        self.cl_dice_weight = 0.1
        self.skeleton_recall_weight = 0.1
        self.topology_resolution = 512
        self.topology_warmup_epochs = 5
        self.deep_sup_weight = 0.3

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--bce-weight', type=float, default=0.7)
        parser.add_argument('--dice-weight', type=float, default=0.3)
        parser.add_argument(
            '--bce-pos-weight', type=float, default=2.0,
            help='foreground weight for BCE in the mask criterion')
        parser.add_argument(
            '--binary-targets', type=int, choices=[0, 1], default=0,
            help='binarize resized training masks at 0.5 (1) or keep the '
                 'official soft bilinear targets (0)')
        parser.add_argument(
            '--deploy-threshold', type=float, default=0.5,
            help='probability threshold used to export the mask')
        parser.add_argument(
            '--deep-sup-weight', type=float, default=0.3,
            help='weight of the Stage-1 all-head mask loss')
        parser.add_argument(
            '--specialized-heads', type=int, choices=[0, 1], default=1,
            help='role-specialised losses on the three auxiliary Stage-2 '
                 'heads; 0 is the native four-head control')
        parser.add_argument(
            '--specialized-head-weight', type=float, default=0.1,
            help='weight of the boundary and recall head losses')
        parser.add_argument(
            '--specialized-topology-weight', type=float, default=0.1,
            help='weight of the topology head loss; set 0 for the '
                 'no-topology control')
        parser.add_argument(
            '--head-fusion', type=int, choices=[0, 1], default=0,
            help='train the zero-initialised linear fusion as an auxiliary '
                 'output; deployment should disable it')
        parser.add_argument(
            '--fusion-loss-weight', type=float, default=1.0,
            help='weight of the fused-head dense mask loss')
        parser.add_argument(
            '--stage2-loss-scope', choices=['all', 'selected'],
            default='all',
            help='dense mask loss on every candidate (all) or only the '
                 'deployment-selected slot (selected)')
        parser.add_argument(
            '--topology-head', choices=['deployment', 'selected', 'all'],
            default='all',
            help='apply topology losses to the fixed deployment head, the '
                 'deployment-selected slot, or all four heads')
        parser.add_argument(
            '--cl-dice-weight', type=float, default=0.1,
            help='binary soft-clDice weight on the topology heads')
        parser.add_argument(
            '--skeleton-recall-weight', type=float, default=0.1,
            help='skeleton-recall weight on the topology heads')
        parser.add_argument(
            '--topology-resolution', type=int, default=512,
            help='resolution used to compute topology losses')
        parser.add_argument(
            '--topology-warmup-epochs', type=int, default=5,
            help='linearly ramp topology losses in over this many epochs')
        parser.add_argument(
            '--head-select', choices=['gating', 'fixed'], default=None,
            help='training uses gating; the derived deployment checkpoint '
                 'sets fixed when the CLI leaves this unset')
        parser.add_argument(
            '--deployment-head', type=int, default=3,
            help='Stage-2 head used by fixed deployment')
        parser.add_argument(
            '--clip-grad-norm', dest='grad_clip', type=float,
            help='gradient clipping norm (official default: 1.0)')
        parser.set_defaults(grad_clip=1.0)

    @staticmethod
    def _resolved_head_select(args):
        requested = getattr(args, 'head_select', None) or 'gating'
        if requested not in ('gating', 'fixed'):
            raise SystemExit(
                f'unsupported head-select {requested!r}; expected gating or '
                'fixed')
        return requested

    def _selected_stage2_index(self):
        """Return the post-gating Stage-2 slot used for deployment."""
        if self.head_select == 'gating':
            return 0
        if self.head_select == 'fixed':
            return int(self.deployment_head)
        raise ValueError(
            f'unsupported head-select {self.head_select!r}')

    def _stage2_loss_indices(self, num_heads):
        """Indices of Stage-2 candidates receiving the dense mask loss."""
        scope = str(getattr(
            getattr(self, 'args', None), 'stage2_loss_scope', 'all'))
        if scope == 'all':
            return list(range(int(num_heads)))
        if scope == 'selected':
            index = self._selected_stage2_index()
            if not 0 <= index < int(num_heads):
                raise ValueError(
                    f'selected Stage-2 index {index} outside [0, '
                    f'{num_heads})')
            return [index]
        raise ValueError(f'unknown stage2-loss-scope: {scope!r}')

    def _topology_indices(self, num_heads):
        """Indices of Stage-2 candidates receiving structural losses."""
        head = str(getattr(
            getattr(self, 'args', None), 'topology_head', 'all'))
        if head == 'all':
            return list(range(int(num_heads)))
        if head == 'selected':
            index = self._selected_stage2_index()
        elif head == 'deployment':
            index = int(self.deployment_head)
        else:
            raise ValueError(f'unknown topology-head: {head!r}')
        if not 0 <= index < int(num_heads):
            raise ValueError(
                f'topology index {index} outside [0, {num_heads})')
        return [index]

    def _deployment_anchor_index(self):
        """Head that keeps full loss weight under deployment aggregation."""
        if self.head_select == 'gating':
            return 0
        return int(self.deployment_head)

    @staticmethod
    def _aggregate_head_losses(losses, indices, anchor):
        """Keep the deployed head at full weight and average the rest.

        The auxiliary heads enter at weight 0.25 (the canonical recipe).
        With a single selected head this is just that head's loss.
        """
        values = [losses[index] for index in indices]
        if len(values) == 1 or anchor not in indices:
            return torch.stack(values).mean()
        auxiliary = [
            losses[index] for index in indices if index != anchor]
        if not auxiliary:
            return losses[anchor]
        return losses[anchor] + 0.25 * torch.stack(auxiliary).mean()

    def selected_candidate_index(self):
        """Public deployment-slot index for candidate diagnostics."""
        return self._selected_stage2_index()

    def build_model(self, args):
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

        self.args = args
        self.head_select = self._resolved_head_select(args)
        self.deployment_head = int(getattr(args, 'deployment_head', 3))
        self.specialized_heads = bool(
            int(getattr(args, 'specialized_heads', 1)))
        self.specialized_head_weight = float(
            getattr(args, 'specialized_head_weight', 0.1))
        self.specialized_topology_weight = float(
            getattr(args, 'specialized_topology_weight', 0.1))
        if self.specialized_topology_weight < 0:
            raise SystemExit(
                '--specialized-topology-weight must be non-negative')
        self.head_fusion = bool(int(getattr(args, 'head_fusion', 0)))
        self.fusion_loss_weight = float(
            getattr(args, 'fusion_loss_weight', 1.0))
        self.head_fusion_mode = 'linear'
        self.head_fusion_warmup_epochs = 0
        self.stage2_loss_scope = str(
            getattr(args, 'stage2_loss_scope', 'all'))
        self.topology_head = str(
            getattr(args, 'topology_head', 'all'))
        self.topology_source = 'native'
        self.cl_dice_weight = float(
            getattr(args, 'cl_dice_weight', 0.1))
        self.skeleton_recall_weight = float(
            getattr(args, 'skeleton_recall_weight', 0.1))
        self.topology_resolution = int(
            getattr(args, 'topology_resolution', 512))
        self.topology_warmup_epochs = int(
            getattr(args, 'topology_warmup_epochs', 5))
        self.deep_sup_weight = float(
            getattr(args, 'deep_sup_weight', 0.3))

        model = build_sam_vit_l(
            checkpoint=checkpoint, use_adapter=True,
            adapter_dim_ratio=args.adapter_dim_ratio)
        decoder = getattr(model, 'mask_decoder', None)
        if (decoder is not None
                and not 0 <= self.deployment_head
                < decoder.num_mask_tokens):
            raise SystemExit(
                f'deployment-head must be in [0, '
                f'{decoder.num_mask_tokens}), got {self.deployment_head}')
        if self.head_fusion:
            model.head_fusion = HeadFusion()
        return model

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'adapter_dim_ratio': float(
                getattr(args, 'adapter_dim_ratio', 0.1)),
            'bce_weight': float(getattr(args, 'bce_weight', 0.7)),
            'dice_weight': float(getattr(args, 'dice_weight', 0.3)),
            'bce_pos_weight': float(
                getattr(args, 'bce_pos_weight', 2.0)),
            'binary_targets': int(getattr(args, 'binary_targets', 0)),
            'deploy_threshold': float(
                getattr(args, 'deploy_threshold', 0.5)),
            'deep_sup_weight': self.deep_sup_weight,
            'specialized_heads': int(self.specialized_heads),
            'specialized_head_weight': self.specialized_head_weight,
            'specialized_topology_weight': self.specialized_topology_weight,
            'head_fusion': int(self.head_fusion),
            'fusion_loss_weight': self.fusion_loss_weight,
            'head_fusion_mode': self.head_fusion_mode,
            'head_fusion_warmup_epochs': self.head_fusion_warmup_epochs,
            'head_select': self.head_select,
            'deployment_head': int(self.deployment_head),
            'stage2_loss_scope': self.stage2_loss_scope,
            'topology_head': self.topology_head,
            'topology_source': self.topology_source,
            'cl_dice_weight': self.cl_dice_weight,
            'skeleton_recall_weight': self.skeleton_recall_weight,
            'topology_resolution': self.topology_resolution,
            'topology_warmup_epochs': self.topology_warmup_epochs,
        }

    def restore_config(self, args, config):
        for key in (
                'adapter_dim_ratio', 'bce_weight', 'dice_weight',
                'bce_pos_weight', 'binary_targets', 'deploy_threshold',
                'deep_sup_weight', 'specialized_heads',
                'specialized_head_weight', 'specialized_topology_weight',
                'head_fusion', 'fusion_loss_weight', 'head_fusion_mode',
                'head_fusion_warmup_epochs', 'deployment_head',
                'stage2_loss_scope', 'topology_head', 'topology_source',
                'cl_dice_weight', 'skeleton_recall_weight',
                'topology_resolution', 'topology_warmup_epochs'):
            if key in config:
                setattr(args, key, config[key])
        saved_size = config.get('input_size')
        if saved_size is not None:
            self.input_size = int(saved_size)
            args.input_size = self.input_size
        if (getattr(args, 'head_select', None) is None
                and config.get('head_select') in ('gating', 'fixed')):
            args.head_select = config['head_select']

    def load_state_dict(self, model, state) -> None:
        model.load_state_dict(state)

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
        return torch.optim.AdamW([
            {'params': decay, 'lr': args.lr,
             'weight_decay': args.weight_decay},
            {'params': no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        ])

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
            mask_np = np.asarray(mask, dtype=np.float32).copy()
            if mask_np.size and float(mask_np.max()) <= 1.0:
                mask_np *= 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) / 255.0
            if int(getattr(self.args, 'binary_targets', 0)) == 1:
                mask_tensor = (mask_tensor > 0.5).float()
            masks.append(mask_tensor)
        images = torch.stack(images).to(self.device)
        return images, torch.stack(masks).to(self.device)

    def _mask_criterion(self, logits, target, bce_weight, dice_weight):
        pos_weight = float(getattr(
            getattr(self, 'args', None), 'bce_pos_weight', 2.0))
        if pos_weight > 0 and pos_weight != 1.0:
            bce = F.binary_cross_entropy_with_logits(
                logits, target,
                pos_weight=logits.new_tensor(pos_weight))
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return bce_weight * bce + dice_weight * dice.mean()

    def _topology_loss(self, stage2, masks):
        """Soft-clDice and Skeleton Recall on the selected Stage-2 heads."""
        cl_w = float(self.cl_dice_weight)
        sk_w = float(self.skeleton_recall_weight)
        if max(cl_w, sk_w) <= 0:
            return stage2.new_zeros(())

        if self._cl_loss is None:
            self._cl_loss = SoftclDiceLoss()
        if sk_w > 0 and self._skeleton_recall_loss is None:
            self._skeleton_recall_loss = SkeletonRecallLoss()

        resolution = int(self.topology_resolution)
        if resolution <= 0:
            raise ValueError('topology-resolution must be positive')
        target = F.interpolate(
            masks, size=(resolution, resolution), mode='nearest')
        indices = self._topology_indices(stage2.shape[1])

        head_totals = {}
        for index in indices:
            logits = F.interpolate(
                stage2[:, index:index + 1],
                size=(resolution, resolution),
                mode='bilinear', align_corners=False)
            per_head = stage2.new_zeros(())
            if cl_w > 0:
                per_head = per_head + cl_w * self._cl_loss(logits, target)
            if sk_w > 0:
                per_head = per_head + sk_w * self._skeleton_recall_loss(
                    logits, target)
            head_totals[index] = per_head
        anchor = (self._deployment_anchor_index()
                  if self.topology_head == 'all'
                  else indices[0])
        total = self._aggregate_head_losses(head_totals, indices, anchor)

        epoch = int(getattr(self, '_epoch', 1))
        warmup = max(0, int(self.topology_warmup_epochs))
        ramp = 1.0 if warmup <= 0 else min(
            1.0, max(0.0, (epoch - warmup) / float(warmup)))
        return total * ramp

    @staticmethod
    def _boundary_weighted_loss(logits, target):
        """Boundary-emphasised full-mask loss for the boundary head."""
        eroded = -F.max_pool2d(
            -target, kernel_size=5, stride=1, padding=2)
        boundary = (target - eroded).clamp(0.0, 1.0)
        weight = 1.0 + 4.0 * boundary
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return bce + dice.mean()

    def _specialized_head_loss(self, stage2, masks):
        """Role losses on the three auxiliary Stage-2 heads.

        Head 1: soft-clDice topology.  Head 2: boundary-weighted dense
        loss.  Head 3: recall-heavy Tversky dense loss.  All heads keep the
        canonical all-dense loss; these terms only add a different emphasis.
        """
        weight = float(self.specialized_head_weight)
        topology_weight = float(self.specialized_topology_weight)
        if weight <= 0 and topology_weight <= 0:
            return stage2.new_zeros(())
        resolution = int(self.topology_resolution)
        target = F.interpolate(
            masks, (resolution, resolution), mode='bilinear',
            align_corners=False)
        heads = F.interpolate(
            stage2, (resolution, resolution), mode='bilinear',
            align_corners=False)
        topology = stage2.new_zeros(())
        if topology_weight > 0:
            if self._specialized_cl_loss is None:
                self._specialized_cl_loss = SoftclDiceLoss()
            topology = self._specialized_cl_loss(heads[:, 1:2], target)
        if self._specialized_tversky is None:
            self._specialized_tversky = TverskyLoss(alpha=0.3, beta=0.7)
        boundary = self._boundary_weighted_loss(heads[:, 2:3], target)
        recall = self._specialized_tversky(heads[:, 3:4], target)
        return (weight * (boundary + recall)
                + topology_weight * topology)

    def _fused_heads(self, model, stage2):
        fusion = getattr(model, 'head_fusion', None)
        if fusion is None:
            raise RuntimeError(
                'head-fusion=1 but model.head_fusion is missing')
        return fusion(stage2)

    def train_step(self, model, batch):
        images, masks = batch
        stage2, _, stage1 = self._forward(
            model, images, return_stage1=True)
        # Keep the native Stage-2 tensor for topology losses; the dense loss
        # uses the 256 -> 1024 training geometry.
        stage2_native = stage2
        masks = F.interpolate(masks, (1024, 1024), mode='bilinear',
                              align_corners=False)
        stage1 = F.interpolate(stage1, (1024, 1024), mode='bilinear',
                               align_corners=False)
        stage2 = F.interpolate(stage2, (1024, 1024), mode='bilinear',
                               align_corners=False)

        bce_weight = float(self.args.bce_weight)
        dice_weight = float(self.args.dice_weight)
        head_indices = self._stage2_loss_indices(stage2.shape[1])
        head_losses = {
            k: self._mask_criterion(stage2[:, k:k + 1], masks,
                                    bce_weight, dice_weight)
            for k in head_indices
        }
        loss = self._aggregate_head_losses(
            head_losses, head_indices, self._deployment_anchor_index())

        if self.deep_sup_weight > 0:
            ds_losses = [
                self._mask_criterion(stage1[:, k:k + 1], masks,
                                     bce_weight, dice_weight)
                for k in range(stage1.shape[1])
            ]
            loss = loss + self.deep_sup_weight * torch.stack(ds_losses).mean()

        loss = loss + self._topology_loss(stage2_native, masks)
        if self.specialized_heads:
            loss = loss + self._specialized_head_loss(
                stage2_native, masks)
        if self.head_fusion:
            fused = F.interpolate(
                self._fused_heads(model, stage2_native),
                (1024, 1024), mode='bilinear', align_corners=False)
            loss = loss + self.fusion_loss_weight * self._mask_criterion(
                fused, masks, bce_weight, dice_weight)
        return loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        stage2, _ = self._forward(
            model, prepared, return_stage1=False)
        if self.head_fusion:
            logits = self._fused_heads(model, stage2)
        else:
            index = self._selected_stage2_index()
            logits = stage2[:, index:index + 1]
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold)

    @torch.no_grad()
    def predict_candidates(self, model, prepared, sample):
        """Return all four post-gating Stage-2 candidates at GT resolution."""
        if not sample.mask_path:
            raise ValueError('candidate diagnostics require a GT mask')
        stage2, _ = self._forward(
            model, prepared, return_stage1=False)
        if stage2.shape[1] != 4:
            raise RuntimeError(
                f'expected four Stage-2 candidates, got {stage2.shape[1]}')
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(
            stage2, size=(height, width), mode='bilinear',
            align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return probs > threshold
