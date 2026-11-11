"""SACM-STC: geometry-consistent multi-candidate SACM prompt-free decoder.

The shipped method keeps the official Stage-2 structure: all four native mask
tokens predict dense masks, and the Stage-1 gating order selects the
deployment candidate.  SACM-STC adds structural consistency to those dense
candidates: Skeleton Recall and soft-clDice are applied to every Stage-2 head,
the four Stage-1 coarse heads receive dense deep supervision, training uses
soft bilinear targets and a foreground-weighted BCE.  The geometry terms
regularize the masks that are actually evaluated; there is no auxiliary
centerline/boundary head and no test-time selector.

For controlled ablations, ``--stage2-loss-scope`` and
``--topology-head selected`` can restrict the dense or structural loss to the
deployed Stage-2 slot.  Defaults remain all-candidate dense loss and the
legacy topology target, so the shipped recipe is unchanged.

An earlier prototype instead specialized the four heads to dense, centerline,
boundary and width, with a typed composer.  That path remains available for
ablation (``--use-typed-stage2 1``) but is not the shipped configuration: the
specialized heads removed the official four-candidate robustness and lost on
the 20-dataset transfer check.  ``sacm_official`` remains an independent
baseline and is not modified.
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
from baseline_zoo.models.sacm_stc.tpd import (
    ImageDensePrompt,
    MorphPrompt,
    TopologyQualityHead,
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
        'Deployment-aligned Stage-2 aggregation: with '
        '--stage2-loss-scope all and --topology-head all, the deployed head '
        'keeps full loss weight and the auxiliary heads contribute a mean '
        'term at --stage2-aux-weight / --topology-aux-weight (default 0.25). '
        'The old equal-weight mean divided the deployed head loss by N; '
        '--stage2-loss-aggregate mean / --topology-aggregate mean reproduce '
        'that ablation. Topology losses also consume the native Stage-2 '
        'resolution before resampling, avoiding the 256->1024->512 blur; '
        '--topology-source training reproduces that legacy path.',
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
        'SACM-STC keeps the official four dense Stage-2 candidates and the '
        'Stage-1 gating order, then adds dense deep supervision on the four '
        'Stage-1 coarse heads and Skeleton Recall plus soft-clDice '
        'consistency on all four Stage-2 dense candidates. These are '
        'consistency terms on the evaluated masks, not auxiliary typed heads.',
        'Training uses the official soft bilinear targets by default; '
        'foreground-weighted BCE (pos_weight 2) compensates the 0.3-8% '
        'foreground ratio of the curvilinear datasets. Binary targets and '
        'unweighted BCE remain selectable (--binary-targets 1, '
        '--bce-pos-weight 1).',
        'AdamW excludes LayerNorm and bias parameters from weight decay; '
        'upstream applied 0.01 to every parameter.',
        'The 1024x1024 resize, no augmentation, 3e-4 learning rate, cosine '
        'to 1e-6, gradient clipping 1.0, BCE 0.7 + Dice 0.3, frozen encoder '
        'except adapters, and the shared 800-epoch budget are preserved.',
        'Checkpoint selection uses validation Dice on the selected mask; '
        'upstream selected by validation F1 on raw logits at 0.5.',
        'Predictions are exported at GT resolution and thresholded after '
        'sigmoid (> 0.5); upstream thresholded raw logits at 0.5.',
        'The typed Stage-2 path (stable type embeddings, auxiliary '
        'centerline/boundary/width heads, typed composer) remains available '
        'as an ablation but is not the shipped configuration. The official '
        'SACM remains an independent baseline row and is not modified.',
    )

    def __init__(self):
        super().__init__()
        self._cl_loss = None
        self._tversky_loss = None
        self._skeleton_recall_loss = None
        self._quality_cl_loss = None
        self.deployment_head = 2
        self.image_dense_prompt = False
        self.image_dense_prompt_mode = 'residual'
        self.morph_prompt = False
        self.topology_quality = False
        self.quality_resolution = 64
        self.quality_warmup_epochs = 5
        self.quality_refine = False
        self.quality_refine_topk = 2
        self.quality_refine_threshold = 0.7
        self.quality_refine_weight = 0.5
        self.quality_refine_stage1 = False
        self.quality_refine_min_score_ratio = 0.5
        self.quality_select_margin = 0.01
        self._last_quality_scores = None

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--bce-weight', type=float, default=0.7)
        parser.add_argument('--dice-weight', type=float, default=0.3)
        parser.add_argument(
            '--bce-pos-weight', type=float, default=2.0,
            help='foreground weight for BCE in the mask criterion '
                 '(shipped all_dense recipe: 2.0)')
        parser.add_argument(
            '--binary-targets', type=int, choices=[0, 1], default=0,
            help='binarize resized training masks at 0.5 (1) or keep the '
                 'official soft bilinear targets (0, shipped default)')
        parser.add_argument(
            '--deploy-threshold', type=float, default=0.5,
            help='probability threshold used to export the mask; can be '
                 'calibrated on the validation split after training')
        parser.add_argument(
            '--stage1-fusion-weight', type=float, default=0.0,
            help='optional logit-space blend of the deep-supervised Stage-1 '
                 'coarse prediction into the Stage-2 deployment logits')
        parser.add_argument(
            '--stage1-fusion-mode', choices=['gated', 'mean'], default='gated',
            help='gated = the Stage-1 head in the same gating slot; mean = '
                 'average the four deep-supervised Stage-1 heads')
        parser.add_argument(
            '--deep-sup-weight', type=float, default=0.3,
            help='weight of the Stage-1 all-head mask loss')
        parser.add_argument(
            '--iou-loss-weight', type=float, default=0.0,
            help='weight of the per-head IoU MSE (shipped all_dense: 0)')
        parser.add_argument(
            '--head-select',
            choices=['auto', 'iou', 'gating', 'gating_soft', 'fixed', 'mean'],
            default=None,
            help='mask-head selection: auto -> iou when iou-loss-weight > 0, '
                 'otherwise gating; fixed -> deployment-head; mean -> average '
                 'the four native dense-head logits; gating_soft -> average '
                 'weighted by the Stage-1 gating confidences')
        parser.add_argument(
            '--deployment-head', type=int, default=2,
            help='Stage-2 head used by fixed deployment and topology losses')
        parser.add_argument(
            '--image-dense-prompt', type=int, choices=[0, 1], default=0,
            help='TPD: add an image-conditioned residual to the prompt-free '
                 'dense embedding (MM-SAM style); zero-initialised')
        parser.add_argument(
            '--image-dense-prompt-mode', choices=['residual', 'replace'],
            default='residual',
            help='residual: no_mask_embed + image prompt; replace: image '
                 'embedding + projected image prompt (MM-SAM style)')
        parser.add_argument(
            '--morph-prompt', type=int, choices=[0, 1], default=0,
            help='POMP: polarity-invariant ON/OFF morphological dense '
                 'prompt (multi-scale white + black top-hat sum)')
        parser.add_argument(
            '--topology-quality', type=int, choices=[0, 1], default=0,
            help='TPD: train a candidate topology-quality head and select '
                 'the deployed candidate with it; four candidates remain '
                 'proposals and receive no dense GT loss')
        parser.add_argument(
            '--quality-loss-weight', type=float, default=0.1,
            help='weight of the scalar candidate-quality regression loss')
        parser.add_argument(
            '--quality-iou-weight', type=float, default=0.34,
            help='IoU component of the candidate-quality target (SAMEO)')
        parser.add_argument(
            '--quality-dice-weight', type=float, default=0.33,
            help='Dice component of the candidate-quality target')
        parser.add_argument(
            '--quality-cldice-weight', type=float, default=0.33,
            help='soft-clDice component of the candidate-quality target')
        parser.add_argument(
            '--quality-resolution', type=int, default=64,
            help='spatial resolution used by the quality head and target')
        parser.add_argument(
            '--quality-warmup-epochs', type=int, default=5,
            help='before this epoch, inference falls back to gating while '
                 'the quality head is still warming up')
        parser.add_argument(
            '--quality-refine', type=int, choices=[0, 1], default=0,
            help='E-SAM-inspired proposal refinement: add only high-'
                 'confidence regions supported by the top-k quality '
                 'candidates')
        parser.add_argument(
            '--quality-refine-topk', type=int, default=2,
            help='number of top-quality candidates used for refinement')
        parser.add_argument(
            '--quality-refine-threshold', type=float, default=0.7,
            help='support probability required before adding a missing '
                 'region')
        parser.add_argument(
            '--quality-refine-weight', type=float, default=0.5,
            help='blend weight for high-confidence proposal support')
        parser.add_argument(
            '--quality-refine-stage1', type=int, choices=[0, 1], default=0,
            help='use the coarse Stage-1 heads as additional high-confidence '
                 'support for under-segmentation repair (E-SAM multi-level '
                 'proposal adaptation)')
        parser.add_argument(
            '--quality-refine-min-score-ratio', type=float, default=0.5,
            help='only candidates whose quality score is at least this '
                 'fraction of the best candidate may contribute refinement '
                 'support; prevents low-score degenerate proposals from '
                 'flooding the mask')
        parser.add_argument(
            '--quality-select-margin', type=float, default=0.01,
            help='only switch away from the gating candidate when its '
                 'quality is below the best by this margin')
        parser.add_argument(
            '--stage2-loss-scope', choices=['all', 'selected'],
            default='all',
            help='Stage-2 dense mask loss: all = every native candidate; '
                 'selected = only the deployment-selected slot (gating slot 0, '
                 'or deployment-head when head-select=fixed). How the all '
                 'candidates are combined is controlled by '
                 '--stage2-loss-aggregate')
        parser.add_argument(
            '--stage2-loss-aggregate', choices=['deployment', 'mean'],
            default='deployment',
            help='only for --stage2-loss-scope all: deployment keeps the '
                 'deployed head at full weight and adds auxiliary heads at '
                 '--stage2-aux-weight; mean reproduces the old equal-weight '
                 'average (deployed head weight 1/N)')
        parser.add_argument(
            '--stage2-aux-weight', type=float, default=0.25,
            help='weight of the mean auxiliary dense loss when '
                 '--stage2-loss-aggregate deployment')
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
            '--topology-source', choices=['native', 'training'],
            default='native',
            help='tensor fed to the topology losses: native keeps the '
                 'Stage-2 decoder resolution before resizing to '
                 'topology-resolution; training reproduces the legacy '
                 '256 -> 1024 -> topology-resolution path')
        parser.add_argument(
            '--topology-head', choices=['deployment', 'selected', 'all'],
            default='deployment',
            help='apply topology losses to one fixed deployment head, the '
                 'deployment-selected slot (selected), or all four heads')
        parser.add_argument(
            '--topology-aggregate', choices=['deployment', 'mean'],
            default='deployment',
            help='only for --topology-head all: deployment keeps the '
                 'deployed head at full topology weight and adds auxiliary '
                 'heads at --topology-aux-weight; mean reproduces the old '
                 'equal-weight average')
        parser.add_argument(
            '--topology-aux-weight', type=float, default=0.25,
            help='weight of the mean auxiliary topology loss when '
                 '--topology-aggregate deployment')
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
            '--use-typed-stage2', type=int, choices=[0, 1], default=0,
            help='use stable Stage-2 task heads + typed composition '
                 '(ablation; shipped all_dense: 0)')
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
            help='legacy pixel-gate temperature (unused by the oracle '
                 'distillation path)')
        parser.add_argument(
            '--typed-gate-distill-weight', type=float, default=0.2,
            help='weight of the GT-oracle blend distillation for the '
                 'image-level composer gate')
        parser.add_argument(
            '--typed-output', choices=['dense', 'compose'], default='compose',
            help='deployment output: dense head alone or the composed mask')
        parser.add_argument(
            '--typed-dense-skeleton-weight', type=float, default=0.0,
            help='Skeleton Recall on the dense head (0 disables)')
        parser.add_argument(
            '--typed-dense-cldice-weight', type=float, default=0.0,
            help='soft-clDice on the dense head (0 disables)')
        parser.add_argument(
            '--typed-dense-boundary-weight', type=float, default=0.0,
            help='soft contour boundary loss on the dense head (0 disables)')
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

    def _selected_stage2_index(self):
        """Return the post-gating Stage-2 slot used for deployment.

        ``gating`` deploys slot 0 after the decoder reorders the candidates.
        ``fixed`` uses ``deployment-head``.  ``gating_soft`` and ``mean``
        deploy an aggregate, and ``iou`` selects a different slot per image;
        those selectors have no single supervised candidate and are rejected
        rather than silently mapping to slot 0.
        """
        if self.head_select == 'gating':
            return 0
        if self.head_select == 'fixed':
            return int(self.deployment_head)
        raise ValueError(
            'selected candidate scope requires head-select=gating or fixed; '
            f'got {self.head_select!r}')

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
            getattr(self, 'args', None), 'topology_head', 'deployment'))
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
        """Head that keeps full loss weight under deployment aggregation.

        For the shipped gating selector this is the post-gating slot 0; for a
        fixed selector it is ``deployment_head``.  Selectors that aggregate or
        choose per image (mean/gating_soft/iou) have no single supervised
        candidate, so use ``deployment_head`` only as the auxiliary anchor.
        """
        head_select = getattr(self, 'head_select', 'gating')
        if head_select == 'gating':
            return 0
        if head_select == 'fixed':
            return int(self.deployment_head)
        return int(self.deployment_head)

    @staticmethod
    def _aggregate_head_losses(losses, indices, anchor, aux_weight,
                               aggregate):
        """Combine per-head losses without diluting the deployed head.

        ``mean`` is the old behaviour: every selected head contributes 1/N,
        so an all-scope run gives the deployed head only 1/4 of its
        selected-scope weight.  ``deployment`` keeps the deployed head at
        weight 1 and adds the mean auxiliary loss at ``aux_weight``.  This
        makes the scope comparison a scope comparison instead of a hidden
        loss-weight comparison.
        """
        values = [losses[index] for index in indices]
        if (str(aggregate) == 'mean' or len(values) == 1
                or anchor not in indices):
            return torch.stack(values).mean()
        auxiliary = [
            losses[index] for index in indices if index != anchor]
        if not auxiliary:
            return losses[anchor]
        return values[indices.index(anchor)] + float(aux_weight) * (
            torch.stack(auxiliary).mean())

    def selected_candidate_index(self):
        """Public deployment-slot index for candidate diagnostics."""
        return self._selected_stage2_index()

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
        self.image_dense_prompt = bool(
            int(getattr(args, 'image_dense_prompt', 0)))
        self.image_dense_prompt_mode = str(
            getattr(args, 'image_dense_prompt_mode', 'residual')).lower()
        if self.image_dense_prompt_mode not in ('residual', 'replace'):
            raise SystemExit(
                'image-dense-prompt-mode must be residual or replace')
        self.morph_prompt = bool(int(getattr(args, 'morph_prompt', 0)))
        self.topology_quality = bool(
            int(getattr(args, 'topology_quality', 0)))
        self.quality_resolution = int(
            getattr(args, 'quality_resolution', 64))
        self.quality_warmup_epochs = int(
            getattr(args, 'quality_warmup_epochs', 5))
        self.quality_refine = bool(
            int(getattr(args, 'quality_refine', 0)))
        self.quality_refine_topk = int(
            getattr(args, 'quality_refine_topk', 2))
        self.quality_refine_threshold = float(
            getattr(args, 'quality_refine_threshold', 0.7))
        self.quality_refine_weight = float(
            getattr(args, 'quality_refine_weight', 0.5))
        self.quality_refine_stage1 = bool(
            int(getattr(args, 'quality_refine_stage1', 0)))
        self.quality_refine_min_score_ratio = float(
            getattr(args, 'quality_refine_min_score_ratio', 0.5))
        self.quality_select_margin = float(
            getattr(args, 'quality_select_margin', 0.01))
        if self.quality_refine_topk <= 0:
            raise SystemExit('quality-refine-topk must be positive')
        if not 0.0 <= self.quality_refine_min_score_ratio <= 1.0:
            raise SystemExit(
                'quality-refine-min-score-ratio must be in [0, 1]')
        if self.quality_select_margin < 0:
            raise SystemExit('quality-select-margin must be non-negative')
        if self.quality_resolution <= 0:
            raise SystemExit('quality-resolution must be positive')
        if self.topology_quality:
            scope = str(getattr(args, 'stage2_loss_scope', 'all'))
            if scope not in ('all', 'selected'):
                raise SystemExit(
                    'topology-quality requires stage2-loss-scope=all or '
                    'selected; the candidates it ranks must be trainable')
            # With scope=all the auxiliary candidates receive dense GT loss
            # (the canonical SACM-STC full recipe), so they are well-trained
            # proposals and the quality head ranks four supervised heads.
            # scope=selected keeps the proposal-only design.
            if int(getattr(args, 'use_stc', 0)) == 1:
                raise SystemExit(
                    'topology-quality is not supported with use_stc')
            if int(getattr(args, 'use_typed_stage2', 0)) == 1:
                raise SystemExit(
                    'topology-quality is not supported with typed Stage-2')
        if self.quality_refine and not self.topology_quality:
            raise SystemExit(
                'quality-refine requires topology-quality=1')
        if self.quality_refine_stage1 and not self.quality_refine:
            raise SystemExit(
                'quality-refine-stage1 requires quality-refine=1')
        self.stage2_loss_aggregate = str(getattr(
            args, 'stage2_loss_aggregate', 'deployment')).lower()
        self.topology_aggregate = str(getattr(
            args, 'topology_aggregate', 'deployment')).lower()
        self.topology_source = str(getattr(
            args, 'topology_source', 'native')).lower()
        self.stage2_aux_weight = float(getattr(
            args, 'stage2_aux_weight', 0.25))
        self.topology_aux_weight = float(getattr(
            args, 'topology_aux_weight', 0.25))
        for name, value in (
                ('stage2_loss_aggregate', self.stage2_loss_aggregate),
                ('topology_aggregate', self.topology_aggregate)):
            if value not in ('deployment', 'mean'):
                raise ValueError(
                    f'{self.id}: unknown {name} {value!r}')
        if self.topology_source not in ('native', 'training'):
            raise ValueError(
                f'{self.id}: unknown topology_source '
                f'{self.topology_source!r}')
        for name, value in (
                ('stage2_aux_weight', self.stage2_aux_weight),
                ('topology_aux_weight', self.topology_aux_weight)):
            if value < 0:
                raise ValueError(
                    f'{self.id}: {name} must be non-negative, got {value}')
        if (self.head_select == 'iou'
                and float(getattr(args, 'iou_loss_weight', 0.0)) <= 0):
            print('[sacm_stc] iou-loss-weight <= 0: falling back to '
                  'gating head 0 for mask selection', flush=True)
            self.head_select = 'gating'
        if (self.topology_quality
                and self.head_select not in ('gating', 'fixed')):
            raise SystemExit(
                'topology-quality requires head-select=gating or fixed; '
                f'got {self.head_select!r}')
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
        if (float(getattr(args, 'stage1_fusion_weight', 0.0)) > 0
                and (int(getattr(args, 'use_stc', 0)) == 1
                     or int(getattr(args, 'use_typed_stage2', 0)) == 1)):
            raise SystemExit(
                'stage1-fusion-weight is not supported with use_stc or '
                'use_typed_stage2; set it to 0 for those paths')
        if int(getattr(args, 'use_stc', 0)) == 1:
            resolution = int(getattr(args, 'stc_resolution', 512))
            if resolution <= 0:
                raise SystemExit('stc-resolution must be positive')
            encoder = getattr(model, 'image_encoder', None)
            embed_dim = getattr(encoder, 'embed_dim', None)
            if embed_dim is None:
                proj = getattr(getattr(encoder, 'patch_embed', None),
                               'proj', None)
                embed_dim = getattr(proj, 'out_channels', None)
            if embed_dim is None:
                raise SystemExit(
                    'cannot infer image-encoder embed_dim for STCHead')
            model.stc_head = STCHead(
                feature_ch=2 * int(embed_dim),
                width=int(getattr(args, 'stc_width', 32)))
        if int(getattr(args, 'use_typed_stage2', 0)) == 1:
            if int(getattr(args, 'use_stc', 0)) == 1:
                raise SystemExit('use-stc and use-typed-stage2 are mutually '
                                 'exclusive')
            model.typed_composer = TypedComposer()
        if self.image_dense_prompt or self.topology_quality:
            encoder = getattr(model, 'image_encoder', None)
            embed_dim = getattr(encoder, 'embed_dim', None)
            if embed_dim is None:
                proj = getattr(getattr(encoder, 'patch_embed', None),
                               'proj', None)
                embed_dim = getattr(proj, 'out_channels', None)
            if embed_dim is None:
                raise SystemExit(
                    'cannot infer image-encoder embed_dim for TPD modules')
        if self.image_dense_prompt:
            # SAM's ViT neck emits the decoder-width embedding (256 for
            # ViT-L), not the 1024-d patch-embedding input.
            prompt_dim = getattr(
                getattr(model, 'mask_decoder', None), 'transformer_dim', 256)
            model.image_dense_prompt = ImageDensePrompt(
                int(prompt_dim), prompt_dim=int(prompt_dim))
        if self.morph_prompt:
            prompt_dim = getattr(
                getattr(model, 'mask_decoder', None), 'transformer_dim', 256)
            model.morph_prompt = MorphPrompt(int(prompt_dim))
        if self.topology_quality:
            model.topology_quality_head = TopologyQualityHead()
        return model

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'adapter_dim_ratio': float(
                getattr(args, 'adapter_dim_ratio', 0.1)),
            'bce_weight': float(getattr(args, 'bce_weight', 0.7)),
            'dice_weight': float(getattr(args, 'dice_weight', 0.3)),
            'bce_pos_weight': float(getattr(args, 'bce_pos_weight', 1.0)),
            'binary_targets': int(getattr(args, 'binary_targets', 1)),
            'deploy_threshold': float(
                getattr(args, 'deploy_threshold', 0.5)),
            'stage1_fusion_weight': float(
                getattr(args, 'stage1_fusion_weight', 0.0)),
            'stage1_fusion_mode': getattr(
                args, 'stage1_fusion_mode', 'gated'),
            'deep_sup_weight': float(getattr(args, 'deep_sup_weight', 0.3)),
            'iou_loss_weight': float(getattr(args, 'iou_loss_weight', 1.0)),
            'image_dense_prompt': int(getattr(
                self, 'image_dense_prompt',
                getattr(args, 'image_dense_prompt', 0))),
            'image_dense_prompt_mode': str(getattr(
                self, 'image_dense_prompt_mode',
                getattr(args, 'image_dense_prompt_mode', 'residual'))),
            'morph_prompt': int(getattr(
                self, 'morph_prompt',
                getattr(args, 'morph_prompt', 0))),
            'topology_quality': int(getattr(
                self, 'topology_quality',
                getattr(args, 'topology_quality', 0))),
            'quality_loss_weight': float(getattr(
                args, 'quality_loss_weight', 0.1)),
            'quality_iou_weight': float(getattr(
                args, 'quality_iou_weight', 0.34)),
            'quality_dice_weight': float(getattr(
                args, 'quality_dice_weight', 0.33)),
            'quality_cldice_weight': float(getattr(
                args, 'quality_cldice_weight', 0.33)),
            'quality_resolution': int(getattr(
                self, 'quality_resolution',
                getattr(args, 'quality_resolution', 64))),
            'quality_warmup_epochs': int(getattr(
                self, 'quality_warmup_epochs',
                getattr(args, 'quality_warmup_epochs', 5))),
            'quality_refine': int(getattr(
                self, 'quality_refine',
                getattr(args, 'quality_refine', 0))),
            'quality_refine_topk': int(getattr(
                self, 'quality_refine_topk',
                getattr(args, 'quality_refine_topk', 2))),
            'quality_refine_threshold': float(getattr(
                self, 'quality_refine_threshold',
                getattr(args, 'quality_refine_threshold', 0.7))),
            'quality_refine_weight': float(getattr(
                self, 'quality_refine_weight',
                getattr(args, 'quality_refine_weight', 0.5))),
            'quality_refine_stage1': int(getattr(
                self, 'quality_refine_stage1',
                getattr(args, 'quality_refine_stage1', 0))),
            'quality_refine_min_score_ratio': float(getattr(
                self, 'quality_refine_min_score_ratio',
                getattr(args, 'quality_refine_min_score_ratio', 0.5))),
            'quality_select_margin': float(getattr(
                self, 'quality_select_margin',
                getattr(args, 'quality_select_margin', 0.01))),
            'head_select': getattr(
                self, 'head_select', self._resolved_head_select(args)),
            'deployment_head': int(getattr(args, 'deployment_head', 2)),
            'stage2_loss_scope': getattr(
                args, 'stage2_loss_scope', 'all'),
            'stage2_loss_aggregate': str(getattr(
                self, 'stage2_loss_aggregate',
                getattr(args, 'stage2_loss_aggregate', 'deployment'))),
            'stage2_aux_weight': float(getattr(
                self, 'stage2_aux_weight',
                getattr(args, 'stage2_aux_weight', 0.25))),
            'cl_dice_weight': float(getattr(args, 'cl_dice_weight', 0.0)),
            'tversky_weight': float(getattr(args, 'tversky_weight', 0.0)),
            'skeleton_recall_weight': float(
                getattr(args, 'skeleton_recall_weight', 0.0)),
            'topology_resolution': int(
                getattr(args, 'topology_resolution', 512)),
            'topology_source': str(getattr(
                self, 'topology_source',
                getattr(args, 'topology_source', 'native'))),
            'topology_head': getattr(args, 'topology_head', 'deployment'),
            'topology_aggregate': str(getattr(
                self, 'topology_aggregate',
                getattr(args, 'topology_aggregate', 'deployment'))),
            'topology_aux_weight': float(getattr(
                self, 'topology_aux_weight',
                getattr(args, 'topology_aux_weight', 0.25))),
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
            'typed_gate_distill_weight': float(
                getattr(args, 'typed_gate_distill_weight', 0.2)),
            'typed_output': getattr(args, 'typed_output', 'compose'),
            'typed_dense_skeleton_weight': float(
                getattr(args, 'typed_dense_skeleton_weight', 0.0)),
            'typed_dense_cldice_weight': float(
                getattr(args, 'typed_dense_cldice_weight', 0.0)),
            'typed_dense_boundary_weight': float(
                getattr(args, 'typed_dense_boundary_weight', 0.0)),
        }

    def restore_config(self, args, config):
        for key in ('adapter_dim_ratio', 'bce_weight', 'dice_weight',
                    'bce_pos_weight', 'binary_targets', 'deploy_threshold',
                    'stage1_fusion_weight', 'stage1_fusion_mode',
                    'deep_sup_weight', 'iou_loss_weight',
                    'image_dense_prompt', 'image_dense_prompt_mode',
                    'morph_prompt',
                    'topology_quality',
                    'quality_loss_weight', 'quality_iou_weight',
                    'quality_dice_weight', 'quality_cldice_weight',
                    'quality_resolution', 'quality_warmup_epochs',
                    'quality_refine', 'quality_refine_topk',
                    'quality_refine_threshold', 'quality_refine_weight',
                    'quality_refine_stage1', 'quality_refine_min_score_ratio',
                    'quality_select_margin',
                    'deployment_head', 'stage2_loss_scope',
                    'stage2_loss_aggregate', 'stage2_aux_weight',
                    'cl_dice_weight', 'tversky_weight',
                    'skeleton_recall_weight', 'topology_resolution',
                    'topology_source', 'topology_head', 'topology_aggregate',
                    'topology_aux_weight', 'topology_warmup_epochs',
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
                    'typed_gate_temperature', 'typed_gate_distill_weight',
                    'typed_output', 'typed_dense_skeleton_weight',
                    'typed_dense_cldice_weight', 'typed_dense_boundary_weight',
                    'calibrated_blend_g'):
            if key in config:
                previous = getattr(args, key, None)
                if (key in {'deploy_threshold', 'stage1_fusion_weight',
                            'stage1_fusion_mode', 'use_stc',
                            'use_typed_stage2'}
                        and previous is not None
                        and previous != config[key]):
                    print(f'[sacm_stc] checkpoint config overrides '
                          f'{key}: {previous!r} -> {config[key]!r}',
                          flush=True)
                setattr(args, key, config[key])
        saved_size = config.get('input_size')
        if saved_size is not None:
            self.input_size = int(saved_size)
            args.input_size = self.input_size
        # Only override the CLI selection when it was left at "auto".
        if (getattr(args, 'head_select', None) is None
                and config.get('head_select')
                in ('iou', 'gating', 'gating_soft', 'fixed', 'mean')):
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
        # Keep the raw [0,1] image for the polarity-invariant morphological
        # prompt; SAM preprocessing normalizes it and would destroy the
        # intensity polarity the ON/OFF features are built from.
        raw_images = images
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
        if self.image_dense_prompt:
            prompt = getattr(model, 'image_dense_prompt', None)
            if prompt is None:
                raise RuntimeError(
                    'image-dense-prompt=1 but model.image_dense_prompt is '
                    'missing')
            residual = prompt(embeddings)
            if residual.shape[1] != dense.shape[1]:
                raise RuntimeError(
                    'image dense prompt produced '
                    f'{residual.shape[1]} channels but SAM dense prompts '
                    f'use {dense.shape[1]}')
            if residual.shape[-2:] != dense.shape[-2:]:
                residual = F.interpolate(
                    residual, size=dense.shape[-2:],
                    mode='bilinear', align_corners=False)
            if self.image_dense_prompt_mode == 'replace':
                # MM-SAM-style dense embedding: image encoder embedding +
                # projected image embedding, instead of a constant no-mask
                # anchor.
                base = embeddings
                if base.shape[1] != dense.shape[1]:
                    raise RuntimeError(
                        'image encoder embeddings have '
                        f'{base.shape[1]} channels but SAM dense prompts '
                        f'use {dense.shape[1]}')
                if base.shape[-2:] != dense.shape[-2:]:
                    base = F.interpolate(
                        base, size=dense.shape[-2:],
                        mode='bilinear', align_corners=False)
                dense = base + residual
            else:
                dense = dense + residual
        if self.morph_prompt:
            morph = getattr(model, 'morph_prompt', None)
            if morph is None:
                raise RuntimeError(
                    'morph-prompt=1 but model.morph_prompt is missing')
            residual = morph(raw_images)
            if residual.shape[1] != dense.shape[1]:
                raise RuntimeError(
                    'morph prompt produced '
                    f'{residual.shape[1]} channels but SAM dense prompts '
                    f'use {dense.shape[1]}')
            if residual.shape[-2:] != dense.shape[-2:]:
                residual = F.interpolate(
                    residual, size=dense.shape[-2:],
                    mode='bilinear', align_corners=False)
            dense = dense + residual
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
            # Accept both 0/255 and literal 0/1 mask encodings; without this
            # the 0/1 case becomes an all-background target after /255.
            if mask_np.size and float(mask_np.max()) <= 1.0:
                mask_np *= 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) / 255.0
            if int(getattr(self.args, 'binary_targets', 1)) == 1:
                mask_tensor = (mask_tensor > 0.5).float()
            masks.append(mask_tensor)
        return (torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device))

    def _mask_criterion(self, logits, target, bce_weight, dice_weight):
        pos_weight = float(getattr(
            getattr(self, 'args', None), 'bce_pos_weight', 1.0))
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

    @staticmethod
    def _oracle_gate_target(dense_logits, structure, target, grid_size=11):
        """Return the best constant dense/structure blend under GT Dice.

        The oracle is computed only from current predictions and GT; it is a
        training teacher for the image-level gate, never used at inference.
        """
        if grid_size < 2:
            raise ValueError('grid_size must be >= 2')
        with torch.no_grad():
            dense_logits = dense_logits.detach()
            structure = structure.detach().clamp(1e-4, 1.0 - 1e-4)
            structure_logits = torch.logit(structure)
            target = target.detach().float()
            grid = torch.linspace(0.0, 1.0, grid_size,
                                  device=dense_logits.device)
            dices = []
            for gate in grid:
                pred = torch.sigmoid(
                    dense_logits + gate * (structure_logits - dense_logits))
                pred = (pred > 0.5).float()
                dims = (1, 2, 3)
                inter = (pred * target).sum(dim=dims)
                denom = pred.sum(dim=dims) + target.sum(dim=dims)
                dices.append((2.0 * inter + 1e-6) / (denom + 1e-6))
            dices = torch.stack(dices, dim=1)  # [B, grid]
            # Use the mean of the whole Dice plateau instead of argmax: a
            # hard argmax breaks ties toward g=0 and systematically teaches
            # the gate to close when several blends are equally good.
            best_value = dices.max(dim=1, keepdim=True).values
            plateau = (dices >= best_value - 1e-9).float()
            best_gate = (grid[None, :] * plateau).sum(dim=1) / (
                plateau.sum(dim=1).clamp_min(1.0))
        return best_gate, dices

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

        # Geometry-consistency supervision on the deployment head.  The
        # auxiliary heads act as geometry teachers, but the topology and
        # boundary losses are applied directly to the dense output so they
        # improve the mask that is actually evaluated.
        dense_skel_w = float(getattr(
            args, 'typed_dense_skeleton_weight', 0.0))
        if dense_skel_w > 0:
            if self._skeleton_recall_loss is None:
                self._skeleton_recall_loss = SkeletonRecallLoss()
            loss = loss + ramp * dense_skel_w * (
                self._skeleton_recall_loss(dense, target))
        dense_cl_w = float(getattr(args, 'typed_dense_cldice_weight', 0.0))
        if dense_cl_w > 0:
            if self._cl_loss is None:
                self._cl_loss = SoftclDiceLoss()
            loss = loss + ramp * dense_cl_w * self._cl_loss(dense, target)
        dense_bnd_w = float(getattr(
            args, 'typed_dense_boundary_weight', 0.0))
        if dense_bnd_w > 0:
            dense_prob = torch.sigmoid(dense)
            eroded = -F.max_pool2d(-dense_prob, 3, 1, 1)
            boundary_pred = (dense_prob - eroded).clamp(1e-4, 1.0 - 1e-4)
            boundary_bce = F.binary_cross_entropy(
                boundary_pred, boundary_gt)
            dims = tuple(range(1, boundary_pred.ndim))
            inter = (boundary_pred * boundary_gt).sum(dim=dims)
            denom = (boundary_pred.sum(dim=dims)
                     + boundary_gt.sum(dim=dims))
            boundary_dice = (
                1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
            loss = loss + ramp * dense_bnd_w * (
                boundary_bce + boundary_dice)

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

        distill_w = float(getattr(
            args, 'typed_gate_distill_weight', 0.2))
        if distill_w > 0:
            best_gate, _ = self._oracle_gate_target(
                dense, outputs['structure'], target)
            predicted = outputs['global_gate'].reshape(-1)
            gate_loss = F.smooth_l1_loss(
                predicted, best_gate.to(predicted.dtype), beta=0.1)
            loss = loss + ramp * distill_w * gate_loss
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
            if tv_w > 0:
                per_head = per_head + tv_w * self._tversky_loss(
                    logits, target)
            if sk_w > 0:
                per_head = per_head + sk_w * self._skeleton_recall_loss(
                    logits, target)
            head_totals[index] = per_head
        aggregate = str(getattr(
            self, 'topology_aggregate',
            getattr(args, 'topology_aggregate', 'deployment')))
        aux_weight = float(getattr(
            self, 'topology_aux_weight',
            getattr(args, 'topology_aux_weight', 0.25)))
        # For topology-head=all the deployed slot keeps full weight; the
        # auxiliary heads only add --topology-aux-weight * mean(aux).  For
        # single-head modes there is no auxiliary term.
        anchor = (self._deployment_anchor_index()
                  if str(getattr(args, 'topology_head', 'deployment')) == 'all'
                  else indices[0])
        total = self._aggregate_head_losses(
            head_totals, indices, anchor, aux_weight, aggregate)

        epoch = int(getattr(self, '_epoch', 1))
        warmup = max(0, int(getattr(args, 'topology_warmup_epochs', 5)))
        ramp = 1.0 if warmup <= 0 else min(
            1.0, max(0.0, (epoch - warmup) / float(warmup)))
        return total * ramp

    def _quality_scores(self, model, stage2):
        """Return ``[B, N]`` candidate quality scores at low resolution."""
        head = getattr(model, 'topology_quality_head', None)
        if head is None:
            raise RuntimeError(
                'topology-quality=1 but model.topology_quality_head is '
                'missing')
        if stage2.ndim != 4 or stage2.shape[1] != 4:
            raise RuntimeError(
                'topology-quality expects four native Stage-2 candidates, '
                f'got {tuple(stage2.shape)}')
        low_res = F.interpolate(
            stage2, size=(self.quality_resolution, self.quality_resolution),
            mode='bilinear', align_corners=False)
        return head(low_res)

    def _quality_target(self, stage2, masks):
        """Per-candidate Dice + soft-clDice target, detached from the graph."""
        if self._quality_cl_loss is None:
            self._quality_cl_loss = SoftclDiceLoss()
        low_res = F.interpolate(
            stage2, size=(self.quality_resolution, self.quality_resolution),
            mode='bilinear', align_corners=False)
        target = F.interpolate(
            masks, size=(self.quality_resolution, self.quality_resolution),
            mode='bilinear', align_corners=False)
        with torch.no_grad():
            return TopologyQualityHead.quality_target(
                low_res, target, self._quality_cl_loss,
                float(getattr(self.args, 'quality_iou_weight', 0.34)),
                float(getattr(self.args, 'quality_dice_weight', 0.33)),
                float(getattr(self.args, 'quality_cldice_weight', 0.33)))

    def _quality_loss(self, model, stage2, masks):
        weight = float(getattr(self.args, 'quality_loss_weight', 0.1))
        if weight <= 0:
            return stage2.new_zeros(())
        scores = self._quality_scores(model, stage2)
        target = self._quality_target(stage2, masks)
        return weight * F.mse_loss(scores, target)

    def _quality_refined_logits(self, stage2, scores, stage1=None,
                                base_index=None):
        """E-SAM-inspired conservative under-segmentation refinement.

        The deployed candidate stays the base mask.  Regions that are missing
        from it but supported by the top-k quality candidates or the coarse
        Stage-1 heads above a high confidence threshold are blended in.  This
        is a lightweight binary adaptation of E-SAM's multi-level proposal /
        under-segmentation repair, not a dense loss on the auxiliary heads.

        ``base_index`` decouples refinement from candidate selection: it is
        the candidate actually deployed by the selector (gating/fixed/iou or
        the quality override).  ``None`` keeps the legacy behaviour of using
        the top-quality candidate as the base.
        """
        if stage2.shape[1] <= 1:
            return stage2[:, :1]
        if (base_index is not None
                and not 0 <= int(base_index) < stage2.shape[1]):
            raise ValueError(
                f'base_index {base_index} outside [0, {stage2.shape[1]})')
        topk = max(1, min(int(self.quality_refine_topk), stage2.shape[1]))
        _, indices = torch.topk(scores, topk, dim=1)
        probs = torch.sigmoid(stage2)
        gathered = torch.gather(
            probs, 1, indices[:, :, None, None].expand(
                -1, -1, probs.shape[2], probs.shape[3]))
        # Non-deployed Stage-2 heads receive no dense GT loss and can
        # degenerate into all-foreground proposals.  A plain max over top-k
        # lets such a proposal flood the mask even when the quality head
        # ranked it far below the best candidate (measured: score 0.02 vs
        # 0.49, fill 0.9, Dice 0.83 -> 0.19).  Keep a proposal only when its
        # predicted quality is comparable to the best one.
        min_ratio = float(getattr(
            self, 'quality_refine_min_score_ratio', 0.5))
        best_score = scores.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        keep = scores >= (min_ratio * best_score)
        keep = torch.gather(keep, 1, indices)
        gathered = gathered * keep[:, :, None, None].to(gathered.dtype)
        # High-confidence support from any quality-vetted proposal, as in
        # E-SAM's under-segmentation repair; not a weighted average that
        # would hide a confident auxiliary proposal behind a weak top-1
        # score.
        support = gathered.max(dim=1, keepdim=True).values
        if stage1 is not None and self.quality_refine_stage1:
            if stage1.shape[-2:] != probs.shape[-2:]:
                stage1 = F.interpolate(
                    stage1, size=probs.shape[-2:],
                    mode='bilinear', align_corners=False)
            stage1_support = torch.sigmoid(stage1).mean(
                dim=1, keepdim=True)
            support = torch.maximum(support, stage1_support)
        if base_index is None:
            base = scores.argmax(dim=1)
        else:
            base = torch.full(
                (probs.shape[0],), int(base_index),
                dtype=torch.long, device=probs.device)
        selected = torch.gather(
            probs, 1, base[:, None, None, None].expand(
                -1, 1, probs.shape[2], probs.shape[3]))
        # "Missing" uses the same deployment threshold as final
        # binarisation, so refinement and thresholding stay consistent.
        deploy_threshold = float(getattr(
            getattr(self, 'args', None), 'deploy_threshold', 0.5))
        extra = (
            (support > float(self.quality_refine_threshold))
            & (selected < deploy_threshold)
        ).to(probs.dtype)
        fused = selected + float(self.quality_refine_weight) * extra * (
            support - selected)
        fused = fused.clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(fused)

    def _quality_candidate_index(self, model, stage2):
        """Select one candidate by predicted topology quality.

        During the short warm-up the head is untrained, so inference keeps
        the gating fallback.  Evaluation after training has no active epoch
        and therefore uses the trained quality head.
        """
        epoch = getattr(self, '_epoch', None)
        warmup = int(getattr(self, 'quality_warmup_epochs', 0))
        if epoch is not None and int(epoch) <= warmup:
            # Refinement must not consume scores from an untrained head, or
            # stale scores left over from a previous phase.
            self._last_quality_scores = None
            return None
        scores = self._quality_scores(model, stage2)
        if scores.shape[0] != 1:
            raise RuntimeError(
                'quality candidate selection expects batch size 1, got '
                f'{scores.shape[0]}')
        self._last_quality_scores = scores.detach()
        best = int(scores.argmax(dim=1)[0].item())
        anchor = self._selected_stage2_index()
        margin = float(getattr(self, 'quality_select_margin', 0.0))
        if (0 <= anchor < scores.shape[1]
                and float(scores[0, anchor]) + margin >= float(
                    scores[0, best])):
            # Keep the gating/deployment candidate when quality scores are
            # too close; this prevents an untrained or noisy quality head
            # from changing the base output.
            return None
        return best

    def train_step(self, model, batch):
        images, masks = batch
        stage2, iou_pred, stage1 = self._forward(
            model, images, return_stage1=True)
        # Keep the native Stage-2 tensor for the STC branch: predict_sample
        # resamples 256 -> stc_resolution directly, while loss_main below
        # uses the 256 -> 1024 training geometry.
        stage2_native = stage2
        masks = F.interpolate(masks, (1024, 1024), mode='bilinear',
                              align_corners=False)
        stage1 = F.interpolate(stage1, (1024, 1024), mode='bilinear',
                               align_corners=False)
        loss_quality = stage2.new_zeros(())
        if self.topology_quality:
            loss_quality = self._quality_loss(model, stage2_native, masks)

        if int(getattr(self.args, 'use_typed_stage2', 0)) == 1:
            # Keep Stage-2 logits at their native 256 resolution.  _typed_loss
            # resamples them directly to typed_resolution, matching
            # predict_sample; going 256 -> 1024 -> 512 would blur the heads
            # and create a train/inference mismatch.
            return self._typed_loss(model, masks, stage2, stage1)

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
        aggregate = str(getattr(
            self, 'stage2_loss_aggregate',
            getattr(self.args, 'stage2_loss_aggregate', 'deployment')))
        aux_weight = float(getattr(
            self, 'stage2_aux_weight',
            getattr(self.args, 'stage2_aux_weight', 0.25)))
        loss_main = self._aggregate_head_losses(
            head_losses, head_indices, self._deployment_anchor_index(),
            aux_weight, aggregate)

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

        loss = (loss_main + deep_sup * loss_ds + iou_weight * loss_iou
                + loss_quality)
        # Native topology supervision avoids the 256 -> 1024 -> 512
        # resampling blur.  The legacy path intentionally passes the 1024-res
        # stretched tensor so the native-vs-legacy control changes only this
        # input, not the loss, targets, or aggregation.
        topology_source = str(getattr(
            self, 'topology_source',
            getattr(self.args, 'topology_source', 'native'))).lower()
        topology_input = (
            stage2_native if topology_source == 'native' else stage2)
        loss = loss + self._topology_loss(topology_input, masks)
        if int(getattr(self.args, 'use_stc', 0)) == 1:
            loss = loss + self._stc_loss(
                model, images, masks, stage2_native, iou_pred)
        return loss

    def prepare_eval_sample(self, sample, device):
        image = load_rgb(sample.image_path).resize(
            (self.input_size, self.input_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(device)

    def predict_sample(self, model, prepared, sample):
        stage1 = None
        stage1_weight = float(getattr(
            self.args, 'stage1_fusion_weight', 0.0))
        use_typed = int(getattr(self.args, 'use_typed_stage2', 0)) == 1
        use_stc = int(getattr(self.args, 'use_stc', 0)) == 1
        need_stage1 = (
            stage1_weight > 0
            or (self.quality_refine and self.quality_refine_stage1)
        )
        if need_stage1 and not use_typed and not use_stc:
            stage2, iou_pred, stage1 = self._forward(
                model, prepared, return_stage1=True)
        else:
            stage2, iou_pred = self._forward(
                model, prepared, return_stage1=False)
        quality_index = None
        if self.topology_quality and not use_typed and not use_stc:
            quality_index = self._quality_candidate_index(model, stage2)
        if use_typed:
            composer = getattr(model, 'typed_composer', None)
            if composer is None:
                raise RuntimeError(
                    'typed Stage-2 requires model.typed_composer')
            resolution = int(getattr(self.args, 'typed_resolution', 512))
            output = getattr(self.args, 'typed_output', 'compose')
            if output == 'dense':
                logits = F.interpolate(
                    stage2[:, 0:1], (resolution, resolution),
                    mode='bilinear', align_corners=False)
            else:
                heads = [
                    F.interpolate(
                        stage2[:, k:k + 1], (resolution, resolution),
                        mode='bilinear', align_corners=False)
                    for k in range(4)
                ]
                blend = getattr(
                    self.args, 'calibrated_blend_g', None)
                logits = composer(
                    *heads, resolution, blend_override=blend)['final']
        elif use_stc:
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
            # Resolve the single candidate this selector deploys, if any.
            # Aggregate selectors (mean / gating_soft) have no single base and
            # are not refined.
            base_index = None
            if quality_index is not None:
                base_index = quality_index
            elif self.head_select == 'gating':
                # Official gating order: slot 0 is the top-gated candidate.
                base_index = 0
            elif self.head_select == 'fixed':
                base_index = int(self.deployment_head)
            elif self.head_select == 'iou':
                base_index = int(iou_pred.argmax(dim=1)[0].item())

            if base_index is not None:
                logits = stage2[:, base_index:base_index + 1]
                # Refinement is independent of the quality override: it runs
                # whenever the feature is enabled and the quality head has
                # produced scores, including when selection kept gating.
                if (self.quality_refine
                        and self._last_quality_scores is not None):
                    logits = self._quality_refined_logits(
                        stage2, self._last_quality_scores, stage1,
                        base_index=base_index)
            elif self.head_select == 'mean':
                # The four native Stage-2 heads are all dense candidates.
                # Averaging their logits is permutation-invariant, so it also
                # works if the official gating order changes per image.
                logits = stage2.mean(dim=1, keepdim=True)
            elif self.head_select == 'gating_soft':
                weights = getattr(
                    model.mask_decoder, 'last_gating_weights', None)
                if weights is None:
                    logits = stage2.mean(dim=1, keepdim=True)
                else:
                    weights = weights.to(stage2.dtype)
                    weights = weights / weights.sum(
                        dim=1, keepdim=True).clamp_min(1e-6)
                    logits = (stage2 * weights[:, :, None, None]).sum(
                        dim=1, keepdim=True)
            else:
                # Defensive fallback: official gating slot 0.
                logits = stage2[:, 0:1]
            if stage1_weight > 0 and stage1 is not None:
                if getattr(self.args, 'stage1_fusion_mode',
                           'gated') == 'mean':
                    stage1_logits = stage1.mean(dim=1, keepdim=True)
                else:
                    stage1_logits = stage1[:, 0:1]
                logits = ((1.0 - stage1_weight) * logits
                          + stage1_weight * stage1_logits)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(logits, size=(height, width),
                               mode='bilinear', align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold)

    @torch.no_grad()
    def predict_candidates(self, model, prepared, sample):
        """Return all four post-gating Stage-2 candidates at GT resolution.

        This is an analysis-only helper used by the candidate-head audit.  It
        deliberately rejects typed/STC/fused deployments because their
        deployed mask is not one of the four native Stage-2 candidates.
        ``candidate 0`` is the deployment slot for the shipped gating recipe;
        callers should use :meth:`selected_candidate_index` rather than
        assuming index 0 for other selectors.
        """
        if not sample.mask_path:
            raise ValueError('candidate diagnostics require a GT mask')
        if int(getattr(self.args, 'use_typed_stage2', 0)) == 1:
            raise RuntimeError(
                'candidate diagnostics do not support typed Stage-2')
        if int(getattr(self.args, 'use_stc', 0)) == 1:
            raise RuntimeError(
                'candidate diagnostics do not support the STC branch')
        if float(getattr(self.args, 'stage1_fusion_weight', 0.0)) > 0:
            raise RuntimeError(
                'candidate diagnostics require stage1-fusion-weight=0')

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
