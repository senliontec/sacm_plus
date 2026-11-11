"""Unified training engine.

The Trainer is model-agnostic: every model-specific behaviour comes from
the ModelSpec. The training-loop logic is moved VERBATIM from the
original train_sam.py (four-term loss — main DiceBCE, Stage-1 deep
supervision, IoU-head MSE, soft-clDice with warmup — plus the optional
extra topology loss; gradient clipping; best-F1 checkpoint selection).
"""

import json
import logging
import os
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from core.console import console, epoch_header, epoch_row, topo_legend
from core.data import letterbox_rect
from core.evaluator import (
    resolve_selection,
    select_mask_fixed,
    select_mask_quality,
)
from core.losses.monitor import topology_loss_monitor
from core.run_artifacts import (
    ARTIFACT_VERSION,
    capture_generator_state,
    capture_rng_state,
    write_json_atomic,
    write_wandb_info,
)
from core.losses.registry import TOPOLOGY_LOSS_REGISTRY
from core.losses.structural_targets import build_structural_targets
from core.losses.topology_utils import resize_target_for_topology
from core.losses.tversky import TverskyLoss
from core.metrics import BINARY_METRICS, binary_metrics, compute_metrics
from core.metrics_advisor import advise
from core.typed_heads import TYPED_HEAD_NAMES
from core.wandb_utils import finish_run, log_scalars, setup_wandb


class ForwardWrapper(nn.Module):
    """Expose the model-specific prompt-free forward to DDP.

    DDP only synchronizes gradients when its own ``forward`` runs.  Calling
    ``model.module(...)`` directly bypasses ``Reducer.prepare_for_backward``
    and silently disables gradient all-reduce, so the actual training forward
    is delegated to this wrapper and DDP wraps the wrapper.
    """

    def __init__(self, model, spec):
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, images):
        images = self.spec.preprocess(self.model, images)
        sparse, dense = self.spec.prompt_builder(
            self.model, images.shape[0])
        return self.spec.forward(
            self.model, images, sparse, dense, return_stage1=True)


def unwrap_sam_model(model):
    """Return the inner Sam from DDP(ForwardWrapper(Sam)) or Sam."""
    inner = getattr(model, 'module', model)
    if isinstance(inner, ForwardWrapper):
        inner = inner.model
    return inner


def calculate_f1_score(pred_masks, true_masks, threshold=0.0):
    """F1 on raw logits (0.0 = probability 0.5, SAM's mask_threshold)."""
    pred_masks = pred_masks.detach().cpu().numpy()
    true_masks = true_masks.detach().cpu().numpy()

    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = (true_masks > 0.5).astype(np.uint8)

    return f1_score(true_masks.reshape(-1), pred_masks.reshape(-1))


def cl_dice_weight_schedule(epoch, warmup_epochs, ramp_epochs):
    """0 before warmup, then a linear ramp to 1 over ramp_epochs.

    Keeps the soft-skeleton gradients out of the early training stage
    where they are dominated by noise on random masks.
    """
    if epoch < warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs) / max(ramp_epochs, 1))


def stage2_soft_weights(iou_pred, mix=0.5, tau=1.0):
    """D8 fix: uniform + IoU-softmax mixture over Stage-2 candidates.

    The weights are detached: the main mask loss trains every candidate
    through the weighted sum, while the IoU head keeps its own MSE
    supervision. ``mix=0`` is equivalent to uniform all-head training;
    ``mix=1`` trusts the IoU head. A small uniform floor (mix<1) prevents
    early IoU-head noise from starving any candidate of gradient.
    """
    k = iou_pred.shape[1]
    uniform = torch.full_like(iou_pred, 1.0 / float(k))
    soft = F.softmax(iou_pred / max(float(tau), 1e-6), dim=1)
    mix = min(max(float(mix), 0.0), 1.0)
    return ((1.0 - mix) * uniform + mix * soft).detach()


def _merge_val_payloads(gathered):
    """Merge per-rank validation payloads into global metrics (rank 0).

    gathered: list of dicts with keys loss_sum/count/tp/fp/fn plus the
    optional full-validation fields cldice/metrics/macro/empty/per_dataset
    (see _validate). F1 comes from the GLOBAL confusion counts (exact
    micro-F1, identical to sklearn on the concatenated shards). Baseline
    style additionally returns macro per-image metrics and per-dataset
    buckets, matching ``src/baseline_zoo/runtime.py::evaluate``.
    """
    avg_val_loss = sum(g['loss_sum'] for g in gathered) / max(
        sum(g['count'] for g in gathered), 1)
    tp = sum(g['tp'] for g in gathered)
    fp = sum(g['fp'] for g in gathered)
    fn = sum(g['fn'] for g in gathered)
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom > 0 else 0.0
    acc = {}
    macro_acc = {}
    empty_all = []
    per_dataset = {}
    for g in gathered:
        for k, v in g.get('metrics', {}).items():
            acc.setdefault(k, []).extend(v)
        for k, v in g.get('macro', {}).items():
            macro_acc.setdefault(k, []).extend(v)
        empty_all.extend(g.get('empty', []))
        for dataset, values in g.get('per_dataset', {}).items():
            bucket = per_dataset.setdefault(dataset, {})
            for k, v in values.items():
                bucket.setdefault(k, []).extend(v)
    cldice_all = [x for g in gathered for x in g.get('cldice', [])]
    with warnings.catch_warnings():
        # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
        warnings.simplefilter('ignore', RuntimeWarning)
        if any(macro_acc.values()):
            # Baseline-style path: macro per-image metrics.
            val_metrics = {k: float(np.nanmean(v))
                           for k, v in macro_acc.items() if v}
            if empty_all:
                val_metrics['empty_prediction_rate'] = float(
                    np.nanmean(empty_all))
            avg_val_cldice = val_metrics.get('cldice', float('nan'))
        else:
            # Rich-table / legacy path.
            avg_val_cldice = (
                float(np.nanmean(cldice_all)) if cldice_all else float('nan'))
            val_metrics = {k: float(np.nanmean(v))
                           for k, v in acc.items() if v}
        val_per_dataset = {}
        for dataset, values in per_dataset.items():
            merged = {}
            for k, v in values.items():
                name = ('empty_prediction_rate'
                        if k == 'empty' else k)
                merged[name] = float(np.nanmean(v))
            val_per_dataset[dataset] = merged
    return (avg_val_loss, f1, avg_val_cldice, val_metrics,
            val_per_dataset)


class Trainer:
    """Unified training engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, soft_cl, topo_loss,
                 optimizer, scheduler, train_loader, val_loader, args,
                 is_ddp=False, train_sampler=None, boundary_loss=None,
                 artifact_info=None, train_generator=None, val_generator=None,
                 resume_state=None, raw_criterion=None):
        self.model = model
        self.spec = spec
        self.device = device
        self.criterion = criterion
        # SCNP wraps the mask criterion.  Auxiliary structural heads and the
        # residual repair must keep the raw Dice/BCE objective; otherwise the
        # same-class neighbour transform is applied to 1-px centerline and
        # boundary targets, which is not the official SCNP objective.
        self.raw_criterion = raw_criterion if raw_criterion is not None else criterion
        self.soft_cl = soft_cl
        self.topo_loss = topo_loss
        self.boundary_loss = boundary_loss
        self.tversky = TverskyLoss(
            alpha=float(getattr(args, 'tversky_alpha', 0.3)),
            beta=float(getattr(args, 'tversky_beta', 0.7)))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.is_ddp = is_ddp
        self.train_sampler = train_sampler
        self.artifact_info = artifact_info or {}
        self.train_config = self.artifact_info.get(
            'train_config_payload', {})
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.resume_state = resume_state or {}
        self.save_dir = os.path.dirname(self.args.save_path) or '.'
        self.last_checkpoint = self.resume_state.get('last_checkpoint')
        self.periodic_checkpoints = []
        self.wandb_info = {}
        self.best_f1_score = 0.0
        # Checkpoint selection metric.  With full validation 'dice' means
        # macro per-image Dice, exactly matching the baseline runner's
        # selection rule.  Under fast_val no macro Dice exists, so the
        # default falls back to micro F1 (binary F1 == binary Dice).
        self.selection_metric = getattr(args, 'selection_metric', 'dice')
        # Baseline runner initializes its best score to -inf; do the same in
        # baseline mode so a first validation epoch with score 0.0 still
        # saves a checkpoint.  Keep the historical 0.0 for table/official.
        log_style = getattr(args, 'log_style', 'table')
        self.best_selection_score = (
            -float('inf') if log_style == 'baseline' else 0.0)
        if self.resume_state:
            self.best_selection_score = float(
                self.resume_state.get(
                    'best_selection_score', self.best_selection_score))
            self.best_f1_score = float(
                self.resume_state.get('best_f1_score', 0.0) or 0.0)
        if (self.selection_metric == 'cldice'
                and getattr(args, 'fast_val', False)):
            logging.warning(
                "selection_metric='cldice' requires full validation; "
                "fast_val=True falls back to dice for checkpoint selection.")
        if getattr(args, 'stage2_supervision', 'soft') == 'wta' and self._rank0:
            logging.warning(
                "D8: stage2_supervision='wta' — only the gating champion "
                "Stage-2 head receives mask loss. Use 'soft' (default) or "
                "'allhead' unless running the D8 baseline/ablation.")
        if (getattr(args, 'stage2_supervision', 'soft') == 'typed'
                and not getattr(args, 'use_typed_heads', False)):
            raise ValueError(
                "stage2_supervision='typed' requires --use_typed_heads true "
                "(the full_typed preset sets both)")
        if (getattr(args, 'use_typed_heads', False)
                and getattr(args, 'stage2_supervision', 'soft') != 'typed'
                and self._rank0):
            logging.warning(
                "use_typed_heads=True but stage2_supervision=%s; typed heads "
                "will receive no type-specific structural loss. Use the "
                "full_typed preset or set stage2_supervision=typed.",
                getattr(args, 'stage2_supervision', 'soft'))

    @property
    def _rank0(self):
        """DDP 下日志/保存只在 rank 0(验证已分片到全部 rank)。"""
        if not self.is_ddp:
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0

    def _forward(self, images):
        # DDP must see the forward call itself; calling .module directly
        # skips DDP's reducer preparation and silently disables all-reduce.
        if self.is_ddp:
            return self.model(images)
        model = self.model
        images = self.spec.preprocess(model, images)
        sparse, dense = self.spec.prompt_builder(model, images.shape[0])
        return self.spec.forward(model, images, sparse, dense, return_stage1=True)

    def _res_scale_for_epoch(self, epoch):
        """Residual-pathway activity for one epoch.

        The old implementation only scaled the extra losses; the module itself
        was still in the forward pass, so the gate/expert pathway was never
        truly frozen.  The returned scale is also pushed into the module.
        """
        if not bool(getattr(self.args, 'use_res_moe', False)):
            return 0.0
        warm = int(getattr(self.args, 'res_warmup_epochs', 30))
        ramp = max(1, int(getattr(self.args, 'res_ramp_epochs', 30)))
        step = max(0, (epoch + 1) - warm)
        return min(1.0, step / float(ramp))

    def _set_res_scale(self, epoch):
        if not bool(getattr(self.args, 'use_res_moe', False)):
            return
        base_model = unwrap_sam_model(self.model)
        decoder = getattr(base_model, 'mask_decoder', None)
        res_moe = getattr(decoder, 'residual_moe', None)
        if res_moe is not None:
            res_moe.res_scale = self._res_scale_for_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        self._set_res_scale(epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)  # DDP: 每 epoch 换采样顺序
        train_loss = 0.0
        comps_sum = {k: 0.0 for k in (
            'loss_main', 'loss_ds', 'loss_iou', 'loss_cl', 'loss_topo',
            'loss_boundary', 'loss_center', 'loss_width', 'loss_area',
            'loss_moe', 'loss_res', 'loss_res_topo', 'loss_gate',
            'loss_res_sparse')}
        # 进度条已移除:逐 epoch 的 rich 宽表 + time 列取代进度反馈
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]",
                          disable=True)

        for images, masks in train_pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            masks_stage2, iou_pred, masks_stage1 = self._forward(images)

            # All losses are computed at 1024 resolution. The GT is already
            # 1024 after transforms.Resize, so this interpolation is a
            # (near-zero-cost) identity — kept deliberately to make the
            # loss block self-contained.
            masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            m2 = F.interpolate(masks_stage2, size=(1024, 1024), mode='bilinear', align_corners=False)
            m1 = F.interpolate(masks_stage1, size=(1024, 1024), mode='bilinear', align_corners=False)

            # TADT regularizer: the decision boundary should be stable under
            # a small threshold perturbation.  This is a soft Dice agreement
            # between the two perturbed surrogates; it is computed on the
            # decoder-resolution logits to keep memory bounded.
            loss_tadt = 0.0
            tadt_weight = float(getattr(
                self.args, 'tadt_consistency_weight', 0.0))
            if tadt_weight > 0 and getattr(self.args, 'use_tadt', False):
                delta = float(getattr(self.args, 'tadt_delta', 0.5))
                b_plus = torch.sigmoid(masks_stage2 - delta)
                b_minus = torch.sigmoid(masks_stage2 + delta)
                inter_t = (b_plus * b_minus).sum(dim=(2, 3))
                denom_t = b_plus.sum(dim=(2, 3)) + b_minus.sum(dim=(2, 3))
                dice_t = (2.0 * inter_t + 1e-6) / (denom_t + 1e-6)
                loss_tadt = (1.0 - dice_t).mean()

            # D8: Stage-2 supervision mode.
            # wta     : original SACM winner-take-all (only gating head 0)
            # allhead : all Stage-2 mask heads (diversity may collapse)
            # soft    : detached uniform + IoU-softmax mixture; every head
            #           gets gradient while the IoU head is trained separately
            stage2_mode = getattr(self.args, 'stage2_supervision', 'soft')
            if stage2_mode == 'wta':
                main_mask = m2[:, 0:1]
                aux_mask = main_mask
                loss_main = self.criterion(main_mask, masks)
            elif stage2_mode == 'allhead':
                head_losses = [self.criterion(m2[:, k:k + 1], masks)
                               for k in range(m2.shape[1])]
                loss_main = sum(head_losses) / float(len(head_losses))
                aux_mask = m2.mean(dim=1, keepdim=True)
            elif stage2_mode == 'soft':
                weights = stage2_soft_weights(
                    iou_pred,
                    mix=getattr(self.args, 'soft_select_mix', 0.5),
                    tau=getattr(self.args, 'soft_select_tau', 1.0),
                )
                main_mask = (m2 * weights[:, :, None, None]).sum(dim=1, keepdim=True)
                aux_mask = main_mask
                loss_main = self.criterion(main_mask, masks)
            elif stage2_mode == 'typed':
                if not getattr(self.args, 'use_typed_heads', False):
                    raise ValueError(
                        "stage2_supervision='typed' requires use_typed_heads")
                if m2.shape[1] != len(TYPED_HEAD_NAMES):
                    raise ValueError(
                        f"typed heads expect {len(TYPED_HEAD_NAMES)} "
                        f"Stage-2 slots, got {m2.shape[1]}")
                # Base Dice/BCE on every typed candidate; the type-specific
                # structural terms below shape the specialization.
                head_losses = [self.criterion(m2[:, k:k + 1], masks)
                               for k in range(m2.shape[1])]
                loss_main = sum(head_losses) / float(len(head_losses))
                aux_mask = m2.mean(dim=1, keepdim=True)
                main_mask = m2[:, 0:1]
            else:
                raise ValueError(
                    f"Unknown stage2_supervision={stage2_mode!r}; "
                    "expected one of wta/allhead/soft/typed")

            # Fixed-head supervision: the deployment head (fixed2) must
            # receive the structural terms directly, not only through the
            # soft mixture of four near-identical heads.
            fixed_w = float(getattr(
                self.args, 'fixed_head_loss_weight', 0.0))
            if fixed_w > 0 and stage2_mode != 'typed' and m2.shape[1] > 2:
                fixed_head = int(getattr(
                    self.args, 'fixed_supervision_head', 2))
                fixed_mask = m2[:, fixed_head:fixed_head + 1]
                loss_main = loss_main + fixed_w * self.criterion(
                    fixed_mask, masks)
                aux_mask = fixed_mask

            # Stage-1 deep supervision: ALL heads vs GT. This is what makes
            # the gating scores meaningful in the first place.
            loss_ds = 0.0
            if self.args.deep_sup_weight > 0:
                for k in range(m1.shape[1]):
                    loss_ds = loss_ds + self.criterion(m1[:, k:k + 1], masks)
                loss_ds = loss_ds / m1.shape[1]

            # IoU head / candidate-ranker supervision.  The ground-truth
            # per-head IoU is shared by the MSE term and the listwise
            # ranking term.
            loss_iou = 0.0
            loss_rank = 0.0
            rank_weight = float(getattr(self.args, 'rank_loss_weight', 0.0))
            iou_gt = None
            if self.args.iou_loss_weight > 0 or (
                    rank_weight > 0 and getattr(self.args, 'use_ranker', False)):
                with torch.no_grad():
                    m2_bin = (m2 > 0).float()
                    gt_bin = (masks > 0.5).float()
                    inter = (m2_bin * gt_bin).sum(dim=(2, 3))
                    union = m2_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3)) - inter
                    # Match the metric convention: IoU/Dice = 1 when BOTH
                    # masks are empty; otherwise use the normal ratio.
                    iou_gt = torch.where(
                        union > 0,
                        inter / union.clamp_min(1e-6),
                        torch.ones_like(inter))
                if self.args.iou_loss_weight > 0:
                    loss_iou = F.mse_loss(iou_pred, iou_gt)
                if rank_weight > 0 and getattr(self.args, 'use_ranker', False):
                    target_rank = torch.softmax(iou_gt / 0.5, dim=1)
                    log_prob = torch.log_softmax(iou_pred, dim=1)
                    listwise = -(target_rank * log_prob).sum(dim=1).mean()
                    # Pairwise regret: a higher-GT-IoU candidate should have
                    # a higher score by at least a small margin.
                    pair_loss = 0.0
                    pair_count = 0
                    for left in range(iou_gt.shape[1]):
                        for right in range(left + 1, iou_gt.shape[1]):
                            diff = iou_gt[:, left] - iou_gt[:, right]
                            valid = diff.abs() > 1e-6
                            if not bool(valid.any()):
                                continue
                            sign = torch.sign(diff)
                            margin = 0.05
                            hinge = torch.relu(
                                margin - sign * (
                                    iou_pred[:, left] - iou_pred[:, right]))
                            pair_loss = pair_loss + hinge[valid].mean()
                            pair_count += 1
                    if pair_count:
                        pair_loss = pair_loss / float(pair_count)
                    loss_rank = listwise + 0.5 * pair_loss

            # Differentiable clDice / boundary / recall terms.  In typed
            # mode each structural term is attached to its own head; the
            # selected/mean mask no longer receives a global clDice term.
            loss_cl = 0.0
            loss_boundary = 0.0
            loss_tversky = 0.0
            mu = self.args.cl_dice_weight * cl_dice_weight_schedule(
                epoch, self.args.cl_dice_warmup, self.args.cl_dice_ramp
            )
            boundary_weight = float(
                getattr(self.args, 'boundary_loss_weight', 0.0))
            boundary_mu = boundary_weight * cl_dice_weight_schedule(
                epoch,
                int(getattr(self.args, 'boundary_loss_warmup', 10)),
                int(getattr(self.args, 'boundary_loss_ramp', 5)))
            tversky_mu = float(getattr(self.args, 'tversky_weight', 0.0)) * \
                cl_dice_weight_schedule(
                    epoch, self.args.cl_dice_warmup, self.args.cl_dice_ramp)

            if stage2_mode == 'typed':
                if mu > 0:
                    loss_cl = self.soft_cl(
                        m2[:, TYPED_HEAD_NAMES.index('thin'):
                           TYPED_HEAD_NAMES.index('thin') + 1], masks)
                if self.boundary_loss is not None and boundary_mu > 0:
                    loss_boundary = self.boundary_loss(
                        m2[:, TYPED_HEAD_NAMES.index('boundary'):
                           TYPED_HEAD_NAMES.index('boundary') + 1], masks)
                if tversky_mu > 0:
                    loss_tversky = self.tversky(
                        m2[:, TYPED_HEAD_NAMES.index('recall'):
                           TYPED_HEAD_NAMES.index('recall') + 1], masks)
            else:
                if mu > 0:
                    loss_cl = self.soft_cl(aux_mask, masks)
                if self.boundary_loss is not None and boundary_mu > 0:
                    loss_boundary = self.boundary_loss(aux_mask, masks)
                if tversky_mu > 0:
                    loss_tversky = self.tversky(aux_mask, masks)

            # Extra topology loss (optional; strict ports)
            loss_topo = 0.0
            if self.topo_loss is not None:
                loss_topo = self.topo_loss(aux_mask, masks)

            # V3 structural auxiliary heads.  Targets are built at the
            # decoder resolution with max-pooled GT so 1-px lines survive.
            loss_center = 0.0
            loss_width = 0.0
            loss_area = 0.0
            loss_aux_boundary = 0.0
            use_aux = bool(
                getattr(self.args, 'use_v3', False)
                or getattr(self.args, 'use_structural_aux', False))
            if use_aux:
                base_model = unwrap_sam_model(self.model)
                decoder = getattr(base_model, 'mask_decoder', None)
                aux = getattr(decoder, 'last_aux_outputs', None)
                center_w = float(getattr(
                    self.args, 'v3_centerline_weight', 0.0))
                aux_boundary_w = float(getattr(
                    self.args, 'v3_boundary_weight', 0.0))
                width_w = float(getattr(
                    self.args, 'v3_width_weight', 0.0))
                area_w = float(getattr(self.args, 'v3_area_weight', 0.0))
                if aux is not None and (center_w > 0 or aux_boundary_w > 0
                                       or width_w > 0):
                    res = aux['centerline'].shape[-1]
                    targets = build_structural_targets(
                        masks, res,
                        width_max=float(getattr(decoder, 'width_max', 16.0)))
                    if center_w > 0 and aux.get('centerline') is not None:
                        # SCNP is for the segmentation mask, not for the
                        # 1-px centerline auxiliary target.
                        loss_center = self.raw_criterion(
                            aux['centerline'], targets['centerline'])
                        cl256 = aux.get('centerline_256')
                        if cl256 is not None:
                            t256 = resize_target_for_topology(
                                targets['centerline'], cl256.shape[-1])
                            loss_center = 0.5 * (
                                loss_center
                                + self.raw_criterion(cl256, t256))
                    if (aux_boundary_w > 0
                            and aux.get('boundary') is not None):
                        loss_aux_boundary = self.raw_criterion(
                            aux['boundary'], targets['boundary'])
                    if width_w > 0 and aux.get('width') is not None:
                        pred_width = torch.sigmoid(aux['width'])
                        fg = targets['width'] > 0
                        if bool(fg.any()):
                            loss_width = F.l1_loss(
                                pred_width[fg], targets['width'][fg])
                if area_w > 0:
                    pred_area = torch.sigmoid(aux_mask).mean(dim=(2, 3))
                    gt_area = masks.mean(dim=(2, 3))
                    loss_area = (pred_area - gt_area).abs().mean()

            # Conv-LoRA noisy-gate load-balancing loss.  The gate already
            # scales its own loss by 1e-2; moe_loss_weight=0 disables it.
            loss_moe = 0.0
            moe_weight = float(getattr(self.args, 'moe_loss_weight', 0.0))
            if moe_weight > 0:
                base_model = unwrap_sam_model(self.model)
                encoder = getattr(base_model, 'image_encoder', None)
                moe = getattr(encoder, 'last_moe_loss', None)
                if moe is not None:
                    loss_moe = moe

            # Topology-aware residual MoE.  The corrected head already
            # receives the base fixed-head loss; this adds a direct
            # residual term and the gate supervision.  Warm-up/ramp keeps
            # the residual pathway inactive while the coarse decoder
            # stabilises.
            loss_res = 0.0
            loss_res_topo = 0.0
            loss_gate = 0.0
            loss_res_sparse = 0.0
            res_scale = 0.0
            if bool(getattr(self.args, 'use_res_moe', False)):
                base_model = unwrap_sam_model(self.model)
                decoder = getattr(base_model, 'mask_decoder', None)
                res_moe = getattr(decoder, 'residual_moe', None)
                if res_moe is not None:
                    res_scale = self._res_scale_for_epoch(epoch)
                    if res_scale > 0:
                        head = int(getattr(
                            self.args, 'res_head_index', 2))
                        res_topo_w = float(getattr(
                            self.args, 'res_topology_weight', 0.0))
                        if m2.shape[1] > head:
                            head_logits = m2[:, head:head + 1]
                            # Repair loss: raw Dice/BCE (SCNP must not leak
                            # into the repair target) plus an explicit
                            # topology objective on the corrected head.
                            loss_res = self.raw_criterion(
                                head_logits, masks)
                            if res_topo_w > 0:
                                if self.topo_loss is not None:
                                    loss_res_topo = loss_res_topo + \
                                        self.topo_loss(head_logits, masks)
                                loss_res_topo = loss_res_topo + \
                                    self.soft_cl(head_logits, masks)
                        if getattr(res_moe, 'has_learned_gate', False):
                            loss_gate = res_moe.gate_loss(masks)
                        loss_res_sparse = res_moe.sparsity_loss()

            loss = (
                loss_main
                + self.args.deep_sup_weight * loss_ds
                + self.args.iou_loss_weight * loss_iou
                + rank_weight * loss_rank
                + mu * loss_cl
                + boundary_mu * loss_boundary
                + tversky_mu * loss_tversky
                + tadt_weight * loss_tadt
                + self.args.topology_loss_weight * loss_topo
                + moe_weight * loss_moe
                + float(getattr(self.args, 'v3_centerline_weight', 0.0))
                * loss_center
                + float(getattr(self.args, 'v3_boundary_weight', 0.0))
                * loss_aux_boundary
                + float(getattr(self.args, 'v3_width_weight', 0.0))
                * loss_width
                + float(getattr(self.args, 'v3_area_weight', 0.0))
                * loss_area
                + res_scale * float(getattr(
                    self.args, 'res_loss_weight', 0.5)) * loss_res
                + res_scale * float(getattr(
                    self.args, 'res_topology_weight', 0.0)) * loss_res_topo
                + res_scale * float(getattr(
                    self.args, 'res_gate_loss_weight', 0.3)) * loss_gate
                + res_scale * float(getattr(
                    self.args, 'res_sparsity_weight', 0.02))
                * loss_res_sparse
            )

            loss.backward()

            # Add gradient clipping to improve stability
            if self.args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.clip_grad_norm
                )

            self.optimizer.step()

            train_loss += loss.item()
            for k in comps_sum:
                comps_sum[k] += locals()[k].item() if isinstance(locals()[k], torch.Tensor) else float(locals()[k])
            train_pbar.set_postfix({'loss': loss.item(), 'cl': mu})

        n = len(self.train_loader)
        comps = {k: v / n for k, v in comps_sum.items()}
        return train_loss / n, comps

    def _val_dataset_name(self, img_name):
        """Dataset name for one validation image (baseline W&B parity)."""
        if isinstance(img_name, (list, tuple)):
            img_name = img_name[0] if img_name else ''
        dataset = getattr(self.val_loader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'dataset_for'):
            return dataset.dataset_for(img_name)
        return 'val'

    def _validate(self, epoch):
        self.model.eval()
        self._set_res_scale(epoch)
        # 验证期在训练缓存之上叠加前向/指标,先清碎片避免 OOM 峰
        # (epoch 5 实测出现过 2.5GB 分配失败)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        fast_val = bool(getattr(self.args, 'fast_val', False))
        log_style = getattr(self.args, 'log_style', 'table')
        baseline_style = log_style == 'baseline'
        if baseline_style and fast_val:
            logging.warning(
                "log_style='baseline' requires full validation; "
                "fast_val=True falls back to loss + F1 only.")
            baseline_style = False
        val_loss = 0.0
        n_batches = 0
        tp = fp = fn = 0
        val_cldice = []
        metrics_acc = {}
        macro_acc = {key: [] for key in BINARY_METRICS}
        empty_acc = []
        per_dataset_acc = {}
        val_has_monitored = False
        vis_items = []  # (image, gt, pred) 供 wandb 可视化
        VAL_ENGINE_IMGS = 50  # 引擎指标子集大小(均值估计,验证提速关键)
        val_selection_raw = getattr(self.args, 'val_selection', 'auto')
        if val_selection_raw == 'fixed':
            val_selection = 'fixed'
        else:
            val_selection = resolve_selection(
                val_selection_raw,
                getattr(self.args, 'iou_loss_weight', 0.0))
        val_empty_guard = bool(getattr(self.args, 'val_empty_guard', False))
        quality_small_weight = float(getattr(
            self.args, 'quality_small_component_weight', 0.5))
        quality_count_weight = float(getattr(
            self.args, 'quality_component_weight', 0.05))
        pred_threshold = float(getattr(self.args, 'pred_threshold', 0.0))

        # DDP 分布式验证(fencing-algs 模式):验证集由 DistributedSampler
        # 分片,每 rank 只算 600/8≈75 张,指标 gather 到 rank 0 汇总。
        # 此前验证只在 rank 0 跑(~16 分钟/epoch),rank 1-7 会阻塞在
        # 下一轮 backward 的 NCCL allreduce 里空等——占显存、GPU 0%,
        # 成为服务器清理器/OOM killer 的击杀目标(实测 7 rank 被杀);
        # 分片后 ~2.5 分钟、全 epoch 8 卡满负荷。
        if self.is_ddp:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                        desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]",
                        disable=True)

        with torch.no_grad():
            for val_idx, (images, masks, original_size, _img_name, fov_mask) in enumerate(val_pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                fov_mask = fov_mask.to(self.device)

                masks_stage2, iou_pred, _masks_stage1 = self._forward(images)

                # Select at decoder resolution, then restore the ORIGINAL GT
                # size before loss/metrics — the same path as test evaluation.
                if val_selection == 'fixed':
                    main_mask, _selected_head = select_mask_fixed(
                        masks_stage2,
                        int(getattr(self.args, 'val_fixed_head', 2)),
                        empty_guard=val_empty_guard,
                        foreground_threshold=pred_threshold)
                elif val_selection == 'quality':
                    main_mask, _selected_head, _scores = select_mask_quality(
                        masks_stage2,
                        iou_pred,
                        foreground_threshold=pred_threshold,
                        small_component_weight=quality_small_weight,
                        component_count_weight=quality_count_weight,
                        empty_guard=val_empty_guard)
                elif val_selection == 'iou':
                    idx = iou_pred.argmax(dim=1)
                    gather = idx[:, None, None, None].expand(
                        -1, 1, masks_stage2.shape[2], masks_stage2.shape[3])
                    main_mask = torch.gather(masks_stage2, 1, gather)
                else:
                    main_mask = masks_stage2[:, 0:1]

                width, height = original_size
                width = width.item() if torch.is_tensor(width) else int(width)
                height = height.item() if torch.is_tensor(height) else int(height)
                if getattr(self.args, 'eval_input_mode', 'resize') == 'aspect_pad':
                    left, top, content_w, content_h = letterbox_rect(
                        width, height, main_mask.shape[-1])
                    main_mask = main_mask[
                        :, :, top:top + content_h, left:left + content_w]
                main_mask = F.interpolate(
                    main_mask, size=(height, width),
                    mode='bilinear', align_corners=False)
                # Official-comparison protocol: no FOV masking.  The final
                # evaluator uses the same switch so checkpoint selection and
                # reported paper metrics share one post-processing rule.
                if not getattr(self.args, 'no_fov', False):
                    main_mask = main_mask * fov_mask
                    masks = masks * fov_mask

                loss = self.criterion(main_mask, masks)
                val_loss += loss.item()
                n_batches += 1
                val_pbar.set_postfix({'loss': loss.item()})

                pred_bin = main_mask[0, 0] > pred_threshold
                true_bin = masks[0, 0] > 0.5
                tp += int((pred_bin & true_bin).sum())
                fp += int((pred_bin & ~true_bin).sum())
                fn += int((~pred_bin & true_bin).sum())

                if not fast_val:
                    if baseline_style:
                        # Baseline parity: per-image metrics at the ORIGINAL
                        # GT resolution, macro-averaged below.  This is the
                        # same metric set as baseline_zoo.runtime.evaluate.
                        m = binary_metrics(
                            pred_bin.detach().cpu().numpy(),
                            true_bin.detach().cpu().numpy())
                        for k in BINARY_METRICS:
                            macro_acc[k].append(m[k])
                        empty_acc.append(m['empty'])
                        dataset = self._val_dataset_name(_img_name)
                        bucket = per_dataset_acc.setdefault(
                            dataset,
                            {key: [] for key in (*BINARY_METRICS, 'empty')})
                        for k, v in m.items():
                            bucket[k].append(v)
                    else:
                        # Rich-table path (legacy): full metric suite on the
                        # ORIGINAL-size main mask. The few-shot val sets are
                        # tiny, so the distance/topology metrics are cheap.
                        # 引擎级指标(秒/图)只在前 50 张验证图上算。
                        engine_subset = (rank == 0) and (val_idx < VAL_ENGINE_IMGS)
                        m = compute_metrics(
                            main_mask[0], masks[0],
                            threshold=pred_threshold,
                            compute_auc=self.args.val_auc and engine_subset,
                            compute_betti_matching=self.args.val_betti_matching and engine_subset,
                            compute_topograph=self.args.val_topograph and engine_subset,
                        )
                        val_cldice.append(m['cldice'])
                        # 保留 NaN 项(空预测时 HD/cD/β 无定义):累积层不动,
                        # 聚合层用 nanmean 跳过 NaN(见下文),列位保持稳定
                        for k, v in m.items():
                            metrics_acc.setdefault(k, []).append(v)
                        # 拓扑损失监控:全部注册损失在当前预测上的值。
                        # 每个 rank 只在其第 1 张验证图上跑(128 输入 + 引擎 64 分辨率)
                        # ——引擎级损失秒级/图;DDP 分片后 8 rank 合计监控 8 张,更稳。
                        if not val_has_monitored:
                            val_has_monitored = True
                            m128 = F.interpolate(main_mask, size=(128, 128), mode='bilinear', align_corners=False)
                            t128 = F.interpolate(masks, size=(128, 128), mode='bilinear', align_corners=False)
                            for k, v in topology_loss_monitor(m128, t128, resolution=64).items():
                                metrics_acc.setdefault(f'topo_{k}', []).append(v)
                if len(vis_items) < 3:
                    vis_img = F.interpolate(
                        images[0:1], size=masks.shape[-2:],
                        mode='bilinear', align_corners=False)[0]
                    vis_items.append((vis_img.detach().cpu(),
                                      masks[0].detach().cpu(),
                                      main_mask[0].detach().cpu()))

        if self.is_ddp:
            # fencing-algs 模式:验证在全部 rank 分片执行,指标在 rank 0
            # 汇总。F1 用混淆计数跨卡汇总(与全量拼接后 sklearn F1 逐位
            # 等价);其余指标把每 rank 的累积字典/列表 gather 到 rank 0
            # 后 nanmean——任何 rank 都不再长时间阻塞于 NCCL。
            # 验证集小于卡数时(如 3-shot 协议层)部分 rank 分片为空:
            # 空分片贡献 0 计数,不参与 F1,其余指标列表同样为空。
            payload = {
                'loss_sum': float(val_loss), 'count': n_batches,
                'tp': tp, 'fp': fp, 'fn': fn,
                'cldice': val_cldice, 'metrics': metrics_acc,
                'macro': macro_acc, 'empty': empty_acc,
                'per_dataset': per_dataset_acc,
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
            if rank == 0:
                (avg_val_loss, f1, avg_val_cldice, val_metrics,
                 val_per_dataset) = _merge_val_payloads(gathered)
            else:
                avg_val_loss = f1 = avg_val_cldice = val_metrics = None
                val_per_dataset = None
            # Every DDP rank must advance ReduceLROnPlateau with the same
            # value, otherwise rank 0 lowers its LR while the other ranks
            # keep the old LR and the replicas diverge.
            loss_payload = [avg_val_loss]
            dist.broadcast_object_list(loss_payload, src=0)
            avg_val_loss = loss_payload[0]
            if self.args.scheduler == 'reduce':
                self.scheduler.step(avg_val_loss)
            if rank != 0:
                return None, None, None, None, None, None
        else:
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom > 0 else 0.0
            avg_val_loss = val_loss / max(n_batches, 1)
            with warnings.catch_warnings():
                # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
                warnings.simplefilter('ignore', RuntimeWarning)
                if any(macro_acc.values()):
                    # Baseline-style path: macro per-image metrics.
                    val_metrics = {k: float(np.nanmean(v))
                                   for k, v in macro_acc.items() if v}
                    if empty_acc:
                        val_metrics['empty_prediction_rate'] = float(
                            np.nanmean(empty_acc))
                    avg_val_cldice = val_metrics.get(
                        'cldice', float('nan'))
                else:
                    # Rich-table / legacy path.
                    avg_val_cldice = (
                        float(np.nanmean(val_cldice))
                        if val_cldice else float('nan'))
                    # nanmean 聚合:引擎指标子集(BM/TE/dAUC/cAh 只算前 50 张)里
                    # 单张 NaN(空 GT/空预测守卫)不再传染整列——此前 np.mean
                    # 导致 BM/TE 两列恒为 "—";全 NaN 时保持 NaN(显示 "—")
                    val_metrics = {k: float(np.nanmean(v))
                                   for k, v in metrics_acc.items() if v}
                val_per_dataset = {}
                for dataset, values in per_dataset_acc.items():
                    merged = {}
                    for k, v in values.items():
                        name = ('empty_prediction_rate'
                                if k == 'empty' else k)
                        merged[name] = float(np.nanmean(v))
                    val_per_dataset[dataset] = merged
            if self.args.scheduler == 'reduce':
                self.scheduler.step(avg_val_loss)
        # 逐 epoch 的 val 数值全部由 rich 宽表呈现,不再刷 INFO 日志

        self._log_val_images(epoch, vis_items)

        # Checkpoint selection.  With full validation, 'dice' selects the
        # macro per-image Dice used by every baseline; under fast_val it
        # falls back to the global micro F1.  'cldice' needs full validation.
        macro_dice = val_metrics.get('dice') if val_metrics else None
        if self.selection_metric == 'cldice':
            selection_score = avg_val_cldice
            if not np.isfinite(selection_score):
                selection_score = (
                    macro_dice if macro_dice is not None else f1)
        elif self.selection_metric == 'dice' and macro_dice is not None:
            selection_score = macro_dice
        else:
            selection_score = f1

        if selection_score > self.best_selection_score:
            self.best_selection_score = selection_score
            self.best_f1_score = f1
            if self.is_ddp:
                inner = self.model.module
                # ForwardWrapper stores the real Sam under .model; keep
                # checkpoints loadable by evaluate.py.
                inner = getattr(inner, 'model', inner)
                state_dict = inner.state_dict()
            else:
                state_dict = self.model.state_dict()
            tmp_path = self.args.save_path + '.tmp'
            torch.save({
                'artifact_version': ARTIFACT_VERSION,
                'epoch': epoch,
                'epoch_completed': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': (
                    self.scheduler.state_dict()
                    if self.scheduler is not None else None),
                'best_selection_score': selection_score,
                'best_epoch': epoch + 1,
                'best_f1_score': f1,
                'val_loss': avg_val_loss,
                'val_cldice': avg_val_cldice,
                'f1_score': f1,
                'data_manifest_sha256': self.artifact_info.get(
                    'data_manifest_sha256'),
                'full_config': self.train_config,
                'wandb': self.wandb_info,
                'resumed_from': self.resume_state.get('resumed_from'),
                'config': {
                    'model': self.args.model,
                    'use_geo_i': self.args.use_geo_i,
                    'use_geo_e': self.args.use_geo_e,
                    'geo_e_layers': self.args.geo_e_layers,
                    'use_coarse_to_fine': self.args.use_coarse_to_fine,
                    'use_fusion_v2': self.args.use_fusion_v2,
                    'use_multi_depth': self.args.use_multi_depth,
                    'use_typed_heads': bool(getattr(
                        self.args, 'use_typed_heads', False)),
                    'init_stage2_from_stage1': getattr(self.args, 'init_stage2_from_stage1', False),
                    'adapter_dim_ratio': self.args.adapter_dim_ratio,
                    'stage2_supervision': getattr(self.args, 'stage2_supervision', 'soft'),
                    'soft_select_mix': getattr(self.args, 'soft_select_mix', 0.5),
                    'soft_select_tau': getattr(self.args, 'soft_select_tau', 1.0),
                    'iou_loss_weight': self.args.iou_loss_weight,
                    'pred_threshold': getattr(self.args, 'pred_threshold', 0.0),
                    'val_selection': getattr(self.args, 'val_selection', 'auto'),
                    'val_fixed_head': int(getattr(
                        self.args, 'val_fixed_head', 2)),
                    'val_empty_guard': bool(getattr(
                        self.args, 'val_empty_guard', False)),
                    'quality_small_component_weight': float(getattr(
                        self.args, 'quality_small_component_weight', 0.5)),
                    'quality_component_weight': float(getattr(
                        self.args, 'quality_component_weight', 0.05)),
                    'topology_loss': getattr(
                        self.args, 'topology_loss', 'none'),
                    'topology_loss_weight': float(getattr(
                        self.args, 'topology_loss_weight', 0.1)),
                    'topology_resolution': int(getattr(
                        self.args, 'topology_resolution', 256)),
                    'selection_metric': self.selection_metric,
                    'seed': getattr(self.args, 'seed', 0),
                    'deterministic': bool(getattr(
                        self.args, 'deterministic', True)),
                    'num_workers': int(getattr(
                        self.args, 'num_workers', 4)),
                    'reproducibility_info': getattr(
                        self.args, 'reproducibility_info', {}),
                    'no_fov': getattr(self.args, 'no_fov', False),
                    'log_style': getattr(self.args, 'log_style', 'table'),
                    'val_interval': getattr(self.args, 'val_interval', 10),
                    'grad_checkpointing': getattr(
                        self.args, 'grad_checkpointing', True),
                    'deep_sup_weight': self.args.deep_sup_weight,
                    'cl_dice_weight': self.args.cl_dice_weight,
                    'scnp': bool(getattr(self.args, 'scnp', False)),
                    'scnp_window': getattr(self.args, 'scnp_window', 3),
                    'scnp_original_weight': getattr(
                        self.args, 'scnp_original_weight', 0.0),
                    'topology_loss': self.args.topology_loss,
                    'topology_loss_weight': self.args.topology_loss_weight,
                    'train_input_mode': getattr(
                        self.args, 'train_input_mode', 'resize'),
                    'eval_input_mode': getattr(
                        self.args, 'eval_input_mode', 'resize'),
                    'foreground_crop_prob': getattr(
                        self.args, 'foreground_crop_prob', 0.0),
                    'boundary_loss_weight': getattr(
                        self.args, 'boundary_loss_weight', 0.0),
                    'boundary_loss_resolution': getattr(
                        self.args, 'boundary_loss_resolution', 512),
                    'tversky_weight': getattr(
                        self.args, 'tversky_weight', 0.0),
                    'tversky_alpha': getattr(
                        self.args, 'tversky_alpha', 0.3),
                    'tversky_beta': getattr(
                        self.args, 'tversky_beta', 0.7),
                },
            }, tmp_path)
            os.replace(tmp_path, self.args.save_path)
            logging.info(
                f'👍New best model saved with {self.selection_metric}='
                f'{selection_score:.4f} (F1={f1:.4f})')

        return (f1, avg_val_loss, avg_val_cldice, val_metrics,
                val_per_dataset, selection_score)

    def _log_val_images(self, epoch, items):
        """验证图可视化(原图/GT/预测三列)上传 wandb;失败不影响训练。"""
        if getattr(self.args, 'log_style', 'table') in ('official', 'baseline'):
            return  # official/baseline modes: no extra W&B artifacts
        run = getattr(self, '_wandb_run', None)
        if run is None or not items:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            import wandb

            n = len(items)
            fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
            if n == 1:
                axes = axes[None, :]
            for row, (img, gt, pred) in enumerate(items):
                axes[row, 0].imshow(img[:, ::4, ::4].permute(1, 2, 0).numpy())
                axes[row, 0].set_title('image')
                axes[row, 1].imshow(gt[0].numpy(), cmap='gray')
                axes[row, 1].set_title('GT')
                axes[row, 2].imshow((pred > 0).float().numpy(), cmap='gray')
                axes[row, 2].set_title('pred')
                for ax in axes[row]:
                    ax.axis('off')
            plt.tight_layout()
            run.log({'val/vis': wandb.Image(fig)}, step=epoch + 1)
            plt.close(fig)
        except Exception:
            pass  # 可视化失败不影响训练

    def _run_config(self):
        """Config snapshot shared by wandb and the local metric history."""
        return {
            'model': self.args.model,
            'preset': self.args.preset,
            'batch_size': self.args.batch_size,
            'epochs': self.args.epochs,
            'adapter_lr': self.args.adapter_lr,
            'new_module_lr': self.args.new_module_lr,
            'decoder_lr': self.args.decoder_lr,
            'deep_sup_weight': self.args.deep_sup_weight,
            'stage2_supervision': getattr(self.args, 'stage2_supervision', 'soft'),
            'soft_select_mix': getattr(self.args, 'soft_select_mix', 0.5),
            'soft_select_tau': getattr(self.args, 'soft_select_tau', 1.0),
            'cl_dice_weight': self.args.cl_dice_weight,
            'scnp': bool(getattr(self.args, 'scnp', False)),
            'scnp_window': getattr(self.args, 'scnp_window', 3),
            'scnp_original_weight': getattr(
                self.args, 'scnp_original_weight', 0.0),
            'iou_loss_weight': self.args.iou_loss_weight,
            'pred_threshold': getattr(self.args, 'pred_threshold', 0.0),
            'val_selection': getattr(self.args, 'val_selection', 'auto'),
            'selection_metric': getattr(self.args, 'selection_metric', 'dice'),
            'no_fov': getattr(self.args, 'no_fov', False),
            'log_style': getattr(self.args, 'log_style', 'table'),
            'val_interval': getattr(self.args, 'val_interval', 10),
            'grad_checkpointing': getattr(
                self.args, 'grad_checkpointing', True),
            'topology_loss': self.args.topology_loss,
            'topology_loss_weight': self.args.topology_loss_weight,
            'train_input_mode': getattr(
                self.args, 'train_input_mode', 'resize'),
            'eval_input_mode': getattr(
                self.args, 'eval_input_mode', 'resize'),
            'foreground_crop_prob': getattr(
                self.args, 'foreground_crop_prob', 0.0),
            'boundary_loss_weight': getattr(
                self.args, 'boundary_loss_weight', 0.0),
            'boundary_loss_resolution': getattr(
                self.args, 'boundary_loss_resolution', 512),
            'tversky_weight': getattr(self.args, 'tversky_weight', 0.0),
            'tversky_alpha': getattr(self.args, 'tversky_alpha', 0.3),
            'tversky_beta': getattr(self.args, 'tversky_beta', 0.7),
            'use_typed_heads': bool(getattr(
                self.args, 'use_typed_heads', False)),
            'use_geo_i': self.args.use_geo_i,
            'use_geo_e': self.args.use_geo_e,
            'geo_e_layers': self.args.geo_e_layers,
            'use_coarse_to_fine': self.args.use_coarse_to_fine,
            'use_fusion_v2': self.args.use_fusion_v2,
            'use_multi_depth': self.args.use_multi_depth,
            'use_res_moe': bool(getattr(self.args, 'use_res_moe', False)),
            'res_num_experts': getattr(self.args, 'res_num_experts', 4),
            'res_hidden': getattr(self.args, 'res_hidden', 24),
            'res_gate_hidden': getattr(self.args, 'res_gate_hidden', 32),
            'res_gate_size': getattr(self.args, 'res_gate_size', 512),
            'res_gate_mode': getattr(self.args, 'res_gate_mode', 'learned'),
            'res_use_head_disagreement': bool(getattr(
                self.args, 'res_use_head_disagreement', True)),
            'res_use_centerline': bool(getattr(
                self.args, 'res_use_centerline', True)),
            'res_gate_input_norm': bool(getattr(
                self.args, 'res_gate_input_norm', True)),
            'res_out_size': getattr(self.args, 'res_out_size', 512),
            'res_head_index': getattr(self.args, 'res_head_index', 2),
            'res_dropout': getattr(self.args, 'res_dropout', 0.0),
            'res_disagreement_tau': getattr(
                self.args, 'res_disagreement_tau', 0.05),
            'res_disagreement_scale': getattr(
                self.args, 'res_disagreement_scale', 0.10),
            'res_loss_weight': getattr(self.args, 'res_loss_weight', 0.5),
            'res_topology_weight': getattr(
                self.args, 'res_topology_weight', 0.0),
            'res_gate_loss_weight': getattr(
                self.args, 'res_gate_loss_weight', 0.3),
            'res_sparsity_weight': getattr(
                self.args, 'res_sparsity_weight', 0.02),
            'res_warmup_epochs': getattr(
                self.args, 'res_warmup_epochs', 30),
            'res_ramp_epochs': getattr(self.args, 'res_ramp_epochs', 30),
            'init_stage2_from_stage1': getattr(self.args, 'init_stage2_from_stage1', False),
            'val_auc': self.args.val_auc,
            'val_betti_matching': self.args.val_betti_matching,
            'val_topograph': self.args.val_topograph,
            'fast_val': getattr(self.args, 'fast_val', False),
            'seed': self.args.seed,
            'deterministic': bool(getattr(
                self.args, 'deterministic', True)),
            'deterministic_strict': bool(getattr(
                self.args, 'deterministic_strict', True)),
            'num_workers': int(getattr(self.args, 'num_workers', 4)),
            'reproducibility_info': getattr(
                self.args, 'reproducibility_info', {}),
        }

    def _checkpoint_config(self):
        """Flat architecture keys plus full args/provenance metadata."""
        config = self._run_config()
        config.update({
            'artifact_version': ARTIFACT_VERSION,
            'adapter_dim_ratio': getattr(
                self.args, 'adapter_dim_ratio', None),
            'data_root': str(getattr(self.args, 'data_root', '')),
            'data_manifest_sha256': self.artifact_info.get(
                'data_manifest_sha256'),
            'base_checkpoint': self.artifact_info.get('base_checkpoint'),
        })
        return config

    def _state_dict(self):
        if self.is_ddp:
            inner = self.model.module
            inner = getattr(inner, 'model', inner)
            return inner.state_dict()
        return self.model.state_dict()

    def _checkpoint_payload(self, history, best_epoch, epoch_completed):
        return {
            'artifact_version': ARTIFACT_VERSION,
            'epoch': int(epoch_completed) - 1,
            'epoch_completed': int(epoch_completed),
            'model_state_dict': self._state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict()
                if self.scheduler is not None else None),
            'best_selection_score': self.best_selection_score,
            'best_f1_score': self.best_f1_score,
            'best_epoch': int(best_epoch),
            'selection_metric': self.selection_metric,
            'config': self._checkpoint_config(),
            'full_config': self.train_config,
            'history': list(history),
            'rng_state': capture_rng_state(),
            'train_generator_state': capture_generator_state(
                self.train_generator),
            'val_generator_state': capture_generator_state(
                self.val_generator),
            'data_manifest_sha256': self.artifact_info.get(
                'data_manifest_sha256'),
            'wandb': self.wandb_info,
            'resumed_from': self.resume_state.get('resumed_from'),
        }

    @staticmethod
    def _torch_save_atomic(payload, path):
        path = str(path)
        tmp_path = path + '.tmp'
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _save_last_checkpoint(self, history, best_epoch, epoch_completed,
                              force=False):
        every = int(getattr(self.args, 'save_last_every', 1) or 0)
        if every <= 0:
            return
        if not force and epoch_completed % every != 0:
            return
        path = os.path.join(self.save_dir, '.resume_state.pth')
        try:
            self._torch_save_atomic(
                self._checkpoint_payload(
                    history, best_epoch, epoch_completed),
                path)
            self.last_checkpoint = os.path.abspath(path)
        except (OSError, RuntimeError) as exc:
            logging.warning('last checkpoint save failed: %s', exc)

    def _remove_temporary_checkpoints(self):
        """Delete crash-resume/periodic snapshots after successful training."""
        last_path = os.path.join(self.save_dir, '.resume_state.pth')
        try:
            if os.path.lexists(last_path):
                os.remove(last_path)
        except OSError as exc:
            logging.warning('last checkpoint cleanup failed: %s', exc)
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        if os.path.isdir(checkpoint_dir):
            try:
                shutil.rmtree(checkpoint_dir)
            except OSError as exc:
                logging.warning('periodic checkpoint cleanup failed: %s', exc)
        try:
            for name in os.listdir(self.save_dir):
                if 'last' in name.lower() and name.endswith('.pth'):
                    os.remove(os.path.join(self.save_dir, name))
        except OSError as exc:
            logging.warning('legacy checkpoint cleanup failed: %s', exc)
        self.last_checkpoint = None
        self.periodic_checkpoints = []

    def _prune_periodic(self):
        keep = max(0, int(getattr(self.args, 'keep_checkpoints', 2)))
        directory = os.path.join(self.save_dir, 'checkpoints')
        if not os.path.isdir(directory):
            return
        files = sorted(
            (name for name in os.listdir(directory)
             if name.startswith('epoch_') and name.endswith('.pth')),
            key=lambda name: int(name[6:-4]) if name[6:-4].isdigit() else -1)
        for name in files[:-keep] if keep else files:
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass
        self.periodic_checkpoints = [
            os.path.abspath(os.path.join(directory, name))
            for name in files[-keep:] if keep
        ]

    def _save_periodic_checkpoint(self, history, best_epoch,
                                  epoch_completed):
        every = int(getattr(self.args, 'save_every', 0) or 0)
        if every <= 0 or epoch_completed % every != 0:
            return
        directory = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f'epoch_{int(epoch_completed):04d}.pth')
        try:
            self._torch_save_atomic(
                self._checkpoint_payload(
                    history, best_epoch, epoch_completed),
                path)
            self._prune_periodic()
        except (OSError, RuntimeError) as exc:
            logging.warning('periodic checkpoint save failed: %s', exc)

    def _write_history(self, history, best_epoch, status='running',
                       last_epoch=None, error=None):
        """Persist the metric history and run metadata atomically.

        Called after every epoch, not only at exit, so an interrupted run
        still has a usable training curve and resume metadata.
        """
        out_dir = self.save_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'metrics_history.json')
        if last_epoch is None:
            last_epoch = history[-1].get('epoch') if history else None
        best_selection = (None if not np.isfinite(
            self.best_selection_score) else self.best_selection_score)
        payload = {
            'artifact_version': ARTIFACT_VERSION,
            'status': status,
            'config': self._run_config(),
            'resolved_config': self.train_config.get('args'),
            'provenance': {
                key: self.artifact_info.get(key)
                for key in (
                    'train_config', 'data_manifest', 'provenance_dir',
                    'data_manifest_sha256', 'git', 'environment',
                    'base_checkpoint',
                )
            },
            'best_epoch': best_epoch,
            'last_epoch': last_epoch,
            'best_f1': self.best_f1_score,
            'best_selection': best_selection,
            'selection_metric': self.selection_metric,
            'history': history,
            'checkpoints': {
                'best': os.path.abspath(self.args.save_path),
                'last': self.last_checkpoint,
                'periodic': list(self.periodic_checkpoints),
            },
            'wandb': self.wandb_info,
            'resumed_from': self.resume_state.get('resumed_from'),
            'error': error,
        }
        try:
            write_json_atomic(path, payload)
        except (OSError, ValueError) as exc:
            logging.warning('metric history save failed: %s', exc)

        info_payload = {
            'artifact_version': ARTIFACT_VERSION,
            'status': status,
            'checkpoint': os.path.abspath(self.args.save_path),
            'best_checkpoint': os.path.abspath(self.args.save_path),
            'last_checkpoint': self.last_checkpoint,
            'checkpoint_dir': os.path.abspath(out_dir),
            'best_epoch': best_epoch,
            'last_epoch': last_epoch,
            'best_selection': best_selection,
            'selection_metric': self.selection_metric,
            'epochs': int(self.args.epochs),
            'preset': getattr(self.args, 'preset', None),
            'seed': int(getattr(self.args, 'seed', 0)),
            'data_root': str(getattr(self.args, 'data_root', '')),
            'reproducibility_info': getattr(
                self.args, 'reproducibility_info', {}),
            'resumed_from': self.resume_state.get('resumed_from'),
            'wandb': self.wandb_info,
            'artifacts': {
                key: self.artifact_info.get(key)
                for key in (
                    'train_config', 'data_manifest', 'provenance_dir',
                    'data_manifest_sha256', 'base_checkpoint',
                )
            },
            'periodic_checkpoints': list(self.periodic_checkpoints),
            'checkpoints': {
                'best': os.path.abspath(self.args.save_path),
                'last': self.last_checkpoint,
                'periodic': list(self.periodic_checkpoints),
            },
            'error': error,
        }
        info_path = os.path.join(out_dir, 'run_info.json')
        try:
            write_json_atomic(info_path, info_payload)
        except (OSError, ValueError) as exc:
            logging.warning('run info save failed: %s', exc)

    def fit(self):
        """Run the full training schedule (wandb-monitored when enabled)."""
        # fencing-algs 模式:rank0 专属操作(wandb init)前先 barrier,
        # 保证所有 rank 从同一起点出发
        if self.is_ddp:
            import torch.distributed as dist
            dist.barrier()
        run = None
        if self._rank0:
            run = setup_wandb(
                self.args,
                name=f"{self.args.model}_{self.args.preset}",
                config=self._run_config(),
            )
            try:
                self.wandb_info = write_wandb_info(self.save_dir, run)
            except OSError as exc:
                logging.warning('wandb metadata save failed: %s', exc)
        self._wandb_run = run
        history = list(self.resume_state.get('history') or [])
        advice_log = {}
        best_epoch = int(self.resume_state.get('best_epoch', -1))
        start_epoch = int(self.resume_state.get('start_epoch', 0))
        log_style = getattr(self.args, 'log_style', 'table')
        official = log_style == 'official'
        baseline_style = log_style == 'baseline'
        # 宽表:紧凑单行(49 列,~290 字符);表头下打一次拓扑损失缩写对照
        topo_names = sorted(TOPOLOGY_LOSS_REGISTRY)
        if not official and not baseline_style:
            for line in epoch_header(topo_names):
                console.print(line, style="bold")
            console.print(topo_legend(topo_names), style="dim")
        if self._rank0:
            self._write_history(
                history, best_epoch, status='running',
                last_epoch=(history[-1].get('epoch') if history else None))

        run_status = 'running'
        failure = None
        try:
            for epoch in range(start_epoch, self.args.epochs):
                t0 = time.time()
                avg_train_loss, comps = self._train_epoch(epoch)
                train_s = time.time() - t0

                if official:
                    # Official SACM's train loss is the main DiceBCE term;
                    # report the same term so the log is comparable instead of
                    # hiding auxiliary deep-supervision/topology terms in it.
                    train_loss = comps.get('loss_main', avg_train_loss)
                    wm = {'epoch': epoch + 1,
                          'train/loss': train_loss,
                          'train/lr': self.optimizer.param_groups[0]['lr']}
                    if self._rank0:
                        print(f'Epoch {epoch+1}/{self.args.epochs} - '
                              f'Average Train Loss: {train_loss:.4f}',
                              flush=True)
                elif baseline_style:
                    wm = {'epoch': epoch + 1,
                          'train/loss': avg_train_loss,
                          'lr': self.optimizer.param_groups[0]['lr']}
                    # Keep per-term loss components visible in baseline-style
                    # logs; otherwise SCNP/ResMoE tuning is blind.
                    for k, v in comps.items():
                        wm[f'train/{k}'] = v
                else:
                    wm = {'epoch': epoch, 'train/loss': avg_train_loss,
                          'lr': self.optimizer.param_groups[0]['lr'],
                          'train/time_s': train_s}
                    for k, v in comps.items():
                        wm[f'train/{k}'] = v

                # DDP 下验证在所有 rank 分片执行(_validate 内部 gather
                # 汇总),日志/保存只在 rank 0;其余 rank 只训练并同步梯度
                best_before = self.best_selection_score
                f1 = vloss = None
                val_metrics = None
                val_per_dataset = {}
                selection_score = None
                val_s = None
                is_best = False
                if (epoch + 1) % self.args.val_interval == 0:
                    val_t0 = time.time()
                    (f1, vloss, avg_val_cldice, val_metrics,
                     val_per_dataset, selection_score) = self._validate(epoch)
                    val_s = time.time() - val_t0
                    if self._rank0:
                        if official:
                            wm.update({'val/loss': vloss, 'val/f1': f1})
                        elif baseline_style:
                            # Same key set and per-dataset layout as the
                            # baseline runner (src/baseline_zoo/runtime.py).
                            wm.update({k: v for k, v in {
                                'val/dice': val_metrics.get('dice'),
                                'val/iou': val_metrics.get('iou'),
                                'val/cldice': val_metrics.get('cldice'),
                                'val/hd95': val_metrics.get('hd95'),
                                'val/empty_prediction_rate':
                                    val_metrics.get('empty_prediction_rate'),
                                'val/selection': selection_score,
                            }.items() if v is not None})
                            for dataset, values in (val_per_dataset or {}).items():
                                for metric in BINARY_METRICS:
                                    if metric in values:
                                        wm[f'val/{dataset}/{metric}'] = values[metric]
                        else:
                            wm.update({'val/loss': vloss, 'val/f1': f1,
                                       'val/cldice': avg_val_cldice,
                                       'val/time_s': val_s})
                            for k, v in val_metrics.items():
                                wm[f'val/{k}'] = v
                        is_best = self.best_selection_score > best_before
                        if official:
                            wm['val/best_f1'] = self.best_f1_score
                            wm['val/best_selection'] = self.best_selection_score
                            wm['val/selection_metric'] = self.selection_metric
                if is_best:
                    best_epoch = epoch + 1

                if self._rank0:
                    if official:
                        if f1 is not None:
                            print(f'Epoch {epoch+1}/{self.args.epochs} - '
                                  f'Average Val Loss: {vloss:.4f}', flush=True)
                            print(f'Epoch {epoch+1}/{self.args.epochs} - '
                                  f'F1 Score: {f1:.4f}', flush=True)
                            if is_best:
                                print(f'👍New best model saved with '
                                      f'{self.selection_metric}: '
                                      f'{self.best_selection_score:.4f} '
                                      f'(F1={self.best_f1_score:.4f})',
                                      flush=True)
                    elif baseline_style:
                        if f1 is not None and val_metrics is not None:
                            print(
                                f'Epoch {epoch+1}/{self.args.epochs} - '
                                f'train_loss: {avg_train_loss:.4f} - '
                                f'val_dice: {val_metrics.get("dice", float("nan")):.4f} - '
                                f'val_iou: {val_metrics.get("iou", float("nan")):.4f} - '
                                f'val_cldice: {val_metrics.get("cldice", float("nan")):.4f} - '
                                f'val_hd95: {val_metrics.get("hd95", float("nan")):.2f} - '
                                f'lr: {self.optimizer.param_groups[0]["lr"]:.7f}',
                                flush=True)
                            if is_best:
                                print(f'👍New best model saved with '
                                      f'{self.selection_metric}='
                                      f'{self.best_selection_score:.4f}',
                                      flush=True)
                    else:
                        for line in epoch_row(epoch + 1, avg_train_loss, comps,
                                              self.optimizer.param_groups[0]['lr'],
                                              time.time() - t0, f1, vloss, val_metrics,
                                              topo_names, train_s=train_s, val_s=val_s):
                            console.print(line, style="bold green" if is_best else "")
                        # Online advisor: metric patterns -> optimization
                        # directions, printed while training (history already
                        # includes this epoch).
                        if val_metrics is not None:
                            # 顾问去重:同一条提示至少间隔 3 个验证 epoch 才重复
                            for sev, msg in advise(history, self.args):
                                last = advice_log.get(msg, -99)
                                if epoch + 1 - last >= 3:
                                    console.print(f"  ⚠ {msg}",
                                                  style="yellow" if sev == 'warn' else "cyan")
                                    advice_log[msg] = epoch + 1

                    log_scalars(run, wm)

                    # **wm first so the explicit 1-based epoch wins (wm stores
                    # the 0-based wandb epoch)
                    history.append({**wm, 'epoch': epoch + 1,
                                    'time_s': round(time.time() - t0, 2),
                                    'train_s': round(train_s, 2),
                                    'val_s': round(val_s, 2) if val_s is not None else None})

                # 每轮末尾同步(fencing-algs 模式):rank 0 的日志/保存
                # 不拖慢其他 rank 进入下一轮,但也不让它们抢跑太远
                if self.is_ddp:
                    dist.barrier()

                # Step the scheduler (except ReduceLROnPlateau, updated during validation)
                if self.scheduler is not None and self.args.scheduler != 'reduce':
                    self.scheduler.step()
                if official and self._rank0:
                    print(f'Current learning rate: '
                          f'{self.optimizer.param_groups[0]["lr"]:.7f}',
                          flush=True)
                # Save after the scheduler step so a resumed run restores the
                # exact next-epoch LR; history is already appended above.
                if self._rank0:
                    self._save_last_checkpoint(
                        history, best_epoch, epoch + 1)
                    self._save_periodic_checkpoint(
                        history, best_epoch, epoch + 1)
                    self._write_history(
                        history, best_epoch, status='running',
                        last_epoch=epoch + 1)
        except BaseException as exc:
            run_status = 'failed'
            failure = exc
            if self._rank0:
                self._write_history(
                    history, best_epoch, status='failed',
                    error=f'{type(exc).__name__}: {exc}')
            raise
        else:
            run_status = 'done'
            if self._rank0:
                # A completed run exposes exactly one model artifact:
                # best_model.pth.  resume-state/periodic files exist only while
                # training is in progress so an interrupted run can resume.
                self._remove_temporary_checkpoints()
                self._write_history(
                    history, best_epoch, status='done',
                    last_epoch=(history[-1].get('epoch') if history else None))
        finally:
            if self._rank0:
                if run is not None:
                    try:
                        run.summary['best_f1'] = self.best_f1_score
                        run.summary['best_selection'] = (
                            None if not np.isfinite(self.best_selection_score)
                            else self.best_selection_score)
                        run.summary['selection_metric'] = self.selection_metric
                    except Exception:
                        pass
                finish_run(run)
            # fencing-algs 模式:训练结束统一收队,再各自退出
            if self.is_ddp:
                dist.barrier()
