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
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from core.console import (
    console,
    print_epoch_footer,
    print_epoch_header,
    print_epoch_row,
)
from core.evaluator import resolve_selection
from core.losses.monitor import topology_loss_monitor
from core.metrics import compute_metrics
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


def _dataset_name_from_image_name(image_name):
    """Derive the 3-shot domain name from a validation filename.

    The split names files as ``<DATASET>_<source...>`` (e.g.
    ``CHASEDB1_Image_04L.png``, ``CrackTree200_...``).  The training/val
    domains have no underscore in their names, so the prefix is the domain.
    """
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    return stem.split('_', 1)[0] if stem else 'unknown'


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

    gathered: list of dicts with keys loss_sum/count/tp/fp/fn/cldice/metrics/
    per_dataset/empty_sum/image_count (see _validate). F1 comes from the
    GLOBAL confusion counts (exact micro-F1, identical to sklearn on the
    concatenated shards); all other metrics are nanmean-aggregated so a NaN
    on one shard never poisons a column.
    """
    avg_val_loss = sum(g['loss_sum'] for g in gathered) / max(
        sum(g['count'] for g in gathered), 1)
    tp = sum(g['tp'] for g in gathered)
    fp = sum(g['fp'] for g in gathered)
    fn = sum(g['fn'] for g in gathered)
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom > 0 else 0.0
    # Empty-prediction rate: same definition as baseline_zoo.runtime.evaluate
    # (fraction of validation images whose prediction mask has no foreground).
    n_empty = sum(int(g.get('empty_sum', 0)) for g in gathered)
    n_images = sum(int(g.get('image_count', 0)) for g in gathered)
    empty_rate = n_empty / n_images if n_images > 0 else float('nan')
    acc = {}
    for g in gathered:
        for k, v in g['metrics'].items():
            acc.setdefault(k, []).extend(v)
    cldice_all = [x for g in gathered for x in g['cldice']]
    per_dataset = {}
    for g in gathered:
        for dataset, values in (g.get('per_dataset') or {}).items():
            bucket = per_dataset.setdefault(dataset, {})
            for key, vals in values.items():
                bucket.setdefault(key, []).extend(vals)
    with warnings.catch_warnings():
        # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
        warnings.simplefilter('ignore', RuntimeWarning)
        avg_val_cldice = float(np.nanmean(cldice_all)) if cldice_all else float('nan')
        val_metrics = {k: float(np.nanmean(v)) for k, v in acc.items() if v}
        val_per_dataset = {
            dataset: {k: float(np.nanmean(v)) for k, v in values.items() if v}
            for dataset, values in per_dataset.items()
        }
    return (avg_val_loss, f1, avg_val_cldice, val_metrics, val_per_dataset,
            empty_rate)


class Trainer:
    """Unified training engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, soft_cl, topo_loss,
                 optimizer, scheduler, train_loader, val_loader, args,
                 is_ddp=False, train_sampler=None, skeleton_recall=None):
        self.model = model
        self.spec = spec
        self.device = device
        self.criterion = criterion
        self.soft_cl = soft_cl
        self.topo_loss = topo_loss
        self.skeleton_recall = skeleton_recall
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.is_ddp = is_ddp
        self.train_sampler = train_sampler
        self.best_f1_score = 0.0
        if getattr(args, 'stage2_supervision', 'soft') == 'wta' and self._rank0:
            logging.warning(
                "D8: stage2_supervision='wta' — only the gating champion "
                "Stage-2 head receives mask loss. Use 'soft' (default) or "
                "'allhead' unless running the D8 baseline/ablation.")

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

    def _train_epoch(self, epoch):
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)  # DDP: 每 epoch 换采样顺序
        train_loss = 0.0
        comps_sum = {k: 0.0 for k in ('loss_main', 'loss_ds', 'loss_iou',
                                     'loss_cl', 'loss_topo', 'loss_srec')}
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
            else:
                raise ValueError(
                    f"Unknown stage2_supervision={stage2_mode!r}; "
                    "expected one of wta/allhead/soft")

            # Stage-1 deep supervision: ALL heads vs GT. This is what makes
            # the gating scores meaningful in the first place.
            loss_ds = 0.0
            if self.args.deep_sup_weight > 0:
                for k in range(m1.shape[1]):
                    loss_ds = loss_ds + self.criterion(m1[:, k:k + 1], masks)
                loss_ds = loss_ds / m1.shape[1]

            # IoU head supervision: ground-truth IoU per head (hard, detached)
            loss_iou = 0.0
            if self.args.iou_loss_weight > 0:
                with torch.no_grad():
                    m2_bin = (m2 > 0).float()
                    gt_bin = (masks > 0.5).float()
                    inter = (m2_bin * gt_bin).sum(dim=(2, 3))
                    union = m2_bin.sum(dim=(2, 3)) + gt_bin.sum(dim=(2, 3)) - inter
                    iou_gt = inter / (union + 1e-6)
                loss_iou = F.mse_loss(iou_pred, iou_gt)

            # Differentiable clDice on the selected mask, with warmup ramp
            loss_cl = 0.0
            mu = self.args.cl_dice_weight * cl_dice_weight_schedule(
                epoch, self.args.cl_dice_warmup, self.args.cl_dice_ramp
            )
            if mu > 0:
                loss_cl = self.soft_cl(aux_mask, masks)

            # Skeleton Recall (ECCV 2024): soft recall over the tubed GT
            # skeleton, which is not differentiated. This directly penalizes
            # missed thin branches that Dice/BCE tolerate as tiny areas.
            loss_srec = 0.0
            mu_srec = 0.0
            if self.skeleton_recall is not None:
                mu_srec = self.args.skeleton_recall_weight * \
                    cl_dice_weight_schedule(
                        epoch, self.args.skeleton_recall_warmup,
                        self.args.skeleton_recall_ramp)
                if mu_srec > 0:
                    loss_srec = self.skeleton_recall(aux_mask, masks)

            # Extra topology loss (optional; strict ports)
            loss_topo = 0.0
            if self.topo_loss is not None:
                loss_topo = self.topo_loss(aux_mask, masks)

            loss = (
                loss_main
                + self.args.deep_sup_weight * loss_ds
                + self.args.iou_loss_weight * loss_iou
                + mu * loss_cl
                + mu_srec * loss_srec
                + self.args.topology_loss_weight * loss_topo
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
            train_pbar.set_postfix({'loss': loss.item(), 'cl': mu,
                                    'srec': round(mu_srec, 3)})

        n = len(self.train_loader)
        comps = {k: v / n for k, v in comps_sum.items()}
        return train_loss / n, comps

    def _validate(self, epoch):
        self.model.eval()
        # 验证期在训练缓存之上叠加前向/指标,先清碎片避免 OOM 峰
        # (epoch 5 实测出现过 2.5GB 分配失败)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        fast_val = bool(getattr(self.args, 'fast_val', False))
        val_loss = 0.0
        n_batches = 0
        tp = fp = fn = 0
        n_empty = 0
        n_images = 0
        val_cldice = []
        metrics_acc = {}
        per_dataset_acc = {}
        val_has_monitored = False
        vis_items = []  # (image, gt, pred) 供 wandb 可视化
        VAL_ENGINE_IMGS = 50  # 引擎指标子集大小(均值估计,验证提速关键)
        val_selection = getattr(self.args, 'val_selection', 'auto')
        val_selection = resolve_selection(
            val_selection, getattr(self.args, 'iou_loss_weight', 0.0))
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
            for val_idx, (images, masks, original_size, img_name, fov_mask) in enumerate(val_pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                fov_mask = fov_mask.to(self.device)
                image_name = (img_name[0]
                              if isinstance(img_name, (list, tuple))
                              else img_name)
                dataset_name = _dataset_name_from_image_name(image_name)

                masks_stage2, iou_pred, _masks_stage1 = self._forward(images)

                # Select at decoder resolution, then restore the ORIGINAL GT
                # size before loss/metrics — the same path as test evaluation.
                if val_selection == 'iou':
                    idx = iou_pred.argmax(dim=1)
                    gather = idx[:, None, None, None].expand(
                        -1, 1, masks_stage2.shape[2], masks_stage2.shape[3])
                    main_mask = torch.gather(masks_stage2, 1, gather)
                else:
                    main_mask = masks_stage2[:, 0:1]

                width, height = original_size
                width = width.item() if torch.is_tensor(width) else int(width)
                height = height.item() if torch.is_tensor(height) else int(height)
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
                n_images += 1
                n_empty += int(pred_bin.sum().item() == 0)

                if not fast_val:
                    # Full metric suite on the ORIGINAL-size main mask. The
                    # few-shot val sets are tiny (6 images in the 3-shot
                    # protocol), so the distance/topology metrics are cheap.
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
                        per_dataset_acc.setdefault(
                            dataset_name, {}).setdefault(k, []).append(v)
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
                'per_dataset': per_dataset_acc,
                'empty_sum': n_empty, 'image_count': n_images,
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
            if rank == 0:
                (avg_val_loss, f1, avg_val_cldice, val_metrics,
                 val_per_dataset, empty_rate) = _merge_val_payloads(gathered)
            else:
                avg_val_loss = f1 = avg_val_cldice = None
                val_metrics = val_per_dataset = None
                empty_rate = None
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
            empty_rate = n_empty / n_images if n_images > 0 else float('nan')
            with warnings.catch_warnings():
                # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
                warnings.simplefilter('ignore', RuntimeWarning)
                avg_val_cldice = float(np.nanmean(val_cldice)) if val_cldice else float('nan')
                # nanmean 聚合:引擎指标子集(BM/TE/dAUC/cAh 只算前 50 张)里
                # 单张 NaN(空 GT/空预测守卫)不再传染整列——此前 np.mean
                # 导致 BM/TE 两列恒为 "—";全 NaN 时保持 NaN(显示 "—")
                val_metrics = {k: float(np.nanmean(v)) for k, v in metrics_acc.items() if v}
                val_per_dataset = {
                    dataset: {
                        k: float(np.nanmean(v))
                        for k, v in values.items() if v
                    }
                    for dataset, values in per_dataset_acc.items()
                }
            if self.args.scheduler == 'reduce':
                self.scheduler.step(avg_val_loss)
        # 逐 epoch 的 val 数值全部由 rich 宽表呈现,不再刷 INFO 日志

        self._log_val_images(epoch, vis_items)

        if f1 > self.best_f1_score:
            self.best_f1_score = f1
            if self.is_ddp:
                inner = self.model.module
                # ForwardWrapper stores the real Sam under .model; keep
                # checkpoints loadable by evaluate.py.
                inner = getattr(inner, 'model', inner)
                state_dict = inner.state_dict()
            else:
                state_dict = self.model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_cldice': avg_val_cldice,
                'val_per_dataset': val_per_dataset,
                'f1_score': f1,
                'config': {
                    'model': self.args.model,
                    'use_geo_i': self.args.use_geo_i,
                    'use_geo_e': self.args.use_geo_e,
                    'geo_e_layers': self.args.geo_e_layers,
                    'use_coarse_to_fine': self.args.use_coarse_to_fine,
                    'use_fusion_v2': self.args.use_fusion_v2,
                    'use_multi_depth': self.args.use_multi_depth,
                    'init_stage2_from_stage1': getattr(self.args, 'init_stage2_from_stage1', False),
                    'adapter_dim_ratio': self.args.adapter_dim_ratio,
                    'stage2_supervision': getattr(self.args, 'stage2_supervision', 'soft'),
                    'soft_select_mix': getattr(self.args, 'soft_select_mix', 0.5),
                    'soft_select_tau': getattr(self.args, 'soft_select_tau', 1.0),
                    'iou_loss_weight': self.args.iou_loss_weight,
                    'pred_threshold': getattr(self.args, 'pred_threshold', 0.0),
                    'val_selection': getattr(self.args, 'val_selection', 'auto'),
                    'no_fov': getattr(self.args, 'no_fov', False),
                    'log_style': getattr(self.args, 'log_style', 'table'),
                    'deep_sup_weight': self.args.deep_sup_weight,
                    'cl_dice_weight': self.args.cl_dice_weight,
                    'scnp': bool(getattr(self.args, 'scnp', False)),
                    'scnp_window': getattr(self.args, 'scnp_window', 3),
                    'scnp_original_weight': getattr(
                        self.args, 'scnp_original_weight', 0.0),
                    'topology_loss': self.args.topology_loss,
                    'topology_loss_weight': self.args.topology_loss_weight,
                },
            }, self.args.save_path)
            logging.info(f'👍New best model saved with F1 score: {self.best_f1_score:.4f}')

        return (f1, avg_val_loss, avg_val_cldice, val_metrics,
                val_per_dataset, empty_rate)

    def _log_val_images(self, epoch, items):
        """验证图可视化(原图/GT/预测三列)上传 wandb;失败不影响训练。"""
        if getattr(self.args, 'log_style', 'table') == 'official':
            return  # official-comparison mode: no extra W&B artifacts
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
            'skeleton_recall_weight': getattr(
                self.args, 'skeleton_recall_weight', 0.0),
            'skeleton_recall_warmup': getattr(
                self.args, 'skeleton_recall_warmup', 10),
            'skeleton_recall_ramp': getattr(
                self.args, 'skeleton_recall_ramp', 5),
            'skeleton_recall_tube': getattr(
                self.args, 'skeleton_recall_tube', True),
            'scnp': bool(getattr(self.args, 'scnp', False)),
            'scnp_window': getattr(self.args, 'scnp_window', 3),
            'scnp_original_weight': getattr(
                self.args, 'scnp_original_weight', 0.0),
            'iou_loss_weight': self.args.iou_loss_weight,
            'pred_threshold': getattr(self.args, 'pred_threshold', 0.0),
            'val_selection': getattr(self.args, 'val_selection', 'auto'),
            'no_fov': getattr(self.args, 'no_fov', False),
            'log_style': getattr(self.args, 'log_style', 'table'),
            'topology_loss': self.args.topology_loss,
            'topology_loss_weight': self.args.topology_loss_weight,
            'use_geo_i': self.args.use_geo_i,
            'use_geo_e': self.args.use_geo_e,
            'geo_e_layers': self.args.geo_e_layers,
            'use_coarse_to_fine': self.args.use_coarse_to_fine,
            'use_fusion_v2': self.args.use_fusion_v2,
            'use_multi_depth': self.args.use_multi_depth,
            'init_stage2_from_stage1': getattr(self.args, 'init_stage2_from_stage1', False),
            'val_auc': self.args.val_auc,
            'val_betti_matching': self.args.val_betti_matching,
            'val_topograph': self.args.val_topograph,
            'fast_val': getattr(self.args, 'fast_val', False),
            'seed': self.args.seed,
        }

    def _write_history(self, history, best_epoch):
        """Persist the per-epoch metric history next to the checkpoint.

        This is the AI-feedback-loop artifact: every metric shown in the
        terminal / uploaded to wandb is also written here as structured
        JSON, so an analysis pass (human or AI) can read exactly which
        metric is weak and target the network change at it.
        """
        out_dir = os.path.dirname(self.args.save_path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'metrics_history.json')
        payload = {
            'config': self._run_config(),
            'best_epoch': best_epoch,
            'best_f1': self.best_f1_score,
            'history': history,
        }
        try:
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)
            logging.info(f"Metric history saved to {path}")
        except OSError as e:
            logging.warning(f"Metric history save failed: {e}")

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
                name=(f"sacm_x_{self.args.model}_{self.args.preset}"
                      f"_seed{self.args.seed}"),
                config=self._run_config(),
            )
        self._wandb_run = run
        history = []
        best_epoch = -1
        log_style = getattr(self.args, 'log_style', 'table')
        official = log_style == 'official'
        if not official:
            # Reuse the baseline's canonical compact table:
            # ep │ tr_loss │ Dice↑ │ clDice↑ │ IoU↑ │ HD95↓ │ lr │ time
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
            print_epoch_header(
                f'sacm_x/{self.args.model}/{self.args.preset}',
                self.args.epochs, self.args.seed, self.device,
                num_params, trainable)

        try:
            for epoch in range(self.args.epochs):
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
                else:
                    # 1-based epoch on the same axis as baseline_zoo.runtime.
                    wm = {'epoch': epoch + 1, 'train/loss': avg_train_loss,
                          'lr': self.optimizer.param_groups[0]['lr'],
                          'train/time_s': train_s}
                    for k, v in comps.items():
                        wm[f'train/{k}'] = v

                # DDP 下验证在所有 rank 分片执行(_validate 内部 gather
                # 汇总),日志/保存只在 rank 0;其余 rank 只训练并同步梯度
                best_before = self.best_f1_score
                f1 = vloss = None
                val_metrics = None
                val_per_dataset = None
                empty_rate = None
                val_s = None
                is_best = False
                if (epoch + 1) % self.args.val_interval == 0:
                    val_t0 = time.time()
                    (f1, vloss, avg_val_cldice, val_metrics, val_per_dataset,
                     empty_rate) = self._validate(epoch)
                    val_s = time.time() - val_t0
                    if self._rank0:
                        if official:
                            wm.update({'val/loss': vloss, 'val/f1': f1})
                            for k, v in (val_metrics or {}).items():
                                wm[f'val/{k}'] = v
                        else:
                            wm.update({'val/loss': vloss, 'val/f1': f1,
                                       'val/cldice': avg_val_cldice,
                                       'val/time_s': val_s})
                            for k, v in val_metrics.items():
                                wm[f'val/{k}'] = v
                        # Same per-domain keys as the baseline runtime:
                        # val/<dataset>/<metric> (Dice, clDice, ...).
                        for dataset, values in (val_per_dataset or {}).items():
                            for k, v in values.items():
                                wm[f'val/{dataset}/{k}'] = v
                        is_best = self.best_f1_score > best_before
                        if official:
                            wm['val/best_f1'] = self.best_f1_score
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
                                print(f'👍New best model saved with F1 score: '
                                      f'{self.best_f1_score:.4f}', flush=True)
                    else:
                        vm = val_metrics or {}
                        row_metrics = {
                            'dice': vm.get('dice', float('nan')),
                            'iou': vm.get('iou', float('nan')),
                            'cldice': vm.get('cldice', float('nan')),
                            'hd95': vm.get('hd95', float('nan')),
                        }
                        print_epoch_row(
                            epoch + 1, avg_train_loss, row_metrics,
                            self.optimizer.param_groups[0]['lr'],
                            time.time() - t0, is_best)

                    if official:
                        wb = wm
                    else:
                        # W&B validation-panel parity with baseline_zoo.runtime:
                        # log exactly the baseline keys (Dice/IoU/clDice/HD95,
                        # empty-prediction rate, selection score, and the same
                        # four per-domain metrics) so one dashboard works for
                        # SACM-X and every baseline.  metrics_history.json
                        # below still keeps the full metric suite.
                        wb = {k: v for k, v in wm.items()
                              if not k.startswith('val/')}
                        if f1 is not None:
                            # SACM-X checkpoint selection is F1 on the global
                            # confusion counts, so val/selection = F1.
                            wb['val/selection'] = f1
                            wb['val/empty_prediction_rate'] = empty_rate
                            for key in ('dice', 'iou', 'cldice', 'hd95'):
                                wb[f'val/{key}'] = (val_metrics or {}).get(
                                    key, float('nan'))
                            for dataset, values in (
                                    val_per_dataset or {}).items():
                                for key in ('dice', 'iou', 'cldice', 'hd95'):
                                    if key in values:
                                        wb[f'val/{dataset}/{key}'] = values[key]
                    log_scalars(run, wb)

                    history.append({**wm, 'epoch': epoch + 1,
                                    'val_per_dataset': val_per_dataset,
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
        finally:
            if self._rank0:
                if run is not None:
                    try:
                        run.summary['best_f1'] = self.best_f1_score
                    except Exception:
                        pass
                finish_run(run)
                self._write_history(history, best_epoch)
            # fencing-algs 模式:训练结束统一收队,再各自退出
            if self.is_ddp:
                dist.barrier()
        if self._rank0 and not official:
            print_epoch_footer()
            if best_epoch > 0:
                console.print(
                    f' [bold]Best[/]  E{best_epoch:03d}  '
                    f'F1={self.best_f1_score:.4f}')
