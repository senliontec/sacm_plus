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
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from core.console import console, epoch_header, epoch_row, topo_legend
from core.losses.monitor import topology_loss_monitor
from core.losses.registry import TOPOLOGY_LOSS_REGISTRY
from core.metrics import compute_metrics
from core.metrics_advisor import advise
from core.wandb_utils import finish_run, log_scalars, setup_wandb


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


def _merge_val_payloads(gathered):
    """Merge per-rank validation payloads into global metrics (rank 0).

    gathered: list of dicts with keys loss_sum/count/tp/fp/fn/cldice/metrics
    (see _validate). F1 comes from the GLOBAL confusion counts (exact micro-F1,
    identical to sklearn on the concatenated shards); all other metrics are
    nanmean-aggregated so a NaN on one shard never poisons a column.
    """
    avg_val_loss = sum(g['loss_sum'] for g in gathered) / max(
        sum(g['count'] for g in gathered), 1)
    tp = sum(g['tp'] for g in gathered)
    fp = sum(g['fp'] for g in gathered)
    fn = sum(g['fn'] for g in gathered)
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom > 0 else 0.0
    acc = {}
    for g in gathered:
        for k, v in g['metrics'].items():
            acc.setdefault(k, []).extend(v)
    cldice_all = [x for g in gathered for x in g['cldice']]
    with warnings.catch_warnings():
        # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
        warnings.simplefilter('ignore', RuntimeWarning)
        avg_val_cldice = float(np.nanmean(cldice_all)) if cldice_all else float('nan')
        val_metrics = {k: float(np.nanmean(v)) for k, v in acc.items() if v}
    return avg_val_loss, f1, avg_val_cldice, val_metrics


class Trainer:
    """Unified training engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, soft_cl, topo_loss,
                 optimizer, scheduler, train_loader, val_loader, args,
                 is_ddp=False, train_sampler=None):
        self.model = model
        self.spec = spec
        self.device = device
        self.criterion = criterion
        self.soft_cl = soft_cl
        self.topo_loss = topo_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.is_ddp = is_ddp
        self.train_sampler = train_sampler
        self.best_f1_score = 0.0

    @property
    def _rank0(self):
        """DDP 下日志/保存只在 rank 0(验证已分片到全部 rank)。"""
        if not self.is_ddp:
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0

    def _forward(self, images):
        # DDP 包装器不转发 Sam 的方法/子模块,spec 函数必须拿到原始模型
        model = self.model.module if self.is_ddp else self.model
        images = self.spec.preprocess(model, images)
        sparse, dense = self.spec.prompt_builder(model, images.shape[0])
        return self.spec.forward(model, images, sparse, dense, return_stage1=True)

    def _train_epoch(self, epoch):
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)  # DDP: 每 epoch 换采样顺序
        train_loss = 0.0
        comps_sum = {k: 0.0 for k in ('loss_main', 'loss_ds', 'loss_iou', 'loss_cl', 'loss_topo')}
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

            # Main mask: the gating winner (index 0 of the reordered heads).
            # Winner-take-all keeps the heads specialized; the IoU head is
            # trained separately and performs selection at inference.
            loss_main = self.criterion(m2[:, 0:1], masks)

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
                loss_cl = self.soft_cl(m2[:, 0:1], masks)

            # Extra topology loss (optional; strict ports)
            loss_topo = 0.0
            if self.topo_loss is not None:
                loss_topo = self.topo_loss(m2[:, 0:1], masks)

            loss = (
                loss_main
                + self.args.deep_sup_weight * loss_ds
                + self.args.iou_loss_weight * loss_iou
                + mu * loss_cl
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
            train_pbar.set_postfix({'loss': loss.item(), 'cl': mu})

        n = len(self.train_loader)
        comps = {k: v / n for k, v in comps_sum.items()}
        return train_loss / n, comps

    def _validate(self, epoch):
        self.model.eval()
        # 验证期在训练缓存之上叠加前向/指标,先清碎片避免 OOM 峰
        # (epoch 5 实测出现过 2.5GB 分配失败)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        val_loss = 0.0
        n_batches = 0
        all_pred_masks = []
        all_true_masks = []
        val_cldice = []
        metrics_acc = {}
        val_has_monitored = False
        vis_items = []  # (image, gt, pred) 供 wandb 可视化
        VAL_ENGINE_IMGS = 50  # 引擎指标子集大小(均值估计,验证提速关键)

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
            for val_idx, (images, masks) in enumerate(val_pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)

                masks_stage2, iou_pred, masks_stage1 = self._forward(images)

                masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                m2 = F.interpolate(masks_stage2, size=(1024, 1024), mode='bilinear', align_corners=False)
                main_mask = m2[:, 0:1]

                loss = self.criterion(main_mask, masks)
                val_loss += loss.item()
                n_batches += 1
                val_pbar.set_postfix({'loss': loss.item()})

                all_pred_masks.append(main_mask)
                all_true_masks.append(masks)

                # Full metric suite on the binarized main mask. The
                # few-shot val sets are tiny (6 images in the 3-shot
                # protocol), so the distance/topology metrics are cheap.
                # The 4 optional extras (auc / persistence engines) are
                # opt-in via CLI flags — they cost seconds per image.
                # 引擎级指标(秒/图)只在前 50 张验证图上算:全量 600 张会
                # 让验证 epoch 膨胀到 ~1 小时(实测 3271-3766s)。
                # DDP 分片后仍由 rank 0 在其本地前 50 张上算(总预算不变)。
                engine_subset = (rank == 0) and (val_idx < VAL_ENGINE_IMGS)
                m = compute_metrics(
                    main_mask[0], masks[0],
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
                    vis_items.append((images[0].detach().cpu(),
                                      masks[0].detach().cpu(),
                                      main_mask[0].detach().cpu()))

        if self.is_ddp:
            # fencing-algs 模式:验证在全部 rank 分片执行,指标在 rank 0
            # 汇总。F1 用混淆计数跨卡汇总(与全量拼接后 sklearn F1 逐位
            # 等价);其余指标把每 rank 的累积字典/列表 gather 到 rank 0
            # 后 nanmean——任何 rank 都不再长时间阻塞于 NCCL。
            pred_cat = torch.cat(all_pred_masks, dim=0) > 0
            true_cat = torch.cat(all_true_masks, dim=0) > 0.5
            payload = {
                'loss_sum': float(val_loss), 'count': n_batches,
                'tp': int((pred_cat & true_cat).sum()),
                'fp': int((pred_cat & ~true_cat).sum()),
                'fn': int((~pred_cat & true_cat).sum()),
                'cldice': val_cldice, 'metrics': metrics_acc,
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
            if rank != 0:
                return None, None, None, None
            avg_val_loss, f1, avg_val_cldice, val_metrics = _merge_val_payloads(gathered)
        else:
            all_pred_masks = torch.cat(all_pred_masks, dim=0)
            all_true_masks = torch.cat(all_true_masks, dim=0)
            f1 = calculate_f1_score(all_pred_masks, all_true_masks)
            avg_val_loss = val_loss / max(n_batches, 1)
            with warnings.catch_warnings():
                # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
                warnings.simplefilter('ignore', RuntimeWarning)
                avg_val_cldice = float(np.nanmean(val_cldice)) if val_cldice else float('nan')
                # nanmean 聚合:引擎指标子集(BM/TE/dAUC/cAh 只算前 50 张)里
                # 单张 NaN(空 GT/空预测守卫)不再传染整列——此前 np.mean
                # 导致 BM/TE 两列恒为 "—";全 NaN 时保持 NaN(显示 "—")
                val_metrics = {k: float(np.nanmean(v)) for k, v in metrics_acc.items() if v}
        # 逐 epoch 的 val 数值全部由 rich 宽表呈现,不再刷 INFO 日志

        if self.args.scheduler == 'reduce':
            self.scheduler.step(avg_val_loss)

        self._log_val_images(epoch, vis_items)

        if f1 > self.best_f1_score:
            self.best_f1_score = f1
            state_dict = (self.model.module.state_dict() if self.is_ddp
                          else self.model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_cldice': avg_val_cldice,
                'f1_score': f1,
                'config': {
                    'model': self.args.model,
                    'use_geo_i': self.args.use_geo_i,
                    'use_geo_e': self.args.use_geo_e,
                    'geo_e_layers': self.args.geo_e_layers,
                    'use_coarse_to_fine': self.args.use_coarse_to_fine,
                    'use_fusion_v2': self.args.use_fusion_v2,
                    'use_multi_depth': self.args.use_multi_depth,
                    'adapter_dim_ratio': self.args.adapter_dim_ratio,
                },
            }, self.args.save_path)
            logging.info(f'👍New best model saved with F1 score: {self.best_f1_score:.4f}')

        return f1, avg_val_loss, avg_val_cldice, val_metrics

    def _log_val_images(self, epoch, items):
        """验证图可视化(原图/GT/预测三列)上传 wandb;失败不影响训练。"""
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
            'cl_dice_weight': self.args.cl_dice_weight,
            'iou_loss_weight': self.args.iou_loss_weight,
            'topology_loss': self.args.topology_loss,
            'topology_loss_weight': self.args.topology_loss_weight,
            'use_geo_i': self.args.use_geo_i,
            'use_geo_e': self.args.use_geo_e,
            'geo_e_layers': self.args.geo_e_layers,
            'use_coarse_to_fine': self.args.use_coarse_to_fine,
            'use_fusion_v2': self.args.use_fusion_v2,
            'use_multi_depth': self.args.use_multi_depth,
            'val_auc': self.args.val_auc,
            'val_betti_matching': self.args.val_betti_matching,
            'val_topograph': self.args.val_topograph,
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
                name=f"{self.args.model}_{self.args.preset}",
                config=self._run_config(),
            )
        self._wandb_run = run
        history = []
        advice_log = {}
        best_epoch = -1
        # 宽表:紧凑单行(49 列,~290 字符);表头下打一次拓扑损失缩写对照
        topo_names = sorted(TOPOLOGY_LOSS_REGISTRY)
        for line in epoch_header(topo_names):
            console.print(line, style="bold")
        console.print(topo_legend(topo_names), style="dim")

        try:
            for epoch in range(self.args.epochs):
                t0 = time.time()
                avg_train_loss, comps = self._train_epoch(epoch)

                wm = {'epoch': epoch, 'train/loss': avg_train_loss,
                      'lr': self.optimizer.param_groups[0]['lr']}
                for k, v in comps.items():
                    wm[f'train/{k}'] = v

                # DDP 下验证在所有 rank 分片执行(_validate 内部 gather
                # 汇总),日志/保存只在 rank 0;其余 rank 只训练并同步梯度
                best_before = self.best_f1_score
                f1 = vloss = None
                val_metrics = None
                is_best = False
                if (epoch + 1) % self.args.val_interval == 0:
                    f1, vloss, avg_val_cldice, val_metrics = self._validate(epoch)
                    if self._rank0:
                        wm.update({'val/loss': vloss, 'val/f1': f1,
                                   'val/cldice': avg_val_cldice})
                        for k, v in val_metrics.items():
                            wm[f'val/{k}'] = v
                        is_best = self.best_f1_score > best_before
                if is_best:
                    best_epoch = epoch + 1

                if self._rank0:
                    log_scalars(run, wm)

                    # **wm first so the explicit 1-based epoch wins (wm stores
                    # the 0-based wandb epoch)
                    history.append({**wm, 'epoch': epoch + 1,
                                    'time_s': round(time.time() - t0, 2)})

                    for line in epoch_row(epoch + 1, avg_train_loss, comps,
                                          self.optimizer.param_groups[0]['lr'],
                                          time.time() - t0, f1, vloss, val_metrics,
                                          topo_names):
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

                # 每轮末尾同步(fencing-algs 模式):rank 0 的日志/保存
                # 不拖慢其他 rank 进入下一轮,但也不让它们抢跑太远
                if self.is_ddp:
                    dist.barrier()

                # Step the scheduler (except ReduceLROnPlateau, updated during validation)
                if self.scheduler is not None and self.args.scheduler != 'reduce':
                    self.scheduler.step()
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
