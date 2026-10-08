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
from sklearn.metrics import f1_score
from tqdm import tqdm

from core.console import console, epoch_header, epoch_row
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


class Trainer:
    """Unified training engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, soft_cl, topo_loss,
                 optimizer, scheduler, train_loader, val_loader, args):
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
        self.best_f1_score = 0.0

    def _forward(self, images):
        images = self.spec.preprocess(self.model, images)
        sparse, dense = self.spec.prompt_builder(self.model, images.shape[0])
        return self.spec.forward(self.model, images, sparse, dense, return_stage1=True)

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        comps_sum = {k: 0.0 for k in ('loss_main', 'loss_ds', 'loss_iou', 'loss_cl', 'loss_topo')}
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")

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
        val_loss = 0.0
        all_pred_masks = []
        all_true_masks = []
        val_cldice = []
        metrics_acc = {}

        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                        desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")

        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                masks_stage2, iou_pred, masks_stage1 = self._forward(images)

                masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                m2 = F.interpolate(masks_stage2, size=(1024, 1024), mode='bilinear', align_corners=False)
                main_mask = m2[:, 0:1]

                loss = self.criterion(main_mask, masks)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

                all_pred_masks.append(main_mask)
                all_true_masks.append(masks)

                # Full metric suite on the binarized main mask. The
                # few-shot val sets are tiny (6 images in the 3-shot
                # protocol), so the distance/topology metrics are cheap.
                # The 4 optional extras (auc / persistence engines) are
                # opt-in via CLI flags — they cost seconds per image.
                m = compute_metrics(
                    main_mask[0], masks[0],
                    compute_auc=self.args.val_auc,
                    compute_betti_matching=self.args.val_betti_matching,
                    compute_topograph=self.args.val_topograph,
                )
                val_cldice.append(m['cldice'])
                # 保留 NaN 项(空预测时 HD/cD/β 无定义):均值会自然传播
                # NaN,显示行保持列位稳定(见 print_metrics_row)
                for k, v in m.items():
                    metrics_acc.setdefault(k, []).append(v)

        all_pred_masks = torch.cat(all_pred_masks, dim=0)
        all_true_masks = torch.cat(all_true_masks, dim=0)
        f1 = calculate_f1_score(all_pred_masks, all_true_masks)

        avg_val_loss = val_loss / len(self.val_loader)
        with warnings.catch_warnings():
            # 全部 NaN(空预测)时 nanmean 会打 "Mean of empty slice"
            warnings.simplefilter('ignore', RuntimeWarning)
            avg_val_cldice = float(np.nanmean(val_cldice)) if val_cldice else float('nan')
        # 逐 epoch 的 val 数值全部由 rich 宽表呈现,不再刷 INFO 日志

        if self.args.scheduler == 'reduce':
            self.scheduler.step(avg_val_loss)

        val_metrics = {k: float(np.mean(v)) for k, v in metrics_acc.items() if v}

        if f1 > self.best_f1_score:
            self.best_f1_score = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
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
        run = setup_wandb(
            self.args,
            name=f"{self.args.model}_{self.args.preset}",
            config=self._run_config(),
        )
        history = []
        best_epoch = -1
        # 宽表:一个表头,每行一个 epoch,每个指标一列(全指标)
        console.print(epoch_header(), style="bold")

        try:
            for epoch in range(self.args.epochs):
                t0 = time.time()
                avg_train_loss, comps = self._train_epoch(epoch)

                wm = {'epoch': epoch, 'train/loss': avg_train_loss,
                      'lr': self.optimizer.param_groups[0]['lr']}
                for k, v in comps.items():
                    wm[f'train/{k}'] = v

                best_before = self.best_f1_score
                f1 = vloss = None
                val_metrics = None
                if (epoch + 1) % self.args.val_interval == 0:
                    f1, vloss, avg_val_cldice, val_metrics = self._validate(epoch)
                    wm.update({'val/loss': vloss, 'val/f1': f1,
                               'val/cldice': avg_val_cldice})
                    for k, v in val_metrics.items():
                        wm[f'val/{k}'] = v
                    is_best = self.best_f1_score > best_before
                else:
                    is_best = False
                if is_best:
                    best_epoch = epoch + 1

                log_scalars(run, wm)

                # **wm first so the explicit 1-based epoch wins (wm stores
                # the 0-based wandb epoch)
                history.append({**wm, 'epoch': epoch + 1,
                                'time_s': round(time.time() - t0, 2)})

                row = epoch_row(epoch + 1, avg_train_loss, comps,
                                self.optimizer.param_groups[0]['lr'],
                                time.time() - t0, f1, vloss, val_metrics)
                console.print(row, style="bold green" if is_best else "")
                # Online advisor: metric patterns -> optimization
                # directions, printed while training (history already
                # includes this epoch).
                if val_metrics is not None:
                    for sev, msg in advise(history, self.args):
                        console.print(f"  ⚠ {msg}",
                                      style="yellow" if sev == 'warn' else "cyan")

                # Step the scheduler (except ReduceLROnPlateau, updated during validation)
                if self.scheduler is not None and self.args.scheduler != 'reduce':
                    self.scheduler.step()
        finally:
            if run is not None:
                try:
                    run.summary['best_f1'] = self.best_f1_score
                except Exception:
                    pass
            finish_run(run)
            self._write_history(history, best_epoch)
