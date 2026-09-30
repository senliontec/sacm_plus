"""Unified training engine.

The Trainer is model-agnostic: every model-specific behaviour comes from
the ModelSpec. The training-loop logic is moved VERBATIM from the
original train_sam.py (four-term loss — main DiceBCE, Stage-1 deep
supervision, IoU-head MSE, soft-clDice with warmup — plus the optional
extra topology loss; gradient clipping; best-F1 checkpoint selection).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from core.metrics import compute_metrics


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
            train_pbar.set_postfix({'loss': loss.item(), 'cl': mu})

        return train_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_pred_masks = []
        all_true_masks = []
        val_cldice = []

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

                # clDice monitor on the binarized main mask
                m = compute_metrics(main_mask[0], masks[0])
                val_cldice.append(m['cldice'])

        all_pred_masks = torch.cat(all_pred_masks, dim=0)
        all_true_masks = torch.cat(all_true_masks, dim=0)
        f1 = calculate_f1_score(all_pred_masks, all_true_masks)

        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_cldice = float(np.nanmean(val_cldice)) if val_cldice else float('nan')
        logging.info(f'Epoch {epoch+1}/{self.args.epochs} - Average Val Loss: {avg_val_loss:.4f}')
        logging.info(f'Epoch {epoch+1}/{self.args.epochs} - F1 Score: {f1:.4f}')
        logging.info(f'Epoch {epoch+1}/{self.args.epochs} - Val clDice: {avg_val_cldice:.4f}')

        if self.args.scheduler == 'reduce':
            self.scheduler.step(avg_val_loss)

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
                    'use_coarse_to_fine': self.args.use_coarse_to_fine,
                    'use_fusion_v2': self.args.use_fusion_v2,
                    'use_multi_depth': self.args.use_multi_depth,
                    'adapter_dim_ratio': self.args.adapter_dim_ratio,
                },
            }, self.args.save_path)
            logging.info(f'👍New best model saved with F1 score: {self.best_f1_score:.4f}')

        return f1, avg_val_loss, avg_val_cldice

    def fit(self):
        """Run the full training schedule."""
        for epoch in range(self.args.epochs):
            avg_train_loss = self._train_epoch(epoch)
            logging.info(f'Epoch {epoch+1}/{self.args.epochs} - Average Train Loss: {avg_train_loss:.4f}')

            if (epoch + 1) % self.args.val_interval == 0:
                self._validate(epoch)

            # Step the scheduler (except ReduceLROnPlateau, updated during validation)
            if self.scheduler is not None and self.args.scheduler != 'reduce':
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'Current learning rate: {current_lr:.7f}')
