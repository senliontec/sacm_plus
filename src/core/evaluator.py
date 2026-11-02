"""Unified evaluation engine.

Model-agnostic via ModelSpec. The evaluation-loop logic is moved
VERBATIM from the original test_sam.py: four-way flip TTA, mask
selection (IoU head or gating), full metric suite per image, PNG
visualization (non-fatal), CSV/summary output.
"""

import csv
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm

from core.data import letterbox_rect
from core.metrics import COMPONENT_KEYS, component_metrics, compute_metrics
from core.postprocess.components import prune_small_components
from core.wandb_utils import finish_run, log_scalars, setup_wandb

METRIC_KEYS = ['dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity',
               'accuracy', 'mcc', 'cldice', 'hd', 'hd95', 'assd', 'asd', 'ravd', 'nsd', 'betti']
AUC_KEYS = ['dice_auc', 'cldice_auc']
BETTI_MATCHING_KEYS = ['betti_matching']
TOPOGRAPH_KEYS = ['topograph_error']


def resolve_selection(requested, iou_loss_weight):
    """Resolve 'auto' without silently using an untrained IoU head.

    auto -> 'iou' only when the IoU head received supervision
    (iou_loss_weight > 0); otherwise 'gating' (head 0), matching the
    original SACM baseline. Explicit 'iou'/'gating'/'quality' are preserved.
    """
    if requested in (None, 'auto'):
        return 'iou' if float(iou_loss_weight or 0.0) > 0 else 'gating'
    if requested not in ('iou', 'gating', 'quality'):
        raise ValueError(
            f"Unknown selection={requested!r}; expected auto/iou/gating/quality")
    return requested


def select_mask(masks, iou_pred, selection, empty_guard=False,
                foreground_threshold=0.5):
    """Select the final mask. 'iou': trained IoU head (argmax);
    'gating': gating-ranked head 0 (original SACM behavior).

    With ``empty_guard``, if the selected candidate is empty (or has
    fewer than one foreground pixel at ``foreground_threshold``), choose
    the candidate with the largest foreground area instead.  This is a
    label-free fallback for thick/dense domains where a head can collapse
    to an empty mask (e.g. TopoMortar).
    """
    if selection == 'iou':
        k = int(iou_pred[0].argmax())
    elif selection == 'gating':
        k = 0
    else:
        raise ValueError(
            f"select_mask received unresolved selection={selection!r}; "
            "call resolve_selection() first")

    if empty_guard and masks.shape[1] > 1:
        areas = (masks[0] > foreground_threshold).flatten(1).sum(dim=1)
        if float(areas[k].item()) <= 0.0:
            k = int(areas.argmax().item())
    return masks[:, k:k + 1]


def select_mask_quality(masks, iou_pred, foreground_threshold=0.5,
                        small_component_weight=0.5,
                        component_count_weight=0.05,
                        empty_guard=True):
    """Label-free candidate selector with a fragmentation penalty.

    The candidate score is ``iou_pred - w_small * small_component_rate
    - w_count * log1p(n_components)``.  It is intentionally simple and
    should be compared against the IoU selector on validation data.
    """
    k_count = int(masks.shape[1])
    scores = []
    for k in range(k_count):
        binary = (masks[0, k] > foreground_threshold).cpu().numpy()
        if binary.any():
            lab, n = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
            small = (int(np.count_nonzero(sizes < 20)) / n) if n else 0.0
        else:
            n, small = 0, 1.0
        score = float(iou_pred[0, k].item())
        score -= float(small_component_weight) * small
        score -= float(component_count_weight) * math.log1p(n)
        scores.append(score)
    k = int(np.argmax(scores))
    if empty_guard:
        areas = (masks[0] > foreground_threshold).flatten(1).sum(dim=1)
        if float(areas[k].item()) <= 0.0:
            k = int(areas.argmax().item())
    return masks[:, k:k + 1], k, scores


def _resize_to_original(masks, width, height, eval_input_mode):
    """Crop letterbox padding and resize masks to the original resolution."""
    if eval_input_mode == 'aspect_pad':
        left, top, content_w, content_h = letterbox_rect(
            width, height, masks.shape[-1])
        masks = masks[:, :, top:top + content_h, left:left + content_w]
    return F.interpolate(
        masks, size=(height, width), mode='bilinear', align_corners=False)


def _quick_binary_metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    denom = 2 * tp + fp + fn
    return {
        'dice': (2.0 * tp / denom) if denom else 1.0,
        'iou': (tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
        'precision': (tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall': (tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def _write_rows(path, rows, fieldnames):
    """Write an optional audit CSV; never fail the main evaluation."""
    try:
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logging.warning("Could not write audit CSV %s: %s", path, exc)


class Evaluator:
    """Unified evaluation engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, args, metric_keys,
                 deep_closing=None):
        self.model = model
        self.spec = spec
        self.device = device
        self.criterion = criterion
        self.args = args
        self.metric_keys = metric_keys
        # Optional Deep Closing refiner (post-processing only).  It never
        # changes the model forward or the raw segmentation loss.
        self.deep_closing = deep_closing
        self.sparse, self.dense = spec.prompt_builder(model, 1)

    def run(self, loader, output_dir):
        """Evaluate a DataLoader over a TestDataset. Returns
        (summary, csv_rows, n_hd95_nan, avg_loss)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        run = setup_wandb(
            self.args,
            name=f"eval_{os.path.basename(self.args.data_root.rstrip('/'))}",
            config={
                'model': self.args.model,
                'preset': self.args.preset,
                'trained_weights': self.args.trained_weights,
                'selection': self.args.selection,
                'tta': self.args.tta,
                'pred_threshold': self.args.pred_threshold,
                'eval_input_mode': getattr(
                    self.args, 'eval_input_mode', 'resize'),
                'empty_guard': bool(getattr(
                    self.args, 'empty_guard', False)),
                'component_metrics': bool(getattr(
                    self.args, 'component_metrics', False)),
                'component_prune_mode': getattr(
                    self.args, 'component_prune_mode', 'off'),
                'audit_heads': audit_heads,
                'audit_thresholds': list(audit_thresholds),
                'save_candidate_masks': save_candidates,
                'nsd_tolerance': self.args.nsd_tolerance,
                'deep_closing': self.deep_closing is not None,
                'deep_closing_checkpoint': getattr(
                    self.args, 'deep_closing_checkpoint', None),
                'deep_closing_threshold': getattr(
                    self.args, 'deep_closing_threshold', None),
            },
        )

        self.model.eval()
        metrics_all = {k: [] for k in self.metric_keys}
        component_all = {k: [] for k in COMPONENT_KEYS}
        loss_values = []
        csv_rows = []
        head_rows = []
        threshold_rows = []
        save_candidates = bool(getattr(self.args, 'save_candidate_masks', False))
        audit_heads = bool(getattr(self.args, 'audit_heads', False)
                           or save_candidates)
        audit_thresholds = getattr(self.args, 'audit_thresholds', None) or []
        candidates_dir = os.path.join(output_dir, 'candidates')
        if save_candidates:
            os.makedirs(candidates_dir, exist_ok=True)

        flips = [
            lambda t: t,
            lambda t: torch.flip(t, dims=[-1]),
            lambda t: torch.flip(t, dims=[-2]),
            lambda t: torch.flip(t, dims=[-1, -2]),
        ]

        test_pbar = tqdm(loader, total=len(loader), desc="Testing")
        viz_warned = False

        with torch.no_grad():
            for model_input, original_mask, original_size, img_name, fov_mask in test_pbar:
                model_input = model_input.to(self.device)
                original_mask = original_mask.to(self.device)
                fov_mask = fov_mask.to(self.device)

                # Model-specific input preprocessing
                model_input = self.spec.preprocess(self.model, model_input)

                # Forward pass (with optional four-way flip TTA)
                if self.args.tta:
                    masks_list, iou_list = [], []
                    for f in flips:
                        masks, iou_pred = self.spec.forward(
                            self.model, f(model_input), self.sparse, self.dense,
                            return_stage1=False,
                        )
                        masks_list.append(f(masks))  # flips are their own inverse
                        iou_list.append(iou_pred)
                    masks_all = torch.stack(masks_list, dim=0).mean(dim=0)
                    iou_all = torch.stack(iou_list, dim=0).mean(dim=0)
                else:
                    masks_all, iou_all = self.spec.forward(
                        self.model, model_input, self.sparse, self.dense,
                        return_stage1=False,
                    )

                width, height = original_size
                width = width.item() if torch.is_tensor(width) else int(width)
                height = height.item() if torch.is_tensor(height) else int(height)
                eval_mode = getattr(self.args, 'eval_input_mode', 'resize')

                if self.args.selection == 'quality':
                    low_res_masks, selected_head, quality_scores = (
                        select_mask_quality(
                            masks_all, iou_all,
                            foreground_threshold=float(
                                getattr(self.args, 'pred_threshold', 0.0)),
                            small_component_weight=float(getattr(
                                self.args, 'quality_small_component_weight',
                                0.5)),
                            component_count_weight=float(getattr(
                                self.args, 'quality_component_weight', 0.05)),
                            empty_guard=bool(getattr(
                                self.args, 'empty_guard', False))))
                else:
                    selected_head = None
                    low_res_masks = select_mask(
                        masks_all, iou_all, self.args.selection,
                        empty_guard=bool(getattr(
                            self.args, 'empty_guard', False)),
                        foreground_threshold=float(
                            getattr(self.args, 'pred_threshold', 0.0)))

                low_res_masks = _resize_to_original(
                    low_res_masks, width, height, eval_mode)
                if not getattr(self.args, 'no_fov', False):
                    low_res_masks = low_res_masks * fov_mask
                    original_mask = original_mask * fov_mask
                gt_binary = (original_mask[0, 0] > 0.5).cpu().numpy()

                loss = self.criterion(low_res_masks, original_mask)
                loss_values.append(loss.item())

                # Deep Closing is a post-processing refiner: metrics and
                # saved masks use the refined binary map, while the logged
                # segmentation loss remains the raw model objective.
                refined_mask = None
                pred_binary = None
                prune_stats = {}
                if self.deep_closing is not None:
                    raw_binary = (
                        low_res_masks[0, 0] > self.args.pred_threshold
                    ).detach().cpu().numpy()
                    refined_mask = self.deep_closing.refine(
                        raw_binary, threshold=0.5)
                    pred_binary = refined_mask.astype(bool)

                prune_mode = getattr(self.args, 'component_prune_mode', 'off')
                if prune_mode != 'off':
                    if pred_binary is None:
                        pred_binary = (
                            low_res_masks[0, 0] > self.args.pred_threshold
                        ).detach().cpu().numpy()
                    if prune_mode == 'relative':
                        pred_binary, prune_stats = prune_small_components(
                            pred_binary,
                            min_area=getattr(
                                self.args, 'component_prune_min_area', 0),
                            min_skeleton=getattr(
                                self.args, 'component_prune_min_skeleton', 0),
                            rel_area=getattr(
                                self.args, 'component_prune_rel_area', 0.0),
                            rel_skeleton=getattr(
                                self.args, 'component_prune_rel_skeleton', 0.0),
                            keep_largest=getattr(
                                self.args, 'component_prune_keep_largest', 1))
                    else:
                        pred_binary, prune_stats = prune_small_components(
                            pred_binary,
                            min_area=getattr(
                                self.args, 'component_prune_min_area', 0),
                            min_skeleton=getattr(
                                self.args, 'component_prune_min_skeleton', 0),
                            keep_largest=getattr(
                                self.args, 'component_prune_keep_largest', 1))

                if pred_binary is not None:
                    metric_logits = torch.from_numpy(
                        ((pred_binary.astype(np.float32) * 2.0 - 1.0) * 10.0)
                    ).to(self.device)[None, None]
                    metric_threshold = 0.0
                else:
                    metric_logits = low_res_masks
                    metric_threshold = self.args.pred_threshold
                    pred_binary = (
                        low_res_masks[0, 0] > metric_threshold
                    ).cpu().numpy()

                m = compute_metrics(
                    metric_logits[0], original_mask[0],
                    threshold=metric_threshold,
                    nsd_tolerance=self.args.nsd_tolerance,
                    compute_auc=self.args.auc,
                    compute_betti_matching=self.args.betti_matching,
                    compute_topograph=self.args.topograph_metric,
                )
                row = {'image': img_name[0]}
                if selected_head is not None:
                    row['selected_head'] = selected_head
                for k in self.metric_keys:
                    if not np.isnan(m[k]):
                        metrics_all[k].append(m[k])
                    row[k] = m[k]

                comp = None
                if getattr(self.args, 'component_metrics', False):
                    comp = component_metrics(pred_binary, gt_binary)
                    for k, v in comp.items():
                        row[k] = v
                        if np.isfinite(v):
                            component_all[k].append(float(v))
                    if prune_stats:
                        row.update({
                            'prune_components_before':
                                prune_stats.get('components_before', 0),
                            'prune_components_removed':
                                prune_stats.get('removed_components', 0),
                            'prune_components_after':
                                prune_stats.get('components_after', 0),
                            'prune_pixels_removed':
                                prune_stats.get('removed_pixels', 0),
                        })

                # Optional per-head audit: Dice/IoU/P/R + component stats,
                # computed at original resolution after the same FOV rule.
                if audit_heads:
                    gt_head = gt_binary
                    for k in range(masks_all.shape[1]):
                        cand = _resize_to_original(
                            masks_all[:, k:k + 1], width, height, eval_mode)
                        if not getattr(self.args, 'no_fov', False):
                            cand = cand * fov_mask
                        cand_bin = (
                            cand[0, 0] > self.args.pred_threshold
                        ).cpu().numpy()
                        q = _quick_binary_metrics(cand_bin, gt_head)
                        c = component_metrics(cand_bin, gt_head)
                        head_rows.append({
                            'image': img_name[0], 'head': k,
                            'iou_head_score': float(iou_all[0, k].item()),
                            **q, **c,
                        })
                        if save_candidates:
                            stem = os.path.splitext(os.path.basename(
                                str(img_name[0])))[0]
                            Image.fromarray(
                                cand_bin.astype(np.uint8) * 255).save(
                                os.path.join(
                                    candidates_dir, f'{stem}_head{k}.png'))

                # Optional selected-mask threshold sweep (lightweight
                # metrics only; useful for calibration, not test tuning).
                if audit_thresholds:
                    for t in audit_thresholds:
                        tbin = (
                            low_res_masks[0, 0] > float(t)
                        ).cpu().numpy()
                        q = _quick_binary_metrics(tbin, gt_binary)
                        c = component_metrics(tbin, gt_binary)
                        threshold_rows.append({
                            'image': img_name[0], 'threshold': float(t),
                            **q, **c,
                        })

                csv_rows.append(row)

                test_pbar.set_postfix({'Dice': f"{m['dice']:.4f}", 'clDice': f"{m['cldice']:.4f}"})

                # Save prediction visualization (non-fatal if matplotlib fails)
                try:
                    pred_mask = pred_binary.astype(np.uint8) * 255

                    width, height = original_size
                    width = width.item() if torch.is_tensor(width) else width
                    height = height.item() if torch.is_tensor(height) else height

                    plt.figure(figsize=(width / 100, height / 100), dpi=100)
                    plt.imshow(pred_mask, cmap='gray')
                    plt.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                    plt.savefig(os.path.join(output_dir, f'{img_name[0]}_result.png'),
                                bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close()
                except Exception as e:
                    if not viz_warned:
                        logging.warning(f"Visualization failed ({e}); PNG output skipped")
                        viz_warned = True

                logging.info(
                    f"Image: {img_name[0]}, Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}, "
                    f"clDice: {m['cldice']:.4f}, HD95: {m['hd95']:.4f}, Loss: {loss.item():.4f}"
                )

        n_hd95_nan = len(csv_rows) - len(metrics_all['hd95'])
        if n_hd95_nan:
            logging.info(f"HD95 undefined (empty mask) for {n_hd95_nan} images; excluded from the mean")

        summary = {}
        for k in self.metric_keys:
            vals = metrics_all[k]
            summary[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))
            logging.info(f"Average {k.capitalize()}: {summary[k][0]:.4f} ± {summary[k][1]:.4f}")

        # Optional fragmentation metrics.  Kept outside metric_keys so the
        # default evaluation remains byte-compatible with the baselines.
        if getattr(self.args, 'component_metrics', False):
            with np.errstate(invalid='ignore'):
                for k, vals in component_all.items():
                    if vals:
                        summary[k] = (float(np.mean(vals)),
                                      float(np.std(vals)))

        if audit_heads:
            _write_rows(
                os.path.join(output_dir, 'head_audit.csv'), head_rows,
                ['image', 'head', 'iou_head_score', 'dice', 'iou',
                 'precision', 'recall', *COMPONENT_KEYS])
        if audit_thresholds:
            _write_rows(
                os.path.join(output_dir, 'threshold_audit.csv'),
                threshold_rows,
                ['image', 'threshold', 'dice', 'iou', 'precision',
                 'recall', *COMPONENT_KEYS])

        avg_loss = float(np.mean(loss_values))
        logging.info(f"Average Loss: {avg_loss:.4f}")

        wm = {'eval/loss': avg_loss}
        for k, (mean, std) in summary.items():
            wm[f'eval/{k}'] = mean
            wm[f'eval/{k}_std'] = std
        log_scalars(run, wm)
        finish_run(run)

        return summary, csv_rows, n_hd95_nan, avg_loss

    @staticmethod
    def save_outputs(output_dir, summary, csv_rows, n_hd95_nan, avg_loss, args):
        """Write results_summary.txt and metrics.csv."""
        with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
            f.write(f"Model: {args.trained_weights}\n")
            f.write(f"Dataset: {args.data_root}\n")
            f.write(f"Num Samples: {len(csv_rows)}\n")
            f.write(f"Selection: {args.selection}\n")
            f.write(f"TTA: {args.tta}\n")
            f.write(f"Eval input mode: {getattr(args, 'eval_input_mode', 'resize')}\n")
            f.write(f"Empty guard: {getattr(args, 'empty_guard', False)}\n")
            f.write(f"No FOV: {getattr(args, 'no_fov', False)}\n")
            f.write(
                "Deep Closing: "
                f"{getattr(args, 'deep_closing_checkpoint', None)}\n")
            for k, (mean, std) in summary.items():
                f.write(f"Average {k.capitalize()}: {mean:.4f} ± {std:.4f}\n")
            f.write(f"Average Loss: {avg_loss:.4f}\n")
            if n_hd95_nan:
                f.write(f"Note: HD95 undefined for {n_hd95_nan} images (empty mask)\n")

        with open(os.path.join(output_dir, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['image'] + list(summary.keys()),
                extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
