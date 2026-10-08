"""Unified evaluation engine.

Model-agnostic via ModelSpec. The evaluation-loop logic is moved
VERBATIM from the original test_sam.py: four-way flip TTA, mask
selection (IoU head or gating), full metric suite per image, PNG
visualization (non-fatal), CSV/summary output.
"""

import csv
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.metrics import compute_metrics
from core.wandb_utils import finish_run, log_scalars, setup_wandb

METRIC_KEYS = ['dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity',
               'accuracy', 'mcc', 'cldice', 'hd', 'hd95', 'assd', 'asd', 'ravd', 'nsd', 'betti']
AUC_KEYS = ['dice_auc', 'cldice_auc']
BETTI_MATCHING_KEYS = ['betti_matching']
TOPOGRAPH_KEYS = ['topograph_error']


def select_mask(masks, iou_pred, selection):
    """Select the final mask. 'iou': trained IoU head (argmax);
    'gating': gating-ranked head 0 (original SACM behavior)."""
    if selection == 'iou':
        k = int(iou_pred[0].argmax())
    else:  # 'gating'
        k = 0
    return masks[:, k:k + 1]


class Evaluator:
    """Unified evaluation engine (model-agnostic via ModelSpec)."""

    def __init__(self, model, spec, device, criterion, args, metric_keys):
        self.model = model
        self.spec = spec
        self.device = device
        self.criterion = criterion
        self.args = args
        self.metric_keys = metric_keys
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
                'nsd_tolerance': self.args.nsd_tolerance,
            },
        )

        self.model.eval()
        metrics_all = {k: [] for k in self.metric_keys}
        loss_values = []
        csv_rows = []

        flips = [
            lambda t: t,
            lambda t: torch.flip(t, dims=[-1]),
            lambda t: torch.flip(t, dims=[-2]),
            lambda t: torch.flip(t, dims=[-1, -2]),
        ]

        test_pbar = tqdm(loader, total=len(loader), desc="Testing")
        viz_warned = False

        with torch.no_grad():
            for model_input, original_mask, original_size, img_name in test_pbar:
                model_input = model_input.to(self.device)
                original_mask = original_mask.to(self.device)

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

                low_res_masks = select_mask(masks_all, iou_all, self.args.selection)

                # Resize prediction to original size
                low_res_masks = F.interpolate(
                    low_res_masks, size=original_size[::-1],
                    mode='bilinear', align_corners=False,
                )

                loss = self.criterion(low_res_masks, original_mask)
                loss_values.append(loss.item())

                m = compute_metrics(
                    low_res_masks[0], original_mask[0],
                    threshold=self.args.pred_threshold,
                    nsd_tolerance=self.args.nsd_tolerance,
                    compute_auc=self.args.auc,
                    compute_betti_matching=self.args.betti_matching,
                    compute_topograph=self.args.topograph_metric,
                )
                row = {'image': img_name[0]}
                for k in self.metric_keys:
                    if not np.isnan(m[k]):
                        metrics_all[k].append(m[k])
                    row[k] = m[k]
                csv_rows.append(row)

                test_pbar.set_postfix({'Dice': f"{m['dice']:.4f}", 'clDice': f"{m['cldice']:.4f}"})

                # Save prediction visualization (non-fatal if matplotlib fails)
                try:
                    pred_mask = (low_res_masks[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255

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
            for k, (mean, std) in summary.items():
                f.write(f"Average {k.capitalize()}: {mean:.4f} ± {std:.4f}\n")
            f.write(f"Average Loss: {avg_loss:.4f}\n")
            if n_hd95_nan:
                f.write(f"Note: HD95 undefined for {n_hd95_nan} images (empty mask)\n")

        with open(os.path.join(output_dir, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image'] + list(summary.keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
