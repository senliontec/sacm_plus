#!/usr/bin/env python3
"""Local diagnostic for the fixed2 ResMoE checkpoint.

For each validation image:
  * run the coarse model with ResMoE disabled;
  * run the trained model with ResMoE enabled;
  * compute Dice / clDice / HD95 before and after repair;
  * compute gate AUROC against hard coarse errors and the topology target;
  * save per-image and aggregate diagnostics.

Usage:
  PYTHONPATH=src python scripts/sacm_x/diagnose_resmoe_local.py \
    --checkpoint results/runs/cvpr8gpu_verify/sacm_x/full_fixed2_resmoe/seed_0/full_fixed2_resmoe/best_model.pth \
    --data-root data/sacm_3shot_seed0 \
    --output-dir results/diagnostics/resmoe_epoch163 \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.data import TestDataset
from core.io import get_prompt_embeddings, predict_all
from core.metrics import compute_metrics
from models.sacm_x.configs import PRESETS
from models.sacm_x.specs import sam_preprocess
from models.sam_network import build_sam_vit_l
from models.sam_network.residual_moe import (
    _resize,
    build_topo_gate_target,
    degrade_coarse,
)


METRICS = ("dice", "cldice", "hd95")


def _load_model(checkpoint_path: Path, device: torch.device):
    preset = PRESETS["full_fixed2_resmoe"]
    model = build_sam_vit_l(
        checkpoint=None,
        use_adapter=True,
        adapter_dim_ratio=preset.get("adapter_dim_ratio", 0.1),
        use_geo_i=preset.get("use_geo_i", True),
        use_geo_e=preset.get("use_geo_e", True),
        geo_e_layers=preset.get("geo_e_layers", "all"),
        use_coarse_to_fine=preset.get("use_coarse_to_fine", True),
        use_fusion_v2=preset.get("use_fusion_v2", True),
        use_multi_depth=preset.get("use_multi_depth", True),
        use_internal_adapter=preset.get("use_internal_adapter", True),
        collect_adapter_features=preset.get("collect_adapter_features", True),
        use_structural_aux=preset.get("use_structural_aux", True),
        high_res_output=preset.get("high_res_output", True),
        high_res_size=preset.get("high_res_size", 512),
        high_res_refine=preset.get("high_res_refine", False),
        use_res_moe=True,
        res_num_experts=preset.get("res_num_experts", 4),
        res_hidden=preset.get("res_hidden", 24),
        res_gate_hidden=preset.get("res_gate_hidden", 32),
        res_gate_size=preset.get("res_gate_size", 512),
        res_gate_mode=preset.get("res_gate_mode", "learned"),
        res_use_head_disagreement=preset.get(
            "res_use_head_disagreement", True),
        res_use_centerline=preset.get("res_use_centerline", True),
        res_gate_input_norm=preset.get("res_gate_input_norm", True),
        res_out_size=preset.get("res_out_size", 512),
        res_head_index=preset.get("res_head_index", 2),
        res_dropout=preset.get("res_dropout", 0.0),
        res_disagreement_tau=preset.get("res_disagreement_tau", 0.05),
        res_disagreement_scale=preset.get("res_disagreement_scale", 0.10),
    )
    payload = torch.load(checkpoint_path, map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    rm = model.mask_decoder.residual_moe
    if rm is None:
        raise SystemExit("checkpoint/model does not contain residual_moe")
    return model, payload, rm


def _forward(model, image: torch.Tensor):
    with torch.no_grad():
        # TestDataset returns [0,1]; sam_preprocess scales to [0,255] before
        # applying SAM's pixel mean/std.  Calling model.preprocess directly
        # here would be a silent scale bug and produce empty masks.
        x = sam_preprocess(model, image)
        sparse, dense = get_prompt_embeddings(model, 1)
        out = predict_all(model, x, sparse, dense, return_stage1=False)
        # predict_all returns (masks_stage2, iou_pred) when return_stage1 is
        # False; keep only the four Stage-2 logits here.
        return out[0] if isinstance(out, (tuple, list)) else out


def _metrics(logits: torch.Tensor, gt: torch.Tensor):
    return compute_metrics(
        logits[0], gt[0], threshold=0.0,
        compute_auc=False, compute_betti_matching=False,
        compute_topograph=False,
    )


def _experts_with_gate(model, rm, image, coarse_head, gt, mode):
    """Apply the trained experts with an oracle (GT-derived) gate.

    This is a diagnostic upper bound only; GT is never used by the method at
    inference.  It answers whether the expert pathway has topology headroom
    when error localization is perfect.
    """
    x = sam_preprocess(model, image)
    size = (rm.out_size, rm.out_size)
    coarse = _resize(coarse_head, size)
    image_r = _resize(x[:, :3], size)
    prob = torch.sigmoid(coarse.detach())
    residuals = []
    with torch.no_grad():
        for kind, expert in zip(rm.expert_kinds, rm.experts):
            degraded = degrade_coarse(prob, kind)
            residuals.append(expert(torch.cat([image_r, degraded], dim=1)))
        residual = torch.stack(residuals, dim=1)
        if rm.last_routing is not None:
            weights = rm.last_routing
        else:
            weights = torch.full(
                (coarse.shape[0], len(residuals)),
                1.0 / float(len(residuals)),
                device=coarse.device, dtype=coarse.dtype)
        mixed = (residual * weights.view(
            weights.shape[0], weights.shape[1], 1, 1, 1)).sum(dim=1)

        gt_small = F.adaptive_max_pool2d(gt, coarse.shape[-2:])
        if mode == "hard":
            gate = ((coarse.detach() > 0) != (gt_small > 0.5)).float()
        elif mode == "topo":
            gate = build_topo_gate_target(coarse, gt_small)
        else:
            raise ValueError(mode)
        out = coarse + gate * mixed
    return out, gate


def _auc(score: np.ndarray, target: np.ndarray):
    """AUC with the Mann-Whitney statistic; avoids an optional dependency."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    n_pos = int(target.sum())
    n_neg = int((~target).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank average for ties.
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    i = 0
    while i < score.size:
        j = i + 1
        while j < score.size and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos_rank = ranks[target].sum()
    return float((pos_rank - n_pos * (n_pos + 1) / 2.0) /
                 (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/sacm_3shot_seed0"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/diagnostics/resmoe_local"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, payload, rm = _load_model(args.checkpoint, device)
    dataset = TestDataset(
        str(args.data_root), split="val", input_mode="resize",
        image_size=1024)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (image, gt, original_size, image_name, _fov) in enumerate(loader):
        if args.limit and idx >= args.limit:
            break
        image = image.to(device)
        gt = gt.to(device)

        # Coarse: exact identity through the new warm-up switch.
        rm.res_scale = 0.0
        coarse_heads = _forward(model, image)
        coarse_head = coarse_heads[:, 2:3]

        # Learned repair.
        rm.res_scale = 1.0
        repaired_heads = _forward(model, image)
        repaired_head = repaired_heads[:, 2:3]

        def to_gt(x):
            return F.interpolate(
                x, size=gt.shape[-2:], mode="bilinear",
                align_corners=False)

        coarse_gt = to_gt(coarse_head)
        repaired_gt = to_gt(repaired_head)
        coarse_m = _metrics(coarse_gt, gt)
        repaired_m = _metrics(repaired_gt, gt)
        oracle_hard, _ = _experts_with_gate(
            model, rm, image, coarse_head, gt, "hard")
        oracle_topo, _ = _experts_with_gate(
            model, rm, image, coarse_head, gt, "topo")
        oracle_hard_m = _metrics(to_gt(oracle_hard), gt)
        oracle_topo_m = _metrics(to_gt(oracle_topo), gt)

        gate = rm.last_gate_mask
        if gate is None:
            gate = torch.zeros_like(coarse_head)
        gate_gt = to_gt(gate)
        pred = coarse_gt > 0
        hard_error = (pred != (gt > 0.5)).float()

        # Topology-aware target, at the coarse resolution, then upsampled.
        target_512 = F.adaptive_max_pool2d(gt, coarse_head.shape[-2:])
        topo_target = build_topo_gate_target(coarse_head, target_512)
        topo_gt = F.interpolate(
            topo_target, size=gt.shape[-2:], mode="nearest")

        row = {
            "image": image_name[0],
            "dataset": Path(image_name[0]).stem,
            "coarse_dice": coarse_m.get("dice", float("nan")),
            "repair_dice": repaired_m.get("dice", float("nan")),
            "coarse_cldice": coarse_m.get("cldice", float("nan")),
            "repair_cldice": repaired_m.get("cldice", float("nan")),
            "coarse_hd95": coarse_m.get("hd95", float("nan")),
            "repair_hd95": repaired_m.get("hd95", float("nan")),
            "oracle_hard_dice": oracle_hard_m.get("dice", float("nan")),
            "oracle_hard_cldice": oracle_hard_m.get("cldice", float("nan")),
            "oracle_hard_hd95": oracle_hard_m.get("hd95", float("nan")),
            "oracle_topo_dice": oracle_topo_m.get("dice", float("nan")),
            "oracle_topo_cldice": oracle_topo_m.get("cldice", float("nan")),
            "oracle_topo_hd95": oracle_topo_m.get("hd95", float("nan")),
            "delta_abs_mean": float(
                (repaired_head - coarse_head).abs().mean()),
            "gate_mean": float(gate_gt.mean()),
            "gate_positive_rate": float((gate_gt > 0.5).float().mean()),
            "gate_auroc_hard": _auc(
                gate_gt.detach().cpu().numpy(),
                hard_error.detach().cpu().numpy()),
            "gate_auroc_topo": _auc(
                gate_gt.detach().cpu().numpy(),
                (topo_gt > 0.5).detach().cpu().numpy()),
            "hard_error_rate": float(hard_error.mean()),
            "topo_target_rate": float(topo_gt.float().mean()),
        }
        rows.append(row)
        print(
            f"{image_name[0]}: "
            f"coarse Dice={row['coarse_dice']:.4f} clDice={row['coarse_cldice']:.4f} "
            f"-> repair Dice={row['repair_dice']:.4f} clDice={row['repair_cldice']:.4f}; "
            f"oracle(topo) Dice={row['oracle_topo_dice']:.4f} "
            f"clDice={row['oracle_topo_cldice']:.4f}; "
            f"gate={row['gate_mean']:.4f} AUROC(hard)={row['gate_auroc_hard']:.3f} "
            f"AUROC(topo)={row['gate_auroc_topo']:.3f}"
        )

    csv_path = args.output_dir / "per_image.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    def mean(key):
        vals = [r[key] for r in rows
                if isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch_completed", payload.get("epoch")),
        "best_epoch": payload.get("best_epoch"),
        "selection_metric": payload.get("selection_metric"),
        "n_images": len(rows),
        "coarse_dice": mean("coarse_dice"),
        "repair_dice": mean("repair_dice"),
        "delta_dice": mean("repair_dice") - mean("coarse_dice"),
        "coarse_cldice": mean("coarse_cldice"),
        "repair_cldice": mean("repair_cldice"),
        "delta_cldice": mean("repair_cldice") - mean("coarse_cldice"),
        "coarse_hd95": mean("coarse_hd95"),
        "repair_hd95": mean("repair_hd95"),
        "delta_hd95": mean("repair_hd95") - mean("coarse_hd95"),
        "oracle_hard_dice": mean("oracle_hard_dice"),
        "oracle_hard_cldice": mean("oracle_hard_cldice"),
        "oracle_hard_hd95": mean("oracle_hard_hd95"),
        "oracle_topo_dice": mean("oracle_topo_dice"),
        "oracle_topo_cldice": mean("oracle_topo_cldice"),
        "oracle_topo_hd95": mean("oracle_topo_hd95"),
        "delta_abs_mean": mean("delta_abs_mean"),
        "gate_mean": mean("gate_mean"),
        "gate_positive_rate": mean("gate_positive_rate"),
        "gate_auroc_hard": mean("gate_auroc_hard"),
        "gate_auroc_topo": mean("gate_auroc_topo"),
        "hard_error_rate": mean("hard_error_rate"),
        "topo_target_rate": mean("topo_target_rate"),
        "per_image_csv": str(csv_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("\nsummary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
