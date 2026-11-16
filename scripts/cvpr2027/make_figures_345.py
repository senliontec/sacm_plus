#!/usr/bin/env python3
"""Draw CVPR 2027 DCD figures 3, 4, and 5 from the 16-dataset records.

Figure 3: DCD rank heatmap across 16 datasets and four metrics.
Figure 4: Seen / Unseen / Novel generalization for selected models.
Figure 5: DCD ablation trade-off on the 16-dataset subset.

Inputs are the CSV files in ``docs/cvpr2027/records``.  The script is
intentionally independent of training/evaluation code and does not alter
any baseline files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS = REPO_ROOT / "docs" / "cvpr2027" / "records"
DEFAULT_OUT = REPO_ROOT / "docs" / "cvpr2027" / "figures"

ROLE_ORDER = ["Seen", "Unseen", "Novel"]
ROLE_ABBR = {"Seen": "S", "Unseen": "U", "Novel": "N"}
METRICS = ["dice", "iou", "cldice", "hd95"]
METRIC_LABELS = {
    "dice": "Dice ↑",
    "iou": "IoU ↑",
    "cldice": "clDice ↑",
    "hd95": "HD95 ↓",
}
MODELS_FIG4 = ["DCD (fixed3)", "SACM official", "SAM-OCTA", "U-Net"]
MODEL_COLORS = {
    "DCD (fixed3)": "#d62728",
    "SACM official": "#1f77b4",
    "SAM-OCTA": "#2ca02c",
    "U-Net": "#7f7f7f",
}
MODEL_MARKERS = {
    "DCD (fixed3)": "o",
    "SACM official": "s",
    "SAM-OCTA": "^",
    "U-Net": "D",
}
ABLATION_LABELS = {
    "fixed3": "DCD\nfixed3",
    "gating_fusion": "gating\n+ fusion",
    "nofusion": "w/o\nfusion",
    "native_control": "native\ncontrol",
    "notopo": "w/o\ntopo",
    "official": "SACM\nofficial",
}
ABLATION_ORDER = ["fixed3", "gating_fusion", "nofusion", "native_control", "notopo", "official"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value) -> float:
    return float(value)


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def figure3_rank_heatmap(records: Path, out_dir: Path) -> None:
    """Heatmap of DCD rank among 11 models for each dataset/metric."""
    rows = read_csv(records / "table2_per_dataset_16.csv")
    counts = {
        r["metric"]: (int(r["top1_count"]), int(r["top3_count"]))
        for r in read_csv(records / "table5_rank_counts_16.csv")
        if r["model"] == "DCD (fixed3)"
    }

    dcd_rows = [r for r in rows if r["model"] == "DCD (fixed3)"]
    datasets = sorted(
        {r["dataset"] for r in dcd_rows},
        key=lambda d: (ROLE_ORDER.index(next(r["role"] for r in dcd_rows if r["dataset"] == d)), d),
    )
    role_of = {r["dataset"]: r["role"] for r in dcd_rows}
    rank = {
        (r["dataset"], r["metric"]): as_float(r["rank_among_11"])
        for r in dcd_rows
    }

    matrix = np.array([[rank[(ds, m)] for m in METRICS] for ds in datasets])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues_r", vmin=1, vmax=11)

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(
        [f"{ds} [{ROLE_ABBR[role_of[ds]]}]" for ds in datasets]
    )
    ax.tick_params(length=0)
    ax.set_title("DCD rank among 11 models on the 16-dataset working subset", pad=12)

    # Role separators and labels
    prev_role = None
    for idx, ds in enumerate(datasets):
        role = role_of[ds]
        if prev_role is not None and role != prev_role:
            ax.axhline(idx - 0.5, color="white", linewidth=2.0)
            ax.axhline(idx - 0.5, color="#333333", linewidth=0.5)
        prev_role = role

    # Rank annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
            color = "white" if value <= 5.0 else "#111111"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rank  (1 = best, 11 = worst)")
    cbar.set_ticks([1, 3, 5, 7, 9, 11])

    count_text = "  ".join(
        f"{METRIC_LABELS[m].replace(' ↑', '').replace(' ↓', '')} "
        f"top1/top3={counts.get(m, ('-', '-'))[0]}/{counts.get(m, ('-', '-'))[1]}"
        for m in METRICS
    )
    fig.text(
        0.5, -0.02, count_text + "    [S=Seen, U=Unseen, N=Novel]",
        ha="center", va="top", fontsize=8, color="#333333",
    )
    save(fig, "fig3_dcd_rank_heatmap_16", out_dir)


def figure4_role_generalization(records: Path, out_dir: Path) -> None:
    """Role curves for DCD and three representative baselines."""
    rows = read_csv(records / "table3_roles_16.csv")
    role_order = ROLE_ORDER + ["All"]
    data: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = {}
    for r in rows:
        model, role, metric = r["model"], r["role"], r["metric"]
        if model not in MODELS_FIG4 or role not in role_order or metric not in METRICS:
            continue
        data.setdefault(model, {}).setdefault(role, {})[metric] = (
            as_float(r["mean"]), as_float(r["std"]), int(r["n_datasets"])
        )

    role_n = {}
    for model in MODELS_FIG4:
        for role in role_order:
            if role in data.get(model, {}):
                role_n[role] = data[model][role]["dice"][2]

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.2), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(role_order))
    for ax, metric in zip(axes, METRICS):
        for model in MODELS_FIG4:
            if model not in data:
                continue
            means, stds = [], []
            for role in role_order:
                mean, std, _ = data[model][role][metric]
                scale = 100.0 if metric != "hd95" else 1.0
                means.append(mean * scale)
                stds.append(std * scale)
            ax.errorbar(
                x, means, yerr=stds, marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model], linewidth=1.4, markersize=4,
                capsize=2.5, label=model,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(n={role_n.get(r, '?')})" for r in role_order])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        if metric != "hd95":
            ax.set_ylabel("%")
        else:
            ax.set_ylabel("pixels")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Role generalization on the 16-dataset subset (4 seeds; mean ± std)", y=1.06)
    fig.tight_layout()
    save(fig, "fig4_role_generalization_16", out_dir)


def figure5_ablation(records: Path, out_dir: Path) -> None:
    """2×2 ablation panels; HD95 is lower-better and is shown raw."""
    rows = {r["arm"]: r for r in read_csv(records / "table6_ablation_16.csv")
            if r.get("arm") in ABLATION_ORDER}
    if set(rows) != set(ABLATION_ORDER):
        raise RuntimeError(f"missing ablation arms: {set(ABLATION_ORDER) - set(rows)}")

    colors = ["#d62728"] + ["#9aa0a6"] * (len(ABLATION_ORDER) - 1)
    x = np.arange(len(ABLATION_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.2), sharex=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRICS):
        means = [as_float(rows[arm][f"{metric}_mean"]) for arm in ABLATION_ORDER]
        stds = [as_float(rows[arm][f"{metric}_std"]) for arm in ABLATION_ORDER]
        if metric != "hd95":
            means = [v * 100.0 for v in means]
            stds = [v * 100.0 for v in stds]
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="#333333",
               linewidth=0.5, capsize=3, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABELS[a] for a in ABLATION_ORDER], fontsize=7.5)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        if metric != "hd95":
            ax.set_ylabel("%")
        else:
            ax.set_ylabel("pixels")
        # Make the fixed3 bar visually identifiable without changing its value.
        ax.patches[0].set_hatch("//")
        ax.patches[0].set_edgecolor("#7f0000")

    fig.suptitle("DCD ablation on the 16-dataset subset (4 seeds; mean ± std)", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_ablation_tradeoff_16", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    style()
    figure3_rank_heatmap(args.records, args.out)
    figure4_role_generalization(args.records, args.out)
    figure5_ablation(args.records, args.out)
    print(f"wrote figures to {args.out}")
    for name in ("fig3_dcd_rank_heatmap_16.pdf", "fig3_dcd_rank_heatmap_16.png",
                 "fig4_role_generalization_16.pdf", "fig4_role_generalization_16.png",
                 "fig5_ablation_tradeoff_16.pdf", "fig5_ablation_tradeoff_16.png"):
        print(args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
