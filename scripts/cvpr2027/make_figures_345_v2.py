#!/usr/bin/env python3
"""CVPR 2027 DCD figures 3-5, version 2 (presentation-focused redesign).

Design choices:
* Fig. 3: rank-strip/lollipop panels instead of a heatmap, so top-1/top-3
  structure is visible without a large colorbar.
* Fig. 4: grouped dot plots instead of connected role curves, because
  Seen/Unseen/Novel are categorical, not a continuous axis.
* Fig. 5: forest plot of paired differences against fixed3 instead of
  zero-based raw bars, so small but meaningful effects are visible.

Inputs are the CSV files under docs/cvpr2027/records.
No training/evaluation/baseline code is imported or modified.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
FIGSIZE = (7.4, 4.9)
DCD_COLOR = "#c0392b"
BASE_COLORS = {
    "SACM official": "#2166ac",
    "SAM-OCTA": "#1b7837",
    "U-Net": "#777777",
}
BASE_MARKERS = {
    "SACM official": "s",
    "SAM-OCTA": "^",
    "U-Net": "D",
}
ARM_LABELS = {
    "gating_fusion": "gating + fusion",
    "nofusion": "w/o fusion",
    "native_control": "native control",
    "notopo": "w/o topology",
    "official": "SACM official",
}
ARM_ORDER = ["official", "native_control", "gating_fusion", "nofusion", "notopo"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value) -> float:
    return float(value)


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
    })


def save(fig, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def figure3_rank_strip(records: Path, out_dir: Path) -> None:
    rows = read_csv(records / "table2_per_dataset_16.csv")
    counts = {
        r["metric"]: (int(r["top1_count"]), int(r["top3_count"]))
        for r in read_csv(records / "table5_rank_counts_16.csv")
        if r["model"] == "DCD (fixed3)"
    }
    dcd = [r for r in rows if r["model"] == "DCD (fixed3)"]
    role_of = {r["dataset"]: r["role"] for r in dcd}
    rank = {(r["dataset"], r["metric"]): fnum(r["rank_among_11"]) for r in dcd}
    datasets = sorted(
        {r["dataset"] for r in dcd},
        key=lambda d: (ROLE_ORDER.index(role_of[d]), d),
    )
    role_starts = {}
    role_ends = {}
    for idx, ds in enumerate(datasets):
        role = role_of[ds]
        role_starts.setdefault(role, idx)
        role_ends[role] = idx

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        for role_idx, role in enumerate(ROLE_ORDER):
            start, end = role_starts[role], role_ends[role]
            if role_idx % 2 == 0:
                ax.axhspan(start - 0.5, end + 0.5, color="#f7f7f7", zorder=0)
            ax.axhline(start - 0.5, color="#d9d9d9", lw=0.6, zorder=1)
        ax.axhline(len(datasets) - 0.5, color="#d9d9d9", lw=0.6, zorder=1)

        # top-3 emphasis
        ax.axvspan(0.5, 3.5, color="#eef5fb", zorder=0)
        ax.axvline(3.5, color="#b8cce0", lw=0.7, ls="--", zorder=1)

        for y, ds in enumerate(datasets):
            r = rank[(ds, metric)]
            ax.plot([1.0, r], [y, y], color="#d0d0d0", lw=1.0, zorder=2)
            ax.scatter(r, y, s=34, color=DCD_COLOR, edgecolor="white",
                       linewidth=0.5, zorder=3)
            label = f"{r:.0f}" if abs(r - round(r)) < 1e-9 else f"{r:.1f}"
            ax.text(r + 0.18, y, label, va="center", ha="left",
                    fontsize=6.5, color="#444444", zorder=4)

        ax.set_xlim(0.5, 11.7)
        ax.set_ylim(len(datasets) - 0.5, -0.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xlabel("Rank  (1 = best)")
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        top1, top3 = counts.get(metric, (0, 0))
        ax.set_title(f"{METRIC_LABELS[metric]}  ({top1}/{top3})")
        if ax_idx == 0:
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels([f"{ds} [{ROLE_ABBR[role_of[ds]]}]" for ds in datasets])
            ax.tick_params(axis="y", length=0)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle("DCD rank among 11 models on the 16-dataset subset", y=1.03)
    fig.text(0.5, -0.01,
             "Titles show Top-1 / Top-3 counts out of 16; shaded band = top-3. "
             "[S=Seen, U=Unseen, N=Novel]",
             ha="center", va="top", fontsize=8, color="#444444")
    save(fig, "fig3_dcd_rank_strip_16_v2", out_dir)


def figure4_role_dot(records: Path, out_dir: Path) -> None:
    rows = read_csv(records / "table3_roles_16.csv")
    roles = ROLE_ORDER + ["All"]
    data = {}
    for r in rows:
        if r["role"] not in roles or r["metric"] not in METRICS:
            continue
        data.setdefault(r["model"], {}).setdefault(r["role"], {})[r["metric"]] = (
            fnum(r["mean"]), fnum(r["std"]), int(r["n_datasets"])
        )
    models = ["DCD (fixed3)", "SACM official", "SAM-OCTA", "U-Net"]
    offsets = {
        "DCD (fixed3)": -0.04,
        "SACM official": -0.22,
        "SAM-OCTA": 0.22,
        "U-Net": 0.44,
    }
    role_n = {role: data["DCD (fixed3)"][role]["dice"][2] for role in roles}

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True)
    axes = axes.ravel()
    xbase = np.arange(len(roles))
    for ax, metric in zip(axes, METRICS):
        for model in models:
            if model not in data:
                continue
            scale = 100.0 if metric != "hd95" else 1.0
            means = [data[model][role][metric][0] * scale for role in roles]
            stds = [data[model][role][metric][1] * scale for role in roles]
            x = xbase + offsets.get(model, 0.0)
            if model == "DCD (fixed3)":
                ax.errorbar(x, means, yerr=stds, fmt="o", ms=7,
                            color=DCD_COLOR, ecolor=DCD_COLOR, elinewidth=1.5,
                            capsize=3, zorder=4, label=model)
            else:
                ax.errorbar(x, means, yerr=stds, fmt=BASE_MARKERS[model], ms=4.5,
                            color=BASE_COLORS[model], alpha=0.9,
                            ecolor=BASE_COLORS[model], elinewidth=1.0,
                            capsize=2, zorder=3, label=model)
        ax.set_xticks(xbase)
        ax.set_xticklabels([f"{role}\n(n={role_n[role]})" for role in roles])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylabel("%" if metric != "hd95" else "pixels")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Role generalization on the 16-dataset subset (4 seeds; mean ± std)", y=1.07)
    fig.tight_layout()
    save(fig, "fig4_role_dot_16_v2", out_dir)


def figure5_ablation_forest(records: Path, out_dir: Path) -> None:
    rows = read_csv(records / "table6c_ablation_paired_deltas_16.csv")
    delta = {
        (r["arm"], r["metric"]): (
            fnum(r["mean_dcd_advantage"]),
            fnum(r["ci95_low"]),
            fnum(r["ci95_high"]),
        )
        for r in rows
    }
    arms = ["official", "native_control", "gating_fusion", "nofusion", "notopo"]
    y = np.arange(len(arms))
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRICS):
        scale = 100.0 if metric != "hd95" else 1.0
        for yi, arm in enumerate(arms):
            mean, low, high = delta[(arm, metric)]
            mean, low, high = mean * scale, low * scale, high * scale
            if low > 0:
                color = "#2c7bb6"
            elif high < 0:
                color = "#d95f02"
            else:
                color = "#9e9e9e"
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]],
                        fmt="o", ms=6, color=color, ecolor=color,
                        elinewidth=1.5, capsize=3, zorder=3)
        ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABELS[a] for a in arms])
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("DCD advantage (pp)" if metric != "hd95" else "DCD advantage (px)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7bb6",
               markersize=6, label="DCD better (CI excludes 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d95f02",
               markersize=6, label="control better (CI excludes 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e",
               markersize=6, label="CI includes 0"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("DCD ablation: paired advantage over fixed3 on the 16-dataset subset", y=1.07)
    fig.text(0.5, -0.01,
             "Positive = DCD fixed3 better. HD95 uses reversed sign so positive is also better. "
             "Bars: 95% t-interval over 4 seeds.",
             ha="center", va="top", fontsize=8, color="#444444")
    fig.tight_layout()
    save(fig, "fig5_ablation_forest_16_v2", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    style()
    figure3_rank_strip(args.records, args.out)
    figure4_role_dot(args.records, args.out)
    figure5_ablation_forest(args.records, args.out)
    print(f"wrote v2 figures to {args.out}")
    for name in (
        "fig3_dcd_rank_strip_16_v2.pdf", "fig3_dcd_rank_strip_16_v2.png",
        "fig4_role_dot_16_v2.pdf", "fig4_role_dot_16_v2.png",
        "fig5_ablation_forest_16_v2.pdf", "fig5_ablation_forest_16_v2.png",
    ):
        print(args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
