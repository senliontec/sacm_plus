"""Shared rich console for training/eval terminal output.

Reference pattern: /home/dev/mywork/project/fencing-algs
(src/train/trainer.py) — a single Console instance, markup-coloured
messages, and a compact aligned per-epoch summary table
(ECG-SAM style, fixed widths joined by " │ ").
"""

from rich.console import Console
from rich.table import Table

console = Console()


# 紧凑行显示用的短名映射(每 epoch 一行全指标)
VAL_METRIC_SHORT = [
    ('dice', 'D'), ('iou', 'IoU'), ('precision', 'P'), ('recall', 'R'),
    ('sensitivity', 'Sen'), ('specificity', 'Spe'), ('accuracy', 'Acc'),
    ('mcc', 'MCC'), ('cldice', 'cD'), ('hd', 'HD'), ('hd95', 'H95'),
    ('assd', 'ASSD'), ('asd', 'ASD'), ('ravd', 'RAVD'), ('nsd', 'NSD'),
    ('betti', 'β'), ('dice_auc', 'dAUC'), ('cldice_auc', 'cAUC'),
    ('betti_matching', 'BM'), ('topograph_error', 'TopoE'),
]


def print_metrics_row(metrics, f1=None, loss=None):
    """One compact line with ALL val metrics (per-epoch display)."""
    parts = []
    if f1 is not None:
        parts.append(f"F1={f1:.4f}")
    if loss is not None:
        parts.append(f"L={loss:.4f}")
    for key, short in VAL_METRIC_SHORT:
        if key in metrics:
            v = metrics[key]
            parts.append(f"{short}={v:.3f}")
    console.print("  " + "  ".join(parts), style="dim cyan")


def print_metrics_table(metrics, title="val metrics"):
    """Full metric suite as a compact table: 4 name/value pairs per row.

    metrics: dict name -> float (NaN values print as 'nan').
    """
    t = Table(show_header=False, box=None, title=f"[bold]{title}[/bold]",
              title_justify="left", padding=(0, 1))
    items = list(metrics.items())
    for i in range(0, len(items), 4):
        row = []
        for name, val in items[i:i + 4]:
            row.append(f"[cyan]{name}[/cyan]")
            row.append(f"{val:.4f}" if val == val else "nan")
        t.add_row(*row)
    console.print(t)
