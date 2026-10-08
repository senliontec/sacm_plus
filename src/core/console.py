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

# 宽表:训练侧 8 列 + F1/L + 全部 20 项 val 指标,time 放最后(元信息)
EPOCH_NAMES = (['ep', 'tr_loss', 'main', 'ds', 'iou', 'cl', 'topo', 'lr',
                'F1', 'L'] + [s for _, s in VAL_METRIC_SHORT] + ['time'])
EPOCH_WIDTHS = ([5, 8, 7, 7, 7, 7, 7, 9, 6, 8] + [6] * len(VAL_METRIC_SHORT) + [7])


def _cell(v, w):
    if v is None:
        return '—'.rjust(w)
    if isinstance(v, float) and v != v:  # NaN -> —
        return '—'.rjust(w)
    if isinstance(v, int):
        return f'{v:>{w}d}'
    if isinstance(v, str):
        return f'{v:>{w}}'
    return f'{v:.4f}'.rjust(w)


def epoch_header():
    """Wide table header: one column per metric."""
    return ' │ '.join(_cell(n, w) for n, w in zip(EPOCH_NAMES, EPOCH_WIDTHS))


def epoch_row(epoch, train_loss, comps, lr, dt, f1, vloss, val_metrics=None):
    """Wide data row: every metric in its own column ('—' when absent)."""
    vm = val_metrics or {}
    vals = [epoch, train_loss,
            comps.get('loss_main'), comps.get('loss_ds'), comps.get('loss_iou'),
            comps.get('loss_cl'), comps.get('loss_topo'), lr,
            f1, vloss]
    for key, _short in VAL_METRIC_SHORT:
        vals.append(vm.get(key))
    vals.append(f'{dt:.0f}s')
    return ' │ '.join(_cell(v, w) for v, w in zip(vals, EPOCH_WIDTHS))


def print_topo_row(metrics):
    """One compact line with all monitored topology-loss values."""
    parts = [f"{k[5:]}={v:.4f}" if v == v else f"{k[5:]}=—"
             for k, v in sorted(metrics.items()) if k.startswith('topo_')]
    if parts:
        console.print("  topo-loss: " + "  ".join(parts), style="dim cyan")


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
            parts.append(f"{short}={v:.3f}" if v == v else f"{short}=—")
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
