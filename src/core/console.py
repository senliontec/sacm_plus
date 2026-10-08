"""Shared rich console for training/eval terminal output.

Reference pattern: /home/dev/mywork/project/fencing-algs
(src/train/trainer.py) — a single Console instance, markup-coloured
messages, and a compact aligned per-epoch summary table
(ECG-SAM style, fixed widths joined by " │ ").
"""

from rich.console import Console
from rich.table import Table

console = Console()


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
