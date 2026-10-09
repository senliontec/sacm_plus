"""Shared rich console for training/eval terminal output.

Reference pattern: /home/dev/mywork/project/fencing-algs
(src/train/trainer.py) — a single Console instance, markup-coloured
messages, and a compact aligned per-epoch summary table
(ECG-SAM style, fixed widths joined by " │ ").
"""

from rich.console import Console
from rich.table import Table

# 宽表(31 列,~200 字符)永不折行:显式 width=500 + soft_wrap=False——
# 日志文件里每 epoch 恰为一行;宽屏终端单行;窄终端横向溢出可接受。
console = Console(soft_wrap=False, width=500)


# 紧凑行显示用的短名映射(每 epoch 一行全指标)。
# 命名约定: cDh/cAUCh = 硬 clDice(评测指标,骨架化后二值计算);
#           scl(训练损失列)= 软 clDice(可微损失,训练用)。
VAL_METRIC_SHORT = [
    ('dice', 'D'), ('iou', 'IoU'), ('precision', 'P'), ('recall', 'R'),
    ('sensitivity', 'Sen'), ('specificity', 'Spe'), ('accuracy', 'Acc'),
    ('mcc', 'MCC'), ('cldice', 'cDh'), ('hd', 'HD'), ('hd95', 'H95'),
    ('assd', 'ASSD'), ('asd', 'ASD'), ('ravd', 'RAVD'), ('nsd', 'NSD'),
    ('betti', 'β'), ('dice_auc', 'dAUC'), ('cldice_auc', 'cAh'),
    ('betti_matching', 'BM'), ('topograph_error', 'TE'),
]

# 宽表基础列(紧凑宽度):训练侧 8 列 + F1/L + 全部 20 项 val 指标
BASE_NAMES = (['ep', 'tr_loss', 'main', 'ds', 'iou', 'scl', 'topo', 'lr',
               'F1', 'L'] + [s for _, s in VAL_METRIC_SHORT])
BASE_WIDTHS = ([4, 7, 6, 6, 6, 6, 6, 6, 5, 6] + [5] * len(VAL_METRIC_SHORT))

# 方向箭头:↑ = 越大越好,↓ = 越小越好(ep/lr/time 为元信息,无箭头)
UP = {'F1', 'D', 'IoU', 'P', 'R', 'Sen', 'Spe', 'Acc', 'MCC', 'cDh', 'NSD',
      'dAUC', 'cAUCh'}
DOWN = {'tr_loss', 'main', 'ds', 'iou', 'scl', 'topo', 'L', 'HD', 'H95',
        'ASSD', 'ASD', 'RAVD', 'β', 'BM', 'TE'}

# 拓扑损失 3-4 字符缩写(表头下打印一次对照表)
TOPO_SHORT = {
    'betti': 'bt', 'ce_clce': 'cCE', 'ce_cldice': 'cCD', 'centerline_ce': 'clC',
    'composed_wasserstein': 'cW', 'decl': 'decl', 'dice_betti': 'dB',
    'dice_cldice': 'dCD', 'dice_topograph': 'dT', 'euler_refine': 'eul',
    'exact_topograph': 'xT', 'hutopo': 'hut', 'satloss': 'sat',
    'topo_cripser': 'tC', 'topo_tcripser': 'tT', 'topograph': 'tG',
    'warping': 'wrp', 'wasserstein': 'wst',
}


def _col_specs(topo_names):
    """(names, widths) with direction arrows;topo losses 全部 ↓。"""
    names, widths = [], []
    for n, w in zip(BASE_NAMES, BASE_WIDTHS):
        if n in UP:
            names.append(n + '↑'); widths.append(w + 1)
        elif n in DOWN:
            names.append(n + '↓'); widths.append(w + 1)
        else:
            names.append(n); widths.append(w)
    for n in topo_names:
        names.append(TOPO_SHORT.get(n, n[:4]) + '↓'); widths.append(5)
    names.append('time'); widths.append(5)
    return names, widths


def topo_legend(topo_names):
    """拓扑损失缩写对照(启动时打印一次)。"""
    return '  ' + '  '.join(f"{TOPO_SHORT.get(n, n[:4])}={n}" for n in topo_names)


def _fmt(v):
    """自适应精度:保证数值宽度 ≤ 5(大数自动降小数位;小于 0.001
    用科学计数法,避免学习率等小量被截成 0.000)。"""
    a = abs(v)
    if a >= 1000:
        return f'{v:.0f}'
    if a >= 100:
        return f'{v:.1f}'
    if a >= 10:
        return f'{v:.2f}'
    if a >= 0.001 or a == 0:
        return f'{v:.3f}'
    return f'{v:.0e}'


def _cell(v, w):
    # 表头与数值统一居中(Excel 风格);数值经 _fmt 保证不超列宽
    if v is None:
        return '—'.center(w)
    if isinstance(v, float) and v != v:  # NaN -> —
        return '—'.center(w)
    if isinstance(v, int):
        return f'{v:^{w}d}'
    if isinstance(v, str):
        return f'{v:^{w}}'
    return _fmt(v).center(w)


def _base_row_vals(epoch, train_loss, comps, lr, dt, f1, vloss, val_metrics):
    vm = val_metrics or {}
    return [epoch, train_loss,
            comps.get('loss_main'), comps.get('loss_ds'), comps.get('loss_iou'),
            comps.get('loss_cl'), comps.get('loss_topo'), lr,
            f1, vloss] + [vm.get(key) for key, _s in VAL_METRIC_SHORT] + \
           [f'{dt:.0f}s']


def epoch_header(topo_names=(), single=True):
    """表头(单行;分隔符带空格,~410 字符)。"""
    names, widths = _col_specs(topo_names)
    return [' │ '.join(_cell(n, w) for n, w in zip(names, widths))]


def epoch_row(epoch, train_loss, comps, lr, dt, f1, vloss, val_metrics=None,
              topo_names=(), single=True):
    """数据行(紧凑单行)。"""
    vm = val_metrics or {}
    vals = [epoch, train_loss,
            comps.get('loss_main'), comps.get('loss_ds'), comps.get('loss_iou'),
            comps.get('loss_cl'), comps.get('loss_topo'), lr,
            f1, vloss]
    for key, _short in VAL_METRIC_SHORT:
        vals.append(vm.get(key))
    for n in topo_names:
        vals.append(vm.get(f'topo_{n}'))
    vals.append(f'{dt:.0f}s')
    _names, widths = _col_specs(topo_names)
    return [' │ '.join(_cell(v, w) for v, w in zip(vals, widths))]


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
