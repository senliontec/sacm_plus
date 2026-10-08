"""Metrics diagnostic report — the analysis step of the AI improvement loop.

Reads metrics_history.json (produced by the Trainer, next to
best_model.pth) and, optionally, results_summary.txt (produced by the
Evaluator), then answers the two questions the loop needs:

  1. 单实验: 哪个指标弱?(健康带 + 训练末期趋势)→ 可疑组件 + 建议改动
  2. 跨实验: 哪个组件贡献最大?(full vs no_X 单模块消融增量,按指标排名)

Usage:
    python scripts/metrics_report.py --history results/full/DRIVE/metrics_history.json
    python scripts/metrics_report.py --root results --out results/tables/report.md
      # --root scans <root>/<preset>/<dataset>/ and compares presets

The healthy bands are coarse literature reference values for curvilinear
segmentation (DRIVE-class difficulty); they are tunable constants at the
top of this file — treat "below band" as a pointer, not a verdict.
"""

import argparse
import json
import os

# ---------------------------------------------------------------- 健康参考带 --
# (val 指标, 粗粒度文献参考;低于下界才提示。改这里即可调整口径)
BANDS = {
    'val/dice': (0.75, 0.98),
    'val/f1': (0.75, 0.98),
    'val/iou': (0.60, 0.96),
    'val/precision': (0.70, 1.0),
    'val/recall': (0.70, 1.0),
    'val/sensitivity': (0.70, 1.0),
    'val/specificity': (0.95, 1.0),
    'val/accuracy': (0.95, 1.0),
    'val/mcc': (0.70, 1.0),
    'val/cldice': (0.72, 0.98),
    'val/nsd': (0.70, 1.0),
    'val/dice_auc': (0.75, 1.0),
    'val/cldice_auc': (0.72, 1.0),
    'val/ravd': (-0.30, 0.30),
    'val/betti': (0.0, 1.0),
    # 距离类没有绝对带(与分辨率相关):只做趋势分析
}

# ------------------------------------------------- 指标 → 可疑组件 / 建议改动 --
COMPONENTS = {
    'val/dice': ('主掩膜质量',
                 ['Stage-2 解码器 / 主损失 DiceBCE', 'CoarseToFineRefiner 掩膜回流',
                  '掩膜选择 (--selection iou vs gating)'],
                 '先看 train/loss_main 是否还在下降;若 dice 与 cldice 同弱,优先怀疑解码器整体;'
                 '若仅 dice 弱而 cldice 正常,查选择器(IoU head)是否选错 head'),
    'val/f1': ('主掩膜质量', ['阈值 0.0 的适用性', '主损失权重 (bce/dice)'],
               'f1 与 dice 口径近似,联动诊断;若 f1 明显低于 dice,查类别不平衡(bce_weight)'),
    'val/iou': ('主掩膜质量', ['Stage-2 解码器'], '与 dice 联动'),
    'val/precision': ('过分割', ['pred_threshold 上调', '背景抑制(负样本)'],
                     'precision 低 + recall 高 = 过分割:提阈值或增强背景监督'),
    'val/recall': ('欠分割', ['pred_threshold 下调', '细结构召回(条带适配器)'],
                   'recall 低 + precision 高 = 欠分割:细线漏检,查 A1/A2 条带适配器与增强'),
    'val/sensitivity': ('欠分割', ['同 recall'], '同 recall'),
    'val/specificity': ('背景纯净度', ['阈值', '背景主导样本的损失权重'], '曲线结构前景占比极小,specificity 波动多为阈值效应'),
    'val/accuracy': ('总体二值化', ['阈值'], '前景占比小时 accuracy 信息量低,参考意义弱于 dice/mcc'),
    'val/mcc': ('总体二值化(均衡口径)', ['阈值', '类别不平衡'], '比 accuracy 可靠;低说明二值化与 GT 结构性不符'),
    'val/cldice': ('拓扑连通性', ['soft-clDice 权重/warmup', 'AdapterFusionV2 层交互',
                                  'MultiDepthPath 语义连续性', '--topology_loss 选择'],
                   'clDice 弱而 dice 正常 = 连通性断裂:优先动拓扑损失(cl_dice_weight / --topology_loss)'),
    'val/betti': ('拓扑结构(0/1 维贝蒂数)', ['--topology_loss betti 家族', '连通性增强'],
                  'betti 误差大 = 分支/环结构错;纯计数指标,与 clDice 互补'),
    'val/hd': ('边界精度', ['MultiDepthPath(已声明补不了细边界)', 'CoarseToFineRefiner',
                            'pred_threshold'],
               'hd 是最大离群点,受单像素噪声支配;与 hd95 一起看'),
    'val/hd95': ('边界精度', ['MultiDepthPath', 'CoarseToFineRefiner', 'pred_threshold'],
                 'hd95 弱 + dice 正常 = 边界毛刺;若同时 recall 低则细线召回问题'),
    'val/assd': ('平均表面距离', ['CoarseToFineRefiner', '边界平滑'], '与 hd95 联动'),
    'val/asd': ('平均表面距离(对称)', ['同 assd'], '同 assd'),
    'val/ravd': ('体积偏差', ['过/欠分割平衡'], '正值=过分割,负值=欠分割;绝对值大先修 dice 再回看'),
    'val/nsd': ('表面贴合', ['边界细化路径', '阈值'], 'NSD 对细结构敏感;弱于 dice 较多时边界质量是主因'),
    'val/dice_auc': ('阈值鲁棒性', ['pred_threshold 选择', 'logit 校准'], 'auc 显著高于单点 dice = 阈值没选对'),
    'val/cldice_auc': ('阈值鲁棒性(拓扑)', ['同 dice_auc'], '同 dice_auc'),
    'val/betti_matching': ('持久同调匹配', ['--topology_loss betti/wasserstein',
                                            '结构级错误定位'], '秒/图级指标;误差大 = 拓扑特征错位'),
    'val/topograph_error': ('组件图临界邻居错误', ['--topology_loss topograph 家族', '连通性'],
                            '与 clDice/betti 三角验证拓扑质量'),
    'train/loss_cl': ('clDice 损失收敛', ['cl_dice_warmup/ramp 节奏', '软骨架梯度质量'],
                      'loss_cl 不降 = 软骨架在细结构上无梯度或 warmup 过晚'),
    'train/loss_ds': ('Stage-1 深监督收敛', ['deep_sup_weight', 'Stage-1 超网络'],
                      'loss_ds 高且不降 = 粗掩膜质量差,gating 排序不可信'),
    'train/loss_iou': ('IoU 头收敛', ['iou_loss_weight', 'IoU head 容量'],
                       'loss_iou 高 = 选择器不可信,推理 argmax 会选错 head'),
    'train/loss_main': ('主损失收敛', ['lr 分组', '过拟合(18 图 few-shot)'],
                        '早停平 = 过拟合风险 R1;查 val 曲线是否与 train 分叉'),
}

# 距离类只做趋势,不进带判
DISTANCE_METRICS = {'val/hd', 'val/hd95', 'val/assd', 'val/asd'}


def load_history(path):
    with open(path) as f:
        return json.load(f)


def val_epochs(history):
    return [h for h in history if 'val/f1' in h or any(k.startswith('val/') for k in h)]


def trend(values):
    """末期趋势:最后 1/3 的线性斜率除以首值(归一化);>0 上升,<0 下降。"""
    if len(values) < 2:
        return 0.0
    tail = values[max(1, len(values) - max(2, len(values) // 3)):]
    xs = list(range(len(tail)))
    n = len(tail)
    slope = (n * sum(x * y for x, y in zip(xs, tail)) - sum(xs) * sum(tail)) / \
        (n * sum(x * x for x in xs) - sum(xs) ** 2) if n > 1 else 0.0
    return slope / (abs(tail[0]) + 1e-12)


def single_report(data):
    """单实验诊断:健康带 + 趋势 → 弱项排序 + 建议。"""
    lines = []
    cfg = data.get('config', {})
    history = data.get('history', [])
    veps = val_epochs(history)
    if not veps:
        return ['(该实验没有验证 epoch 数据 — 用 --val_interval 1 训练或换带验证的实验)\n']

    final = {k: v for k, v in veps[-1].items() if k.startswith('val/')}
    lines.append(f"## 单实验诊断: {cfg.get('model', '?')} / preset={cfg.get('preset', '?')}")
    lines.append(f"- 总 epoch: {len(history)}, 验证 epoch: {len(veps)}, best_f1={data.get('best_f1')}")
    lines.append("")

    issues = []
    for k, v in sorted(final.items()):
        if k in DISTANCE_METRICS:
            continue
        lo, hi = BANDS.get(k, (None, None))
        if lo is not None and v < lo:
            issues.append((k, v, 'below_band', lo))
        elif hi is not None and v > hi:
            issues.append((k, v, 'above_band', hi))
    # 趋势(所有指标)
    for k in sorted(final):
        series = [h[k] for h in veps if k in h]
        if len(series) >= 3:
            t = trend(series)
            if t < -0.01:
                issues.append((k, series[-1], 'declining', t))

    if not issues:
        lines.append("✅ 全部指标落在健康参考带内且末期无下降趋势。")
    else:
        # 弱项排序:below_band > declining > above_band
        order = {'below_band': 0, 'declining': 1, 'above_band': 2}
        issues.sort(key=lambda x: (order.get(x[2], 9), x[1]))
        lines.append("### 弱项排序(可疑组件 → 建议)")
        lines.append("")
        seen = set()
        for k, v, kind, ref in issues:
            if k in seen:
                continue
            seen.add(k)
            comp = COMPONENTS.get(k)
            if comp is None:
                continue
            group, suspects, advice = comp
            if kind == 'below_band':
                lines.append(f"1. **{k} = {v:.4f}**(低于参考带下限 {ref:.2f})— {group}")
            elif kind == 'declining':
                lines.append(f"1. **{k} = {v:.4f}**(末期趋势下降 {ref:+.0%})— {group}")
            else:
                lines.append(f"1. **{k} = {v:.4f}**(高于参考带上限 {ref:.2f})— {group}")
            lines.append(f"   - 可疑组件: {'; '.join(suspects)}")
            lines.append(f"   - 建议: {advice}")
            lines.append("")
    return lines


def compare_report(root):
    """跨实验对比:每指标各预设终值表 + 单模块消融增量。"""
    rows = []
    for preset in sorted(os.listdir(root)):
        pdir = os.path.join(root, preset)
        if not os.path.isdir(pdir):
            continue
        for ds in sorted(os.listdir(pdir)):
            path = os.path.join(pdir, ds, 'metrics_history.json')
            if not os.path.exists(path):
                continue
            data = load_history(path)
            veps = val_epochs(data.get('history', []))
            if not veps:
                continue
            rows.append((preset, ds, {k: v for k, v in veps[-1].items()
                                      if k.startswith('val/')}))
    if len(rows) < 2:
        return ['(跨实验对比需要至少 2 个预设的 metrics_history.json)\n']

    lines = ['## 跨实验对比(消融矩阵)']
    keys = sorted({k for _, _, m in rows for k in m})
    by_preset = {}
    for preset, ds, m in rows:
        by_preset.setdefault(preset, []).append(m)

    lines.append('')
    lines.append('### 各预设指标终值(多数据集平均)')
    lines.append('')
    lines.append('| preset | ' + ' | '.join(keys) + ' |')
    lines.append('|---|' + '---|' * len(keys))
    for preset in sorted(by_preset):
        agg = {}
        for k in keys:
            vals = [m[k] for m in by_preset[preset] if k in m]
            agg[k] = sum(vals) / len(vals) if vals else float('nan')
        cells = []
        for k in keys:
            v = agg[k]
            cells.append(f'{v:.4f}' if v == v else '—')
        lines.append(f'| {preset} | ' + ' | '.join(cells) + ' |')

    # 单模块消融:full vs no_X(no_* 预设名即组件开关)
    lines.append('')
    lines.append('### 单模块消融增量(full − no_X,正 = 组件有贡献)')
    lines.append('')
    full = by_preset.get('full', [])
    if full:
        full_agg = {}
        for k in keys:
            vals = [m[k] for m in full if k in m]
            full_agg[k] = sum(vals) / len(vals) if vals else float('nan')
        deltas = []
        for preset in sorted(by_preset):
            if not preset.startswith('no_'):
                continue
            base = {}
            for k in keys:
                vals = [m[k] for m in by_preset[preset] if k in m]
                base[k] = sum(vals) / len(vals) if vals else float('nan')
            comp = preset[3:]
            for k in keys:
                d = full_agg.get(k, float('nan')) - base.get(k, float('nan'))
                if d == d:  # 非 NaN
                    deltas.append((comp, k, d))
        # 按 clDice/dice 增量排序输出每个组件的关键增量
        for comp in sorted({d[0] for d in deltas}):
            subs = [d for d in deltas if d[0] == comp]
            main = [f"{k}={d:+.4f}" for _, k, d in subs
                    if k in ('val/dice', 'val/cldice', 'val/hd95')
                    and abs(d) > 1e-9]
            if main:
                lines.append(f"- **{comp}**: {'; '.join(main)}")
    else:
        lines.append('(没有 full 预设的历史,无法计算消融增量)')
    lines.append('')
    return lines


def main():
    parser = argparse.ArgumentParser(description='指标诊断报告:最弱指标 → 可疑组件 → 建议改动')
    parser.add_argument('--history', type=str, default=None,
                        help='单个 metrics_history.json 路径')
    parser.add_argument('--root', type=str, default=None,
                        help='实验根目录(<root>/<preset>/<dataset>/,与 run_experiments 布局一致)')
    parser.add_argument('--out', type=str, default=None,
                        help='输出 markdown 文件(默认打印到终端)')
    args = parser.parse_args()

    lines = ['# 指标诊断报告', '']
    if args.history:
        lines += single_report(load_history(args.history))
    if args.root:
        lines += compare_report(args.root)
    if not args.history and not args.root:
        parser.error('需要 --history 或 --root 之一')

    text = '\n'.join(lines)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'报告已写入 {args.out}')
    else:
        print(text)


if __name__ == '__main__':
    main()
