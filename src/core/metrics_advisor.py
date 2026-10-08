"""Online training advisor: metric trajectories -> optimization directions.

Consumes the same per-epoch metric dicts the Trainer already produces
(and writes to metrics_history.json) and emits concrete, rule-based
suggestions WHILE training — not after. Each rule encodes one
metric-pattern -> direction mapping; patterns and thresholds are
deliberately conservative (only fire on clear signals) and tunable
constants at the top.

Design notes:
  - Inputs: train_wm (this epoch), val_series (list of per-val-epoch
    dicts, chronologically), args (weights/flags).
  - Output: list of (severity, message) — severity in {'info','warn'}.
  - Pure function, no I/O — the Trainer prints; metrics_watch.py reuses
    the same rules offline.
"""

# ---------------------------------------------------------------- 阈值常量 --
STALL_TOLERANCE = 0.005      # 相对改善低于此值视为停滞
STALL_EPOCHS = 3             # 连续停滞次数
OVERFIT_EPOCHS = 2           # val 连续反升次数判定过拟合
IOU_HEAD_BAD = 0.1           # train/loss_iou 高于此且不降 = 选择器不可信
CL_LOSS_STUCK = 0.05         # train/loss_cl 停滞阈值
AUC_GAP = 0.05               # auc 与单点差距超过此值 = 阈值没选对


def _rel_change(prev, cur):
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / abs(prev)


def advise(history, args=None):
    """Return [(severity, message)] for the current training state.

    history: full per-epoch metric list (the same entries written to
        metrics_history.json — train-only epochs lack val/ keys).
    """
    out = []
    val_series = [h for h in history if any(k.startswith('val/') for k in h)]
    if len(val_series) < 2:
        return out

    def last(k):
        for h in reversed(val_series):
            if k in h:
                return h[k]
        return None

    def series(k):
        return [h[k] for h in history if k in h]

    # 1) 过拟合: val loss 连续反升而 train loss 仍在降
    vl = series('val/loss')
    tl = series('train/loss')
    if len(vl) >= OVERFIT_EPOCHS + 1 and len(tl) >= 2 and tl[-1] < tl[-2]:
        if all(vl[-(i + 1)] > vl[-(i + 2)] for i in range(OVERFIT_EPOCHS)):
            out.append(('warn', '过拟合信号: val/loss 连续上升而 train/loss 下降 — '
                               '考虑增强增广/降 decoder_lr/提前停止'))

    # 2) 拓扑项停滞: dice 还在涨,cldice 连续停滞
    if args is None or getattr(args, 'cl_dice_weight', 0) > 0 or getattr(args, 'topology_loss', 'none') != 'none':
        cd = series('val/cldice')
        dc = series('val/dice')
        if len(cd) >= STALL_EPOCHS + 1 and len(dc) >= 2:
            stalls = all(_rel_change(cd[-(i + 2)], cd[-(i + 1)]) < STALL_TOLERANCE
                         for i in range(STALL_EPOCHS))
            if stalls and _rel_change(dc[-2], dc[-1]) > STALL_TOLERANCE:
                out.append(('warn', '拓扑项停滞: cldice 连续 %d 个验证 epoch 无改善而 dice 仍在涨 — '
                                   '检查 cl_dice_weight/warmup,或换 --topology_loss' % STALL_EPOCHS))

    # 3) 欠分割 / 过分割(精度-召回分叉)
    p = last('val/precision')
    r = last('val/recall')
    if p is not None and r is not None:
        if p > 0.9 and r < 0.6:
            out.append(('warn', '欠分割信号: precision %.2f 高而 recall %.2f 低 — '
                               '细线漏检,方向: 条带适配器/降低 pred_threshold/增强细结构召回' % (p, r)))
        elif p < 0.7 and r > 0.85:
            out.append(('warn', '过分割信号: precision %.2f 低而 recall %.2f 高 — '
                               '方向: 提高 pred_threshold/加强背景监督' % (p, r)))

    # 4) IoU 选择器不可信
    li = last('train/loss_iou')
    if li is not None and li > IOU_HEAD_BAD:
        out.append(('warn', '选择器风险: train/loss_iou = %.3f 偏高 — '
                           'IoU head 可能选错 head,考虑 iou_loss_weight 或 head 容量' % li))

    # 5) 软骨架梯度问题: cl 损失停滞
    lc = series('train/loss_cl')
    if len(lc) >= STALL_EPOCHS + 1 and all(v > CL_LOSS_STUCK for v in lc[-(STALL_EPOCHS + 1):]):
        if all(_rel_change(lc[-(i + 2)], lc[-(i + 1)]) < STALL_TOLERANCE
               for i in range(STALL_EPOCHS)):
            out.append(('warn', 'clDice 损失停滞于 %.3f — 软骨架在细结构上梯度不足,'
                               '检查 cl_dice_warmup/ramp 节奏' % lc[-1]))

    # 6) 阈值鲁棒性: auc 明显高于单点
    da = last('val/dice_auc')
    dv = last('val/dice')
    if da is not None and dv is not None and da - dv > AUC_GAP:
        out.append(('info', '阈值未选对: dice_auc %.3f 比单点 dice %.3f 高 %.2f — '
                           '调 pred_threshold 可白捡收益' % (da, dv, da - dv)))

    return out
