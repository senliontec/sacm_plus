# U-Net 移植审计（手工 pilot）

> **Post-A0 重审（2026-10-26）：** 本报告表格仍是修复前证据；A0 已 pass
> （pack v3），当前代码级 gate 状态见 [`../A0_REAUDIT.md`](../A0_REAUDIT.md)。

- 官方仓库：`3rd/Pytorch-UNet`
- 官方 commit：`21d7850f2af30a9695bbeea75f3136aa538cfc4a`
- 移植版本：`src/baseline_zoo/models/unet/` + `src/baseline_zoo/adapters/unet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 没有冻结 calibration pack，严格 07 只能输出 `<blocked>`；
> 本文是 provisional 手工 pilot，不进入主表，也不代表官方指标已复现。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-unet-01 | `UNet`/`DoubleConv`/`Down`/`Up`/`OutConv` 结构 | `unet/unet_model.py:5-92`; `unet/unet_parts.py:8-98` | 结构/层名逐字一致；否则 state_dict 无法互载 |
| S-unet-02 | state_dict、参数量、BN buffers | 同上 | key/shape/buffer 同一；否则同权重前向不成立 |
| S-unet-03 | forward 输出 logits `(B,1,H,W)` | `unet/unet_model.py:80-92` | 固定输入输出一致；否则 loss/grad 不可比 |
| S-unet-04 | BCEWithLogits + Dice；Dice 用 `reduce_batch_first=True`，对 B,H,W 求一个全局 Dice | `train.py:98-105`; `utils/dice_score.py:5-17,25-28` | 移植必须全局归约；改成逐图均值会在非均匀 batch 改变 loss |
| S-unet-05 | 单类分支 `n_classes=1` | `train.py:98-105`; `train.py:197` | 单类 BCE+前景 Dice；移植保持单类 |
| S-unet-06 | optimizer：RMSprop lr 1e-5/wd 1e-8/momentum .999，梯度裁剪 1.0 | `train.py:77-79,114-118` | 官方配方；移植改用统一 3-shot AdamW |
| S-unet-07 | scheduler：`ReduceLROnPlateau('max', patience=5)` | `train.py:79,143` | 每个 val 轮按 Dice 调 lr；移植用每 epoch cosine |
| S-unet-08 | 输入：`img_scale=0.5` 动态缩放，2D 图像保持单通道 | `train.py:35,197`; `utils/data_loading.py:64-91` | 官方可随分辨率变化；移植固定 512、RGB |
| S-unet-09 | 预处理：BICUBIC 图像/NEAREST mask、`/255`、mask 值映射 | `utils/data_loading.py:64-91` | 张量语义一致；否则输入尺度不同 |
| S-unet-10 | 训练无增强 | `utils/data_loading.py:93-112`; `train.py:47-55` | 官方无随机增强 |
| S-unet-11 | split：固定 generator 的随机 10% val，`workers=os.cpu_count()` | `train.py:47-55` | 官方数据/RNG 语义；移植为 18/6 协议 |
| S-unet-12 | val：`evaluate()` 每若干 step，sigmoid>0.5，逐图 Dice | `train.py:127-151`; `evaluate.py:23-34` | 逐图指标；移植 val Dice 选择 |
| S-unet-13 | checkpoint：每个 epoch 保存（由 `save_checkpoint` 控制） | `train.py:152-165` | 官方无 val-best 选择；移植按 val Dice |
| S-unet-14 | predict：bilinear 上采样 logits 到原图，再 sigmoid>0.5 | `predict.py:21-34` | 顺序不可交换；移植保持 logits 上采样 |
| S-unet-15 | 指标/导出：官方 Dice/F1 图级；项目统一 bool mask@GT | `evaluate.py:23-34`; `predict.py:31-34` | 导出分辨率/口径差异必须披露 |
| S-unet-16 | 来源/许可证：GPL-3.0、文件清单 | `src/baseline_zoo/models/unet/UPSTREAM.md:3-9` | 可追溯；否则 G0 失败 |

### 关键原文证据

```text
# 3rd/Pytorch-UNet/utils/dice_score.py:10-17
sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
inter = 2 * (input * target).sum(dim=sum_dim)
sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
dice = (inter + epsilon) / (sets_sum + epsilon)
return dice.mean()

# 3rd/Pytorch-UNet/utils/dice_score.py:25-28
def dice_loss(input, target, multiclass=False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
```

```text
# src/baseline_zoo/adapters/unet.py:23-35：docstring 声称 global，实际逐图
inter = 2.0 * (prob * target).sum(dim=(1, 2, 3))
sets_sum = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
dice = (inter + 1e-6) / (sets_sum + 1e-6)
return 1.0 - dice.mean()
```

模型文件 `diff -q 3rd/Pytorch-UNet/unet/unet_model.py src/baseline_zoo/models/unet/unet_model.py`
与 `unet_parts.py` 均无输出（逐字一致）；见 03 state/forward 探针。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-unet-01 | `equivalent` | `wrapper_change` | G1 | 官方模型文件 | 复制后逐字一致，仅包路径不同 | 无 | UPSTREAM | 01,02 |
| D-unet-02 | `equivalent` | `wrapper_change` | G1/G2 | 官方 state_dict | key/shape/buffer/参数相同，forward diff=0 | 无 | 探针 | 02,03 |
| D-unet-03 | **`differ`** | `bug` | **G2** | Dice 对 B,H,W 一次全局归约 | 逐图 Dice 后 `mean()` | 非均匀 batch 的 loss/grad 不同；见 03 | **未写，docstring 反而写错** | 04 |
| D-unet-04 | `intentional_deviation` | `protocol_adaptation` | G1/G4 | 默认 2 类 CE+multiclass Dice | 1 类 BCE+前景 Dice | 任务头不同但不影响二值推理 | deviations #4 | 05 |
| D-unet-05 | `intentional_deviation` | `official_recipe` | G3 | RMSprop 1e-5/5 epoch/batch1 | AdamW 1e-3/wd1e-4/batch4/cosine/200 | 训练轨迹完全不同，按项目预算解释 | deviations #3 | 06,07 |
| D-unet-06 | `intentional_deviation` | `protocol_adaptation` | G3/G4 | 0.5 动态缩放、单通道 | 固定 512、RGB | 输入分布/分辨率不同；单类头已披露 | deviations #2,#5 | 08,09 |
| D-unet-07 | `intentional_deviation` | `official_recipe` | G3 | 无增强 | flips+rot90 | 数据随机流不同 | deviations #6 | 10 |
| D-unet-08 | `intentional_deviation` | `protocol_adaptation` | G3/G5 | 随机 10% val、每 epoch 存盘 | 18/6 val、val-Dice best | split/选择语义不同 | deviations #7 | 11,13 |
| D-unet-09 | `differ` | `wrapper_change` | G3 | `workers=cpu_count()`，`shuffle=True`，全局 RNG | `workers=0`，`Random(seed+epoch)` | 数据顺序与浮点累加不同 | **未写** | 11 |
| D-unet-10 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | val 每若干 step，逐图 Dice | 每 epoch val Dice + 项目指标 | 选择点/评测频率不同 | deviations #7 | 12 |
| D-unet-11 | `equivalent` | `wrapper_change` | G4 | logits bilinear→sigmoid>0.5 | 同顺序 | 二值结果一致 | 适配器注释 | 14 |
| D-unet-12 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 官方 PNG/图级指标 | Bool mask@GT，项目指标 | 分辨率/指标口径不同 | deviations 间接说明 | 15 |
| D-unet-13 | `differ` | `wrapper_change` | G0 | 官方仓库路径 | UPSTREAM 仍写 `baselines/unet` | 文档路径过期，无代码影响 | UPSTREAM | 16 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-unet-01 | `covered_by_discrepancy` | D-01 |
| S-unet-02 | `preserved` | D-01, D-02 + 探针 |
| S-unet-03 | `preserved` | D-02 + forward diff=0 |
| S-unet-04 | `covered_by_discrepancy` | D-03 |
| S-unet-05 | `covered_by_discrepancy` | D-04 |
| S-unet-06 | `covered_by_discrepancy` | D-05 |
| S-unet-07 | `covered_by_discrepancy` | D-05 |
| S-unet-08 | `covered_by_discrepancy` | D-06 |
| S-unet-09 | `preserved` | D-06 + 预处理代码 |
| S-unet-10 | `covered_by_discrepancy` | D-07 |
| S-unet-11 | `covered_by_discrepancy` | D-08, D-09 |
| S-unet-12 | `covered_by_discrepancy` | D-10 |
| S-unet-13 | `covered_by_discrepancy` | D-08 |
| S-unet-14 | `preserved` | D-11 |
| S-unet-15 | `covered_by_discrepancy` | D-12 |
| S-unet-16 | `covered_by_discrepancy` | D-13 |

closure = 16/16。D-unet-03 是实质未披露偏差；D-unet-09 未披露 RNG/worker 差异。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；无下载、无训练，仅一次 forward/backward）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_unet_hetero_probe.py'
```

结果见 `03_numeric_probe.json`：

```text
state keys/shapes equal:            True / True
params official/port:               31,037,633 / 31,037,633
buffers official/port:              11,794 / 11,794
official loads port state:          True
forward max_abs_diff (B2,64x64):    0.0
homogeneous B=2 loss off/port:      1.3757247924804688 / 1.3757764101028442
homogeneous loss absdiff:           5.161762237548828e-05
homogeneous grad maxdiff:           5.734153091907501e-06
heterogeneous B=4 forward maxdiff:  0.0
per-image dice official/port:       完全相同（4 项逐项相等）
official global dice:               0.428108811378479
port mean dice:                     0.3160777688026428
dice diff:                          0.11203104257583618
loss official/port:                 1.26997971534729 / 1.3820106983184814
loss absdiff:                       0.1120309829711914
grad maxdiff:                       0.007257301360368729
repeat runs:                        两次完全一致（确定性 CPU）
```

结论：结构/state/forward 精确一致；**loss 公式不一致**。B=2 均匀 batch 的 5e-5 只是
掩盖效应，B=4 非均匀 batch 的 0.112 loss / 0.0073 grad 远超声明的
`atol=1e-4`（loss/grad）容差，因此 `G2=fail`，不是浮点噪声。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 5（`train.py:30`） | 200 元数据 | 不同，已披露 |
| batch | 1（`train.py:31`） | 4 | 不同，已披露 |
| input | `img_scale=0.5` 动态（`train.py:35`） | 固定 512 | 不同，已披露 |
| optimizer | RMSprop 1e-5/wd1e-8/mom .999（`:77-79`） | AdamW 1e-3/wd1e-4 | 不同，已披露 |
| scheduler | ReduceLROnPlateau（`:79,143`） | cosine/epoch | 不同，已披露 |
| grad clip | 1.0（`:114-118`） | 共享 runtime 默认值 | 需按 runtime 配置核对 |
| loss | BCE + 全局 Dice（`:98-105`） | BCE + 逐图 Dice | **bug，未披露** |
| train data | 官方数据目录，随机 10% val（`:47-55`） | 18/6 3-shot | 协议适配 |
| workers/GPU | cpu_count / 单进程 | 0 / 单卡 | worker 差异未披露 |
| seed | 无显式 seed（split generator 固定 0） | `seed+epoch` | RNG 语义不同 |
| selection | 每 epoch 存盘（`:152-165`） | 6 图 val Dice | 协议适配 |

结论：G3 不能 pass。优化器/预算差异本身是已披露的项目选择，但 loss 归约与 worker/RNG
差异会改变训练动力学，且前者未披露。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | scale=1 的 BICUBIC/NEAREST、`/255`（`evaluate.py:13-30`） | 512 BICUBIC、mask NEAREST、`/255` | 分辨率不同，已披露 |
| checkpoint | 官方每 epoch 权重 | port val-Dice best | 协议适配 |
| 输出 | logits→bilinear→原图→sigmoid>0.5（`predict.py:26-33`） | 同顺序（`adapters/unet.py:118-134`） | preserved |
| threshold | 0.5 | 0.5 | same |
| metric | 官方逐图 Dice | 项目 Dice/IoU/clDice/HD95 | 口径不同，已披露 |
| export | 官方 PNG | bool mask@GT（`runtime.py:774-786`） | 已披露 |
| 分辨率 | 原图（`predict.py:27`） | 原图 | same |

G4 可 pass（推理顺序和阈值有明确代码证据），但 G5 的协议适配必须与 G3 的 loss 问题分开。

---

## 07. 门禁结论（pre-fix snapshot）

> **Superseded:** this is the pre-fix gate snapshot. Post-A0 status is in
> [`../A0_REAUDIT.md`](../A0_REAUDIT.md).


```text
A0 = unverified            # 无冻结 calibration pack；正式 07 = <blocked>
G0 = pass                  # 本地 commit/文件/GPL 可追溯；UPSTREAM 路径过期
G1 = pass                  # 模型文件逐字一致；state_dict/shape/buffer 完全一致
G2 = fail                  # Dice 归约公式不同；非均匀 batch loss/grad 超容差
G3 = unverified            # optimizer/split 已披露，但 workers/RNG 与 loss 未闭合
G4 = pass                  # logits 上采样→sigmoid→0.5，有代码证据
G5 = pass                  # 18/6、混合域、val 选择均已披露
G6a = unverified           # 未跑 >=20 optimizer steps 的同状态路径
G6b = N/A                  # 无官方 U-Net checkpoint
G6c = unverified           # 未做全量训练/保存 masks/重算指标
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：修复或披露 D-unet-03；补做 20-step G6a；补 G6c；修正 UPSTREAM 路径；
把 workers/RNG 差异写入 deviations 或 context pack；冻结 A0 后才能给正式门禁。

## 最终判断

- 结构、state_dict、forward、推理上采样：**逐位一致**，有数值证据。
- **训练 loss 不是官方实现**：官方全局 batch Dice，移植逐图平均；这会让小 batch/稀疏
  mask 的梯度明显不同，当前 `G2=fail`。
- 因此不能标 `path_verified`，更不能声称官方指标复现；`claim_label=none`。


## Post-A0 re-audit (2026-10-26)

- A0: pass (`a0-cal-2026-10-26-v3`, `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`).
- Resolved: `D-unet-03` global Dice; after-fix probe loss/grad diff `0.0`. `D-unet-09` workers/RNG disclosed; `D-unet-13` UPSTREAM path fixed.
- Post-fix gates: G0 pass, G1 pass, **G2 pass**, G3 unverified, G4 pass, G5 pass, **G6a pass**, G6b N/A, G6c unverified.
- G6a evidence: `03_g6a_20step.json` — 20 steps, shared AdamW, real 3-shot batch hash, loss/grad/param/forward max diff `0.0`; the batch is 128/batch2 and this is path-only, not the official RMSprop recipe.
- G6c seed0 evidence: `03_g6c_seed0.md` / `03_g6c_seed0.json` — full 200-epoch train, 3580 masks, independent eval repeat with 0 mask-hash differences and identical `paper_metrics.csv`. This is one seed, so seed variance (1–4) is still missing.
- `main_table=blocked`: G3 recipe evidence and seeds 1–4 variance remain.
