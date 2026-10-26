# DSCNet 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet`
- 官方 commit：`0e35b2b6ea748a5e8ce7557ea3e580b9951471d6`
- 移植版本：`src/baseline_zoo/models/dscnet/` + `src/baseline_zoo/adapters/dscnet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 = `<blocked>`；本文是 provisional 手工 pilot。
> 模型/loss 有精确同 state 证据；数据增强、mean/std 和 worker/RNG 是未闭合的
> 训练/推理适配项。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-ds-01 | `DSCNet`/`DSConv`/`cross_loss` 文件 | `S3_DSCNet.py`, `S3_DSConv.py`, `S3_Loss.py` | 复制后数学一致 |
| S-ds-02 | state_dict、参数量、buffers | 同上 | key/shape 一致 |
| S-ds-03 | forward 输出 logits/offset 路径 | `S3_DSCNet.py`; `S3_DSConv.py` | 同权重输出一致 |
| S-ds-04 | DSConv device | `S3_DSConv.py` | 移植传 `f.device`，官方用构造期 device |
| S-ds-05 | soft BCE `cross_loss` | `S3_Loss.py` | loss/grad 一致 |
| S-ds-06 | AdamW lr1e-4, betas(.9,.95), wd 默认 .01 | `S3_Train_Process.py:105-110` | 优化器一致 |
| S-ds-07 | ReduceLROnPlateau 构造但不 step | `S3_Train_Process.py:111-117,129` | 实际恒定 lr |
| S-ds-08 | 训练/测试 mean/std 分别计算 | `S0_Main.py:35-36`; `S1_Pre_Getmeanstd.py:15-42` | 训练用 `Tr_`，推理用 `Te_` |
| S-ds-09 | 训练 dataloader：`Tr_Meanstd` + 随机 crop224 | `S3_Dataloader.py:41-54` | 输入尺度/裁剪一致 |
| S-ds-10 | MONAI 增强：Orientation/RandAffine/Elastic/Noise/Smooth | `S3_Data_Augumentation.py:15-64` | 同 RNG 下张量一致 |
| S-ds-11 | batch1、400 epoch、ROI224、从 200 epoch 验证 | `S0_Main.py:180-191` | 官方预算 |
| S-ds-12 | 从 epoch 200 用 **test** Dice 选择 best | `S3_Train_Process.py:124-143` | 官方 test 选择；移植改 val |
| S-ds-13 | patch 推理 stride112、四次核权重 | `S3_Train_Process.py:205-290` | 覆盖/权重一致 |
| S-ds-14 | 阈值 >0.5、Dice@原图 | `S3_Train_Process.py:293-315` | 阈值/指标一致 |
| S-ds-15 | 导出 NIfTI 概率图 | `S3_Train_Process.py:283-290` | 格式/分辨率 |
| S-ds-16 | 无 TTA | 官方 inference | 移植保持 |
| S-ds-17 | 来源：MIT LICENSE | `models/dscnet/LICENSE`, `UPSTREAM.md:3-10` | 可追溯 |
| S-ds-18 | 随机性：无显式 seed；DataLoader 默认 workers0 | `S3_Train_Process.py:99-103` | 无 seed 调用；workers 默认 0 |

### 关键原文证据

```text
# 3rd/DSCNet/.../DSCNet/S0_Main.py:35-36
Getmeanstd(args, args.Tr_Image_dir, args.Tr_Meanstd_name)
Getmeanstd(args, args.Te_Image_dir, args.Te_Meanstd_name)

# S3_Dataloader.py:41-44（训练）
mean, std = np.load(self.args.root_dir + self.args.Tr_Meanstd_name)
# S3_Train_Process.py:187-188（推理）
mean, std = np.load(args.root_dir + args.Te_Meanstd_name)
image = (image - mean) / std
```
官方训练用训练集统计量，**推理用测试集统计量**；移植统一用 18 图 train split 统计量。

```text
# 3rd/DSCNet/.../DSCNet/S3_Train_Process.py:208-209
map_kernal = 1 / ((a[0] - shape[0] // 2)**4 +
                  (a[1] - shape[1] // 2)**4)
```
中心处 `1/0`，会产生 inf/nan；移植改为有限 `1/(1+d^4)`。

```text
# S3_Train_Process.py:101-103
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
# 全文件搜索 manual_seed/random.seed/np.random.seed：无匹配；
# DataLoader 未传 num_workers，取默认 0。
```
官方没有显式种子；移植在外层 `seed_everything(args.seed)`。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-ds-01 | `equivalent` | `wrapper_change` | G1/G2 | 模型/DSConv/loss | 复制；仅相对 import | state/forward/loss/grad 精确一致 | UPSTREAM | 01,02,03,05 |
| D-ds-02 | `equivalent` | `bug_fix` | G2 | `DSC` 使用构造期 device | 使用 `f.device` | 数值不变，CPU/GPU 可搬移 | UPSTREAM | 04 |
| D-ds-03 | **`differ`** | `protocol_adaptation` | G3/G4 | 训练 `Tr_` 统计量；推理 `Te_` 统计量 | 统一 18 图 train 统计量，存 checkpoint | 推理输入分布不同；官方 test 统计量不可用 | UPSTREAM 写了 test mean/std；deviations 只写 train | 08,09 |
| D-ds-04 | `differ` | `wrapper_change` | G3 | MONAI Orientation/Affine/Elastic/Noise/Smooth | PIL flips/rot90/±20° + noise | 增强分布/RNG 不同 | deviations #1 | 10 |
| D-ds-05 | `differ` | `protocol_adaptation` | G3/G4 | NIfTI/SimpleITK I/O | PIL 灰度 PNG | 元数据/灰度范围可能不同 | deviations #1 | 09 |
| D-ds-06 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | test Dice 从 200 epoch 选 best | 6 图 val Dice | 移植避免 test 泄漏；选择点/指标不同 | deviations #3 | 11,12 |
| D-ds-07 | `equivalent` | `official_recipe` | G3 | 构造 ReduceLROnPlateau，step 被注释 | 不构造、恒定 lr | 实际 lr 轨迹相同 | deviations #4 | 07 |
| D-ds-08 | `intentional_deviation` | `bug_fix` | G4 | 四次核中心 1/0，边界循环可能漏最后 patch | `1/(1+d^4)` + 显式 last starts | 避免 inf/nan；覆盖更完整 | deviations #5 | 13 |
| D-ds-09 | `intentional_deviation` | `protocol_adaptation` | G4 | NIfTI 概率图 | GT bool PNG | 格式/分辨率不同 | deviations #6 | 15 |
| D-ds-10 | `equivalent` | `official_recipe` | G4 | 无 TTA | 无 TTA | same | deviations #7 | 16 |
| D-ds-11 | `unverified` | `wrapper_change` | G3 | 无显式 seed、DataLoader workers0 | 外层次序 seed、workers0 | 数据/RNG 未做同 seed 探针 | **未写** | 18 |
| D-ds-12 | `equivalent` | `wrapper_change` | G0 | MIT LICENSE | 复制 MIT | 可追溯 | UPSTREAM | 17 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-ds-01 | `preserved` | D-01 |
| S-ds-02 | `preserved` | D-01 |
| S-ds-03 | `preserved` | D-01 + forward probe |
| S-ds-04 | `covered_by_discrepancy` | D-02 |
| S-ds-05 | `preserved` | D-01 + loss/grad probe |
| S-ds-06 | `preserved` | adapter.py `make_optimizer` |
| S-ds-07 | `covered_by_discrepancy` | D-07 |
| S-ds-08 | `covered_by_discrepancy` | D-03 |
| S-ds-09 | `covered_by_discrepancy` | D-03, D-05 |
| S-ds-10 | `covered_by_discrepancy` | D-04 |
| S-ds-11 | `covered_by_discrepancy` | D-06 |
| S-ds-12 | `covered_by_discrepancy` | D-06 |
| S-ds-13 | `covered_by_discrepancy` | D-08 |
| S-ds-14 | `preserved` | adapter 阈值 >0.5 |
| S-ds-15 | `covered_by_discrepancy` | D-09 |
| S-ds-16 | `preserved` | D-10 |
| S-ds-17 | `covered_by_discrepancy` | D-12 |
| S-ds-18 | `unverified` | D-11 |

closure = 18/18；D-ds-11 仍是 unverified。G3 的主要未闭合项是 D-ds-03/D-ds-04/D-ds-11。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；无下载、无训练）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
state keys/shapes equal:        True / True
params official/port:           2,094,169 / 2,094,169
buffers official/port:          518 / 518
official loads port state:      True
forward maxdiff (B1,64x64):     0.0
loss official/port:             0.6880562901496887 / 0.6880562901496887
loss absdiff:                   0.0
grad maxdiff:                   1.30385160446167e-07
missing grads:                  []
both-grad-None params:          28（两侧同为未使用的 x/y DSConv 分支）
```

结论：模型/损失/梯度路径 `G2=pass`。28 个 `grad=None` 在官方与移植完全相同，
不是移植漏梯度。数据增强、mean/std、worker 不在该探针覆盖范围，故 G3 仍不能 pass。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 400（`S0_Main.py:186-191`） | 400 | same |
| batch | 1（`:183`） | 1 | same |
| input | ROI 224（`:181`） | 224 | same |
| optimizer | AdamW 1e-4, betas(.9,.95), wd .01 | 同 | same |
| scheduler | ReduceLROnPlateau 构造；step 注释 | 恒定 lr | 实际 same |
| loss | cross_loss soft BCE | 同 | same（探针） |
| mean/std | 训练 `Tr_`，推理 `Te_` | 统一 18 图 train | **不同** |
| transforms | MONAI 随机增强 | PIL 简化增强 | **不同** |
| train data | 官方 DRIVE train/test | 18/6 3-shot | 协议适配 |
| workers/GPU | workers0 / 单卡 | workers0 / 单卡 | same |
| seed | 无显式 | 外层 seed | 未验证 |
| selection | test Dice（200 epoch 后） | val Dice | 协议适配 |

G3 不能 pass：mean/std 和增强是实质差异，且没有同 seed 数据探针。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval preprocessing | NIfTI、`Te_Meanstd`、无增强 | PIL 灰度、train mean/std | **不同** |
| checkpoint | 官方 test-Dice best | val-Dice best | 协议适配 |
| patch 推理 | stride112、四次核 1/0 | stride112、有限四次核、last patch | bug_fix，已披露 |
| postprocess | 概率/归一化后 >0.5 | >0.5 | same |
| metric | 官方 Dice@原图 | 项目 Dice/IoU/clDice/HD95 | 口径不同 |
| export | NIfTI 概率 | GT bool PNG | 协议适配 |
| 分辨率 | 原图 | GT | 协议适配 |

G4 可 pass 的前提是把 mean/std 与 kernel 修复视为已披露适配；不能声称与官方
推理逐位相同。没有官方 checkpoint，G6b=N/A。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # MIT、本地 commit/文件可追溯
G1 = pass                  # state/shape/buffer 完全一致
G2 = pass                  # forward 0；loss 0；grad 1.3e-7；28 个双侧 None
G3 = unverified            # mean/std、MONAI→PIL、seed/worker 未闭合
G4 = pass                  # 推理路径与修复有代码证据（非逐位官方）
G5 = pass                  # 18/6、test→val 选择、导出/指标均披露
G6a = unverified           # 无 20-step 路径
G6b = N/A                  # 无官方 checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：明确 mean/std 的训练/推理口径（当前 adapter deviations 只写训练统计量）；
补增强/数据同 seed 探针；补 G6a/G6c；A0。没有官方权重，不能声称 checkpoint parity。

## 最终判断

- 模型/DSConv/loss：**精确一致**（forward 0，loss 0，grad 1.3e-7）。
- 训练/推理并非逐位官方：mean/std 口径、MONAI→PIL 增强、kernel 修复、test→val 选择
  都是实质适配，其中 mean/std 只在 UPSTREAM 完整披露。
- 当前 `G2=pass`，但 `G3=unverified`；只能作为 path-level 参考，不能进主表。

## Post-review fixes (2026-10-26)

- `--dsc-stride` greater than `--input-size` now raises; previously it silently left uncovered stripes as background.
- Test: `test_patch_prediction_rejects_stride_gap`.
