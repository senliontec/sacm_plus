# CS2Net 移植审计（手工 pilot）

- 官方仓库：`3rd/CS-Net`
- 官方 commit：`f1f8bbb011ebabc39b3ce6b552b8a0d881e587f6`
- 移植版本：`src/baseline_zoo/models/cs2net/csnet.py` + `src/baseline_zoo/adapters/cs2net.py`
- 项目 commit：`4bd89806cf1230a4aec46a87f89851d49d2619b6`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 仓库还没有冻结 calibration pack，严格协议下 A0 未通过，
> 因此本报告是手工 pilot，不是“已校准审计”，不能直接作为主表证据。
> A0 通过前，下面的 G0–G6c 只用于定位问题，不作为最终门禁。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 |
|---|---|---|---|
| S-01 | 模型结构、初始化、`channels=3` / `classes=1` | `3rd/CS-Net/model/csnet.py:182-244` | 必须逐字一致 |
| S-02 | state_dict、参数量、BN buffers | 同上 | 键/shape/buffer 一致 |
| S-03 | forward 输出、sigmoid | `model/csnet.py:224-244` | 固定输入输出一致 |
| S-04 | loss：`MSELoss + dice_coeff_loss` | `train.py:58,76-78`; `utils/dice_loss_single_class.py` | 数值和梯度一致 |
| S-05 | Adam lr=1e-4, weight_decay=5e-4 | `train.py:23,57` | 一致 |
| S-06 | poly LR：`base*(1-epoch/epochs)^0.9` | `train.py:47-50,95` | 每个 epoch 的 lr 序列一致 |
| S-07 | train transform：center-square→512、±40°、ImageEnhance、RandomCrop、flip | `dataloader/drive.py:93-131` | 同 RNG 下张量一致 |
| S-08 | eval transform：center-square→512 | `dataloader/drive.py:99-126` | 张量一致 |
| S-09 | 数据：DRIVE train/test | `dataloader/drive.py:16-33`; `train.py:62,112` | 项目协议明确替代 |
| S-10 | DataLoader：batch=8, workers=2, shuffle=True；2×GPU DataParallel | `train.py:56,63` | 训练动力学需说明 |
| S-11 | checkpoint 选择、test/val 语义 | `train.py:100-128` | 官方 bug/泄漏需披露 |
| S-12 | predict：center_crop、Otsu、保存 | `predict.py:143-180`; `utils/misc.py:36-43` | 可运行等价 |
| S-13 | Otsu 阈值 | `utils/misc.py:36-43` | 阈值一致 |
| S-14 | 指标和导出 | `train.py:81-85,110-128`; `predict.py` | 项目指标口径需披露 |
| S-15 | 来源/许可证/文件清单 | `UPSTREAM.md` | 可追溯 |

---

## 02. 移植对照与差异

### 结论先行

**模型实现层面：一致。** 官方和移植的 `csnet.py` diff 为空；state_dict key/shape、
参数量、buffer 数完全一致；同一 `state_dict` 下 forward、loss、梯度、优化器 step、
LR 序列、DRIVE 预处理、Otsu 阈值均通过数值探针。

**完整训练层面：不是官方逐位复现。** 主要差异是单卡 vs 官方双卡 DataParallel、
DataLoader worker/RNG、官方 val/test 选择 bug 的处理，以及 18/6 混合数据协议。

### 差异表

| id | level | gate | category | 官方行为 | 移植行为 | 影响 | 披露 |
|---|---|---|---|---|---|---|---|
| D-01 | L1 | G1 | `wrapper_change` | `CSNet(classes=1, channels=3)` | 同结构，diff 为空 | 无 | `UPSTREAM.md` 已写 |
| D-02 | L1 | G1/G2 | `wrapper_change` | 官方整模型 pickle | 统一 state_dict | 无 forward 影响 | 已写 |
| D-03 | L2 | G2 | `wrapper_change` | 自定义 `DiceCoeff` 逐图循环 | 向量化 per-image Dice | loss/grad diff ≤1.5e-8 | 需在 deviations 明确 |
| D-04 | L3 | G3 | `official_recipe` | Adam 1e-4/5e-4；poly LR | 同；LambdaLR 平移一位 | LR 序列完全一致 | `BASELINE_AUDIT.md` 已写 |
| D-05 | L3 | G3/G5 | `protocol_adaptation` | DRIVE 20 张训练、test 评测 | 18/6 3-shot 混合域；6 张 val | 训练数据/验证语义不同 | `deviations` 已写 |
| D-06 | L3 | G3 | `protocol_adaptation` | `DataParallel(device_ids=[0,1])` + batch 8 | 单 GPU + batch 8 | BN 统计/训练轨迹不同 | **未写** |
| D-07 | L3 | G3 | `wrapper_change` | `num_workers=2`，全局 `random`，无 seed | `num_workers=0`，`Random(seed+epoch)` | 数据顺序/增强随机流不同 | **未写** |
| D-08 | L4 | G4 | `bug_fix` | `model_eval` 只返回第一张 test 图；best 条件不可达 | 6 张 val 全评；val Dice 选择 | 选择语义不同 | `deviations` 已写 |
| D-09 | L4 | G4 | `bug_fix` | `predict.py:164` 引用未定义 `center_crop`；`utils/model_eval.py` 缺失；`scipy.misc.imsave` 已移除 | 复用 center-square；skimage Otsu；GT 尺寸恢复 | 推理路径被重写 | `deviations` 已写 |
| D-10 | L4 | G4 | `wrapper_change` | cv2 Otsu | skimage Otsu | 测试图上阈值/二值完全一致 | 需在 deviations 明确 |
| D-11 | L4 | G4 | `protocol_adaptation` | 512 概率图 + 255 PNG | GT 分辨率 bool mask | 导出分辨率不同 | `deviations` 已写 |
| D-12 | L0 | G0 | `wrapper_change` | 官方 repo 路径 | `UPSTREAM.md` 仍写 `baselines/cs2net` | 文档路径过期，无数值影响 | **未写** |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | D-01 |
| S-02 | `preserved` | D-02 |
| S-03 | `preserved` | forward probe |
| S-04 | `covered_by_discrepancy` | D-03 |
| S-05 | `preserved` | D-04 |
| S-06 | `preserved` | D-04 / LR probe |
| S-07 | `covered_by_discrepancy` | D-05, D-07 |
| S-08 | `preserved` | eval probe |
| S-09 | `covered_by_discrepancy` | D-05 |
| S-10 | `covered_by_discrepancy` | D-05, D-06, D-07 |
| S-11 | `covered_by_discrepancy` | D-08 |
| S-12 | `covered_by_discrepancy` | D-09 |
| S-13 | `preserved` | D-10 |
| S-14 | `covered_by_discrepancy` | D-11 |
| S-15 | `covered_by_discrepancy` | D-12 |

closure = 15/15；但 D-06/D-07/D-12 是 **undocumented**，因此 G3/G0 不能算完整 pass。

---

## 03. 数值探针结果

详见 `03_numeric_probe.json`。关键结果：

```text
state_dict keys equal:              True
state_dict shapes equal:            True
param count official/port:          8,925,603 / 8,925,603
buffer count official/port:         6,164 / 6,164
init max_abs_diff:                  0.0
forward 512x512 max_abs_diff:       0.0
forward 256x256 max_abs_diff:       0.0
one-batch loss max_abs_diff:        ~0.0
one-batch grad max_abs_diff:        ~1.5e-8
one-step param max_abs_diff:        1.4e-5  (< optimizer_step_atol 1e-4)
20-step loss max_abs_diff:          9.5e-7
20-step param max_abs_diff:         6.6e-4  (cumulative; Adam amplification)
LR sequence max_abs_diff:           0.0
DRIVE train preprocess image diff:  0.0
DRIVE train preprocess mask diff:   0.0
DRIVE eval preprocess diff:         0.0
cv2 vs skimage Otsu:                127.0 vs 127.0 (binary equal)
inference binary mask:              equal
```

20-step 参数漂移是累计量，不是单步超差；单步参数差 1.4e-5 在协议容差内，
loss/grad 也远低于容差。这个漂移来自浮点归约顺序 + Adam 对小梯度的放大，
不是模型/损失公式不同。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 1000 | 1000 | same |
| batch size | 8 | 8 | same（但 GPU 数不同） |
| input | 512 | 512 | same |
| optimizer | Adam 1e-4, wd 5e-4 | Adam 1e-4, wd 5e-4 | same |
| scheduler | poly 0.9 per epoch | LambdaLR 平移一位 | same（LR 探针 0 diff） |
| loss | MSE + dice | MSE + vectorized dice | equivalent within 1e-7 |
| train data | DRIVE 20 | 18 张 3-shot 混合域 | protocol_adaptation |
| val/test | DRIVE test | 6 张 val + 20 test | protocol_adaptation |
| workers | 2 | 0 | **diff，未披露** |
| GPUs | DataParallel 2 卡 | 单卡 | **diff，未披露** |
| seed | 无显式 seed | 共享 runtime 固定 seed | protocol_adaptation |
| checkpoint | 每 100 epoch + 不可达 best 分支 | 每 epoch val Dice best | protocol_adaptation / bug_fix |

结论：**G3 不能 pass**，因为 D-06/D-07 还没有进入 deviations/ledger。
补上披露后，G3 可以按“单卡 + 项目 3-shot 协议”标为已披露适配，但仍不是官方双卡复现。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | center-square→512 | 同 | same（DRIVE 张量 diff=0） |
| 输出 | 512 概率图→PNG→cv2 Otsu | 512→skimage Otsu→GT 尺寸 | equivalent (Otsu 同阈值) |
| predict 入口 | `center_crop` 未定义，`utils.model_eval` 缺失 | 自包含 center-square + Otsu | bug_fix，已披露 |
| 输出分辨率 | 512 | GT 尺寸 | protocol_adaptation，已披露 |
| 指标 | accuracy/sensitivity | Dice/IoU/clDice/HD95 | protocol_adaptation，已披露 |
| checkpoint | 整模型 pickle | state_dict | protocol_adaptation，已披露 |

推理路径层面的实现一致；导出/指标属于项目协议适配。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # calibration pack 未冻结，严格协议直接 blocked
G0 = pass                  # 来源/许可证/文件可追溯；UPSTREAM 路径需修正
G1 = pass                  # diff 空、state_dict/shape/init 完全一致
G2 = pass                  # forward/loss/grad/one-step 数值通过
G3 = unverified            # 单卡 vs 双卡 DataParallel、workers/RNG 未披露
G4 = pass                  # 推理重写和导出差异已披露
G5 = pass                  # 18/6、混合域、val split 已披露
G6a = pass (single-GPU)    # 固定 batch 20-step loss/grad 路径通过
G6b = N/A                  # 无官方 checkpoint
G6c = unverified           # 未做全量训练/评测

main_table = blocked
claim_label = path_verified
```

## 最终判断

- **模型、损失、优化器、LR、DRIVE 预处理、Otsu、推理二值化：与官方一致**，有数值证据。
- **完整训练不是官方逐位复现**：官方双卡 DataParallel、workers=2、无 seed；移植单卡、
  workers=0、固定 seed；官方 val/test 选择逻辑有 bug，移植改为 val Dice。
- 在严格协议下，当前只能标 **path_verified**，不能标 `table_metric_reproduced`，
  也不能声称“官方指标已复现”。
- 下一步（按优先级）：
  1. 把 D-06/D-07 写入 `adapter.deviations` 或 context pack；
  2. 修正 `UPSTREAM.md` 的 `baselines/cs2net` 路径；
  3. 跑一次 G6c 全量训练/评测；
  4. 冻结 A0 calibration pack 后再跑正式审计。
