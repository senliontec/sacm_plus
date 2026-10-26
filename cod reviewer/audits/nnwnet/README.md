# nnWNet 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/nnWNet`
- 官方 commit：`cfcd2e09754fc832c3099a2ed39d86a90007dbcb`
- 移植版本：`src/baseline_zoo/models/nnwnet/` + `src/baseline_zoo/adapters/nnwnet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 = `<blocked>`；本文是 provisional 手工 pilot。
> 模型/state/forward 有精确证据；训练 loss/优化器来自 nnU-Net 外部实现，未做同状态
> loss/grad 探针，因此 G2 不能 pass。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-nnw-01 | `WNet2D` 结构、5 层 deep-supervision 输出 | `nnUNetTrainer_WNet2D.py:170-329` | 结构与输出数一致；否则 state/forward 不同 |
| S-nnw-02 | `InitWeights_He(1e-2)` 初始化 | `:351-374` | 同 seed 权重一致 |
| S-nnw-03 | trainer 构造签名/`unpack_dataset` | `:333-349` | 适配 nnU-Net 2.8.1；否则 snapshot/恢复异常 |
| S-nnw-04 | `initial_lr=1e-2`, 200 epochs, deep supervision | `:333-349` | 官方训练预算 |
| S-nnw-05 | loss/优化器继承 nnU-Net（DC+CE、SGD、AMP） | `nnunetv2` runtime；官方 trainer 不覆写 | 外部实现；不可仅凭静态代码断言 |
| S-nnw-06 | planner/preprocessing 看整个 dataset | nnU-Net 官方流程 | 规范化统计不得看 val；否则泄漏 |
| S-nnw-07 | 官方每数据集 5-fold CV | nnU-Net CLI `-fold`；adapter 自定义 | 与项目 18/6 单 fold 不同 |
| S-nnw-08 | batch/patch 由 plans 决定，不由 adapter 元数据 | nnU-Net planner/`configuration_manager` | 不能把 `train_batch_size=1` 当作有效 batch |
| S-nnw-09 | 内部 RNG/数据增强/AMP | nnU-Net 训练器 | 不可用外层 seed 保证 bitwise |
| S-nnw-10 | 数据 staging：RGB PNG，`NaturalImage2DIO` | `adapters/nnwnet.py:96-126` | 输入范围/通道正确 |
| S-nnw-11 | checkpoint：EMA pseudo-Dice on val | nnU-Net 官方 | 选择语义 |
| S-nnw-12 | inference：官方 `nnUNetPredictor`，fold0、gaussian、mirroring | `adapters/nnwnet.py:257-338` | 不得重写 TTA/滑窗 |
| S-nnw-13 | 输出 0/1 label，GT 分辨率 | 官方 predictor + `:330-338` | 导出格式/分辨率 |
| S-nnw-14 | 外部依赖 `nnunetv2==2.8.1`；`.safetensors` 拒绝 | `UPSTREAM.md:3-15`; `adapters/nnwnet.py:277-291` | 版本/格式不匹配会失败 |
| S-nnw-15 | `--resume` 不支持 | `adapters/nnwnet.py:49-68` | 重跑重新 staging/planning |
| S-nnw-16 | 来源/许可证 | `src/baseline_zoo/models/nnwnet/UPSTREAM.md` | URL 未记录、无 LICENSE |

### 关键原文证据

```text
# 3rd/nnWNet/nnUNetTrainer_WNet2D.py:333-349
class nnUNetTrainer_WNet2D(nnUNetTrainer):
    def __init__(..., unpack_dataset: bool = True, device=...):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True
        self.initial_lr = 1e-2
        self.num_epochs = 200
```

```python
# src/baseline_zoo/adapters/nnwnet.py:192-202：先 planner，后写 split
command = [... 'plan_and_preprocess_entrypoints', '-d', '501', '-c', '2d', ...]
subprocess.run(command, env=env, check=True, cwd=workdir)
self._write_splits(preprocessed, train_cases, val_cases)
```
planner/preprocessing 在 `_write_splits` 之前运行，因此 6 张 val 已进入
`imagesTr` 并参与规范化统计/planning。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-nnw-01 | `equivalent` | `wrapper_change` | G1 | `WNet2D` | 复制；仅 trainer 构造签名调整 | state/shape/buffer/5 输出 forward diff=0 | UPSTREAM | 01,02 |
| D-nnw-02 | `equivalent` | `wrapper_change` | G1 | `unpack_dataset` 参数传给 base | 删除该参数，匹配 nnU-Net 2.8.1 | 构造兼容；训练逻辑未改 | UPSTREAM | 03 |
| D-nnw-03 | **`fail`** | `protocol_adaptation` | G5 | planner 只看训练数据 | planner/preprocessing 看 24 例（含 6 val） | val 归一化统计泄漏；不可当 held-out | deviations #2 | 06 |
| D-nnw-04 | `intentional_deviation` | `protocol_adaptation` | G5 | 每数据集 5-fold CV | 单 pooled 18/6、fold0 | 数据/选择语义不同 | deviations #1 | 07 |
| D-nnw-05 | `differ` | `wrapper_change` | G3 | batch/patch 由 plans 决定 | 元数据写 batch1，但 trainer 不读取 | 实际 batch 未记录/未核对 | **未写** | 08 |
| D-nnw-06 | `unverified` | `official_recipe` | G3 | nnU-Net 内部 RNG/增强/AMP | 外层 seed；nnU-Net 另有 RNG | seed 不是 bitwise 保证 | deviations #4 | 09 |
| D-nnw-07 | `intentional_deviation` | `wrapper_change` | G3 | 支持 continue training | `--resume` 直接拒绝 | 重跑新 plan；不影响单次训练 | deviations #5 | 15 |
| D-nnw-08 | `equivalent` | `wrapper_change` | G4 | 官方 predictor/TTA/滑窗 | 直接调用 `nnUNetPredictor` | 未重写推理 | code | 12 |
| D-nnw-09 | `intentional_deviation` | `protocol_adaptation` | G4 | 官方输出 label map | GT bool PNG | 导出格式不同 | deviations #6 | 13 |
| D-nnw-10 | `differ` | `missing_asset` | G4/G6b | 官方 nnU-Net 2.x 可能写 `.safetensors` | 仅接受 `.pth`，否则 SystemExit | 资产格式不兼容 | UPSTREAM | 14 |
| D-nnw-11 | `differ` | `missing_asset` | G0 | 上游 URL/License | 未记录；注明待补 | 来源/许可不可核验 | UPSTREAM | 16 |
| D-nnw-12 | `unverified` | `missing_asset` | G2/G6a | 官方 loss/优化器 | 未做 loss/grad 探针 | 无 G2 完整证据 | — | 05,09 |
| D-nnw-13 | `intentional_deviation` | `protocol_adaptation` | G5 | 官方 natural-image reader | 24 例转 RGB PNG | 输入范围/通道不同但已披露 | deviations #7 | 10 |
| D-nnw-14 | `equivalent` | `official_recipe` | G3/G4 | EMA pseudo-Dice 选 best | 沿用官方行为（在 6 图 val 上） | 选择实现一致，数据语义不同 | deviations #3 | 11 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-nnw-01 | `preserved` | D-01 + 探针 |
| S-nnw-02 | `preserved` | D-01 |
| S-nnw-03 | `covered_by_discrepancy` | D-02 |
| S-nnw-04 | `preserved` | 官方 trainer 常量 + adapter 继承 |
| S-nnw-05 | `covered_by_discrepancy` | D-12 |
| S-nnw-06 | `covered_by_discrepancy` | D-03 |
| S-nnw-07 | `covered_by_discrepancy` | D-04 |
| S-nnw-08 | `covered_by_discrepancy` | D-05 |
| S-nnw-09 | `covered_by_discrepancy` | D-06, D-12 |
| S-nnw-10 | `covered_by_discrepancy` | D-13 |
| S-nnw-11 | `covered_by_discrepancy` | D-14 |
| S-nnw-12 | `preserved` | D-08 |
| S-nnw-13 | `covered_by_discrepancy` | D-09 |
| S-nnw-14 | `covered_by_discrepancy` | D-10 |
| S-nnw-15 | `covered_by_discrepancy` | D-07 |
| S-nnw-16 | `covered_by_discrepancy` | D-11 |

closure = 16/16。D-nnw-03 是协议 fail，不是普通 adaptation；D-nnw-05 未披露。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；只构造模型，不训练、不下载）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
state keys/shapes equal:        True / True
params official/port:           7,044,042 / 7,044,042
buffers official/port:          10,268 / 10,268
official loads port state:      True
forward outputs official/port:  5 / 5
all-output forward maxdiff:     0.0
official file imports:          True
loss/grad probe:                NOT RUN（loss/优化器继承 nnU-Net；无训练状态）
```

结论：`G1=pass`（模型/初始化结构层面）；**`G2=unverified`**，因为没有同状态
loss/grad 证据；不能用 forward diff=0 代替训练数值门禁。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 200（`:348`） | 200（默认可覆盖） | same |
| batch/patch | plans 决定 | plans 决定；adapter 元数据 batch=1 不生效 | **需以 plans 为准** |
| input | nnU-Net 2D plans | 512 staging，plans 实际 patch | 外部 planner |
| optimizer | nnU-Net SGD, initial_lr 1e-2, wd 3e-5 | 继承 | 外部实现 |
| scheduler | nnU-Net poly | 继承 | 外部实现 |
| loss | DC+CE + deep supervision | 继承 | 外部实现 |
| AMP | nnU-Net 默认 | 继承 | 外部实现 |
| transforms | nnU-Net augmentation | 继承 | 外部实现 |
| train data | 官方数据集 5-fold | 18/6 pooled，fold0 | 协议适配 |
| planner | 只看 train split | 看 24 例（含 val） | **G5 fail** |
| workers/GPU | nnU-Net 多进程/单卡 | 单卡；num_processes=1 | 近似 |
| seed | nnU-Net 内部 | 外层 seed + 内部 RNG | 不保证 bitwise |
| selection | EMA pseudo-Dice | 官方行为，6 图 val | 选择语义适配 |

G3 不能 pass：loss/optimizer 未做数值探针，batch 实际值未核对，seed 不可复现。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval preprocessing | nnU-Net plans 预处理 | 由官方 predictor 读取 plans | 外部实现 |
| checkpoint | 官方 fold `checkpoint_best.pth` | 找 `.pth` 的 best/final | 格式一致；safetensors 拒绝 |
| postprocess | sliding window + gaussian + mirroring | 官方 predictor 直接复用 | preserved |
| threshold | argmax/官方 export | 官方 export 后 `label>0` | 等价 |
| metric | 官方 pseudo-Dice/验证 | 项目 Dice/IoU/clDice/HD95 | 口径不同 |
| export | 0/1 label map | bool mask@GT PNG | 协议适配 |
| 分辨率 | 原图 | GT | 协议适配 |

推理路径复用官方 predictor，G4 可 pass（静态证据），但没有官方 checkpoint 的
固定输入矩阵输出，G6b 仍为 N/A。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # 本地 commit/文件；URL/LICENSE 缺失
G1 = pass                  # state/shape/buffer/5 输出 forward 精确一致
G2 = unverified            # 无 loss/grad 同状态探针
G3 = unverified            # batch/seed/内部 RNG 未闭合
G4 = pass                  # 官方 nnUNetPredictor/TTA 直接复用
G5 = fail                  # planner/preprocessing 看到 6 张 val（泄漏）
G6a = unverified           # 无 20-step 路径
G6b = N/A                  # 无官方 nnWNet checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：修复 val 泄漏（planner 前写 split 或仅用 train 统计）；记录 plans 的实际
batch/patch；补 loss/grad 或 20-step G6a；补 G6c；URL/LICENSE；A0。

## 最终判断

- 模型/state/forward 精确一致，官方 trainer 插件也被复用；这是本模型最强的证据。
- 但**协议上 val 参与了 planner/normalization**，不是 held-out；G5=fail。
- 训练 loss/优化器由 nnU-Net 外部实现承担，未做数值探针；G2=unverified。
- 不能进入主表，也不能声称官方指标复现。
