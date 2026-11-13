# JTFN 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/JTFN`
- 官方 commit：`5b23b242d37e58698258b05ca1ce75f6f9a03b8c`
- 移植版本：`src/baseline_zoo/models/jtfn/` + `src/baseline_zoo/adapters/jtfn.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 = `<blocked>`；本文是 provisional 手工 pilot。
> 模型/loss 有精确同 state 证据；学习率调度、增强和 worker/RNG 仍是未闭合项。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-jt-01 | `JTFN` 模型/解码器 | `models/jtfn.py:8-...` | 复制逐字一致 |
| S-jt-02 | `modules.py`/backbone (`base64`, `resbase32`) | `models/modules.py`; `models/backbone/*` | 复制一致 |
| S-jt-03 | forward：3 steps × mask/boundary × 多尺度输出 | `models/jtfn.py` | 24 个输出逐项一致 |
| S-jt-04 | `compute_objective` 多尺度 BCE（mask+boundary） | `models/jtfn.py` | loss/grad 一致 |
| S-jt-05 | Adam lr 1e-3, wd 5e-4, betas .9/.999, eps 1e-8 | `train.py:48-55` | 优化器一致 |
| S-jt-06 | StepLR 每 **iteration** step 500、gamma .9 | `train.py:62-63`; `:167-168` | 每 batch 更新 lr；移植按 epoch 近似 |
| S-jt-07 | ImageNet mean/std 归一化 | `data/drive.py:14-30` | 输入张量一致 |
| S-jt-08 | 训练增强：flips、任意 ±180°、ColorJitter | `data/drive.py:77-109` | 同 RNG 下增强一致 |
| S-jt-09 | 训练 crop 256/`img_mode=crop` | `config/train/JTFN_DRIVE.yaml:16-18`; `data/drive.py:44-61` | crop 顺序/大小一致 |
| S-jt-10 | boundary=`abs(mask-maxpool3)`≥0.5 | `tools/produce_boundary.py:18-24` | 边界目标一致 |
| S-jt-11 | val/test 都读 `test.txt` | `data/drive.py:18,146-159` | 官方选择在 test split |
| S-jt-12 | 每 epoch val F1 best 保存 | `train.py:100-124` | 官方 F1 选择 |
| S-jt-13 | DataLoader batch2、workers8、drop_last、DataParallel | `config/...:21-22`; `data/__init__.py:24-40`; `train.py:37-41` | 训练动力学 |
| S-jt-14 | test：`img_mode=same`、全分辨率、sigmoid>0.5、无 TTA | `test.py:60-95` | 上采样/裁剪/阈值一致 |
| S-jt-15 | 指标：F1/precision/recall/quality/completeness | `test.py:70-80`; `utils/Evaluator` | 官方口径 |
| S-jt-16 | 导出：原图二值 mask | `test.py`; `predict.py`（无） | 分辨率/格式 |
| S-jt-17 | 无官方 checkpoint | `UPSTREAM.md` | 只能 scratch 训练 |
| S-jt-18 | 来源/许可证 | `models/jtfn/UPSTREAM.md:7-66` | 有 URL/commit；无 LICENSE |
| S-jt-19 | 数据格式：DRIVE `.tif`/离线 boundary | `data/drive.py:121-140,152-159` | 项目 PNG 适配 |

### 关键原文证据

```text
# 3rd/JTFN/train.py:48-55,62-63
optimizer = optim.Adam(model.parameters(), lr=args.base_lr,
                       betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.gamma)

# 3rd/JTFN/train.py:160-168
loss = loss_dict['total_loss']
optimizer.zero_grad()
loss.backward()
optimizer.step()
if args.lr_update:
    scheduler.step()
```
`lr_step=500`（`config/train/JTFN_DRIVE.yaml:30`）是按 **batch** 计数。

```text
# 3rd/JTFN/data/drive.py:18
self.split = 'val' if split in ['val', 'test'] else 'train'
# data/drive.py:146-159
if self.split == 'train': meta_file = .../train.txt
elif self.split == 'val' or self.split == 'test': meta_file = .../test.txt
```
官方 val/test 共用 DRIVE test 列表，构成官方 test-based selection。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-jt-01 | `equivalent` | `wrapper_change` | G1 | 模型/modules/backbone | 仅改相对 import | 无行为变化 | UPSTREAM | 01,02 |
| D-jt-02 | `equivalent` | `wrapper_change` | G2 | 24 输出/多尺度 BCE | 复制 | forward diff=0，loss/grad 精确 | UPSTREAM | 03,04 |
| D-jt-03 | `equivalent` | `official_recipe` | G2/G3 | boundary 3×3 maxpool 差分 | 同公式（adapter `_boundary`） | 边界目标一致 | deviations #5 | 10 |
| D-jt-04 | `intentional_deviation` | `official_recipe` | G3 | Adam 1e-3/5e-4，1000 epoch，batch2 | 同（默认），可覆盖 epochs | 预算/优化器一致 | UPSTREAM | 05 |
| D-jt-05 | **`differ`** | `official_recipe` | G3 | StepLR 每 500 iteration | `step_size=max(1,round(500/steps_per_epoch))` 每 epoch | lr 轨迹不同；无法逐位复现 | deviations #3 | 06 |
| D-jt-06 | `differ` | `official_recipe` | G3 | 任意 ±180° 旋转、ColorJitter [1,2.1]/[.5,1.5] | 90°/±20°、brightness 0.8–1.4；RNG 不同 | 数据分布/顺序不同 | deviations #4 | 08 |
| D-jt-07 | `unverified` | `wrapper_change` | G3/G4 | torchvision RandomCrop（小图先 pad）；ImageNet norm | `load_rgb`；小图直接 BICUBIC resize 到 256；大图随机 crop | 小图几何处理不同；未见同张量探针 | 部分 | 07,09,19 |
| D-jt-08 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | val/test 共用 test.txt，F1 选择 | 18/6 val，Dice 选择 | 选择语义不同，移植无 test 泄漏 | deviations #2 | 11,12 |
| D-jt-09 | `differ` | `wrapper_change` | G3 | workers8、drop_last、DataParallel | workers0、单卡、共享 runtime | 数据顺序/BN 不同 | **未完整披露** | 13 |
| D-jt-10 | `intentional_deviation` | `wrapper_change` | G4 | 全分辨率 `same`，无 padding | pad 到 32 倍数，预测裁回 | 边界像素可能不同 | deviations #6 | 14 |
| D-jt-11 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 官方 F1/quality 等 | 项目 Dice/IoU/clDice/HD95，GT bool | 指标/导出不同 | UPSTREAM/deviations | 15,16 |
| D-jt-12 | `unverified` | `missing_asset` | G6b | 无官方 JTFN checkpoint | scratch 训练 | 无法 checkpoint parity | UPSTREAM | 17 |
| D-jt-13 | `differ` | `missing_asset` | G0 | 上游无 LICENSE 文件 | UPSTREAM 明示无 LICENSE | 再分发需作者许可 | UPSTREAM | 18 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-jt-01 | `preserved` | D-01 |
| S-jt-02 | `preserved` | D-01 |
| S-jt-03 | `preserved` | D-02 + 24 输出探针 |
| S-jt-04 | `preserved` | D-02 + loss/grad 探针 |
| S-jt-05 | `preserved` | D-04 |
| S-jt-06 | `covered_by_discrepancy` | D-05 |
| S-jt-07 | `covered_by_discrepancy` | D-07 |
| S-jt-08 | `covered_by_discrepancy` | D-06 |
| S-jt-09 | `covered_by_discrepancy` | D-07 |
| S-jt-10 | `preserved` | D-03 |
| S-jt-11 | `covered_by_discrepancy` | D-08 |
| S-jt-12 | `covered_by_discrepancy` | D-08 |
| S-jt-13 | `covered_by_discrepancy` | D-09 |
| S-jt-14 | `covered_by_discrepancy` | D-10 |
| S-jt-15 | `covered_by_discrepancy` | D-11 |
| S-jt-16 | `covered_by_discrepancy` | D-11 |
| S-jt-17 | `covered_by_discrepancy` | D-12 |
| S-jt-18 | `covered_by_discrepancy` | D-13 |
| S-jt-19 | `covered_by_discrepancy` | D-07 |

closure = 19/19。G3 的三项主要缺口是 D-jt-05/D-jt-06/D-jt-09。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；无下载、无训练）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
state keys/shapes equal:            True / True
params official/port:               41,926,369 / 41,926,369
buffers official/port:              25,264 / 25,264
official loads port state:          True
forward outputs:                    3 steps × 2 kinds × 4 scales = 24
all-24-output forward maxdiff:      0.0
loss official/port:                 4.177380084991455 / 4.177380084991455
loss absdiff:                       0.0
grad maxdiff:                       5.820766091346741e-11
missing grads:                      []
```

结论：模型和 loss 路径 `G2=pass`。探针没有覆盖数据增强/调度器，因此不能据此
声称官方训练可复现。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 1000（YAML `epochs`） | 1000（默认） | same |
| batch | 2 | 2 | same |
| input | 256 crop | 256 crop | same |
| optimizer | Adam 1e-3, wd5e-4 | 同 | same |
| scheduler | StepLR 每 500 iteration, gamma .9 | 每 epoch 近似 | **不同** |
| loss | JTFN 多尺度 BCE | 同 | same（探针） |
| train transform | 任意旋转/ColorJitter/flips | 简化增强 | **不同** |
| val/test | 官方 test.txt，F1 | 18/6 val，Dice | 协议适配 |
| workers/GPU | 8 / DataParallel | 0 / 单卡 | **未完整披露** |
| seed | 未显式 | 共享 runtime seed | 不同 |
| checkpoint | val F1 best（官方选 test） | val Dice best | 协议适配 |

G3 不能 pass：调度器和增强是实质差异，且没有同 RNG 数据探针；worker/GPU 差异未完整披露。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | `same` 全分辨率、ImageNet norm（`data/drive.py`） | 原图 norm，pad 32 | 协议/实现差异 |
| checkpoint | 官方 F1 best | port val-Dice best | 协议适配 |
| postprocess | sigmoid>0.5（`test.py:90-93`） | 同（`adapters/jtfn.py:194-196`） | preserved |
| threshold | 0.5 | 0.5 | same |
| metric | F1/precision/recall/quality | 项目 Dice/IoU/clDice/HD95 | 口径不同 |
| export | 原图二值 | GT bool | 协议适配 |
| 分辨率 | 原图 | GT | 协议适配 |

G4 可 pass（阈值/输出有代码证据，padding 已披露），但没有官方 checkpoint 输出。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # URL/commit/文件；无 LICENSE
G1 = pass                  # 复制一致，state/shape/buffer 完全一致
G2 = pass                  # 24 输出 forward 0 diff；loss 0 diff；grad 5.8e-11
G3 = unverified            # StepLR 位置、简化增强、workers/DataParallel 未闭合
G4 = pass                  # sigmoid/0.5/裁剪路径有代码证据
G5 = pass                  # 18/6、test→val 选择、导出/指标均披露
G6a = unverified           # 无 20-step 路径
G6b = N/A                  # 无官方 checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：把 D-jt-05/06/09 写全；补同 RNG 预处理/增强张量探针；补 G6a/G6c；
LICENSE；A0。

## 最终判断

- 模型、24 输出、loss、梯度：**精确一致**，这是强证据。
- 训练并非官方逐位复现：StepLR 按 batch→按 epoch、增强简化、RNG/worker 不同。
- 当前 `G2=pass`，但 `G3=unverified`；不能进主表或声称官方指标复现。

## Post-review fixes (2026-10-26)

- Added `--jtfn-eval-max-side` (default 1024): oversized eval images are downscaled before forward and predictions are upsampled to GT, avoiding full-resolution OOM on the 11 GiB host.
- This is a disclosed memory adaptation, not official full-resolution inference; high-resolution metrics may change.
- Test: `test_eval_downscales_oversized_image_and_restores_gt_shape`.
- Remaining: G3/G6a and the official full-resolution path.
