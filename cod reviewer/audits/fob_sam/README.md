# FoB-SAM 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/FoB_SAM`
- 官方 commit：`40d0e23d75f81f07bb7f34cb641db51ef490d025`（2026-05-31）
- 移植版本：`src/baseline_zoo/adapters/fob_sam.py` + `src/baseline_zoo/models/fob_sam/`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 = unverified。** 没有冻结 calibration pack；本报告是手工 pilot。
> `main_table=blocked`。官方评测协议与移植支持集协议不同，指标不可同表比较。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 |
|---|---|---|---|
| S-01 | `FewShotSeg`/`IDR`/`SPG`/`SPR`/`Head`/`MaskedAttention` 图 | `models/FoB.py:294-329`; port `model.py:294-329` | 结构与参数名一致 |
| S-02 | ResNet-101 + frozen SAM ViT-H；ResNet COCO 初始化 | `models/encoder.py`; `train.py:62-67`; `SAM.py:96-114` | state_dict/初始化一致 |
| S-03 | loss = `heatmap_loss*1000 + L2/10000 + rac_loss + foreground_loss` | `models/FoB.py:355-431` | loss/梯度一致 |
| S-04 | 39001 名义 iterations；实际 13×3000=39000；γ=0.98 / 3000 | `scripts/train_CHAOST2.sh:15-20`; `config.py:57-68`; `train.py:72-74,132-166` | 优化步数/LR 序列 |
| S-05 | Adam lr 1e-4，无 weight decay；ISIC 覆盖 lr 1e-5 | `train.py:70`; `config.py:75-79`; `train.py:101` | 优化器超参 |
| S-06 | 训练监督：`USE_GT=False` 时用 supervoxel 伪类；1-shot/1-query | `config.py:48-62`; `scripts/train_CHAOST2.sh:13`; `datasets.py:237-248` | 监督来源 |
| S-07 | 3D volume 切片 episode；每 volume 归一化 | `datasets.py:234-235,252-280`; `train.py:142-155` | 输入/采样 |
| S-08 | gamma 与 geometry 增强独立分支 | `datasets.py:303-313` | 增强语义 |
| S-09 | test support 来自 test fold 内的 held-out volume | `datasets.py:30-37,80-110`; `test.py:265-266` | 支持集协议 |
| S-10 | 只评测含标签切片；query label 仅作空前景 guard | `datasets.py:60-63`; `models/FoB.py:359,424-429` | 查询/阈值语义 |
| S-11 | SAM prompt 推理：best_pred_idx=0（非 ISIC），threshold 0.96，256×256 | `SAM.py:72-75,92-114`; `models/FoB.py:426` | prompt/掩码 |
| S-12 | 官方输出 256 volume mask；指标 per-class Dice/IoU | `test.py:283-315` | 导出/指标 |
| S-13 | 论文 vs 代码差异（RAC/ring/threshold/budget/loss 等） | `UPSTREAM.md:55-80` | 代码忠实并披露 |
| S-14 | 来源/许可证/权重 | `README.md:4-5,150-156`; `LICENSE` | 可追溯 |

---

## 02. 移植对照与差异

### 结论先行

**模型图、prompt 路径、损失、梯度：逐位一致。** 官方/移植 common state 688/688、
params 54,554,688、init maxdiff 0；单步 loss 7480.87548828125 相同、346 个梯度
全 0 diff；20-step loss/state 全 0 diff；SAM wrapper 的 mask IoU=1.0。
**训练和评测协议不是官方复现**：官方用 3D volume + supervoxel 伪标签
（`USE_GT=False`），test support 来自 test fold 的 held-out volume 的 3 个切片；
移植用 2D GT 对、训练集/结构 donor support，并评测所有 query 图。G5=fail。

### 关键源码证据

```text
# 3rd/FoB_SAM/config.py:48-54
min_size = 200
max_slices = 3
use_gt = False  # True - GT; False - supervoxel
test_label = [1, 4]
n_part = 3  # evaluation, i.e. 3 chunks
```
```text
# 3rd/FoB_SAM/dataloaders/datasets.py:30-37,60-63（support/query 都在 test fold）
self.image_dirs = [elem for ... if idx in self.FOLD[args['eval_fold']]]
self.support_dir = self.image_dirs[idx[args['supp_idx']]]
...
idx = lbl.sum(axis=(1, 2)) > 0
sample['image'] = torch.from_numpy(img[idx])
```
```text
# 3rd/FoB_SAM/test.py:265-266
support_sample = test_dataset.getSupport(
    label=label_val, all_slices=False, N=_config['n_part'])
```
```text
# src/baseline_zoo/adapters/fob_sam.py:304-313
 donor = self._support_dataset(name)
 pairs = [sample for sample in self._train_by_dataset[donor]
          if self._resize_mask(sample.mask_path).sum() > 0]
...
return pairs[:shots]
```
```text
# src/baseline_zoo/adapters/fob_sam.py:415-429（所有 query 图，支持集来自训练 split/donor）
support_images, support_masks = self._query_support(sample.dataset)
...
'query_label': torch.ones(1, SIZE, SIZE, device=device),
```

### 差异表

| id | verdict | gate | category | 官方行为 | 移植行为 | 影响 | 披露 |
|---|---|---|---|---|---|---|---|
| D-01 | `intentional_deviation` | G3 | `protocol_adaptation` | 3D volume、supervoxel 伪类（`USE_GT=False`） | 2D image pairs、GT masks | 监督信号/采样空间不同 | `deviations` 已写 |
| D-02 | `intentional_deviation` | G3 | `protocol_adaptation` | 3D 相邻切片 episode、同 volume | 2D 同域/ donor support | episode 语义不同 | 已写 |
| D-03 | `differ` | G5 | `protocol_adaptation` | support 来自 test fold 的 held-out volume 的 3 个切片 | 训练 split 同域 1-shot 或 donor（DRIVE/CrackTree200） | 支持集分布/难度不同，指标不可比 | 部分披露；官方 test-fold 协议未写 |
| D-04 | `differ` | G5 | `protocol_adaptation` | 只评测含标签切片，空前景不产生样本 | 评测所有 query 图；全 1 dummy label 绕过 guard | 空图 FP 进入指标 | **未写** |
| D-05 | `intentional_deviation` | G4 | `bug_fix` | 真实 query label 进入 forward 仅作 guard | all-ones dummy，避免测试 GT 泄漏 | 空前景行为不同 | 已写 |
| D-06 | `equivalent` | G1 | `wrapper_change` | `sin_pos` 和 NLL weight hard-code CUDA | 注册 buffer/可设备迁移 | 多 1 个 buffer；数值一致 | 已写 |
| D-07 | `equivalent` | G3/G5 | `official_recipe` | 论文与代码多处不一致（RAC/ring/0.96/budget 等） | 跟随代码而非论文 | 不能称为论文公式复现 | 已写 |
| D-08 | `equivalent` | G3 | `wrapper_change` | 39001 名义 steps，实际 39000；scheduler 每 iter | 13×3000=39000；scheduler 每 epoch | 步数/LR 序列等价 | 未见精确说明 |
| D-09 | `intentional_deviation` | G3 | `protocol_adaptation` | 每 volume 归一化；gamma/geometry 独立 | 每 image 归一化/灰度标准化；独立分支 | 输入分布不同 | 已写 |
| D-10 | `unverified` | G6b | `missing_asset` | HF 公开训练 checkpoint；本地未下载 | 无本地官方权重 | G6b 无法执行 | 已写 |
| D-11 | `differ` | G0 | `wrapper_change` | 上游 URL/commit 未写入 port 文档；路径 `baselines/FoB_SAM` 过期 | 本地 commit/license 可追溯 | 文档 provenance 不完整 | **未写** |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | state/forward probe |
| S-02 | `preserved` | D-06；state probe |
| S-03 | `preserved` | loss/grad probe |
| S-04 | `covered_by_discrepancy` | D-08 |
| S-05 | `preserved` | optimizer source；无 wd |
| S-06 | `covered_by_discrepancy` | D-01 |
| S-07 | `covered_by_discrepancy` | D-01, D-02, D-09 |
| S-08 | `covered_by_discrepancy` | D-09 |
| S-09 | `covered_by_discrepancy` | D-03 |
| S-10 | `covered_by_discrepancy` | D-04, D-05 |
| S-11 | `preserved` | SAM wrapper probe |
| S-12 | `covered_by_discrepancy` | D-03, D-04 |
| S-13 | `covered_by_discrepancy` | D-07 |
| S-14 | `covered_by_discrepancy` | D-10, D-11 |

closure = 14/14；D-03/D-04/D-11 未完整披露，G5 不能 pass。

---

## 03. 数值探针结果

官方用 `3rd/FoB_SAM/models/FoB.py` + 本地 ResNet-101 COCO 权重；移植用
`baseline_zoo.models.fob_sam`。固定 seed、确定性算法（NLL 无确定性实现，
`warn_only=True`）、TF32 off。batch=1，同一 support/query 张量。

```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/fob_probe.py
keys official/port/common: 688 / 689 / 688
only port: ['masked_attention.sin_pos']; shape mismatch []; init maxdiff 0.0
params 54,554,688 both; buffers 313/314; trainable 54,554,688 both
one-step loss official/port: 7480.87548828125 / 7480.87548828125 (diff 0.0)
parts prompt/RAC/fg: 7476.75 / 1.4317444562911987 / 2.693800449371338 both
grad tensors 346/346, grad maxdiff 0.0
20-step loss maxdiff 0.0 (7480.875 → 154.2859), final state maxdiff 0.0
```
```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/fob_sam_probe.py
preprocess_equal True; mask_equal True; mask IoU 1.0; mask sums 4729/4729
official SAM score 0.7191361784934998; mask shape 256×256
```

数值上模型与训练一步路径完全一致。G5 的失败来自**支持集/评测样本协议**，
不是模型或 loss 实现。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| 名义 steps | 39001，实际 39000 | 13×3000=39000 | equivalent |
| batch/GPU | 1 / 单卡 | 1 / 单卡 | same |
| workers | 16（`scripts/train_CHAOST2.sh:8,38`）；config 默认 0 | 0（`runtime.py:851`） | diff，RNG 流不同 |
| optimizer | Adam 1e-4，无 wd | Adam 1e-4，wd 0 | same |
| scheduler | MultiStepLR γ=.98，每 3000 iter | 每 3000-iter epoch，γ=.98 | 步数等价 |
| loss | heatmap×1000 + L2/10000 + RAC + fg | 同 | 数值一致 |
| supervision | supervoxel 伪标签（`USE_GT=False`） | GT mask | **materially different** |
| episode | 3D volume 相邻切片，1-shot/1-query | 2D image pair，1-shot/1-query | protocol adaptation |
| normalize | per volume mean/std | per image mean/std 或 ImageNet | diff，已披露 |
| augment | gamma/geometry 独立 | 同分支 | 语义一致 |
| data | CHAOST2/SABS/ISIC 3D | 18 图 6 域 2D + 20-dataset | protocol adaptation |
| selection | 手动 checkpoint/固定 snapshot | val Dice best | diff，已披露 |
| init | ResNet-101 COCO + SAM ViT-H | 同 | same |

**G3 结论：unverified。** GT vs supervoxel、2D vs 3D 是实质性监督差异；
即使模型/损失逐位一致，也不能称为官方训练配方复现。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | 3D volume 切片 + 每 volume 归一化（`datasets.py:43-47,80-86`） | 2D resize 256 + 每图归一化（`fob_sam.py:185-210,415-429`） | protocol adaptation |
| support | test fold 内 held-out volume，3 个含标签切片 | 训练 split 同域 1-shot 或 donor | **协议不同** |
| query | test fold volume，只取含标签切片 | 20-dataset 全部图像 | **协议不同** |
| query label | 真实 GT 仅用于空前景 guard | 全 1 dummy | 行为不同，已披露 |
| SAM prompt | best_pred_idx 0，threshold 0.96，256×256 | 同 | same |
| mask 输出 | 256 volume mask | 256→GT 双线性 | 导出不同，已披露 |
| metric | per-class Dice/IoU，volume 级 | 共享 image Dice/IoU/clDice/HD95 | protocol adaptation |
| checkpoint | HF 官方权重（本地无） | 本地训练权重 | G6b 不可做 |

模型推理与 prompt 生成在固定 support/query 下与官方一致；但官方 test-fold
支持集、3-slice 支持和 labeled-slice 评测没有复现，因此 **G5=fail**，不是
“等价但未校准”。

---

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # commit/Apache-2.0/本地来源可追溯；D-11 文档待修
G1  = pass                  # 688 common keys、params 一致、仅多 sin_pos buffer
G2  = pass                  # forward/loss/grad/20-step 全 0 diff；SAM IoU=1.0
G3  = unverified            # GT vs supervoxel、2D vs 3D 监督不同
G4  = pass                  # prompt/mask 路径有数值证据；协议差异归 G5
G5  = fail                  # test-fold support / 3-slice / labeled-slice 未复现
G6a = pass                  # 同权重同 episode 20-step loss/state 0 diff
G6b = N/A                   # 本地无官方 checkpoint；HF 权重未下载
G6c = unverified            # 未做全量训练/评测

main_table = blocked
claim_label = path_verified  # 仅指模型/loss/prompt 路径，不含官方 3D eval 协议
```

## 最终判断

- 模型、loss、prompt 和 20-step 优化路径与官方逐位一致，state/梯度证据充分。
- 不能把移植结果与 FoB 论文表直接比较：监督信号、support 来源、评测切片和
  指标口径都不同；G5=fail，不是“已披露的等价适配”。
- D-03/D-04 必须写进 adapter/context pack，并在论文中明确标注
  “2D GT + donor support”而非官方 “3D supervoxel + test-fold support”。
- 待办：1) 补 D-03/D-04/D-11；2) 冻结 A0；3) 获取官方 HF checkpoint 后补
  G6b；4) 跑 G6c 全量训练/评测。
