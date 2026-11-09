# CWSAM 移植审计（手工 pilot）

- 官方仓库：`3rd/CWSAM`（origin: `xypu98/CWSAM`）
- 官方 commit：`7bbb32ad3769ffc3f41f4361a074c55a7299fa27`
- 移植版本：`src/baseline_zoo/models/cwsam/` + `src/baseline_zoo/adapters/cwsam.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25
- prompt 语义：`supervised_prompt_free_sam`（空 sparse prompt + 学习到的 no-mask dense prompt；不是 oracle-prompted）

> **A0=unverified。** 没有冻结 calibration pack；本报告是手工 pilot，不是已校准审计。
> A0 通过前 G0–G6c 只用于定位问题，不能单独作为主表证据。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-01 | 官方 SAR config：5 类、ViT-B、num_classes=5、ignore_bg | `configs/...5cls...yaml:5,74,96-103` | 类数/backbone 改变会改输出头与可训练参数 |
| S-02 | class-weighted CE `[1.5,1,0.5,1.9,0.1]`；`criterionIOU` 定义但 backward 未用 | `models/sam.py:157-166,278-284`；config `:76-77` | 权重/类数错会改梯度；漏用/误用 IOU 项也会改 loss |
| S-03 | prompt-free：`sparse=empty`、dense=`no_mask_embed` 广播 | `models/sam.py:221-244` | 若用 GT prompt 则语义变 oracle；空 prompt 实现错会改输出 |
| S-04 | per-block Adapter + prompt_generator；encoder 只训练这两类 | `train.py:374-378` | 冻结策略错误会训练到 base encoder |
| S-05 | 直接 resize 1024、ImageNet norm、mask NEAREST；训练无增强（flip 被注释） | `datasets/wrappers.py:80-89,131-155` | 预处理/增强不同会改变输入分布 |
| S-06 | AdamW lr=2e-4（config 未给 wd，AdamW 默认 0.01）；Cosine 200、eta_min=1e-7 | config `:104-109`；`train.py:296-300,356` | lr/scheduler 错会改收敛 |
| S-07 | DDP 2 GPU、每卡 batch 1（有效 batch 2）、workers 8、DistributedSampler | `README.md:32,36-38`；`train.py:35,99-101,359-365` | 单卡/有效 batch 不同会改训练动力学 |
| S-08 | 用 `result1`（eval_type=seg → S-measure）选 best | `train.py:411-425`；config `:53,128-132` | 用 val Dice 选择属协议改变 |
| S-09 | eval 直接 1024；COD 指标用 sigmoid 概率，mIoU 用 argmax over class channels | `test.py:157-159,178-190`；`datasets/wrappers.py:73-103` | 分辨率/阈值/通道决策不同会改边界 |
| S-10 | 官方指标：S-measure/E-measure/weighted-F/MAE + SegmentationMetric mIoU/OA 等 | `train.py:125-132,204-280`；config `:53` | 换成 Dice/IoU 等需明确披露 |
| S-11 | 导出 1024 RGB JPG；test 严格加载作者 checkpoint | `test.py:188-192,348-350` | 分辨率/导出格式/权重来源不同需披露 |
| S-12 | 官方训练权重 `save/.../model_epoch_best.pth` 与 base SAM ViT-B | README `:47`；config `:54` | 缺权重则不能复现官方资产；G6b=N/A |
| S-13 | 来源/许可证/文件清单 | `UPSTREAM.md:3-24`（模型目录） | URL/LICENSE 缺失必须明确记录 |
| S-14 | 项目协议：18 train / 6 val / 20 test roots、GT 导出、prompt-free 角色 | `cod reviewer/README.md:63-96`；`runtime.py:469-485,740-785` | 与官方 SAR train/test 不同，必须记录 |
| S-15 | 2-class 化后类别语义：背景/前景，输出 2 通道 | 移植 config/adapter | 5→2 类属 output space 改变，需显式披露 |

## 02. 移植对照与差异

| id | level/gate | category | verdict | documented | 官方 vs 移植 | 证据 | 数值影响 |
|---|---|---|---|---|---|---|---|
| D-01 | L1/G1/G3/G5 | `wrapper_change` | `intentional_deviation` | yes | 官方 5 类 ViT-B → 移植 2 类 ViT-B（ViT-L 可选） | `configs/models/cwsam.yaml`；`cwsam.py` deviations | matched 2-class ViT-B 下无结构差；ViT-L 通过 `--cwsam-encoder vit_l` |
| D-02 | L3/G3 | `official_recipe` | `intentional_deviation` | yes | 5 类加权 CE → 2 类 bg 0.1/fg 1.0 CE + 0.5 soft Dice | `models/sam.py`；`cwsam.py` `train_step` | 梯度权重不同；无法映射官方 5 类权重；可用 `--cwsam-dice-weight 0` 回退 |
| D-03 | L3/G3 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 2 GPU × batch1（有效 2）→ 单 GPU batch1 | `cwsam.py` deviations | 训练动力学不同；已披露、未量化 |
| D-04 | L3/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方 workers=8；移植 workers=0 | `cwsam.py` deviations | 数据加载顺序/RNG 流不同；已披露 |
| D-05 | L3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方按 S-measure 选 best；移植 6 图 val Dice | `train.py:411-425`；`cwsam.py:45-46` | best 权重选择不同 |
| D-06 | L3/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方无 grad checkpointing；移植启用 | `cwsam.py:47-48,80`；`UPSTREAM.md:20-21` | 只改显存，不改计算 |
| D-07 | L4/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 1024 COD 指标 + mIoU；移植 GT Dice/IoU/clDice/HD95 | `cwsam.py` deviations（指标口径条目） | 指标集与分辨率均改变；已完整披露，不可与官方数值横向比较 |
| D-08 | L4/G4 | `official_recipe` | `equivalent` | code | 官方 argmax over 2 logits ≡ 移植 softmax(fg)>0.5 | `test.py:181-183`；`cwsam.py:120-127` | 2 类决策边界精确等价；插值分辨率另见 D-07 |
| D-09 | L0/G6b | `missing_asset` | `unverified` | yes | 官方训练 checkpoint 缺失；移植自训 val Dice | `test.py:348-350`；README `:47`；`UPSTREAM.md:3-7` | G6b=N/A |
| D-10 | L0/G0 | `doc_drift` | `documented_limitation` | yes | URL/commit 已记录；上游快照无 LICENSE | `UPSTREAM.md:3-8` | 无数值影响；再分发前需补 license |
| D-11 | L3/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方 DistributedSampler + 无显式 seed；移植固定 seed 42 | `cwsam.py` deviations | 随机流不同；已披露、未量化 |
| D-12 | L2/G2 | `wrapper_change` | `equivalent` | probe | prompt-free infer + postprocess 路径 | `models/sam.py:221-276`；`models/cwsam/sam.py` | forward diff=0 |
| D-13 | L3/G3 | `official_recipe` | `equivalent` | code | ImageNet norm、无增强、mask NEAREST | `datasets/wrappers.py:80-89,142-150`；`cwsam.py:91-108` | 同预处理；未跑张量探针 |
| D-14 | L4/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 1024 RGB JPG；移植 GT bool mask | `test.py:188-192`；`cwsam.py:49-50` | 导出分辨率/格式不同 |

**关键证据（3–5 行）**

```text
# 官方 5 类加权 CE（models/sam.py:160-164）
if loss_weight is not None:
    pos_weight = torch.tensor(loss_weight,dtype=torch.float)
    self.criterionBCE = torch.nn.CrossEntropyLoss(pos_weight)
else:
    self.criterionBCE = torch.nn.CrossEntropyLoss()
# 官方配置 loss_weight=[1.5,1,0.5,1.9,0.1]；移植 2 类直接 neutral CE
```

```text
# 官方 prompt-free infer（models/sam.py:221-228）
sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), ...)
dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(...)
# 移植 model/cwsam/sam.py:221-244 保留同一路径
```

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `covered_by_discrepancy` | D-01 |
| S-02 | `covered_by_discrepancy` | D-02 |
| S-03 | `preserved` | D-12 |
| S-04 | `preserved` | `train.py:374-378`；`cwsam.py:75-80`；D-06（仅显存） |
| S-05 | `preserved` | D-13 |
| S-06 | `preserved` | 官方 AdamW 默认 wd=0.01；`cwsam.py:34-35,82-89` |
| S-07 | `covered_by_discrepancy` | D-03, D-04, D-11 |
| S-08 | `covered_by_discrepancy` | D-05 |
| S-09 | `covered_by_discrepancy` | D-07, D-08 |
| S-10 | `covered_by_discrepancy` | D-07 |
| S-11 | `covered_by_discrepancy` | D-09, D-14 |
| S-12 | `covered_by_discrepancy` | D-01, D-09 |
| S-13 | `covered_by_discrepancy` | D-10 |
| S-14 | `covered_by_discrepancy` | D-01, D-14 |
| S-15 | `covered_by_discrepancy` | D-01, D-02 |

closure = 15/15；D-03/D-04/D-07/D-11 已完整披露；G3/G4 仍因实际训练协议
与指标口径不同而不能声称官方复现。

## 03. 数值探针结果

详见 `03_numeric_probe.json`。探针为**随机初始化 + 官方 state_dict strict 复制到移植侧**；
只证明 matched-config 实现等价，不是官方权重/训练结果，也不是 G6a。

```text
命令（GPU 1，tsai 环境）：
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_cwsam.py
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_cwsam_grad_small.py
环境绕过：官方 models/mmseg/__init__.py 依赖 mmcv（未安装），仅在 harness 内用
synthetic `models` 包 + namespace package 绕过 import；官方文件未改。

state_dict keys/shapes equal:  True / True；missing=0 unexpected=0
param count official/port:     350,511,652 / 350,511,652
buffer count official/port:    1 / 1；strict copy=True
forward:                       prompt-free 1x3x1024x1024
image_encoder max_abs_diff:    0.0（shape 1x256x64x64）
logits shape / max_abs_diff:   1x2x1024x1024 / 0.0
one-step loss official/port:   0.6987874507904053 / 0.6987874507904053 (abs 0.0)
grad max_abs_diff:             2.9802322387695312e-08（166 个比较张量）
grad norm official/port:       1.6917170713762673 / 1.691717038307889
```

loss/grad 是 **reduced-width** 配置（2 类、inp 256、embed 256、depth 4、heads 4），
两侧都用 neutral CE。**未测 optimizer step；未用官方 class-weighted CE；
不是官方 5/10 类 ViT-B 配置。**

## 04. 训练配方审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 200 | 200 | same |
| batch | 1/GPU × 2 GPU = 2 | 1 × 1 GPU = 1 | differ，已披露 |
| input | 1024 | 1024 | same |
| optimizer | AdamW lr 2e-4, wd 0.01（默认） | AdamW lr 2e-4, wd 0.01 | same |
| scheduler | Cosine 200, eta_min 1e-7 | 同 | same（代码） |
| loss | 5 类 weighted CE | 2 类 bg 0.1/fg 1.0 CE + 0.5 soft Dice | intentional_deviation |
| backbone | ViT-B (embed 768/depth 12) | ViT-B 默认（ViT-L 可选） | intentional_deviation |
| transform | resize 1024、ImageNet、无增强 | 同 | same |
| data | SAR 5 类 train/test | 18/6 3-shot | protocol_adaptation |
| workers | 8 | 0 | differ，已披露 |
| GPUs | DDP 2 | 单卡 1 | differ，已披露 |
| seed | 无显式 seed | 42 | differ，已披露 |
| selection | S-measure best | 6 图 val Dice | protocol_adaptation |
| freeze | encoder 除 prompt_generator/Adapter | 同 | same |
| grad ckpt | 无 | 启用 | intentional（memory only） |

结论：**G3=unverified**——D-03/D-04/D-11 已进入 deviations；5→2 类、
bg 0.1/fg 1.0 CE + 0.5 soft Dice 已披露，但有效 batch、workers、seed
仍与官方不同，不能称为官方 SAR recipe 复现。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | resize 1024 + ImageNet | 同 | same |
| prompt | prompt-free（空 sparse + no_mask_embed） | 同 | same |
| checkpoint | 官方 `model_epoch_best.pth`（缺失） | 自训 best val Dice | missing asset |
| postprocess | 1024，argmax over channels | softmax(fg)>0.5，双线性回 GT | 2 类决策等价；分辨率不同 |
| threshold | argmax（无显式阈值） | 0.5（2 类等价） | equivalent |
| metric | S/E/wF/MAE + mIoU/OA @1024 | Dice/IoU/clDice/HD95 @GT | intentional_deviation，已披露 |
| export | 1024 RGB JPG | GT bool mask | intentional_deviation（已披露） |

**G4=unverified**：prompt-free、normalization、2 类决策规则与官方一致；但官方 eval 是
1024 上的 COD 指标 + mIoU，移植是 GT 上的 Dice/IoU/clDice/HD95。该指标口径差异现已
在 deviations 中完整披露，但仍不能与官方数值横向比较。

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # commit/remote 可追溯；URL/LICENSE 缺失已记录
G1  = pass                  # matched 2-class ViT-L: keys/shape/param/buffer/strict copy 全同
G2  = pass (scoped)         # forward 0 diff；reduced-width neutral CE loss/grad 3e-8；step 未测
G3  = unverified            # 有效 batch/workers/seed 未披露；5→2 类/权重 CE 属适配
G4  = unverified            # metric 集未披露
G5  = pass                  # 18/6/20、GT 导出、prompt-free 角色；2 类适配已披露
G6a = unverified            # 仅单步探针，未做 N>=20 optimizer steps
G6b = N/A                   # 官方训练 checkpoint 不存在
G6c = unverified            # 未全量训练/评测

main_table = blocked
claim_label = none
```

仍缺：官方训练权重、metric 集披露、DDP/有效 batch/workers/seed 披露、20-step G6a、
全量训练与评测产物。**不能声称官方 SAR 指标已复现。**

## 最终一行状态

`cwsam: G0=pass G1=pass G2=pass G3=unverified G4=unverified G5=pass G6a=unverified G6b=N/A G6c=unverified`

2026-11-09 update: `G6c=partial` — the checkpoint-remap + zero-init port
trains stably for seeds 0–2 at 20 epochs (test All Dice 0.4838 ± 0.0095,
IoU 0.3492, clDice 0.5661, HD95 82.85).  This is a diagnostic, not the
official 800-epoch/200-epoch protocol; the two initialisation deviations
must be disclosed in any paper comparison.

## Post-review fixes (2026-10-26)

- Added disclosure for official 2-GPU/batch-1/workers=8 vs single-GPU/batch-1/workers=0.
- Class-weighted CE remains a disclosed, unresolved high-risk adaptation.

## Live training audit and checkpoint-loading fix (2026-11-09)

### Confirmed bug: SAM no-mask embedding and PE buffer were never loaded

CWSAM moved `no_mask_embed` and `pe_layer` from SAM's `prompt_encoder` to the
top-level `SAM` module but kept loading the Meta SAM checkpoint with
`strict=False`.  Consequently:

- `no_mask_embed.weight` was re-initialised by PyTorch as N(0, 1), whereas
  SAM's trained embedding is N(0, 1/sqrt(256)) (checkpoint std 0.0527).
- `pe_layer.positional_encoding_gaussian_matrix` got a fresh random buffer
  instead of the checkpoint buffer.

The first effect is dominant: at initialisation the decoder logits reached
`min=-1372.1, max=685.4` (class-0 mean `-1043.2`, class-1 mean `+370.2`),
so softmax was saturated and training collapsed to all background.

The port now remaps both keys in `CWSAMAdapter.load_pretrained`
(`SAM_KEY_REMAP` / `_remap_sam_state`).  With the remap the same sample gives
`min=-23.9, max=8.5`, and the missing-key count drops from 182 to 180.

### Before vs after the key remap (seed 0, 20 epochs, val = 6 images)

| run | E1 train loss | best Dice | clDice | IoU | HD95 |
|---|---|---|---|---|---|
| pre-fix | 64.61 | 0.1297 | 0.1039 | 0.0805 | 364.1 |
| remap only | 4.08 | 0.4490 | 0.5385 | 0.3125 | 30.93 |

The pre-fix run hit an all-background collapse around E8–E12
(Dice 0.0003–0.0007, loss ~0.90).  The remap-only loss decreases monotonically
from 4.08 to 0.49 and val Dice from 0.079 to 0.449.

### Remaining seed instability → zero-init the new class readout

The remap alone was still seed-dependent, because `cls_upscaling` is new and
randomly initialised upstream; its output scale changes the initial logits:

| seed | E1 train loss | best val Dice | outcome |
|---|---|---|---|
| 0 | 4.08 | 0.4490 | learns |
| 1 | 11.01 | 0.1067 | all-background collapse |
| 2 | 5.91 | 0.4518 | learns |

The port therefore zero-initialises the final `cls_upscaling` projection
(`--cwsam-cls-init zero`, default; `official` restores upstream init).  This
starts every seed at logit 0 without changing the forward graph:

| seed | E1 train loss | best val Dice | clDice | IoU | HD95 |
|---|---|---|---|---|---|
| 0 | 1.93 | 0.4832 | 0.5933 | 0.3390 | 21.96 |
| 1 | 1.88 | 0.4741 | 0.5440 | 0.3413 | 44.05 |
| 2 | 3.05 | 0.4906 | 0.5776 | 0.3560 | 40.61 |
| 3 | 3.47 | 0.4991 | 0.5778 | 0.3532 | 33.09 |
| mean ± sd | — | 0.4868 ± 0.0106 | 0.5732 ± 0.0207 | 0.3474 ± 0.0085 | 34.93 ± 9.78 |

All seeds now converge to a similar operating point
(val Dice 0.487 ± 0.011, clDice 0.573 ± 0.021, IoU 0.347 ± 0.008); the
seed-1 collapse is gone.

### Other fix

`save_config`/`restore_config` now persist the binary loss weights
(`bg_weight`, `fg_weight`, `dice_weight`) and the readout initialisation
(`cls_init`); previously a resumed run silently fell back to defaults.

### Caveats

- Both fixes are deliberate numerical/initialisation deviations from the
  upstream training script, which would leave these modules randomly
  initialised.  They restore the intended SAM initialisation / a stable new
  readout and are recorded in `CWSAMAdapter.deviations`.  The forward graph is
  unchanged.  `--cwsam-sam-key-remap off` and `--cwsam-cls-init official`
  reproduce the upstream load/initialisation for a separate audit run.
- 20 epochs on the 6-image val split is a diagnostic, not paper evidence.
  The full 800-epoch run and the 20-dataset test evaluation must be rerun
  after these fixes.
- The upstream model code itself is unchanged and still matches
  `3rd/CWSAM/models/mmseg/models/sam/` (only imports/checkpointing differ).

### 20-dataset test evaluation (seed 0, 20 epochs, remap only, official readout init)

| role | run | Dice | IoU | clDice | HD95 |
|---|---|---|---|---|---|
| All | pre-fix | 0.1052 | 0.0632 | 0.0885 | 330.15 |
| All | post-fix | 0.4500 | 0.3147 | 0.5191 | 84.70 |
| Seen | pre-fix | 0.1424 | 0.0878 | 0.1187 | 294.34 |
| Seen | post-fix | 0.4551 | 0.3171 | 0.5511 | 28.03 |
| Unseen | pre-fix | 0.1368 | 0.0816 | 0.1130 | 237.25 |
| Unseen | post-fix | 0.5823 | 0.4251 | 0.6464 | 84.76 |
| Novel | pre-fix | 0.0259 | 0.0142 | 0.0257 | 489.82 |
| Novel | post-fix | 0.2686 | 0.1652 | 0.3174 | 141.28 |

The remap-only checkpoint has higher Dice on **20/20 datasets** (deltas +0.13
to +0.68; e.g. OCTA-500 +0.676, DeepCrack +0.553, XCAD +0.524).  This rules out
a single-dataset accident.  These are 20-epoch diagnostics, not the 800-epoch
paper results; the full run must be rerun with this loader fix.

### 20-dataset test evaluation with remap + zero-init

| seed | role | Dice | IoU | clDice | HD95 |
|---|---|---|---|---|---|
| 0 | All | 0.4738 | 0.3360 | 0.5503 | 78.57 |
| 1 | All | 0.4927 | 0.3613 | 0.5796 | 87.92 |
| 2 | All | 0.4851 | 0.3504 | 0.5682 | 82.06 |
| mean ± sd | All | 0.4838 ± 0.0095 | 0.3492 ± 0.0127 | 0.5661 ± 0.0148 | 82.85 ± 4.73 |

The zero-init 3-seed mean test Dice (0.4839) is above the remap-only seed-0
value (0.4500), and the seed-1 collapse is gone.  Role breakdown for seed 0:
Seen 0.4792 / Unseen 0.6113 / Novel 0.2850 Dice.
