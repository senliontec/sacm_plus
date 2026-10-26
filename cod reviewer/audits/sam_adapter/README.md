# SAM-Adapter 移植审计（手工 pilot）

> **Post-A0 重审（2026-10-26）：** 本报告表格仍是修复前证据；A0 已 pass
> （pack v3），当前代码级 gate 状态见 [`../A0_REAUDIT.md`](../A0_REAUDIT.md)。

- 官方仓库：`3rd/SAM-Adapter-PyTorch`（`tianrun-chen/SAM-Adapter-PyTorch`）
- 官方 commit：**不可用**（`.git` 指向缺失的 `../../.git/modules/3rd/SAM-Adapter-PyTorch`）
- 移植版本：`src/baseline_zoo/models/sam_adapter/` + `src/baseline_zoo/adapters/sam_adapter.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25
- prompt 语义：`supervised_prompt_free_sam`（空 sparse prompt + 学习到的 no-mask dense prompt；训练/评测都不使用 GT prompt）

> **A0=unverified。** 没有冻结 calibration pack；本报告是手工 pilot，不是已校准审计。
> A0 通过前 G0–G6c 只用于定位问题，不能单独作为主表证据。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-01 | ViT-L SAM + FFT high-pass prompt generator + 逐 block prompt 注入；prompt-free decoder | config `:62-92`；`models/sam.py:95-151` | 结构/参数量一致；换 prompt 生成方式会改输出 |
| S-02 | 从 SAM ViT-L `strict=False` 加载；prompt generator/位置编码新初始化 | `train.py:171-176`；`UPSTREAM.md:68-72` | 初始化错误会改变早期训练 |
| S-03 | loss = BCEWithLogits + `1-IoU`（config `loss: iou`） | config `:66`；`models/sam.py:144-146,248-254` | 漏 IoU 项或换 reduction 会改梯度 |
| S-04 | 冻结 image_encoder，只训练 prompt_generator、decoder、pe_layer、no_mask_embed | `train.py:174-176` | 误训 base encoder 会改参数量/收敛 |
| S-05 | AdamW lr=2e-4；config 未给 wd，AdamW 默认 wd=0.01；Cosine 20、eta_min=1e-7 | config `:93-98`；`train.py:104-108,159` | wd/lr/scheduler 错会改训练轨迹 |
| S-06 | 4 GPU DDP、每卡 batch 2（有效 8）、workers 8 | `README.md:47-50`；`train.py:16-19,33-35,162-168` | 单卡/有效 batch 不同会改动力学 |
| S-07 | resize 1024、ImageNet norm、mask NEAREST；训练 random flip p=0.5（尽管 YAML `augment: false`） | `datasets/wrappers.py:38-47,74-79,97-103` | flip/归一化不同会改输入分布 |
| S-08 | `data_norm`（sub/div 0.5）被传入但 executable train/eval 不使用；实际用 wrapper ImageNet | `train.py:151-155`；`test.py:40-47,111-115`；`datasets/wrappers.py:74-79` | 误加/漏加 data_norm 会改变输入尺度 |
| S-09 | prompt-free infer：`bs=1` 空 sparse + no_mask_embed，broadcast 到 image batch | `models/sam.py:193-215` | 用 GT prompt 会变 oracle；broadcast 错会 crash/改输出 |
| S-10 | postprocess 1024→input_size→original_size | `models/sam.py:217-246` | 分辨率/裁剪错会改边界 |
| S-11 | 按 `result1`（eval_type=cod → S-measure）选 best；每 epoch 验证/保存 | `train.py:205-226`；config `:98,104-105` | 用 val Dice 选择属协议改变 |
| S-12 | eval：sigmoid，COD S/E/wF/MAE @1024；`data_norm` 未应用 | `test.py:40-89`；config `:44` | 换指标/分辨率需披露 |
| S-13 | 作者 SAM-Adapter 权重缺失；test 严格加载作者 checkpoint | `test.py:107-109`；`UPSTREAM.md:84-89` | 缺权重则不能复现官方资产；G6b=N/A |
| S-14 | 来源/URL/MIT LICENSE；无 commit | `UPSTREAM.md:3-15` | commit 缺失使来源不能完整追溯 |
| S-15 | 项目协议：18 train / 6 val / 20 test roots、GT 导出、prompt-free 角色 | `cod reviewer/README.md:63-96`；`runtime.py:469-485,740-785` | 与官方 COD train/test 不同，必须记录 |

## 02. 移植对照与差异

| id | level/gate | category | verdict | documented | 官方 vs 移植 | 证据 | 数值影响 |
|---|---|---|---|---|---|---|---|
| D-01 | L0/G0 | `doc_drift` | `unverified` | yes | `.git` gitdir 指向缺失模块，无可用 commit | `UPSTREAM.md:10-11`；`3rd/SAM-Adapter-PyTorch/.git` | 无数值影响；G0 不能 pass |
| D-02 | L0/G6b | `missing_asset` | `unverified` | yes | 作者权重缺失；移植从 SAM ViT-L 重训 | `UPSTREAM.md:84-89`；`sam_adapter.py:39-42` | G6b=N/A |
| D-03 | L3/G3 | `protocol_adaptation` | `differ` | **no** | 官方 4 GPU × batch2（有效 8）；移植单 GPU batch2（有效 2） | `README.md:47-50`；config `:14`；`runtime.py:838-840,481-485` | 训练动力学不同；未量化 |
| D-04 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方 workers=8；移植 workers=0 | `train.py:33-35`；`runtime.py:481-485` | 数据顺序/RNG 流不同 |
| D-05 | L3/G3 | `official_recipe` | `differ` | **no / 文档写错** | 官方 AdamW 只传 lr=2e-4 → 默认 wd=0.01；移植 wd=0.0，UPSTREAM 却写 “no weight decay” | config `:93-96`；`utils.py:90-99`；`sam_adapter.py:34-36,123-126`；`runtime.py:845` | 实际正则化不同；G3 阻断 |
| D-06 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方无显式 seed；移植 seed=42 | `runtime.py:372-377,850` | 随机流不同；未量化 |
| D-07 | L3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 COD S-measure 选 best；移植 6 图 val Dice | `train.py:205-226`；`sam_adapter.py:43-46` | best 权重选择不同 |
| D-08 | L4/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 COD S/E/wF/MAE @1024；移植 Dice/IoU/clDice/HD95 @GT | `test.py:75-81`；`UPSTREAM.md:91-96`；`sam_adapter.py:47-48` | 指标集/分辨率不同；已明确披露 |
| D-09 | L3/G3 | `official_recipe` | `equivalent` | yes | YAML `augment:false` 但代码 flip p=0.5；移植保留 | `datasets/wrappers.py:97-103`；`sam_adapter.py:49-51,137-139` | 同增强；RNG 流另见 D-06 |
| D-10 | L3/G3 | `official_recipe` | `equivalent` | yes | `data_norm` 传入但不使用；移植用 wrapper ImageNet | `test.py:40-47,111-115`；`UPSTREAM.md:71-74`；`sam_adapter.py:52-54,146-149` | 输入尺度同 |
| D-11 | L2/G2 | `wrapper_change` | `equivalent` | yes | prompt-free `bs=1` 广播路径 | `models/sam.py:193-215`；`UPSTREAM.md:78-80`；`model.py:137-165` | 1024 ViT-L forward diff=0 |
| D-12 | L4/G4 | `wrapper_change` | `equivalent` | code | postprocess / sigmoid / threshold 0.5 | `models/sam.py:217-246`；`sam_adapter.py:181-193` | 路径同；分辨率适配见 D-08 |
| D-13 | L3/G3 | `official_recipe` | `equivalent` | yes | 20 epochs、AdamW 2e-4、Cosine 20 eta 1e-7 | config `:93-98`；`sam_adapter.py:34-36,123-130` | 配方同（wd 除外，见 D-05） |
| D-14 | L3/G3 | `wrapper_change` | `equivalent` | yes | grad checkpointing 默认关；移植默认关 | `UPSTREAM.md:55-58`；`sam_adapter.py:55-58,75` | 计算不变 |
| D-15 | L4/G4 | `missing_asset` | `unverified` | yes | 官方 test 严格加载作者权重；移植加载自训 best | `test.py:107-109`；`sam_adapter.py:39-42` | 不能验证作者权重；G6b=N/A |

**关键证据（3–5 行）**

```text
# 官方 AdamW 只收到 lr（configs/cod-sam-vit-l.yaml:93-96）
optimizer:
  name: adamw
  args:
    lr: 0.0002
# utils.py:96  Optimizer(param_list, **optimizer_spec['args'])
# => torch.optim.AdamW 默认 weight_decay=0.01；移植 official_weight_decay=0.0
```

```text
# 官方训练 flip 不受 augment 控制（datasets/wrappers.py:97-100）
# random filp
if random.random() < 0.5:
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
# 移植 sam_adapter.py:137-139 保留同一 p=0.5
```

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | D-11；探针 state_dict/forward |
| S-02 | `preserved` | D-11；`strict=False` 加载 |
| S-03 | `preserved` | 探针 BCE+IoU loss/grad |
| S-04 | `preserved` | `train.py:174-176`；`sam_adapter.py:116-121`；D-14（grad ckpt off） |
| S-05 | `covered_by_discrepancy` | D-05, D-13 |
| S-06 | `covered_by_discrepancy` | D-03, D-04, D-06 |
| S-07 | `preserved` | D-09 |
| S-08 | `preserved` | D-10 |
| S-09 | `preserved` | D-11 |
| S-10 | `preserved` | D-12 |
| S-11 | `covered_by_discrepancy` | D-07 |
| S-12 | `covered_by_discrepancy` | D-08 |
| S-13 | `covered_by_discrepancy` | D-02, D-15 |
| S-14 | `covered_by_discrepancy` | D-01 |
| S-15 | `covered_by_discrepancy` | D-07, D-08 |

closure = 15/15；D-03/D-04/D-05/D-06 未完整披露，G3 不能 pass。

## 03. 数值探针结果

详见 `03_numeric_probe.json`。探针为**随机初始化 + 官方 state_dict strict 复制到移植侧**；
只证明 matched-config 实现等价，不是作者权重复现，也不是 G6a。

```text
命令（GPU 1，tsai 环境）：
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_adapter.py
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_adapter_grad_small.py
环境绕过：官方 models/mmseg/__init__.py 依赖 mmcv（未安装），仅在 harness 内用 synthetic
`models` 包 + namespace package 绕过 import；官方文件未改。

state_dict keys/shapes equal:  True / True；missing=0 unexpected=0
param count official/port:     312,453,412 / 312,453,412
buffer count official/port:    1 / 1；strict copy=True
image_encoder max_abs_diff:    0.0（1x256x64x64）
infer shape / max_abs_diff:    1x1x1024x1024 / 0.0（prompt-free）
one-step loss official/port:   1.5202431678771973 / 1.5202431678771973 (abs 0.0)
grad max_abs_diff:             3.725290298461914e-09（129 个比较张量）
grad norm official/port:       0.5811172751352593 / 0.5811172620920639
```

loss/grad 是 **reduced-width** 配置（inp 256、embed 256、depth 4、heads 4，prompt-free）；
两侧都用官方 `BCEWithLogitsLoss + _iou_loss`。移植 model 本身没有 `_iou_loss`，
adapter 的 `train_step` 才加这一项，因此探针从官方 `models.sam` 导入该函数并在两侧同用。
**未测 optimizer step；未用作者权重；未测 ViT-L/1024 全宽 loss/grad。**

## 04. 训练配方审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 20 | 20 | same |
| batch | 2/GPU × 4 GPU = 8 | 2 × 1 GPU = 2 | **differ，未披露** |
| input | 1024 | 1024 | same |
| optimizer | AdamW lr 2e-4, wd 0.01（默认） | AdamW lr 2e-4, wd 0.0 | **differ，未披露/文档写错** |
| scheduler | Cosine 20, eta_min 1e-7 | 同 | same（代码） |
| loss | BCE + 1-IoU | 同（adapter 实现） | equivalent（探针 3.7e-9） |
| backbone | ViT-L, FFT prompt generator | 同 | same |
| transform | resize 1024, ImageNet, flip p=0.5, mask NEAREST | 同 | same |
| data | COD train/test | 18/6 3-shot | protocol_adaptation |
| workers | 8 | 0 | **differ，未披露** |
| GPUs | DDP 4 | 单卡 1 | **differ，未披露** |
| seed | 无显式 seed | 42 | **differ，未披露** |
| selection | COD S-measure best | 6 图 val Dice | protocol_adaptation |
| 初始化 | SAM ViT-L + 作者 adapter 权重 | SAM ViT-L 重训 | missing asset（已披露） |
| grad ckpt | 无 | 默认关 | same |

结论：**G3=unverified**——D-05（wd）是实质性配方差异且文档写反；D-03/D-04/D-06 未披露。
20 epoch/优化器/lr/cosine/冻结/loss 本身与官方一致。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | resize 1024 + ImageNet；mask NEAREST | 同 | same |
| prompt | prompt-free（空 sparse + no_mask_embed） | 同 | same |
| checkpoint | 作者 SAM-Adapter 权重（缺失），strict load | 自训 best val Dice | missing asset（已披露） |
| postprocess | 1024→original_size | 1024→GT | equivalent |
| threshold | sigmoid，COD 指标不阈值化 | 0.5 for Dice | 随指标协议改变 |
| metric | COD S/E/wF/MAE @1024 | Dice/IoU/clDice/HD95 @GT | intentional_deviation（已披露） |
| export | 1024 预测 | GT bool mask | intentional_deviation（已披露） |

**G4=pass**：eval transform、prompt-free 路径、postprocess 与官方一致；checkpoint 缺失、
指标集和 GT 导出差异都在 `UPSTREAM.md:84-96` 与 adapter deviations 中明确写出。
这不代表作者权重或官方指标已复现（G6b=N/A，G6c=unverified）。

## 07. 门禁结论（pre-fix snapshot）

> **Superseded:** this is the pre-fix gate snapshot. Post-A0 status is in
> [`../A0_REAUDIT.md`](../A0_REAUDIT.md).


```text
A0  = unverified            # calibration pack 未冻结
G0  = unverified            # 无可用 commit，来源不能完整追溯
G1  = pass                  # matched ViT-L 1024: keys/shape/param/buffer/strict copy 全同
G2  = pass (scoped)         # forward 0 diff；reduced-width BCE+IoU loss/grad 3.7e-9；step 未测
G3  = unverified            # wd 0.01→0.0、有效 batch、workers、seed 未披露
G4  = pass                  # eval 路径一致；metric/export/权重差异已披露
G5  = pass                  # 18/6/20、GT 导出、prompt-free 角色已披露
G6a = unverified            # 仅单步探针，未做 N>=20 optimizer steps
G6b = N/A                   # 作者 checkpoint 不在仓库
G6c = unverified            # 未全量训练/评测

main_table = blocked
claim_label = none
```

仍缺：可用 commit、作者权重、weight decay 修正/披露、有效 batch/workers/seed 披露、
20-step G6a、全量训练与评测产物。**不能声称官方指标已复现。**

## 最终一行状态

`sam_adapter: G0=unverified G1=pass G2=pass G3=unverified G4=pass G5=pass G6a=unverified G6b=N/A G6c=unverified`


## Post-A0 re-audit (2026-10-26)

- A0: pass (`a0-cal-2026-10-26-v3`, `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`).
- Resolved: `D-05` AdamW weight decay `0.01`; `D-03`/`D-04`/`D-06` 4-GPU, workers and seed differences disclosed.
- Post-fix gates: G0 unverified, G1 pass, G2 pass, G3 unverified, G4 pass, G5 pass, G6a unverified, G6b N/A, G6c unverified.
- `main_table=blocked`: G0 provenance and G3/G6a/G6c remain.
