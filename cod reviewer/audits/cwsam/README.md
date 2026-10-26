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
| D-01 | L1/G1/G3/G5 | `wrapper_change` | `intentional_deviation` | yes | 官方 5 类 ViT-B → 移植 2 类 ViT-L | config `:5,74,96-103`；`cwsam.py:32,38-44,58-59` | matched 2-class ViT-L 下无结构差 |
| D-02 | L3/G3 | `official_recipe` | `intentional_deviation` | yes | 5 类加权 CE → 2 类 neutral CE | `models/sam.py:157-166`；`cwsam.py:38-42,110-115` | 梯度权重不同；无法映射官方 5 类权重 |
| D-03 | L3/G3 | `protocol_adaptation` | `differ` | **no** | 官方 2 GPU × batch1（有效 2）→ 单 GPU batch1 | `README.md:32,36-38`；config `:17`；`train.py:99-101`；`cwsam.py:36` | 训练动力学不同；未量化 |
| D-04 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方 workers=8；移植 workers=0 | `train.py:99-101`；`runtime.py:481-485` | 数据加载顺序/RNG 流不同 |
| D-05 | L3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方按 S-measure 选 best；移植 6 图 val Dice | `train.py:411-425`；`cwsam.py:45-46` | best 权重选择不同 |
| D-06 | L3/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方无 grad checkpointing；移植启用 | `cwsam.py:47-48,80`；`UPSTREAM.md:20-21` | 只改显存，不改计算 |
| D-07 | L4/G4 | `protocol_adaptation` | `differ` | partial | 官方 1024 COD 指标 + mIoU；移植 GT Dice/IoU/clDice/HD95 | `train.py:125-132,204-280`；`cwsam.py:49-50,120-127` | 指标集与分辨率均改变；只披露了 GT 恢复 |
| D-08 | L4/G4 | `official_recipe` | `equivalent` | code | 官方 argmax over 2 logits ≡ 移植 softmax(fg)>0.5 | `test.py:181-183`；`cwsam.py:120-127` | 2 类决策边界精确等价；插值分辨率另见 D-07 |
| D-09 | L0/G6b | `missing_asset` | `unverified` | yes | 官方训练 checkpoint 缺失；移植自训 val Dice | `test.py:348-350`；README `:47`；`UPSTREAM.md:3-7` | G6b=N/A |
| D-10 | L0/G0 | `doc_drift` | `differ` | yes | URL 未记录；无 LICENSE | `UPSTREAM.md:3-7` | 无数值影响；再分发前需补 |
| D-11 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方 DistributedSampler + 无显式 seed；移植固定 seed 42 | `train.py:99,396`；`runtime.py:372-377,850` | 随机流不同；未量化 |
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

closure = 15/15；D-03/D-04/D-07/D-11 未完整披露，G3/G4 不能 pass。

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
| batch | 1/GPU × 2 GPU = 2 | 1 × 1 GPU = 1 | **differ，未披露** |
| input | 1024 | 1024 | same |
| optimizer | AdamW lr 2e-4, wd 0.01（默认） | AdamW lr 2e-4, wd 0.01 | same |
| scheduler | Cosine 200, eta_min 1e-7 | 同 | same（代码） |
| loss | 5 类 weighted CE | 2 类 neutral CE | intentional_deviation |
| backbone | ViT-B (embed 768/depth 12) | ViT-L (1024/24) | intentional_deviation |
| transform | resize 1024、ImageNet、无增强 | 同 | same |
| data | SAR 5 类 train/test | 18/6 3-shot | protocol_adaptation |
| workers | 8 | 0 | **differ，未披露** |
| GPUs | DDP 2 | 单卡 1 | **differ，未披露** |
| seed | 无显式 seed | 42 | **differ，未披露** |
| selection | S-measure best | 6 图 val Dice | protocol_adaptation |
| freeze | encoder 除 prompt_generator/Adapter | 同 | same |
| grad ckpt | 无 | 启用 | intentional（memory only） |

结论：**G3=unverified**——D-03/D-04/D-11 未进入 deviations；5→2 类和无条件 CE 已披露，
但有效 batch、workers、seed 不同，不能称为官方 SAR recipe 复现。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | resize 1024 + ImageNet | 同 | same |
| prompt | prompt-free（空 sparse + no_mask_embed） | 同 | same |
| checkpoint | 官方 `model_epoch_best.pth`（缺失） | 自训 best val Dice | missing asset |
| postprocess | 1024，argmax over channels | softmax(fg)>0.5，双线性回 GT | 2 类决策等价；分辨率不同 |
| threshold | argmax（无显式阈值） | 0.5（2 类等价） | equivalent |
| metric | S/E/wF/MAE + mIoU/OA @1024 | Dice/IoU/clDice/HD95 @GT | **differ，未完整披露** |
| export | 1024 RGB JPG | GT bool mask | intentional_deviation（已披露） |

**G4=unverified**：prompt-free、normalization、2 类决策规则与官方一致；但官方 eval 是
1024 上的 COD 指标 + mIoU，移植是 GT 上的 Dice/IoU/clDice/HD95，metric 差异只在
deviations 中以“restored to GT resolution”带过，没有列出指标口径。

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
