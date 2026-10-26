# TopoLoRA-SAM 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/Seglab`
- 官方 commit：`7de3a5fbad992363e690190eb82dffbe8a3c1a0b`（2026-01-02）
- 移植版本：`src/baseline_zoo/adapters/topolora.py` + `src/baseline_zoo/models/topolora/`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 = unverified。** 没有冻结 calibration pack；本报告是手工 pilot，G0–G6c
> 只用于定位问题，不能作为主表证据。`main_table=blocked`。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 |
|---|---|---|---|
| S-01 | ViT-B + LoRA(r8/alpha16/mlp,fc1,fc2) + ConvAdapter(256,3) + trainable decoder，prompt-free | `configs/models/sam_topolora.yaml:1-17`; `seglab/models/sam_peft/model.py:18-49` | 图与参数名一致 |
| S-02 | 仅 LoRA、adapter、mask decoder 可训练；总/可训练参数量 | `model.py:24-49`; `seglab/train.py:152-153` | state_dict/参数量一致 |
| S-03 | forward：反 ImageNet 归一化→[0,255]→SAM preprocess→encoder→adapter→空 prompt→decoder→384 logits | `model.py:51-79` | 固定输入数值一致 |
| S-04 | loss = BCE 1.0 + Dice 1.0 + clDice 0.5，boundary 0 | `configs/models/sam_topolora.yaml:19-23`; `seglab/models/__init__.py:61-79` | loss/梯度一致 |
| S-05 | AdamW lr 1e-4 / wd 1e-4 | `configs/base.yaml:50-53`; `seglab/models/__init__.py:307-314` | 优化器超参一致 |
| S-06 | cosine，T_max = max_epochs = 50 | `configs/base.yaml:11-12,55-56`; `seglab/models/__init__.py:316-319` | LR 序列一致 |
| S-07 | batch 1 × accumulate 4、clip 1.0、16-mixed | `configs/models/sam_topolora.yaml:25-30`; `configs/base.yaml:15-19` | 有效 batch/数值路径 |
| S-08 | train transform：RandomScale 0.8–1.2、pad、crop 384、flip、ColorJitter、ImageNet norm | `seglab/data/transforms.py:21-42` | 同 RNG 下张量一致 |
| S-09 | 数据：HF DRIVE train，seed 0，val_ratio 0.2，workers 4 | `configs/base.yaml:1-2,21-33`; `seglab/train.py:46-80` | split/loader 语义 |
| S-10 | 50 epochs，val/dice max 选择 checkpoint，最后 test | `seglab/train.py:172-199` | 选择规则一致 |
| S-11 | eval：Resize 384 + ImageNet norm，sigmoid > 0.5，384 分辨率指标 | `seglab/data/transforms.py:43-49`; `seglab/models/__init__.py:99-118` | 阈值/分辨率一致 |
| S-12 | 来源、许可证、权重、RNG | `UPSTREAM.md:3-12,52-53`; `configs/base.yaml:2` | 可追溯、可复现 |

---

## 02. 移植对照与差异

### 结论先行

**模型图、损失、梯度：逐位一致。** 官方与移植的 `lora.py`、`adapters.py`、
`topo_losses.py` `diff` 为空；`transforms.py` 仅 `fill/fill_mask`→
`value/mask_value` API 改名。固定权重下 state_dict 370/370、init maxdiff 0；
forward/loss/grad/20-step 均为 0 diff。**训练协议不是官方逐位复现**：无 AMP、
共享 18/6 split，且 `runtime` 对最后一个不完整 accumulation 组做了官方
Lightning 没有的 rescale（未披露）。

### 关键源码证据

```text
# 3rd/Seglab/configs/models/sam_topolora.yaml:6-11,19-23
  lora: {enabled: true, r: 8, alpha: 16, dropout: 0.0,
         target_keywords: ["mlp", "fc1", "fc2"]}
loss: {bce_weight: 1.0, dice_weight: 1.0,
       cldice_weight: 0.5, boundary_weight: 0.0}
```
```text
# src/baseline_zoo/adapters/topolora.py:176-179
prob = torch.sigmoid(logits)
bce = F.binary_cross_entropy_with_logits(logits, target)
return bce + dice_loss(prob, target) + 0.5 * cldice_loss(prob, target)
```
```text
# src/baseline_zoo/runtime.py:156-169（官方 Lightning 无此 rescale）
if pending:
    if pending < accum:
        scale = accum / pending
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
```

### 差异表

| id | verdict | gate | category | 官方行为 | 移植行为 | 影响 | 披露 |
|---|---|---|---|---|---|---|---|
| D-01 | `differ` | G3 | `wrapper_change` | Lightning `precision: 16-mixed` | plain PyTorch fp32 | 速度/数值细节；图不变 | `deviations` 已写 |
| D-02 | `differ` | G5 | `protocol_adaptation` | HF DRIVE + val_ratio 0.2；test=validation split | 共享 18/6 split + 20-dataset 协议 | 数据/指标不可与官方直接比 | 已写 |
| D-03 | `intentional_deviation` | G3 | `wrapper_change` | Lightning Trainer/logger/ModelCheckpoint | 共享 runtime；无 logger/artifact | 训练编排不同 | 已写 |
| D-04 | `intentional_deviation` | G3 | `wrapper_change` | `seed_everything(0)`；Albumentations 无 per-Compose seed | 每 batch 用共享 RNG seed 全局 Python/NumPy | 增强随机流不同 | 已写 |
| D-05 | `equivalent` | G2 | `wrapper_change` | `fill=0, fill_mask=0` | `value=0, mask_value=0` | 零填充语义相同 | 已写 |
| D-06 | `intentional_deviation` | G4/G5 | `protocol_adaptation` | 384 概率图/Png；无 TTA | GT 分辨率双线性导出；无 TTA | 导出分辨率不同 | 已写 |
| D-07 | `intentional_deviation` | G3 | `bug_fix` | 每个 microbatch loss/accum；末组不 rescale | 末组先乘 `accum/pending` 再 step | 18 张/accum4 时末组 2 张更新×2 | **未写** |
| D-08 | `equivalent` | G3 | `wrapper_change` | optimizer 接收 `self.parameters()`（含冻结参数） | 仅接收 `requires_grad` 参数 | 冻结参数无梯度，effective 更新相同 | 未见文档 |
| D-09 | `unverified` | G6b | `missing_asset` | 官方仓库无 TopoLoRA 训练权重 | 从 SAM ViT-B 起训 | 无法做 checkpoint 对齐 | 已写 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | D-01/D-05；state/forward probe |
| S-02 | `preserved` | structure probe；D-08 |
| S-03 | `preserved` | forward probe (maxdiff 0) |
| S-04 | `preserved` | loss/grad probe (maxdiff 0) |
| S-05 | `preserved` | D-08；源码同超参 |
| S-06 | `preserved` | 源码 T_max/余弦同为 50 |
| S-07 | `covered_by_discrepancy` | D-01, D-07 |
| S-08 | `covered_by_discrepancy` | D-04, D-05 |
| S-09 | `covered_by_discrepancy` | D-02 |
| S-10 | `covered_by_discrepancy` | D-02, D-06 |
| S-11 | `covered_by_discrepancy` | D-06 |
| S-12 | `covered_by_discrepancy` | D-09 |

closure = 12/12；但 D-07、D-08 是 **undocumented**，G3 不能 pass。

---

## 03. 数值探针结果

官方模型用本地 `3rd/Seglab` + 移植 vendored Meta SAM 构建，加载同一个
`checkpoints/sam_vit_b_01ec64.pth`；仅替换缺失的 `seglab.models`/registry 导入。
batch=1、固定 seed、TF32 off、cuDNN deterministic。

```text
$ PYTHONPATH=src /home/dev/anaconda3/envs/tsai/bin/python /tmp/topolora_structure.py
keys 370 370 only off [] only port []
shape mismatch [] init maxdiff 0.0
official total 94606640 trainable 4929508 lora 737280 decoder 4058340 adapter 133888 prompt_encoder 6220
port     total 94606640 trainable 4929508 lora 737280 decoder 4058340 adapter 133888 prompt_encoder 6220
```
```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/topolora_probe.py
one-step loss official/port: 1.7147350311279297 / 1.7147350311279297  diff 0.0
grad tensors 167 / 167, grad maxdiff 0.0, state after forward maxdiff 0.0
20-step loss maxdiff 0.0 (first 1.7147350, 3.1516562, 1.9142759; last 1.6147943)
20-step final state maxdiff 0.0
```

LoRA 可训练参数为 **737,280**；总可训练 **4,929,508**，与官方完全一致。
20-step 使用 AdamW lr/weight_decay 与官方相同，但**不是** Lightning accumulation
路径（D-07 未进入该探针）。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 50 | 50 | same |
| batch/accum | 1 × 4 | 1 × 4 | same（末组 D-07 除外） |
| precision | 16-mixed | fp32 | diff，已披露 |
| optimizer | AdamW 1e-4 / 1e-4 | AdamW args.lr / args.weight_decay | same defaults |
| scheduler | CosineAnnealingLR T_max=50 | 同 | same |
| loss | 1×BCE + 1×Dice + 0.5×clDice | 同 | 数值一致 |
| transforms | RandomScale/Pad/Crop/flip/ColorJitter/ImageNet | 同（API 改名） | 语义一致 |
| data | HF DRIVE；16 train/4 val（val_ratio .2） | 18 train/6 val（每域 3+1） | protocol adaptation |
| workers/GPU | 4 / 单卡 | 共享 loader / 单卡 | 随机流不同 |
| selection | val/dice max | val Dice max | same |
| init | 官方 SAM ViT-B | 同 | same |

**G3 结论：unverified。** D-07 会改变末组更新；D-08 虽无数值影响但未记录；
无 AMP 与共享 split 已披露，可在补充 disclosure 后重新评估。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | Resize 384 + ImageNet norm | 同 | same |
| 输出分辨率 | 384 logits | 384 logits → GT 双线性 | export 不同，已披露 |
| threshold | sigmoid > 0.5 | sigmoid > 0.5 | same |
| TTA | 无 | 无 | same |
| checkpoint | val/dice best | val Dice best | same rule |
| 测试集 | HF DRIVE validation split | 20-dataset 协议 | 不可直接比 |
| 指标 | 官方 meter/CSV + boundary/clDice | 共享 evaluator Dice/IoU/clDice/HD95 | protocol adaptation |

推理实现本身与官方 forward/阈值一致；差异集中在导出分辨率和测试协议。

---

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # commit/license/来源可追溯，UPSTREAM 与代码一致
G1  = pass                  # 370/370 keys，shape/init 0 diff，LoRA 计数一致
G2  = pass                  # forward/loss/grad/20-step 全 0 diff
G3  = unverified            # D-07 末组 accumulation rescale 未披露；D-08 未记录
G4  = pass                  # forward/threshold/导出路径有证据且差异已披露
G5  = pass                  # 18/6、20-dataset、无 TTA 已披露；A0 另行 blocked
G6a = pass (single GPU)     # 同权重同 batch 20-step loss/state 0 diff
G6b = N/A                   # 无官方 TopoLoRA 训练 checkpoint
G6c = unverified            # 未做全量训练/评测

main_table = blocked
claim_label = path_verified  # 仅覆盖模型/损失/梯度数值路径，不含官方指标复现
```

## 最终判断

- 模型、LoRA/adapter/decoder 结构、损失、梯度、20-step 训练路径与官方**逐位一致**，
  有 state_dict、参数量和 GPU 数值证据。
- 不能标 `table_metric_reproduced`：无官方训练 checkpoint、无 G6c 全量复现，
  且 D-07 使项目 18 张训练的末组更新与官方 Lightning 语义不同。
- 待办：1) 在 `deviations` 记录 D-07/D-08；2) 冻结 A0；3) 跑 G6c；
  4) 若有官方权重再补 G6b。
