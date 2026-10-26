# SACM (official) 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/SACM`
- 官方 commit：`986504957e3f41f7fc02e610e2ad9ad27c43300f`（2026-07-07，本地 git 可核验）
- 移植版本：`src/baseline_zoo/adapters/sacm_official.py` + `src/baseline_zoo/models/sacm_official/`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 = unverified。** 没有冻结 calibration pack；本报告是手工 pilot。
> `main_table=blocked`。官方仓库无 LICENSE，UPSTREAM 未记录 commit/路径过期，
> 因此 **G0 不能 pass**。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 |
|---|---|---|---|
| S-01 | SAM ViT-L + adapter_dim_ratio 0.1；encoder 仅 adapter 可训练 | `train_sam.py:154-175`; `build_sam.py` | 图/冻结策略一致 |
| S-02 | prompt encoder 冻结，mask decoder 全训练 | `train_sam.py:169-175` | state_dict/参数量 |
| S-03 | zero prompt forward，decoder 接收 adapter_features | `train_sam.py:256-302` | forward 数值 |
| S-04 | loss = 0.7×BCEWithLogits + 0.3×Dice，1024 分辨率 | `train_sam.py:55-91,197-198,304-307` | loss/梯度 |
| S-05 | AdamW 两组 adapter_lr/decoder_lr，wd 0.01 | `train_sam.py:220-224,415-417` | 超参/参数组 |
| S-06 | cosine，T_max=60，eta_min=1e-6 | `train_sam.py:226-232,418,428` | LR 序列 |
| S-07 | batch 1、clip 1.0、train Resize 1024 | `train_sam.py:183-192,311-316,413,420` | 有效 batch/输入 |
| S-08 | 每 10 epoch val；按 F1 最优保存；loss/F1 用 raw logits | `train_sam.py:329-405`; `train_sam.py:36-53` | 选择/val 语义 |
| S-09 | test 输入 1024；预测 resize 回原图；raw logit > 0.5 算 F1 | `test_sam.py:151-166,251-274`; `test_sam.py:36-50` | 阈值/分辨率 |
| S-10 | test 数据 split、数据根目录语义 | `test_sam.py:199-203`; `train_sam.py:93-118` | 评测 split 明确 |
| S-11 | 数据协议：4 个内部曲线结构数据集 | `README.md:27-31`; `train_sam.py:188-192` | 训练/测试数据 |
| S-12 | 来源/许可证/seed | `UPSTREAM.md:3-14`; `train_sam.py:148-155` | provenance/复现 |
| S-13 | vendored 代码差异仅 checkpointing/import | `UPSTREAM.md:7-11`; `image_encoder.py:64-65,146-151` | 计算不变 |

---

## 02. 移植对照与差异

### 结论先行

**模型图、损失、梯度、20-step 路径：逐位一致。** 722/722 state keys、init maxdiff
0、总参 325,291,829、可训练 17,007,337；forward maxdiff 0；单步 loss
0.6403343677520752 相同、318 个梯度全 0；20-step loss/state 全 0。
**评测/文档不是官方复现**：adapter 声称 “upstream evaluates at 1024”，但官方
`test_sam.py:261` 明确把预测 resize 回原图；官方 test split 也有歧义。G5=fail。

### 关键源码证据

```text
# 3rd/SACM/train_sam.py:163-175
for name, param in sam.image_encoder.named_parameters():
    if 'adapter' not in name:
        param.requires_grad = False
for param in sam.prompt_encoder.parameters():
    param.requires_grad = False
for param in sam.mask_decoder.parameters():
    param.requires_grad = True
```
```text
# 3rd/SACM/train_sam.py:220-232,418-428
optimizer = optim.AdamW([
  {'params': adapter_params, 'lr': args.adapter_lr, 'weight_decay': args.weight_decay},
  {'params': mask_decoder_params, 'lr': args.decoder_lr, 'weight_decay': args.weight_decay}])
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
# defaults: epochs=60, adapter_lr=decoder_lr=3e-4, wd=0.01, min_lr=1e-6
```
```text
# 3rd/SACM/test_sam.py:251-274（预测回原图；阈值是 raw logits）
low_res_masks, iou_predictions = sam.mask_decoder(...)
low_res_masks = F.interpolate(low_res_masks, size=original_size[::-1], mode='bilinear', align_corners=False)
f1 = calculate_f1_score(low_res_masks, original_mask)
pred_mask = (low_res_masks[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
```
```text
# src/baseline_zoo/adapters/sacm_official.py:39-44（与官方源码矛盾）
'Predictions are restored to the original GT resolution for the '
'shared evaluator; upstream evaluates at 1024.',
'Upstream test_sam.py thresholds raw logits at 0.5; this port '
'applies sigmoid first (logit > 0). ...'
```
```text
# 3rd/SACM/test_sam.py:199-203 vs train_sam.py:93-94
val_dataset = SegmentationDataset(args.data_root)   # split 默认 'train'
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
```

### 差异表

| id | verdict | gate | category | 官方行为 | 移植行为 | 影响 | 披露 |
|---|---|---|---|---|---|---|---|
| D-01 | `intentional_deviation` | G4/G5 | `protocol_adaptation` | 每 10 epoch val，按 F1 best；val loss/F1 raw logits | 每 epoch val，按 Dice best | 选择规则不同 | 已写 |
| D-02 | `intentional_deviation` | G4 | `bug_fix` | test 用 raw logits > 0.5（未 sigmoid） | sigmoid > 0.5 | 二值结果通常更合理，但不等价 | 已写 |
| D-03 | `differ` | G5 | `bug_fix` | test 预测 resize 回 `original_size[::-1]` | port 也回 GT 分辨率 | 实际实现一致 | **文档写反，未披露** |
| D-04 | `differ` | G5 | `protocol_adaptation` | `SegmentationDataset` 默认 split='train'，test 未显式传 split | 20-dataset held-out 测试 | split 语义不可比/官方歧义 | **未写** |
| D-05 | `intentional_deviation` | G3/G5 | `protocol_adaptation` | 4 个内部曲线数据集，train/val | 18/6 3-shot + 20-dataset | 数据协议不同 | 已写 |
| D-06 | `equivalent` | G3 | `wrapper_change` | 官方无 activation checkpointing | 每 block checkpoint（可开关） | 内存下降，计算不变 | 已写 |
| D-07 | `intentional_deviation` | G3/G5 | `protocol_adaptation` | 无 seed；workers 4，shuffle | 固定 seed；workers 0，shuffle | 训练轨迹/复现性不同 | **未写** |
| D-08 | `unverified` | G0 | `missing_asset` | 无 LICENSE；UPSTREAM 有 URL（`UPSTREAM.md:3`）但无 commit 记录，路径 `baselines/sacm` 过期 | 本地 commit 可核验，但文档不完整 | provenance/再分发风险 | UPSTREAM 仅写“no LICENSE” |
| D-09 | `unverified` | G6b | `missing_asset` | 官方无训练 checkpoint | 本地无官方权重 | G6b 无法执行 | 已写 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | structure probe |
| S-02 | `preserved` | structure probe |
| S-03 | `preserved` | forward probe |
| S-04 | `preserved` | loss/grad probe |
| S-05 | `preserved` | D-06；源码同超参 |
| S-06 | `preserved` | 源码同 T_max/eta_min |
| S-07 | `preserved` | 源码同 batch/clip/1024 |
| S-08 | `covered_by_discrepancy` | D-01 |
| S-09 | `covered_by_discrepancy` | D-02, D-03 |
| S-10 | `covered_by_discrepancy` | D-03, D-04 |
| S-11 | `covered_by_discrepancy` | D-05 |
| S-12 | `covered_by_discrepancy` | D-07, D-08 |
| S-13 | `preserved` | 源码 diff 仅 checkpointing/import |

closure = 13/13；D-03（错误披露）、D-04、D-07、D-08 仍未关闭。

---

## 03. 数值探针结果

官方用 `3rd/SACM/segment_anything` 构建 ViT-L，移植用 vendored 副本；同一
`checkpoints/sam_vit_l_0b3195.pth`。官方 backward 需要 activation
checkpointing（与移植 `grad_checkpointing=True` 等价），否则 11 GiB OOM。
batch=1、固定 seed、TF32 off、deterministic。

```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/sacm_probe2.py
keys 722/722; only off []; only port []; shape mismatch []; init maxdiff 0.0
total 325,291,829 both; trainable 17,007,337 both; adapter 12,184,352 both
mask_decoder 4,822,985 both; frozen 308,284,492 both
forward shape (1,1,256,256), maxdiff 0.0, sign_equal True, bit_equal True
one-step loss official/port 0.6403343677520752 / 0.6403343677520752 (diff 0.0)
grads 318/318, grad maxdiff 0.0; peak MiB 8354.95 / 7108.04
```
```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/sacm_20step_compare.py  # 由保存的 20-step state 比较
keys 360/360; trainable elems 17,007,337; loss maxdiff 0.0; state maxdiff 0.0
first5: 0.6536693, 2.7518022, 0.7031066, 0.6361142, 1.4312135; last 0.7679344
```

数值路径完全一致；差异在评测协议和 provenance，不在模型/损失。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 60 | 60 | same |
| batch/GPU | 1 / 单卡 | 1 / 单卡 | same |
| optimizer | AdamW 两组 3e-4，wd .01 | 两组 args.lr/wd，默认同 | same |
| scheduler | cosine T_max=60, eta_min=1e-6 | 同 | same |
| loss | 0.7 BCE + 0.3 Dice @1024 | 同 | 数值一致 |
| clip | 1.0 | 1.0 | same |
| input | Resize 1024 + ToTensor | PIL bilinear 1024 /255 | equivalent |
| selection | 每 10 epoch F1 best | 每 epoch Dice best | diff，已披露 |
| data | 4 内部数据集 train/val | 18/6 3-shot + 20-dataset | protocol adaptation |
| workers | 4 | 0 | diff，未写 |
| seed | 无 | 固定 seed | diff，未写 |
| checkpoint | `best_model.pth`（F1） | val Dice best | diff，已披露 |

**G3 结论：unverified。** 模型超参、loss、schedule、batch、clip 均一致；但
D-07 的 seed/workers 差异未写入文档，按 04 的确定性与 DataLoader 项不能 pass。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | Resize 1024 + ToTensor | 同 | same |
| output | low_res logits @256→原图双线性 | 同→GT 双线性 | same |
| threshold | raw logits > 0.5 | sigmoid > 0.5 | 有意修正，已披露 |
| metric | F1 per image/aggregate | 共享 Dice/IoU/clDice/HD95 | protocol adaptation |
| test split | `SegmentationDataset` 默认 `train`，语义歧义 | 20-dataset held-out | **不可比** |
| checkpoint | 官方 `best_model.pth`（本地无） | 本地训练权重 | G6b 不可做 |
| 分辨率声明 | 实际回原图（`test_sam.py:261`） | 回 GT（正确） | adapter 文字写反 |

实现路径与官方一致；G4 通过，G5 因 **错误的上游分辨率声明** 和 test split
歧义而 fail。

---

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = unverified            # 无 LICENSE；UPSTREAM 有 URL 但无 commit，路径过期
G1  = pass                  # 722/722 keys、shape/init 0 diff
G2  = pass                  # forward/loss/grad/20-step 全 0 diff
G3  = unverified            # 60ep/3e-4/.01/cosine/clip/.7-.3 保留；D-07 未披露
G4  = pass                  # forward/阈值/GT-res 导出有证据
G5  = fail                  # D-03 上游分辨率写反；D-04 test split 歧义未披露
G6a = pass                  # 同权重同 batch 20-step loss/state 0 diff
G6b = N/A                   # 官方无训练 checkpoint
G6c = unverified            # 未做全量训练/评测

main_table = blocked
claim_label = path_verified  # 仅指模型/loss/20-step 路径，不含官方评测协议
```

## 最终判断

- 模型、损失、优化器、schedule 和 20-step 训练路径与官方逐位一致，证据充分。
- **必须修正 adapter 的 D-03 错误声明**：官方 test 不是在 1024 评测，而是在
  `test_sam.py:261` resize 回原图；当前 “upstream evaluates at 1024” 是错误披露。
- G0 未通过：无 LICENSE，UPSTREAM 有 URL 但没有 commit，`baselines/sacm` 路径过期；
  在再分发或投稿前必须补齐。
- G5 fail 还来自 test split 歧义（官方默认 `train`）；未写入的 seed/workers
  差异同时阻断 G3。
- 待办：1) 修 D-03/D-04/D-07/D-08；2) 冻结 A0；3) 跑 G6c；4) 若获取官方
  checkpoint 再补 G6b。
