# SAMed 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/SAMed`
- 官方 commit：`e351840481f51d0c7dee50007f29d02197b63d99`（2023-09-02）
- 移植版本：`src/baseline_zoo/adapters/samed.py` + `src/baseline_zoo/models/samed/`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 = unverified。** 没有冻结 calibration pack；本报告是手工 pilot。
> `main_table=blocked`，不能据此声称官方 Synapse 指标已复现。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 |
|---|---|---|---|
| S-01 | 类别数：官方默认 8 类，decoder token = num_classes+1 | `train.py:25-26`; `trainer.py:53-54` | 类别语义/输出通道 |
| S-02 | LoRA rank 4，注入 image encoder 的 q/v；encoder 冻结 | `train.py:49,99-100`; `sam_lora_image_encoder.py:83-108` | 注入位置/参数一致 |
| S-03 | prompt encoder + mask decoder 可训练，无 prompt | `train.py:94-109`; `trainer.py:78` | state_dict/参数量 |
| S-04 | forward 返回 low_res_logits 与 masks；decoder 不切片 | `mask_decoder.py:95-111`; `trainer.py:78` | 输出 shape/数值 |
| S-05 | loss = 0.2×CE + 0.8×Dice，监督 128×128 low-res | `trainer.py:22-27,74-79`; `utils.py:61-98` | loss/梯度一致 |
| S-06 | 优化器：源码默认 SGD；`--AdamW` 时 AdamW wd 0.1 | `trainer.py:55-62`; `train.py:53` | 优化器语义 |
| S-07 | warmup 250 iterations；poly `(1-i/max_i)^0.9` 每 iteration | `trainer.py:55-58,83-95` | LR 序列 |
| S-08 | 200 epochs、stop 160；batch 12/GPU × 2 GPU | `train.py:29-35`; `trainer.py:38,48-51` | 有效 batch/步数 |
| S-09 | 训练数据 224→512 order3 zoom；随机 rot/flip 或 ±20° | `datasets/dataset_synapse.py:13-47` | 预处理/增强 |
| S-10 | 无 best-checkpoint 规则；保存 epoch 20 间隔和 stop epoch | `trainer.py:115-132` | 选择规则 |
| S-11 | eval：native→224→512 order3，argmax@512，zoom 回 native order0 | `utils.py:114-140` | 推理/导出 |
| S-12 | 官方 8 类 LoRA checkpoint 公开，但不在本地仓库 | `README.md:53-63` | checkpoint 兼容性 |
| S-13 | 来源/许可证 | `UPSTREAM.md:3-10`; `LICENSE` | 可追溯 |

---

## 02. 移植对照与差异

### 结论先行

**二分类模型图与官方 `num_classes=1` 构造逐位一致。** 官方/移植 state_dict
350/350、shape 全同、init maxdiff 0；forward/loss/grad/20-step 均 0 diff。
**与官方默认 8 类模型不等价**：官方 8 类 decoder 多 982,247 参数，且公开
checkpoint 与二分类 head 不兼容。**训练协议不是官方复现**：官方源码默认
SGD，README 训练命令用 `--AdamW`；移植固定 AdamW，且未披露这个源码默认值。

### 关键源码证据

```text
# 3rd/SAMed/train.py:25-35
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--stop_epoch', type=int, default=160)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--n_gpu', type=int, default=2)
```
```text
# 3rd/SAMed/trainer.py:55-62（源码默认 SGD）
if args.AdamW:
    optimizer = optim.AdamW(..., lr=b_lr, ..., weight_decay=0.1)
else:
    optimizer = optim.SGD(..., lr=b_lr, momentum=0.9, weight_decay=0.0001)
```
```text
# src/baseline_zoo/adapters/samed.py:88-94,113-116
return build_samed(image_size=self.input_size, num_classes=1, ...)
return torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=args.weight_decay)
```
```text
# 3rd/SAMed/segment_anything/modeling/mask_decoder.py:102-111
# if multimask_output: ... masks = masks[:, mask_slice, :, :]
# 以上切片被注释；num_multimask_outputs=num_classes=1 时返回 2 个 token
return masks, iou_pred
```

### 差异表

| id | verdict | gate | category | 官方行为 | 移植行为 | 影响 | 披露 |
|---|---|---|---|---|---|---|---|
| D-01 | `intentional_deviation` | G1/G2 | `protocol_adaptation` | 默认 8 类 + 9 mask tokens；公开 8 类权重 | 二分类 + 2 tokens；Δ total 982,247 | 不能加载/比较官方 8 类权重 | `UPSTREAM.md:76-81` 已写 |
| D-02 | `differ` | G3 | `official_recipe` | 源码默认 SGD momentum .9 / wd 1e-4；README 用 `--AdamW` | 固定 AdamW wd 0.1 | 优化轨迹不同；README 可辩护但未披露 | **未写** |
| D-03 | `intentional_deviation` | G3 | `protocol_adaptation` | batch 24、2×GPU DataParallel、200/stop160 | batch 4、单卡、160 epochs | 有效 batch/BN 统计不同 | 部分披露；DataParallel 未写 |
| D-04 | `intentional_deviation` | G3/G4 | `protocol_adaptation` | 无 best 规则，测 epoch 159 | 6 图 val Dice best | 选择语义不同 | 已写 |
| D-05 | `intentional_deviation` | G3 | `protocol_adaptation` | 250 iteration warmup + per-iteration poly | 5-epoch warmup + per-epoch poly | LR 曲线不同 | 已写 |
| D-06 | `intentional_deviation` | G3/G4 | `protocol_adaptation` | 224→512 order3 zoom；CT window | 直接 BICUBIC 512；无 CT window | 输入分布/重采样不同 | 已写 |
| D-07 | `intentional_deviation` | G4 | `protocol_adaptation` | argmax@512 后 zoom native；无 TTA | softmax>0.5 at GT res；无 TTA | 二分类阈值等价，导出不同 | 已写 |
| D-08 | `intentional_deviation` | G3 | `wrapper_change` | NumPy RNG；worker_init seed+worker_id | Python `random` per-sample RNG | 增强随机流不同 | **未写** |
| D-09 | `unverified` | G6b | `missing_asset` | 官方 8 类 LoRA checkpoint（Google Drive） | 本地无 checkpoint，二分类不兼容 | G6b 无法执行 | 已写 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `covered_by_discrepancy` | D-01 |
| S-02 | `preserved` | structure probe；D-03 |
| S-03 | `preserved` | structure/forward probe |
| S-04 | `preserved` | D-01；forward probe |
| S-05 | `preserved` | loss/grad probe |
| S-06 | `covered_by_discrepancy` | D-02 |
| S-07 | `covered_by_discrepancy` | D-05 |
| S-08 | `covered_by_discrepancy` | D-03 |
| S-09 | `covered_by_discrepancy` | D-06, D-08 |
| S-10 | `covered_by_discrepancy` | D-04 |
| S-11 | `covered_by_discrepancy` | D-07 |
| S-12 | `covered_by_discrepancy` | D-01, D-09 |
| S-13 | `preserved` | 来源/许可证已核对 |

closure = 13/13；D-02/D-03(DataParallel)/D-08 未完整披露，G3 不能 pass。

---

## 03. 数值探针结果

官方用 `3rd/SAMed` 的 `sam_model_registry['vit_b'](num_classes=1)` + `LoRA_Sam`
构建，加载同一 `checkpoints/sam_vit_b_01ec64.pth`；binary 口径对齐移植的
`build_samed(num_classes=1)`。batch=1、固定 seed、TF32 off、deterministic。

```text
$ /home/dev/anaconda3/envs/tsai/bin/python /tmp/samed_probe2.py
keys 350/350; only off []; only port []; shape mismatch []; init maxdiff 0.0
total 91,210,222 both; trainable 3,931,374 both; LoRA 147,456 both
mask_decoder 3,777,698 both; prompt_encoder 6,220 both
official 8-class: total 92,192,469; trainable 4,913,621; decoder 4,759,945 (+982,247)
forward low (1,2,128,128) maxdiff 0.0; masks (1,2,512,512) maxdiff 0.0; argmax equal
one-batch loss 2.3708856105804443 both, diff 0.0; grads 151/151 maxdiff 0.0
20-step loss maxdiff 0.0; final state maxdiff 0.0; final argmax equal
first5 losses: 2.3708856, 1.8726397, 5.7646933, 1.1757257, 1.7244835; last 0.4073141
```

注意：该探针验证的是**官方代码的二分类构造**，不是官方发布的 8 类模型；
128×128 监督、LoRA rank 4 与 decoder 2-token 行为均一致。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 200，stop 160 | 160 | 相关但步数不同 |
| batch/GPU | 12×2=24，DataParallel | 4，单卡 | diff，部分披露 |
| input | 数据 224→512 order3 | 原图直接 BICUBIC→512 | diff，已披露 |
| optimizer | 源码默认 SGD；README `--AdamW` | AdamW 5e-3 / 0.1 | **diff，未完整披露** |
| scheduler | 250-it warmup + per-iter poly | 5-epoch warmup + per-epoch poly | diff，已披露 |
| loss | 0.2 CE + 0.8 Dice @128 | 同 | 数值一致 |
| augment | NumPy rot/flip 或 ±20° | 同几何增强，Python RNG | 语义同，随机流不同 |
| data | Synapse 2211 slices，8 类 | 18 图，二分类，20-dataset 测试 | protocol adaptation |
| workers/GPU | 8 / 2 GPU | 共享 loader / 1 GPU | diff |
| selection | 无 best；测 epoch 159 | val Dice best | diff，已披露 |
| init | SAM ViT-B | 同 | same |
| AMP/TTA | 无 / 无 | 无 / 无 | same |

**G3 结论：unverified。** 主要阻塞是 D-02：官方源码默认 SGD，而移植使用
AdamW 且未说明依据；D-03 DataParallel 和 D-08 RNG 也未完整披露。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | native→224 order3→512 order3 | 直接 BICUBIC→512 | diff，已披露 |
| 输出 | low-res logits → masks@512 | 同模型输出@512 | forward 一致 |
| postprocess | argmax softmax@512 → zoom native order0 | softmax foreground>0.5 → GT 双线性 | 二分类等价，导出不同 |
| threshold | argmax（二分类等价 0.5） | 0.5 | equivalent |
| checkpoint | 8 类官方 LoRA（本地无） | 二分类训练权重 | 不兼容 |
| metric | per-class DSC/HD @native | 共享 Dice/IoU/clDice/HD95 | protocol adaptation |
| TTA | 无 | 无 | same |

模型推理图的数值一致；评测重采样、导出分辨率和指标口径属于项目协议适配。

---

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # 官方 commit/MIT/LICENSE/来源可追溯
G1  = pass                  # 二分类图 350/350 keys、shape/init 0 diff
G2  = pass                  # 二分类 forward/loss/grad/20-step 全 0 diff
G3  = unverified            # D-02 AdamW vs 源码默认 SGD；D-03/D-08 未完整披露
G4  = pass                  # 推理/阈值/导出差异已披露且有 forward 证据
G5  = pass                  # 8→2 类、18/6、指标替换已披露；8 类指标不可比
G6a = pass (single GPU)     # 同权重同 batch 20-step loss/state 0 diff
G6b = N/A                   # 无兼容的二分类官方 checkpoint；8 类权重不可加载
G6c = unverified            # 未做全量训练/评测

main_table = blocked
claim_label = path_verified  # 仅指二分类模型/损失/训练路径，不含官方 8 类结果
```

## 最终判断

- 官方 `num_classes=1` 图与移植二分类图**逐位一致**：350 keys、3,931,374
  trainable、LoRA 147,456，forward/loss/grad/20-step 全 0 diff。
- 不能把移植结果当作官方 SAMed（8 类）复现：官方 8 类 decoder 多 982,247
  参数，公开 checkpoint 形状不兼容。
- G3 仍为 unverified：必须明确 D-02（源码默认 SGD vs README AdamW）、
  DataParallel 和 RNG；之后才能按“单卡二分类适配”重开 G3。
- 待办：1) 补齐 D-02/D-03/D-08 disclosure；2) 冻结 A0；3) 跑 G6c；
  4) 若需要 8 类对照，单独审计官方 8 类 checkpoint 路径。

## Post-review fixes (2026-10-26)

- Low-resolution target is now `input_size/4` (64 at 256, 128 at 512), fixing non-default input-size training.
- Poly LR horizon is `max(epochs, 200)`; the final epoch is no longer assigned lr=0.
- Tests: `test_low_res_tracks_input_size`, `test_poly_scheduler_does_not_zero_last_epoch`.
- Remaining: AdamW vs source-default SGD ambiguity and G3 evidence remain.
