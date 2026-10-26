# SAM-OCTA 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/SAM-OCTA`（origin: `ShellRedia/SAM-OCTA`）
- 官方 commit：`9edc56247a3e30c96e5c6f1e8fe01250b8338527`
- 移植版本：`src/baseline_zoo/models/sam_octa/` + `src/baseline_zoo/adapters/sam_octa.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25
- prompt 语义：`supervised_oracle_prompted_sam`（训练/测试的 prompt 都由 GT 生成；不是 prompt-free）

> **A0=unverified。** 没有冻结 calibration pack；本报告是手工 pilot，不是已校准审计。
> A0 通过前 G0–G6c 只用于定位问题，不能单独作为主表证据。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-01 | SAM ViT-H 默认 + LoRA rank 4；LoRA 只加在 qkv/proj | `options.py:13`；`train_sam_octa.py:56-66` | 同 config 下 key/shape/param 一致；错 rank/挂载点会改变可训练参数 |
| S-02 | 从官方 SAM checkpoint 初始化，LoRA B 置零 | `train_sam_octa.py:56-67`；`sam_lora_image_encoder.py:175-177` | 初始 forward 等于 base SAM；初始化错误会改变早期训练 |
| S-03 | 输入为 OCTA 的三层投影图（OPL_BM/ILM_OPL/FULL） | `dataset.py:29-41` | 3 通道语义是投影层，不是 RGB；换通道会改输入分布 |
| S-04 | 5 种增强，各 p=0.3：亮度对比、CLAHE、垂直/水平翻转、AdvancedBlur | `dataset.py:46-54` | 增强集合/概率一致；多/少一种会改训练分布 |
| S-05 | GT prompt：每个连通分量 1 正点 + 8 像素环内 1 负点；`is_local=False` | `dataset.py:67-79`；`prompt_points.py:40-67` | 必须从 GT 生成；改点数/环宽会改 oracle prompt 强度 |
| S-06 | loss：FAZ/Capillary 用纯 Dice；Artery/Vein/LargeVessel 用 0.8×Dice+0.2×clDice | `train_sam_octa.py:69-71`；`loss_functions.py:4-14,16-73` | 官方是**按标签类型条件化**的；无条件 clDice 会改变非血管类梯度 |
| S-07 | AdamW `lr=1`、wd=1e-4；LambdaLR warmup `epochs//5`、0.98 衰减、下限 1e-5 | `train_sam_octa.py:83-89` | 公式/序列一致；lr 或 warmup 错会改变收敛 |
| S-08 | 100 epochs；batch=GPU 数（默认 1）；workers 未传（=0）；10-fold CV | `options.py:7-10`；`train_sam_octa.py:73-79,125-142` | 官方按 fold 训练/评测；换 split 属协议改变 |
| S-09 | 每 10 epoch 保存，无 val 选择；人工挑 checkpoint | `options.py:9`；`train_sam_octa.py:140-141` | 用 val Dice 选择属协议改变 |
| S-10 | 最长边 resize 到 1024；prompt 坐标同步变换；SAM preprocess 归一化+pad | `train_sam_octa.py:144-149`；`sam_lora_image_encoder.py:179-201` | 非方形输入官方不额外 pad；移植若改顺序会移动预测 |
| S-11 | sigmoid 输出在 original_size；阈值 0.8 | `sam_lora_image_encoder.py:199-201`；`test_sam_octa.py:86` | 阈值/分辨率一致；0.5 会显著改变 Dice |
| S-12 | 指标 Dice/Jaccard/Hausdorff，在原始 label 分辨率计算 | `options.py:17-18`；`test_sam_octa.py:89-92` | 官方 Hausdorff 不是 HD95；换指标集需披露 |
| S-13 | 官方基础权重 `sam_weights/*` 与训练权重 `3M_LargeVessel_Global.pth` | `train_sam_octa.py:56-61`；`test_sam_octa.py:54-65` | 缺权重则不能复现官方资产；G6b=N/A |
| S-14 | 来源/许可证 | `UPSTREAM.md:3-12`（模型目录） | URL 未记录、无 LICENSE 必须明确披露 |
| S-15 | 项目协议：18 train / 6 val / 20 test roots、GT 导出、oracle 角色 | `cod reviewer/README.md:63-96`；`runtime.py:469-485,740-785` | 与官方 OCTA-500 10-fold 不同，必须记录 |

## 02. 移植对照与差异

| id | level/gate | category | verdict | documented | 官方 vs 移植 | 证据 | 数值影响 |
|---|---|---|---|---|---|---|---|
| D-01 | L1/G1/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方默认 ViT-H；移植 ViT-L | `options.py:13`；`sam_octa.py:53-54,69-74` | matched ViT-L 下无结构差 |
| D-02 | L3/G3/G5 | `protocol_adaptation` | `intentional_deviation` | yes | 3 层 OCTA 投影 → RGB 或灰度复制 3 通道 | `dataset.py:29-41`；`sam_octa.py:45-47,110-112` | 未量化；输入分布不同 |
| D-03 | L3/G3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方非方形不额外 pad；移植 longest-side+normalize+pad，postprocess 裁回 | `sam_octa.py:48-52,128-132` | 未量化；非方形边界不同 |
| D-04 | L3/G3 | `official_recipe` | `differ` | **no** | 官方仅对 Artery/Vein/LargeVessel 加 0.2 clDice；移植对所有类无条件 0.8 Dice+0.2 clDice | `train_sam_octa.py:69-71`；`sam_octa.py:60-61,156-160` | 未量化；非血管类 loss 改变 |
| D-05 | L3/G3/G5 | `protocol_adaptation` | `intentional_deviation` | partial | 官方 OCTA-500 10-fold；移植 18/6 3-shot + 20 test | `options.py:10`；`train_sam_octa.py:54,126`；`sam_octa.py:45-47,55-56` | 训练/测试样本与 fold 语义不同 |
| D-06 | L3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方每 10 epoch 保存、无 val 选择；移植 6 图 val Dice | `options.py:9`；`train_sam_octa.py:140-141`；`sam_octa.py:55-56` | best 权重选择不同 |
| D-07 | L3/G3 | `official_recipe` | `equivalent` | code | AdamW lr=1/wd=1e-4；warmup/0.98/floor 公式相同 | `train_sam_octa.py:85-89`；`sam_octa.py:82-97` | 公式同；未跑 LR 序列探针 |
| D-08 | L2/G2 | `wrapper_change` | `equivalent` | probe | forward/prompt/postprocess/sigmoid 路径 | `sam_lora_image_encoder.py:179-201`；`models/sam_octa/lora.py` | 1024 ViT-L forward diff=0 |
| D-09 | L4/G4 | `protocol_adaptation` | `differ` | **no** | 官方 Dice/Jaccard/Hausdorff；移植 Dice/IoU/clDice/HD95 | `options.py:17-18`；`runtime.py:217-264` | Hausdorff→HD95，指标集不同 |
| D-10 | L0/G6b | `missing_asset` | `unverified` | yes | 官方训练权重缺失；移植从 SAM ViT-L 重训 | `test_sam_octa.py:54-65`；`UPSTREAM.md:3-19` | G6b=N/A |
| D-11 | L0/G0 | `doc_drift` | `differ` | yes | URL 未记录；无 LICENSE | `UPSTREAM.md:3-5` | 无数值影响；再分发前需补 |
| D-12 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方全局 random 无 seed、DataParallel(1 GPU)；移植固定 seed 42、单模块 | `train_sam_octa.py:66`；`runtime.py:372-377,850` | 未量化；数据/prompt 随机流不同 |

**关键证据（3–5 行）**

```text
# 官方按标签类型切换 loss（train_sam_octa.py:69-71）
self.loss_func = DiceLoss()
if args.label_type in ["Artery", "Vein", "LargeVessel"]:
    self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(x, y)
# 移植无条件使用 0.8*Dice+0.2*clDice（sam_octa.py:160）
```

```text
# 官方输入=三层 OCTA 投影（dataset.py:29-40）
layers = ["OPL_BM", "ILM_OPL", "FULL"]
for layer in layers:
    image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
images.append(np.array(image_channels))
```

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `covered_by_discrepancy` | D-01 |
| S-02 | `preserved` | D-01, 探针 strict copy |
| S-03 | `covered_by_discrepancy` | D-02 |
| S-04 | `preserved` | `dataset.py:46-54`；`sam_octa.py:99-108` |
| S-05 | `preserved` | `prompt_points.py`；forward 探针 1 pos+1 neg |
| S-06 | `covered_by_discrepancy` | D-04 |
| S-07 | `preserved` | D-07 |
| S-08 | `covered_by_discrepancy` | D-05, D-12 |
| S-09 | `covered_by_discrepancy` | D-06 |
| S-10 | `covered_by_discrepancy` | D-03, D-08 |
| S-11 | `preserved` | D-08；`sam_octa.py:165-169` |
| S-12 | `covered_by_discrepancy` | D-09 |
| S-13 | `covered_by_discrepancy` | D-10 |
| S-14 | `covered_by_discrepancy` | D-11 |
| S-15 | `covered_by_discrepancy` | D-05 |

closure = 15/15；D-04/D-05/D-09/D-12 未完整披露，G3/G4 不能 pass。

## 03. 数值探针结果

详见 `03_numeric_probe.json`。探针为**随机初始化 + 官方 state_dict strict 复制到移植侧**；
只证明实现等价，不证明官方权重/训练结果，也不是 G6a。

```text
命令（GPU 1，tsai 环境）：
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_octa.py
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_octa_grad_small.py
环境绕过：官方 import 的 icecream 未安装，仅在 harness 内 stub（debug 输出），官方文件未改。

base keys/shapes equal:        True / True
state_dict keys/shapes equal:  True / True；missing=0 unexpected=0
param count official/port:     312,736,048 / 312,736,048；strict copy=True
preprocess max_abs_diff:       0.0
image_encoder max_abs_diff:    0.0
output shape / max_abs_diff:   1x1x1024x1024 / 0.0（2 points: 1 pos, 1 neg）
one-step loss official/port:   0.7004100680351257 / 0.7004100680351257 (abs 0.0)
grad max_abs_diff:             2.546585164964199e-11（16 个比较张量）
grad norm official/port:       0.0010289705607168585 / 0.0010289705500456906
```

loss/grad 是 **reduced-width** 配置（img 256、embed 128、depth 4、heads 4、rank 4），
官方/移植都用 `DiceLoss + clDiceLoss`。**未测 optimizer step；未测 ViT-H/1024 全宽；
未复现官方 class-conditional loss。**

## 04. 训练配方审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 100 | 100 | same |
| batch | 1（`len(device.split(','))`） | 1 | same |
| input | 最长边 1024 | 最长边 1024 | same |
| optimizer | AdamW lr=1, wd=1e-4 | AdamW lr=1, wd=1e-4 | same（代码） |
| scheduler | warmup 20, 0.98, floor 1e-5 | 同公式 | same（代码） |
| loss | 类别条件 Dice / +clDice | 无条件 0.8D+0.2clD | **differ，未披露** |
| 输入通道 | 3 OCTA 投影 | RGB/灰度复制 | protocol_adaptation |
| 增强 | 5×p=0.3 | 5×p=0.3 | same（RNG 流不同） |
| prompt | GT 1 pos+1 neg | 同 | same |
| 数据 | OCTA-500 10-fold | 18/6 3-shot | protocol_adaptation |
| workers | 0 | 0 | same |
| GPUs | DataParallel, 1 GPU | 单模块, 1 GPU | equivalent |
| seed | 无显式 seed | 42 | **differ，未披露** |
| selection | 每 10 epoch 保存 | 6 图 val Dice | protocol_adaptation |
| 初始化 | 官方 SAM ckpt + LoRA | SAM ViT-L + LoRA | missing asset / ViT-L adaptation |

结论：**G3=unverified**——D-04（条件化 loss）未披露，D-05/D-12 的 fold/seed 差异未完整记录；
补披露后也只能算“项目协议下的适配”，不是官方 10-fold 复现。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | longest-side 1024 + SAM preprocess | 同（非方形额外 pad，已披露） | protocol_adaptation |
| prompt | GT 1 pos + 1 neg（8 像素环） | 同 | same（oracle-prompted） |
| checkpoint | 官方 `3M_LargeVessel_Global.pth`（缺失） | 自训 best val Dice | missing asset |
| postprocess | sigmoid at original_size | 同 | same |
| threshold | 0.8 | 0.8 | same |
| metric | Dice/Jaccard/Hausdorff @ original | Dice/IoU/clDice/HD95 @ GT | **differ，未披露** |
| export | 每样本 `.npy`（原分辨率） | GT bool mask | equivalent（分辨率一致） |

**G4=unverified**：threshold/resolution/postprocess 与官方一致，但指标集从
Dice/Jaccard/Hausdorff 改为 Dice/IoU/clDice/HD95，adapter/UPSTREAM 未写这一条。

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # commit/remote 可追溯；URL/LICENSE 缺失已记录
G1  = pass                  # matched ViT-L rank4: keys/shape/param/strict copy 全同
G2  = pass (scoped)         # forward 0 diff；reduced-width loss/grad 2.5e-11；step 未测
G3  = unverified            # class-conditional loss、10-fold、seed 差异未完整披露
G4  = unverified            # metric 集未披露
G5  = pass                  # 18/6/20、GT 导出、oracle prompt 角色；fold 替换已声明
G6a = unverified            # 仅单步探针，未做 N>=20 optimizer steps
G6b = N/A                   # 官方训练 checkpoint 不存在
G6c = unverified            # 未全量训练/评测

main_table = blocked
claim_label = none
```

仍缺：官方训练权重、class-conditional loss 披露、10-fold→项目协议的显式说明、
metric 集披露、20-step G6a、全量训练/评测产物。**不能声称官方指标已复现。**

## 最终一行状态

`sam_octa: G0=pass G1=pass G2=pass G3=unverified G4=unverified G5=pass G6a=unverified G6b=N/A G6c=unverified`

## Post-review fixes (2026-10-26)

- Final prediction prompt sampling is seeded from `--seed` before the first eval sample; training validation does not reseed.
- `prepare_train_batch` now raises for `--batch-size > 1` instead of silently dropping samples.
- Tests: `test_sam_octa_eval_seed_and_batch_guard`.
- Remaining: class-conditional clDice and official metric set are not reproduced; G3/G4 remain unverified.
