# SAM-Med2D 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方仓库：`3rd/SAM-Med2D`（origin: `OpenGVLab/SAM-Med2D`）
- 官方 commit：`bfd2b93b1158100c8abd81f61766a2de92c1c175`
- 移植版本：`src/baseline_zoo/models/sam_med2d/` + `src/baseline_zoo/adapters/sam_med2d.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25
- prompt 语义：`supervised_oracle_prompted_sam`（训练/评测均用 GT 生成 box/point；不是 prompt-free）

> **A0 状态：unverified。** 没有冻结 calibration pack；本报告是手工 pilot，不是已校准审计，
> 不能单独作为主表证据。A0 通过前 G0–G6c 只用于定位问题。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-01 | ViT-L + 256 输入 + per-block Adapter 结构；state_dict 映射 | `segment_anything/build_sam.py:45-129` | 键/shape/参数量/buffer 一致；不一致则 G1 fail |
| S-02 | 初始化/RNG；Adapter 从随机开始 | `segment_anything/build_sam.py:93-129` | 同 seed 下结构一致；shared 权重可 strict copy |
| S-03 | BGR 读取 + SAM mean/std（非 ImageNet） | `DataLoader.py:57-58`；`utils.py:161-168` | 预处理逐像素一致；改 RGB 会整体偏移 |
| S-04 | 训练 transform：小于 256 双维 pad，否则 Resize(256,256,NEAREST)，无随机增强 | `utils.py:161-168` | 同输入张量一致；误加归一化/随机增强会改变分布 |
| S-05 | 训练 mask/prompt 采样：`mask_num=5`、GT box、单点采样 | `DataLoader.py:145-179`；`utils.py:117-158` | 二值数据重复同一 mask 5 次；多类官方数据应有 5 个不同标签 |
| S-06 | 训练 prompt 交替：50% box / 50% point；8 轮 error-feedback refinement | `train.py:112-117,164-202` | 轮数、init_mask_num 区间、point_list 一致；减少轮数会降低训练信号 |
| S-07 | loss = 20×Focal + Dice + MaskIoU | `utils.py:336-358` | loss/grad 同量级；漏 IoU 项会改变梯度 |
| S-08 | Adam lr=1e-4，无 weight decay；encoder 只训练 Adapter | `train.py:119-123,225` | 参数更新集合一致；误冻 decoder 会不收敛 |
| S-09 | DataLoader batch=2, workers=4, shuffle=True；全局 random/np.random 无 seed | `train.py:245-246`；`DataLoader.py:153` | 数据顺序/prompt 随机流一致；workers 不同会改变 RNG 消费时序 |
| S-10 | checkpoint 选择：best **训练 loss**（无 val） | `train.py:265-275` | 官方无验证集；用 val Dice 选择属于协议改动 |
| S-11 | 评测 prompt：GT box；multimask max-IoU；256→原图后处理 | `test.py:33,91-110,175-183` | 评测 prompt 必须来自 GT；错用 point/阈值会改指标 |
| S-12 | 阈值 0.5、GT 分辨率导出、官方 Dice/IoU | `test.py:183,190`；`metrics.py:26-44`；`utils.py:222-227` | 阈值/分辨率一致；metric 口径需披露 |
| S-13 | 官方 checkpoint `pretrain_model/sam-med2d_b.pth`；`load_from` 排除三个 decoder 组 | `train.py:34`；`build_sam.py:132-170` | 缺权重则无法复现官方初始化；移植继承整个 decoder 属有意差异 |
| S-14 | 来源、许可证、UPSTREAM 路径 | `UPSTREAM.md:3-10` | URL/commit/LICENSE 可追溯；路径过期属文档问题 |
| S-15 | 项目协议：18 train / 6 val / 20 test roots、GT 导出、oracle 角色 | `cod reviewer/README.md:63-96`；`runtime.py:481-485,740-785` | 协议可核对；缺失披露则 G5 不能 pass |

## 02. 移植对照与差异

| id | level/gate | category | verdict | documented | 官方 vs 移植 | 证据（file:line） | 数值影响 |
|---|---|---|---|---|---|---|---|
| D-01 | L1/G1/G3 | `wrapper_change` | `intentional_deviation` | yes | 官方默认 `vit_b`；移植 ViT-L、256 adapter | `train.py:33`；`adapters/sam_med2d.py:84-90`；`UPSTREAM.md:14-17` | matched ViT-L 下无结构差 |
| D-02 | L2/G2 | `wrapper_change` | `equivalent` | probe | 同一随机 state_dict 复制到两侧 | `03_numeric_probe.json` | forward ≤2.9e-6；loss/grad ≤2.1e-6 |
| D-03 | L3/G3 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 5 个不同标签；二值协议重复同一 mask 5 次 | `DataLoader.py:150-176`；`adapters/sam_med2d.py:120-138` | 二值数据上等价；多类数据不等价 |
| D-04 | L3/G3 | `wrapper_change` | `differ` | **no** | 官方全局 `random`/`np.random` 无 seed；移植用 `Random(seed+epoch)` 控制 box/point、轮数，但 mask/point 采样仍走全局 `np.random` | `train.py:112-117,164-175`；`adapters/sam_med2d.py:193,210-231` | 未量化；训练轨迹不同 |
| D-05 | L3/G3 | `official_recipe` | `equivalent` | code | 官方 Adam 传全部参数；移植只传 trainable | `train.py:225`；`adapters/sam_med2d.py:99-102` | 冻结参数无梯度，更新集合等价；无 step 探针 |
| D-06 | L3/G3 | `protocol_adaptation` | `differ` | **no** | 官方 workers=4、无显式 seed；移植 workers=0、固定 seed | `train.py:245-246`；`runtime.py:481-485,850-851` | 未量化；数据顺序/随机流不同 |
| D-07 | L3/G4 | `protocol_adaptation` | `intentional_deviation` | yes | 官方用训练 loss 选 best；移植用 6 图 val Dice | `train.py:265-275`；`runtime.py:607-656` | 选择准则不同；不可比 best 权重 |
| D-08 | L4/G4 | `wrapper_change` | `equivalent` | probe/code | 官方 GT box + multimask max-IoU + 256→原图；移植相同 | `test.py:33,102-110,183`；`adapters/sam_med2d.py:257-277` | forward 路径等价 |
| D-09 | L4/G4 | `protocol_adaptation` | `differ` | partial | 官方 Dice/IoU（eps=1e-7）@GT；移植共享评测器 Dice/IoU/clDice/HD95 @GT | `metrics.py:26-44`；`runtime.py:217-264` | 额外指标不影响 Dice/IoU；公式/空掩膜差异未逐项披露 |
| D-10 | L0/G2/G6b | `missing_asset` | `intentional_deviation` | yes | 官方 SAM-Med2D checkpoint 缺失；移植从 vanilla SAM ViT-L 开始；`load_from` 继承整个 decoder | `train.py:34`；`build_sam.py:132-170`；`models/.../build_sam.py:132-152`；`UPSTREAM.md:14-25` | decoder 初始化与官方不同；G6b N/A |
| D-11 | L0/G0 | `doc_drift` | `differ` | **no** | UPSTREAM 仍写 `baselines/sam-med2d`，实际为 `3rd/SAM-Med2D` | `UPSTREAM.md:4` | 无数值影响 |
| D-12 | L3/L5 | `protocol_adaptation` | `intentional_deviation` | yes | 官方 data_demo/JSON manifests → 项目 18/6 3-shot + 20 test | `DataLoader.py:140-150`；`UPSTREAM.md:56-58`；`runtime.py:475-485` | 数据分布/样本数不同 |

**关键证据（3–5 行）**

```text
# 官方 prompt refinement（train.py:164-178）
point_num = random.choice(args.point_list)
batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
init_mask_num = np.random.randint(1, args.iter_point - 1)
for iter in range(args.iter_point):
    if iter == init_mask_num or iter == args.iter_point - 1: setting_prompt_none(...)
# 移植（adapters/sam_med2d.py:222-235）用 rng.choice / rng.randint(1, iter_point-2)，区间同为 1..6
```

```text
# 官方 load_from 排除 decoder 组（build_sam.py:134-137）
except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
new_state_dict = {k: v for k, v in state_dicts.items() if k in sam_dict.keys() and ...}
# 移植 build_sam.py:143-152 只跳过 shape 不兼容项，因此继承整个 SAM decoder
```

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-01 | `preserved` | D-01, D-02 |
| S-02 | `preserved` | D-01, D-02 |
| S-03 | `preserved` | 探针 preprocess diff=0 |
| S-04 | `preserved` | D-03（仅二值复用） |
| S-05 | `covered_by_discrepancy` | D-03, D-04 |
| S-06 | `covered_by_discrepancy` | D-04 |
| S-07 | `preserved` | 探针 loss/grad |
| S-08 | `preserved` | D-05 |
| S-09 | `covered_by_discrepancy` | D-04, D-06 |
| S-10 | `covered_by_discrepancy` | D-07 |
| S-11 | `preserved` | D-08 |
| S-12 | `covered_by_discrepancy` | D-09 |
| S-13 | `covered_by_discrepancy` | D-10 |
| S-14 | `covered_by_discrepancy` | D-11 |
| S-15 | `covered_by_discrepancy` | D-12 |

closure = 15/15。D-04/D-06/D-09/D-11 仍属未完整披露，因此 G3/G4 不能给完整 pass。

## 03. 数值探针结果

详见 `03_numeric_probe.json`。探针均为 **随机初始化 + 官方 state_dict 严格复制到移植侧**；
不是官方权重复现，也不是 G6a。

```text
命令（GPU 1，tsai 环境）：
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_med2d.py
  CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/probe_sam_med2d_grad.py

state_dict keys/shapes equal:        True / True；missing=0 unexpected=0
param count official/port:           950,139,184 / 950,139,184
buffer count official/port:          3 / 3；strict copy: True
preprocess max_abs_diff:             0.0
image_encoder max_abs_diff:          2.86102294921875e-06
box masks / low-res / iou:           1.75e-07 / 2.16e-07 / 1.49e-08
point masks / none masks:            1.90e-07 / 1.83e-07
one-step loss official/port:         3.0393178462982178 / 3.0393178462982178 (abs 0.0)
grad max_abs_diff:                   2.0837851479882374e-06（264 个比较张量）
grad norm official/port:             7.585201869670705 / 7.5852017420259426
```

结论：在 matched ViT-L/256 配置下，结构、forward、官方 `FocalDiceloss_IoULoss` 的 loss/grad
都在 1e-5 内。**未比较 optimizer step，未用官方 SAM-Med2D 权重，未做 20-step G6a。**

## 04. 训练配方审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 15 | 15 | same |
| batch | 2 | 2 | same |
| input | 256 | 256 | same |
| optimizer | Adam 1e-4, wd=0 | Adam 1e-4, wd=0 | same |
| scheduler | 默认 None | None | same |
| loss | 20×Focal+Dice+IoU | 同 | same（探针 0 diff） |
| mask_num / iter_point | 5 / 8 | 5 / 8 | same |
| point_list / prompt | [1,3,5,9]；box/point 50-50；GT 生成 | 同 | same（RNG 路径不同） |
| transform | BGR、SAM stats、pad/resize、无增强 | 同 | same |
| 数据 | 官方 JSON manifests | 18/6 3-shot | protocol_adaptation |
| workers | 4 | 0 | **diff，未披露** |
| seed | 无显式 seed | 共享 runtime seed + epoch RNG | **diff，未披露** |
| checkpoint 选择 | best train loss | 6 图 val Dice | protocol_adaptation |
| 初始化 | 官方 SAM-Med2D ckpt | vanilla SAM ViT-L | **missing asset** |

结论：**G3=unverified**——D-04/D-06（workers/RNG）没有进入 deviations；
即使补上，也不是官方数据/初始化复现。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | BGR+SAM stats，256 pad/resize | 同 | same |
| prompt | GT box | GT box | same（oracle-prompted） |
| multimask | max-IoU 选 mask | 同 | same |
| postprocess | 256→原图；<256 双维裁剪 | 同 | same |
| threshold | 0.5 | 0.5 | same |
| checkpoint | 官方 SAM-Med2D（缺失） | 自行训练的 best val Dice | missing asset / protocol |
| metric | Dice/IoU（eps）@GT | Dice/IoU/clDice/HD95（medpy）@GT | 部分披露 |
| export | GT 分辨率 mask | GT bool mask | same |

**G4=pass（带范围）**：eval transform/prompt/postprocess/threshold/export 与官方一致；
官方 Dice/IoU 由共享 evaluator 保留，额外 clDice/HD95 不影响官方两项。但 metric 公式/空掩膜
约定只以 “shared evaluator” 概括，建议在 deviations 中写全。

## 07. 门禁结论（provisional）

```text
A0  = unverified            # calibration pack 未冻结
G0  = pass                  # commit/remote/LICENSE/URL 可追溯；缺官方权重已记录
G1  = pass                  # matched ViT-L: keys/shape/param/buffer/strict copy 全同
G2  = pass (scoped)         # forward/loss/grad 通过；optimizer step 未单测，无官方权重
G3  = unverified            # workers/RNG 差异未披露；官方数据/初始化缺失
G4  = pass (scoped)         # eval 路径一致；metric 仅部分披露
G5  = pass                  # 18/6/20、manifest hash、GT 导出、oracle 角色
G6a = unverified            # 只有单步探针，未做 N>=20 optimizer steps
G6b = N/A                   # 官方 SAM-Med2D checkpoint 不存在
G6c = unverified            # 未全量训练/评测

main_table = blocked
claim_label = none
```

仍缺：官方 SAM-Med2D 权重、metric 口径逐项披露、workers/RNG 披露、20-step G6a、
全量训练与评测产物。**不能声称官方指标已复现。**

## 最终一行状态

`sam_med2d: G0=pass G1=pass G2=pass G3=unverified G4=pass G5=pass G6a=unverified G6b=N/A G6c=unverified`
