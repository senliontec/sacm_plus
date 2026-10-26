# SAM2-UNet 移植审计（手工 pilot）

- 官方：`3rd/SAM2-UNet`，commit `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e`
- 移植：`src/baseline_zoo/models/sam2_unet/**` + `src/baseline_zoo/adapters/sam2_unet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`probe_standard_sam`；Apache-2.0 `LICENSE`；**附录/探针，不进主表**。
- 资产：`checkpoints/sam2_hiera_large.pt` **缺失**；官方无 SAM2-UNet 训练 checkpoint。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 只用于定位问题。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-s2-01 | 官方 commit/文件/许可证/SAM2 权重 | `UPSTREAM.md:1-7`; `train.py:46` | 来源可追溯；权重可加载 |
| S-s2-02 | Hiera-L + Adapter/prompt_learn + RFB + U-Net，三输出 | `SAM2UNet.py:124`; `sam2/build_sam.py:30-40` | 同 key/shape/param/buffer |
| S-s2-03 | 20 epochs、batch 12、AdamW all params、lr 1e-3/wd 5e-4、cosine eta 1e-7 | `train.py:23-27,48-49` | 训练配方逐项一致 |
| S-s2-04 | `structure_loss` 三项相加（weighted BCE + weighted IoU） | `train.py:31-39,58-62` | 同 loss 数值/梯度 |
| S-s2-05 | 352 bilinear image、bicubic soft mask、ImageNet norm、flips | `dataset.py:19-32,56-80` | 同几何/数值/增强 |
| S-s2-06 | DataLoader shuffle、workers 8、无 seed、无 `model.train()` | `train.py:44,74-86`; `build_sam.py:38-39` | 训练模式/随机流需说明 |
| S-s2-07 | 无 val，每 5 epoch/末尾保存快照 | `train.py:69-71` | 选择语义 |
| S-s2-08 | test strict load、GT 上采样、sigmoid、min-max 0-255、不二值化 | `test.py:25-47` | 推理/导出语义 |
| S-s2-09 | py_sod_metrics 概率图指标 | `eval.py:1-45` | 官方指标口径 |
| S-s2-10 | 项目角色 appendix/probe | `inventory`; `UPSTREAM.md:5-7` | 不做 L6、不进主表 |
| S-s2-11 | 官方方法 checkpoint | 无 | G6b N/A |

## 02. 移植对照与差异

```text
# 3rd/SAM2-UNet/train.py:23-27,44,51-71
epoch=20; lr=1e-3; batch_size=12; weight_decay=5e-4
DataLoader(..., shuffle=True, num_workers=8)
for epoch in range(args.epoch):
    for ...:
        pred0, pred1, pred2 = model(x)
        loss = structure_loss(pred0,target)+structure_loss(pred1,target)+...
# train.py never calls model.train()

# src/baseline_zoo/adapters/sam2_unet.py:140-201
freeze encoder; re-enable prompt_learn; train RFB/decoder/head
AdamW(requires_grad params, lr, wd); CosineAnnealingLR(T_max=epochs, eta_min=1e-7)
masks = bicubic 352 soft; structure_loss x3
```

| id | verdict | category | gate | 官方 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-s2-01 | `unverified` | `missing_asset` | G2/G6a | SAM2 Hiera-L 权重缺失；移植训练必须显式提供，`--random-init` 仅 smoke | 已披露（`adapter.py:85-88,123-132`; `UPSTREAM.md:63-71`） | 无同权重 forward；G2/G6a 只能 unverified |
| D-s2-02 | `intentional_deviation` | `protocol_adaptation` | G3 | 官方 20 epochs/batch12/无 val；移植 200 epochs/batch4/val Dice 选择 | 已披露（`adapter.py:65-68`） | 训练预算/有效 batch 不同；不是官方复现 |
| D-s2-03 | `intentional_deviation` | `wrapper_change` | G3 | 官方 build 后 `model.eval()` 且训练循环不 `train()`；移植 runtime 每 epoch `model.train()` | 已披露（`adapter.py:69-71`; `UPSTREAM.md:51-54`） | BN/dropout 模式不同；Hiera 无 BN，decoder BN 受影响 |
| D-s2-04 | `equivalent` | `wrapper_change` | G3 | 官方 AdamW 收全部参数；移植只收 `requires_grad` 参数 | 已披露（`adapter.py:72-74`） | 冻结 Hiera 无梯度，优化器状态等价；无数值训练对照 |
| D-s2-05 | `differ` | `reproducibility` | G3 | 官方 workers=8、无 seed；移植 workers=0、协议 seed | workers 未披露；seed 未在 deviations 明写 | 数据顺序/增强随机流不同；未测 |
| D-s2-06 | `intentional_deviation` | `protocol_adaptation` | G4 | 官方输出 min-max 概率图、不二值化、SOD 指标；移植 sigmoid>0.5、GT bool、项目指标 | 阈值/导出已披露（`adapter.py:78-81`）；指标替换属项目协议 | 概率图→二值 mask 会改变边界指标；未做双口径对照 |
| D-s2-07 | `equivalent` | `wrapper_change` | G1 | 官方/移植 model 仅 import 重写 + `device` 参数；`structure_loss` 用 `reduction` 替代已移除的 `reduce` | 已披露（`UPSTREAM.md:29-54`; `adapter.py:33-39`） | 两侧 1,192 keys / 216,529,891 params / 204 buffers 全同 |
| D-s2-08 | `unverified` | `missing_asset` | G6b | 官方无 SAM2-UNet 训练权重 | 已披露（`UPSTREAM.md`） | G6b=N/A；无 checkpoint parity |
| D-s2-09 | `unverified` | `missing_asset` | G6c | 无全量项目训练/评测产物 | — | G6c=unverified |
| D-s2-10 | `intentional_deviation` | `profile` | G0/G6 | 附录/探针，不进主表、不做 L6 | 已披露（`adapter.py:63-64`; `UPSTREAM.md:5-7`） | 只能附录，不能作为 SACM 主表 baseline |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-s2-01 | `unverified` | D-s2-01；SAM2 权重缺失 |
| S-s2-02 | `preserved` | D-s2-07；03 state-dict probe |
| S-s2-03 | `covered_by_discrepancy` | D-s2-02 |
| S-s2-04 | `preserved` | D-s2-07；`structure_loss` 等价 |
| S-s2-05 | `preserved` | 官方/移植 image bilinear + mask bicubic + norm + flips |
| S-s2-06 | `covered_by_discrepancy` | D-s2-03, D-s2-05 |
| S-s2-07 | `covered_by_discrepancy` | D-s2-02 |
| S-s2-08 | `covered_by_discrepancy` | D-s2-06 |
| S-s2-09 | `covered_by_discrepancy` | D-s2-06；官方 SOD 指标未复现 |
| S-s2-10 | `covered_by_discrepancy` | D-s2-10 |
| S-s2-11 | `not_applicable` | D-s2-08；无官方方法 checkpoint |

closure = 11/11；open 实质项是 SAM2 权重缺失、workers/seed 未披露、指标口径替换。

## 03. 数值探针结果

```bash
/home/dev/anaconda3/envs/tsai/bin/python /tmp/sam2_structure.py
```

```text
official/port: params 216,529,891 / 216,529,891
               buffers 204 / 204
               keys 1,192 / 1,192
               exact keys/shapes/dtypes True
               missing 0; extra 0; shape mismatches none
SAM2 checkpoint: checkpoints/sam2_hiera_large.pt absent
```

`G2=unverified`：缺 SAM2 Hiera-L 权重，不能用 random-init 做 parity；没有固定输入
forward/输出对照。`G6a=unverified`；`G6c=unverified`（未跑全量项目训练/评测）。
本模型按 `probe_standard_sam` 只审 L1–L5，不做 L6。

## 04. 训练配置审计

| 项 | 官方 SAM2-UNet | 移植 | 判定 |
|---|---|---|---|
| epochs | 20 | 200（项目预算） | intentional，已披露 |
| batch | 12 | 4（Hiera-L 显存） | intentional，已披露 |
| input | 352×352 | 352×352 | same |
| optimizer | AdamW all params, 1e-3, wd 5e-4 | AdamW requires_grad, 1e-3, wd 5e-4 | equivalent |
| scheduler | cosine, `T_max=epochs`, eta 1e-7 | 同 | same |
| loss | 3 × structure_loss | 同（`reduction='none'`） | preserved |
| transforms | bilinear image / bicubic soft mask / H+V flips / ImageNet | 同 | preserved |
| data | 官方 train image/mask 目录 | 项目 18 张 train + 6 val | protocol adaptation |
| workers | 8 | 0 | **differ，未披露** |
| seed | 无（seed 代码被注释） | 协议 seed | 未在 deviations 明写 |
| train mode | build 后 eval，循环不 train() | runtime 每 epoch train() | intentional，已披露 |
| selection | 无 val；每 5 epoch/末尾快照 | 每 epoch val Dice best | intentional，已披露 |

G3=unverified：缺权重无法做同初始化训练对照；workers/seed 未进入 deviations；
200/4/val 和 train-mode 差异本身已披露，但不构成官方复现。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | 352 bilinear + ImageNet norm | 同 | preserved |
| checkpoint | strict load，全模型 | 全 state dict；无 SAM2 权重也能预测 | equivalent |
| output | GT 上采样 → sigmoid → min-max 0–255 概率图 | GT 上采样 → sigmoid >0.5 bool | intentional，已披露 |
| metric | py_sod_metrics（adaptive/dynamic F, Smeasure, MAE, ...） | Dice/IoU/clDice/HD95 | protocol adaptation；未做双口径对照 |
| export | 255 灰度 PNG | GT bool mask | protocol adaptation，已披露 |
| resolution | GT | GT | same |
| TTA | 无 | 无 | same |

G4=pass：阈值/导出替换在 adapter 中明确披露；项目指标替换属于全局评测协议，
不是隐藏的官方指标复用。G5=pass：使用共享 18/6 split，无 support/donor，无泄漏。

## 07. 门禁结论（provisional）

```text
A0  = unverified
G0  = pass                  # repo/commit/LICENSE 可追溯；SAM2 权重缺失已披露
G1  = pass                  # 1,192 keys、216,529,891 params、204 buffers 全同
G2  = unverified            # 缺 SAM2 Hiera-L 权重
G3  = unverified            # 无同权重训练；workers/seed 未披露
G4  = pass                  # 阈值/导出/指标替换已披露
G5  = pass                  # 18/6 split；无 support/donor；无泄漏
G6a = unverified            # 缺 SAM2 权重/同初始化训练
G6b = N/A                   # 无官方 SAM2-UNet checkpoint
G6c = unverified            # 未跑全量训练/评测

main_table = blocked        # 且 profile 本身为附录/探针，不进主表
claim_label = none
official_metric_match = unverified
```

仍缺：补 SAM2 Hiera-L 权重；把 workers=0/seed 写入 deviations；补同权重 forward
与 20-step G6a；如要比较官方 SOD 指标，需另存概率图并做双口径；跑全量 G6c；
冻结 A0 calibration pack。
