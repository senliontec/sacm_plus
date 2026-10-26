# KnowSAM 移植审计（手工 pilot）

> **Post-A0 重审（2026-10-26）：** 本报告表格仍是修复前证据；A0 已 pass
> （pack v3），当前代码级 gate 状态见 [`../A0_REAUDIT.md`](../A0_REAUDIT.md)。

- 官方：`3rd/KnowSAM`，commit `b5de98d7f5538e39447d8f6e143c8292875cd22e`
- 移植：`src/baseline_zoo/models/knowsam/**` + `src/baseline_zoo/adapters/knowsam.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`semi_supervised_prompt_free_sam`；官方无 LICENSE，URL 未记录。
- 资产：SAM ViT-B `sam_vit_b_01ec64.pth` 存在；官方无 KnowSAM 训练 checkpoint。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 只用于定位问题。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-kn-01 | commit/文件/权重/许可证 | `UPSTREAM.md:3-5`; repo | 来源、SAM ViT-B、LICENSE 可追溯 |
| S-kn-02 | SGDL + SAM ViT-B 结构/state dict | `Model/model.py`; `Model/sam/` | key/shape/param/buffer 一致 |
| S-kn-03 | 12 labeled + 12 unlabeled、batch 24、50k iter | `train_semi_SAM.py:47-57,105-109` | 训练 batch/预算；3+3 是扩展 |
| S-kn-04 | losses、Adam/SGD、LR、consistency/MixUp、best SGDL Dice | `trainer.py:31-47,178-299` | 官方配方逐项保留 |
| S-kn-05 | weak/strong/valid transforms（256）与每 200 iter 验证 | `transforms.py:31-69`; `train_semi_SAM.py:125-129` | 几何/插值/选择一致 |
| S-kn-06 | unlabeled pool 语义 | `dataset.py:26-35`; `train_semi_SAM.py:105-109` | 不得含 test；否则半监督泄漏 |
| S-kn-07 | seed / workers | `train_semi_SAM.py:112-113,136-139` | 官方 seed=2024、workers 2/1 |
| S-kn-08 | 预测模型 `bilinear` 架构 | `prediction_ACDC.py:149`; `prediction.py:74` | 训练/预测 flag 需一致 |
| S-kn-09 | fusion、zoom、输出分辨率、largest-CC | `prediction_ACDC.py:162-186` | argmax@256 → nearest zoom → GT → CC |
| S-kn-10 | prompt-free 语义 | `trainer.py:242-257`; `prediction_ACDC.py` | 不得用 GT box/point |
| S-kn-11 | 项目 support/donor | 官方无 support | semi-supervised，support/donor N/A |

## 02. 移植对照与差异

```text
# src/baseline_zoo/adapters/knowsam.py:127-135,145-160
if row.get('split') not in ('train', 'val'): continue   # test rows NOT excluded
for domain in SOURCE_DOMAINS:                            # first domains win
    for sample in RawPairDataset(domain_root).samples:
        if count >= unlabeled_limit: break               # default 384

# 3rd/KnowSAM/prediction_ACDC.py:149,176-186
SGDL_model = KnowSAM(args, bilinear=True)...
out_SGDL = torch.argmax(fusion_map_soft, dim=1)...
pred_SGDL = zoom(out_SGDL, ..., order=0)
disc_pred = getLargestCC(disc_pred)

# src/baseline_zoo/adapters/knowsam.py:287,318-325
model = KnowSAM(kargs, bilinear=False)...
prob = torch.softmax(fusion, dim=1)[0, 1]
prob = F.interpolate(prob, GT, mode='bilinear'); pred = prob > 0.5
```

| id | verdict | category | gate | 官方 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-kn-01 | `intentional_deviation` | `protocol_adaptation` | G3 | 官方 12+12/batch24；移植 3+3/batch6 | 已披露（`adapter.py:55-57`） | 有效 labeled/unlabeled batch 变小；无训练数值对照 |
| D-kn-02 | `differ` | 数据泄漏 | G3/G4/G5 | unlabeled pool 只排除 train/val，**包含完整 test 图** | **未披露**；文档称“six full training pools” | **166/384** 是 test：DRIVE 16/16、CHASEDB1 20/20、DCA1 130/130；image+mask 全像素相等 |
| D-kn-03 | `differ` | `protocol_adaptation` | G3/G5 | 384 cap 只覆盖前 4 域，CrackTree/CORN 为 0；218/384 是非 test train-pool 图 | 部分披露（cap 在 `adapter.py:60`；四域结果未写） | 16+24+130+214=384；218 张作 unlabeled 合法，但“六域”覆盖不成立 |
| D-kn-04 | `differ` | `postprocess` | G4 | 官方每个类做 `getLargestCC`；移植不做 | **未披露** | 小连通域保留；指标方向未测 |
| D-kn-05 | `differ` | `postprocess` | G4 | 官方 argmax@256 + `zoom(order=0)`；移植 bilinear 概率上采样 + >0.5 | 部分披露（仅 `bilinear=False` 与 GT 导出） | 融合/插值/阈值不同；adapter 的“upstream 在 256 报指标”对 ACDC 路径不准确 |
| D-kn-06 | `differ` | `wrapper_change` | G3 | workers 官方 2/1；移植 0/0 | **未披露** | 数据加载并行度不同，样本集合不变 |
| D-kn-07 | `intentional_deviation` | `reproducibility` | G3 | 官方实际 seed 2024（`--seed` 未用于 seeding）；移植协议 seed | 已披露（`adapter.py:62-68`） | 随机流不同；已固定 |
| D-kn-08 | `equivalent` | `wrapper_change` | G4 | 每 200 iter 验证、best SGDL Dice；移植调用同一 Trainer | 已披露（`adapter.py:66-68`） | 选择逻辑保留；未全量跑 |
| D-kn-09 | `unverified` | `missing_asset` | G6b/G6c | 官方无训练 checkpoint；无全量项目训练/评测 | 已披露 | G6b=N/A；G6c=unverified |
| D-kn-10 | `unverified` | `missing_provenance` | G0 | 官方 URL 未记录、无 LICENSE | 已记录（`UPSTREAM.md:3-5`） | G0 只能 pass(caveat) |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-kn-01 | `covered_by_discrepancy` | D-kn-09, D-kn-10 |
| S-kn-02 | `preserved` | 03 SGDL state-dict probe；SAM 源文件仅 import/checkpoint guard 差异 |
| S-kn-03 | `covered_by_discrepancy` | D-kn-01 |
| S-kn-04 | `preserved` | D-kn-08；同一 Trainer 代码 |
| S-kn-05 | `preserved` | D-kn-08；官方 transform 直接调用 |
| S-kn-06 | `covered_by_discrepancy` | D-kn-02, D-kn-03 |
| S-kn-07 | `covered_by_discrepancy` | D-kn-06, D-kn-07 |
| S-kn-08 | `covered_by_discrepancy` | D-kn-05 |
| S-kn-09 | `covered_by_discrepancy` | D-kn-04, D-kn-05 |
| S-kn-10 | `preserved` | 训练/推理均无 GT prompt；`super_prompt` 是可学习 prompt |
| S-kn-11 | `not_applicable` | semi-supervised，无 support/donor |

closure = 11/11；open 实质项：**D-kn-02 数据泄漏**、D-kn-04 缺 CC、D-kn-06 未披露。

## 03. 数值探针结果

```bash
/home/dev/anaconda3/envs/tsai/bin/python /tmp/knowsam_structure.py
/home/dev/anaconda3/envs/tsai/bin/python /tmp/knowsam_leak_check.py
```

```text
G1 SGDL: 5,250,856 params; 78 buffers; 244 keys; exact keys/shapes/dtypes True
         SAM modeling source: image_encoder/mask_decoder/prompt_encoder diff empty;
         sam.py/build_sam.py only import path + checkpoint guard
candidates after train/val exclusion:
  DRIVE 16, CHASEDB1 24, DCA1 130, CREMI 369, CrackTree200 202, CORN 1173
selected 384: DRIVE 16 + CHASEDB1 24 + DCA1 130 + CREMI 214; others 0
test leaks: 166/384; DRIVE 16/16, CHASEDB1 20/20, DCA1 130/130,
            CREMI 0/123, CrackTree200 0/202, CORN 0/152
pixel equality: all 166 image+mask pairs equal; 0 mismatches
non-test selected: 218/384 (214 CREMI + 4 CHASEDB1), allowed train-pool images
```

结论：完整 test 图（DRIVE/CHASEDB1/DCA1）进入了 unlabeled pool。即使不使用 test mask，
半监督伪标签/一致性训练仍直接接触 test 图像，构成 G3/G4/G5 泄漏。218 张非 test 图
本身允许，但说明“六域 full pools”的覆盖声明不成立。

`G2=unverified`：无官方训练 checkpoint，也未做同权重固定输入 forward/loss 探针；
`G6a=unverified`；`G6c=unverified`（未跑全量训练/评测）。

## 04. 训练配置审计

| 项 | 官方 KnowSAM | 移植 | 判定 |
|---|---|---|---|
| iterations | 50,000 | 50,000（config 默认） | same |
| batch | 24（12 + 12） | 6（3 + 3） | intentional，已披露；有效 batch 不同 |
| model | SAM ViT-B adapter + UNet/VNet/SGDL | 同 | preserved（SGDL state dict 全同） |
| SAM lr | 1e-4 Adam | 同 | same |
| SGDL lr | 0.01 SGD, momentum .9, wd 1e-4 | 同 | same |
| loss/consistency/MixUp/schedule | 官方 Trainer | 同一 Trainer 代码 | preserved |
| transforms | weak/strong/valid 256 | 同 | preserved |
| unlabeled pool | 官方 per-dataset `patients_to_slices` | 六域 pool，但含 test/覆盖截断 | **fail（泄漏）** |
| validation | 每 200 iter，best SGDL Dice | 同 | preserved |
| workers | 2 train / 1 val | 0 / 0 | **differ，未披露** |
| seed | 2024（实际） | 协议 seed | intentional，已披露 |
| selection | SGDL val Dice | 同 | preserved |

G3=fail：D-kn-02 是数据泄漏，D-kn-03 覆盖截断未披露，D-kn-06 workers 未披露；
3+3 本身已披露，但不能抵消泄漏。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | valid_test 256 nearest | 同 | preserved |
| checkpoint | SGDL best（`bilinear=False` 训练） | SGDL best，`bilinear=False` | trainer 架构一致 |
| 官方 prediction | `prediction_ACDC.py` 用 `bilinear=True`，与 trainer checkpoint 不自洽；`prediction.py` 也 True | 用 `bilinear=False` | 移植取训练架构，已披露 |
| fusion/resize | argmax@256 → `zoom(order=0)` → GT | bilinear 概率 → GT | **differ** |
| postprocess | `getLargestCC` | 无 | **differ，未披露** |
| threshold | argmax（约 0.5） | >0.5 | 不等价于 CC |
| export | ACDC 路径 GT nearest；prediction.py 256 | GT bool | protocol adaptation；adapter 表述不准确 |
| metric | 官方 Dice/IoU/HD95/ASD | 项目 Dice/IoU/clDice/HD95 | 项目协议 |
| TTA | 无 | 无 | same |

G4=fail：D-kn-04 缺 largest-CC、D-kn-05 fusion/zoom 不同，且 test 泄漏使任何 test 指标
都不是 held-out 证据。G5=fail：unlabeled pool 含 test；cap 只覆盖 4 域。

## 07. 门禁结论（pre-fix snapshot）

> **Superseded:** this is the pre-fix gate snapshot. Post-A0 status is in
> [`../A0_REAUDIT.md`](../A0_REAUDIT.md).


```text
A0  = unverified
G0  = pass (caveat)         # commit/源码可追溯；无 LICENSE、URL 未记录
G1  = pass                  # SGDL 5,250,856 params / 78 buffers / 244 keys 全同
G2  = unverified            # 无官方训练 checkpoint；未做同权重 forward/loss
G3  = fail                  # 166/384 test 图进入 unlabeled pool；workers 未披露
G4  = fail                  # 泄漏 + 缺 largest-CC + fusion/resize 未披露
G5  = fail                  # unlabeled pool 含 test；cap 只覆盖 4 域
G6a = unverified            # 无 checkpoint/同初始化训练探针
G6b = N/A                   # 无官方方法 checkpoint
G6c = unverified            # 未跑全量训练/评测

main_table = blocked
claim_label = none
official_metric_match = unverified
```

仍缺：重建排除 test 的 unlabeled pool，并披露 4 域 cap；补 largest-CC 与官方
argmax/nearest zoom 路径，或完整披露替代协议；把 workers/seed/3+3 写入 deviations；
补同权重 G2 与 ≥20-step G6a；跑全量 G6c；冻结 A0 calibration pack。


## Post-A0 re-audit (2026-10-26)

- A0: pass (`a0-cal-2026-10-26-v3`, `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`).
- Resolved: test-image exclusion (`D-kn-02`), largest-CC (`D-kn-04`), argmax + scipy `zoom(order=0)` (`D-kn-05`); 384-cap/workers disclosed (`D-kn-03`/`D-kn-06`).
- Post-fix unlabeled pool: CHASEDB1 4, CREMI 246, CORN 134; DRIVE/DCA1/CrackTree200 have no non-test unlabeled images.
- Post-fix gates: G0 pass, G1 pass, G2 unverified, G3 unverified, G4 unverified, G5 pass, G6a unverified, G6b N/A, G6c unverified.
- `main_table=blocked`: no post-fix numerical probes and G6c.
