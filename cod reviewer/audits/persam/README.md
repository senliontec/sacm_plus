# PerSAM 移植审计（手工 pilot）

> **Post-A0 重审（2026-10-26）：** 本报告表格仍是修复前证据；A0 已 pass
> （pack v3），当前代码级 gate 状态见 [`../A0_REAUDIT.md`](../A0_REAUDIT.md)。

- 官方：`3rd/Personalize-SAM`，commit `a7e87245156e0b8efcba10a15a745e699c679d9d`
- 移植：`src/baseline_zoo/models/persam/{algorithm.py,model.py,UPSTREAM.md}` + `src/baseline_zoo/adapters/persam.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`training_free_support_based_sam`；SAM ViT-H `sam_vit_h_4b8939.pth`
  SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`；MIT `LICENSE.txt`。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 是手工 pilot 的定位结论，不能单独作为主表证据。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-persam-01 | SAM ViT-H fork、权重、state_dict | `persam.py:69-79`; `per_segment_anything/` | 同结构/同权重；否则全链路无效 |
| S-persam-02 | reference image/mask 读取、三通道 mask | `persam.py:62-66` | 0/255 RGB；否则取不到前景 |
| S-persam-03 | 参考前景均值 + 归一化/raw 双路径 | `persam.py:90-93` | 相似度用 normalized，first predict 用 raw |
| S-persam-04 | cosine sim、4× 上采样、top-1 正负点 | `persam.py:109-125,183-201` | 同 support/target 应得同 points |
| S-persam-05 | attn_sim + target_embedding 首预测 | `persam.py:127-139` | 两个 prompt 都要进第一次 predict |
| S-persam-06 | mask-input + box/mask-input 两段精修 | `persam.py:142-163` | 段数、顺序、multimask 一致 |
| S-persam-07 | 空 mask 路径 | `persam.py:151-155` | 官方无 guard，空 mask 在 `x.min()` 崩溃 |
| S-persam-08 | one-shot 语义 | `persam.py:54-56,97` | 官方单参考；项目 3-shot 是扩展 |
| S-persam-09 | support/donor 来源 | 官方对象目录 | 项目只允许 train support，不得 test/val 泄漏 |
| S-persam-10 | 输出分辨率/阈值/TTA/确定性 | `persam.py:165-180` | 原分辨率、bool mask、无 TTA |
| S-persam-11 | 方法 checkpoint | 官方无 PerSAM 训练权重 | 只加载 SAM base；G6b N/A |

## 02. 移植对照与差异

```text
# 3rd/Personalize-SAM/persam.py:90-93,133-139
target_embedding = target_feat.mean(0).unsqueeze(0)
target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
target_embedding = target_embedding.unsqueeze(0)
...
masks, scores, logits, _ = predictor.predict(...,
    attn_sim=attn_sim, target_embedding=target_embedding)  # raw

# src/baseline_zoo/models/persam/algorithm.py:46-48,75-81
embedding = target_feat.mean(0).unsqueeze(0)
embedding = embedding / embedding.norm(dim=-1, keepdim=True)
return embedding.unsqueeze(0)          # normalized, used for BOTH
```

| id | verdict | category | gate | 官方 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-persam-01 | `differ` | `wrapper_change` / 内部 bug | G2/G6a | 官方首次 predict 用 raw mean；移植在相似度和首预测都用 normalized | **未披露**；`UPSTREAM.md:35-40` 只说序列保留 | 最终 mask IoU 0.9999866（3 px）；首 mask IoU 0.9993839（137 px）；首 logits max diff 0.716；非零 |
| D-persam-02 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 官方 1 shot；移植默认 3 shots 独立预测 + 0.5 majority | 已披露（`adapter.py:40-43,70`） | 官方无 3-shot 模式；未做数值对照 |
| D-persam-03 | `equivalent` | `wrapper_change` | G2 | 官方 BGR2RGB mask；移植把 0/1 mask 重复成 0/255 RGB | 已披露（`adapter.py:48-49`） | 对平面二值 mask 等价；探针正常取前景 |
| D-persam-04 | `intentional_deviation` | `protocol_adaptation` | G5 | 官方对象目录；移植 seen 用 3 train shots，unseen/novel 用 donor | 已披露（`adapter.py:44-45`） | `support.py:27-79` 仅 train，无 test/val support |
| D-persam-05 | `intentional_deviation` | `bug_fix` | G2/G4 | 官方空 mask 崩溃；移植首/次 mask 空时提前返回 | 已披露（`adapter.py:50-51`） | 探针两路非空；空路径未数值对照 |
| D-persam-06 | `equivalent` | `wrapper_change` | G3/G4/G6a | 输出原分辨率 bool mask、阈值 0.5、无 TTA；prediction 不 seed | 大部分已披露（`UPSTREAM.md:49-53,70-72`） | 固定 DRIVE 对 3 次重复全等 |
| D-persam-07 | `equivalent` | `wrapper_change` | G1 | SAM fork 仅去 TinyViT/vit_t、改 `torch.load`、类型注解 | 部分披露（`UPSTREAM.md:28-34`） | 594 keys/641,090,608 params/3 buffers 全同 |
| D-persam-08 | `unverified` | `missing_asset` | G6b | 官方无方法 checkpoint；只加载 SAM ViT-H | 已披露（`UPSTREAM.md:68-71`） | 无 checkpoint parity；G6b=N/A |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-persam-01 | `preserved` | D-07；state-dict probe |
| S-persam-02 | `covered_by_discrepancy` | D-03 |
| S-persam-03 | `covered_by_discrepancy` | D-01 |
| S-persam-04 | `preserved` | 03 `same_points=true` |
| S-persam-05 | `covered_by_discrepancy` | D-01 |
| S-persam-06 | `covered_by_discrepancy` | D-01（后两段同逻辑） |
| S-persam-07 | `covered_by_discrepancy` | D-05 |
| S-persam-08 | `covered_by_discrepancy` | D-02 |
| S-persam-09 | `covered_by_discrepancy` | D-04 |
| S-persam-10 | `covered_by_discrepancy` | D-06 |
| S-persam-11 | `not_applicable` | D-08；无官方方法权重 |

closure = 11/11；实质 open item 是 **D-persam-01 未披露**。因此本报告不能作为主表证据。

## 03. 数值探针结果

```bash
CUDA_VISIBLE_DEVICES=1 /home/anaconda3/envs/tsai/bin/python /tmp/persam_probe.py
/home/anaconda3/envs/tsai/bin/python /tmp/persam_structure.py
```

固定 support `data/sacm_3shot/train/images/DRIVE_40_training.tif` + mask，target
`data/sacm_3shot/test/DRIVE/images/DRIVE_21_training.tif`（584×565），SAM ViT-H：

```text
structure: 641,090,608 params; 3 buffers; 594 keys; exact keys/shapes/dtypes True
embedding: raw_norm 1.44396877; normalized_norm 0.99999994; cosine(raw,norm) 1.0
           port vs official normalized max_abs_diff 0.0
           port vs official raw max_abs_diff 0.12520298
official raw vs normalized: final IoU 0.99998656 (223137/223138 px, 3 diff)
                            first IoU 0.99938393 (222241/222378 px, 137 diff)
                            first logits max_abs_diff 0.71615314; score diff 0.00388288
                            same points True; empty shortcuts False/False
port actual vs official raw: final IoU 0.99998656 (223138/223137 px, 3 diff)
port vs official normalized: final IoU 1.0 (0 diff)
determinism: 3 repeats equal True; runtime 7.76 s
```

以最终二值 mask 的 path tolerance（IoU≥0.999）计，`G2=pass`、`G6a=pass`（仅此
1 support/1 target，3 repeats）；首阶段不是零差异，不能声称逐 logits/逐 tensor 一致。
`G6c=unverified`（未跑全量项目评测）。

## 04. 训练配置审计

| 项 | 官方 PerSAM | 移植 | 判定 |
|---|---|---|---|
| 模式 | training-free | training-free | same |
| loss/optimizer/epochs/lr | 无 | 无 | N/A |
| support | 1 | 默认 3（可 `--support-shots 1`） | 3-shot extension，已披露 |
| support 来源 | 官方对象目录 | 18 train；seen 3 shots，unseen/novel donor | protocol_adaptation，已披露 |
| vote | 无 | 0.5 majority | 非官方模式，已披露 |
| 输入/输出 | 原图 | 原图 | same |
| SAM 权重 | ViT-H 4b8939 | 同 | same（SHA 已记录） |
| seed | 无 | training-free；prediction 未 seed | 固定对 3 repeats 确定，全量未证 |
| batch/workers/GPU | 1 / 官方 driver | 1 / 0 / 单卡 | protocol adaptation |
| selection | 无 | 无 | N/A |

G3=pass：无训练项；support 只来自 train，donor 已披露，固定 DRIVE 对重复确定。
但 3-shot 是扩展，不能作为官方 one-shot 对照。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | RGB 原图，无 resize | 同 | same |
| prompt | support-derived points，无 target GT | 同 | same（探针 same points） |
| checkpoint | SAM ViT-H | 同 | same |
| postprocess | 首预测 + 两段精修 | 同序；空 mask guard | equivalent；空路径未数值对照 |
| embedding | raw（首预测）/normalized（sim） | normalized（两者） | **differ，未披露** |
| threshold | SAM bool / 0.5 | 同 | same |
| export | 彩色 PNG 原分辨率 | bool mask GT 分辨率 | protocol_adaptation |
| metric | demo 不报项目指标 | Dice/IoU/clDice/HD95 | 项目协议 |
| TTA | 无 | 无 | same |

G4=pass：导出、指标、3-shot 均为已披露 protocol adaptation；D-01 属内部 L2 路径，
不是 eval transform。但 D-01 仍阻断主表。

## 07. 门禁结论（pre-fix snapshot）

> **Superseded:** this is the pre-fix gate snapshot. Post-A0 status is in
> [`../A0_REAUDIT.md`](../A0_REAUDIT.md).


```text
A0  = unverified
G0  = pass                  # repo/commit/LICENSE/SAM SHA 可追溯
G1  = pass                  # 594 keys、641,090,608 params、3 buffers 全同
G2  = pass (final-mask)     # final IoU 0.9999866；首段 raw/norm 差异未披露
G3  = pass                  # training-free；train-only support；3 repeats
G4  = pass                  # adaptations 已披露；D-01 属 L2 路径
G5  = pass                  # 18/6 train support + donor；无 test/val support
G6a = pass (1 support/1 target, 3 repeats)
G6b = N/A                   # 无官方方法 checkpoint
G6c = unverified            # 未做全量项目训练/评测

main_table = blocked
claim_label = path_verified
official_metric_match = unverified
```

仍缺：修复/披露 D-persam-01；在 20 数据集、3 shots 下重跑 G6a（只能算 extension）；
补空 mask/全背景 support；跑全量 G6c 并冻结 artifacts；冻结 A0 calibration pack。


## Post-A0 re-audit (2026-10-26)

- A0: pass (`a0-cal-2026-10-26-v3`, `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`).
- Resolved: `D-persam-01` raw-vs-normalized reference embedding split now matches official `persam.py:90-92`.
- Post-fix gates: G0 pass, G1 pass, G2 pass (final-mask caveat), G3 pass, G4 pass, G5 pass, G6a unverified (post-fix re-run pending), G6b N/A, G6c unverified.
- `main_table=blocked`: the pre-fix G6a pass cannot be reused after the first-stage change; G6c is unrun.
