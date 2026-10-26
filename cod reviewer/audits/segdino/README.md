# SegDINO 移植审计（手工 pilot）

- 官方仓库：`3rd/segdino`
- 官方 commit：`c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41`
- 移植版本：`src/baseline_zoo/models/segdino/` + `src/baseline_zoo/adapters/segdino.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`
- 审计日期：2026-10-25

> **A0 状态：unverified。** 严格 07 = `<blocked>`；本文是 provisional 手工 pilot。
> 官方 `dinov3` hub repo 不在本地，且上游不提供 SegDINO 分割 checkpoint；不能构造
> 完整官方模型，**G1/G2/G3/G4 均为 unverified**。

---

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-sd-01 | `DPT`/`DPTHead` 结构 | `dpt.py:58-90`; `blocks.py` | 头结构一致；否则 state_dict 不兼容 |
| S-sd-02 | 中间层索引 small/base/large | `dpt.py:67-75` | small/base `[2,5,8,11]`，large `[4,11,17,23]` |
| S-sd-03 | backbone 接口 `get_intermediate_layers` | `dpt.py:83-86` | hub DINOv3 API；timm 无此方法 |
| S-sd-04 | DINOv3 hub repo/`dino_ckpt` | `train_segdino.py:171-175,197-200`; `test_segdino.py:230-242` | `torch.hub.load(repo_dir,...)`；外部本地 `.pth` |
| S-sd-05 | forward：DPT→bilinear 回 patch×16 | `dpt.py:83-90` | 输出 logits shape/插值一致 |
| S-sd-06 | loss `BCEWithLogitsLoss` | `train_segdino.py:72-92` | loss/grad 一致 |
| S-sd-07 | AdamW lr1e-4/wd1e-4 | `train_segdino.py:200-209` | 优化器一致 |
| S-sd-08 | 50 epoch、batch4（README）、256、无 scheduler、无增强 | `README.md:65-95`; `train_segdino.py:156-180` | 官方配方 |
| S-sd-09 | parser 默认 batch8，README 用 4 | `train_segdino.py:161-163` | 默认命令与 README 不一致 |
| S-sd-10 | 预处理：cv2 BGR→RGB、BICUBIC antialias、ImageNet norm、mask NEAREST>0.5 | `dataset.py:11-39` | 张量一致 |
| S-sd-11 | dataset train/test split | `train_segdino.py:214-245` | 官方用 test 当 val |
| S-sd-12 | 每 epoch 在 test split 上选 val Dice best | `train_segdino.py:248-276` | 官方 test 选择 |
| S-sd-13 | DataLoader workers16、train batch8、drop_last | `train_segdino.py:156-180,230-245` | 训练动力学 |
| S-sd-14 | test 默认 1024，README 256 | `test_segdino.py:193-220`; `README.md:65-95` | 分辨率不一致 |
| S-sd-15 | 推理：sigmoid>0.5，指标在输入分辨率 | `test_segdino.py:110-174` | 官方不恢复 GT 分辨率 |
| S-sd-16 | checkpoint：`state_dict` 或整模型，`load_ckpt_flex` | `test_segdino.py:178-190` | 加载格式 |
| S-sd-17 | 无官方 SegDINO 分割 checkpoint | `UPSTREAM.md` | 只能自训 |
| S-sd-18 | 来源/许可证：MIT；URL 未记录 | `models/segdino/UPSTREAM.md:3-14` | G0 有缺口 |
| S-sd-19 | seed 42 / workers16 / 单卡 | `train_segdino.py:162,167,182-186,233-245` | 随机性 |

### 关键原文证据

```text
# 3rd/segdino/dpt.py:83-90
features = self.backbone.get_intermediate_layers(
    x, n = self.intermediate_layer_idx[self.encoder_size]
)
out = self.head(features, patch_h, patch_w)
out = F.interpolate(out, (patch_h * 16, patch_w * 16),
                    mode='bilinear', align_corners=True)
```

```python
# src/baseline_zoo/models/segdino/dpt.py:83-98：timm fallback
if hasattr(self.backbone, 'get_intermediate_layers'):
    features = self.backbone.get_intermediate_layers(x, n=indices)
else:
    features = self.backbone.forward_intermediates(
        x, indices=indices, norm=True, output_fmt='NLC',
        intermediates_only=True)
```
探针显示本地 timm backbone `get_intermediate_layers=False`，实际走 fallback；
这不是官方 hub API 的逐字路径。

```text
# 3rd/segdino/train_segdino.py:214-245
val_dataset = FolderDataset(..., split="test", ...)
...
if val_dice > best_val_dice: save(...)
```
官方训练把 test split 当 val 并据此选 checkpoint。

---

## 02. 移植对照与差异

| id | verdict | category | gate | 官方行为 | 移植行为 | 影响 | 披露 | S |
|---|---|---|---|---|---|---|---|---|
| D-sd-01 | `equivalent` | `wrapper_change` | G1 | `DPT`/`DPTHead`/blocks | 复制；相对 import + timm fallback | 头 state/shape 探针一致；完整模型仍缺 | UPSTREAM | 01,02 |
| D-sd-02 | **`unverified`** | `wrapper_change` | G1/G2 | hub `get_intermediate_layers` | timm `forward_intermediates` | 特征选择/归一化可能不同；无完整同权重 forward | UPSTREAM | 03,05 |
| D-sd-03 | `unverified` | `missing_asset` | G0/G1/G2 | 本地 `dinov3` hub repo + 外部 `.pth` | 本地 `checkpoints/dinov3/pytorch_model.bin` + timm | 官方 hub 全模型不可构造 | deviations #2 | 04,17 |
| D-sd-04 | `equivalent` | `official_recipe` | G1 | 中间层索引表 | 同一张表，按 backbone size 选择 | large 不再误用 base indices | deviations #8 | 02 |
| D-sd-05 | `equivalent` | `wrapper_change` | G1 | DPT head | 复制；head keys/shapes 一致 | 头层面一致；不含 backbone | UPSTREAM | 01,05 |
| D-sd-06 | `unverified` | `wrapper_change` | G3/G4 | torchvision BICUBIC antialias | PIL BICUBIC（默认 resize） | 未见同图张量探针 | **未写** | 10 |
| D-sd-07 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | test split 当 val，Dice 选择 | 6 图 val，Dice | 移植避免 test 泄漏 | deviations #1 | 11,12 |
| D-sd-08 | `intentional_deviation` | `official_recipe` | G3 | parser 默认 batch8；README 4 | batch4 | 与 README 一致，与 parser 默认不同 | deviations #5 未写 parser 8 | 08,09,13 |
| D-sd-09 | `intentional_deviation` | `protocol_adaptation` | G4 | test 默认 1024；README 256 | 256 | 分辨率/边界不同 | deviations #3 | 14 |
| D-sd-10 | `intentional_deviation` | `protocol_adaptation` | G4 | 指标在输入分辨率，不恢复 GT | sigmoid→bilinear→GT→0.5 | 指标口径/分辨率不同 | deviations #4 | 15 |
| D-sd-11 | `unverified` | `wrapper_change` | G4 | `load_ckpt_flex` 用 `strict=False`，缺键仅警告 | `restore_config` 强制 random-init 以避开 gated 下载；runtime 用 `strict=True` 载入 | 完整 ckpt 可覆盖；缺键直接失败，无同权重验证 | deviations #7 未写 strict | 16 |
| D-sd-12 | `unverified` | `missing_asset` | G6b | 无官方任务 checkpoint | scratch | 无 checkpoint parity | UPSTREAM | 17 |
| D-sd-13 | `differ` | `missing_asset` | G0 | MIT LICENSE；URL 未记录 | 复制 MIT；UPSTREAM 仍无 URL | 来源不可核验 | UPSTREAM | 18 |
| D-sd-14 | `unverified` | `wrapper_change` | G3 | seed42、workers16、单卡 | 协议 seed、workers0、单卡 | 数据/RNG 不同 | 部分 | 13,19 |
| D-sd-15 | `unverified` | `missing_asset` | G2 | 官方完整模型 | 只有 DPT head 结构探针 | 无 full-model loss/grad/forward | — | 01,03,05,06 |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-sd-01 | `covered_by_discrepancy` | D-01, D-05, D-15 |
| S-sd-02 | `covered_by_discrepancy` | D-01, D-04 |
| S-sd-03 | `covered_by_discrepancy` | D-02, D-15 |
| S-sd-04 | `covered_by_discrepancy` | D-03 |
| S-sd-05 | `covered_by_discrepancy` | D-02, D-05, D-15 |
| S-sd-06 | `unverified` | D-15 |
| S-sd-07 | `preserved` | adapter `make_optimizer` |
| S-sd-08 | `preserved` | D-08 |
| S-sd-09 | `covered_by_discrepancy` | D-08 |
| S-sd-10 | `covered_by_discrepancy` | D-06 |
| S-sd-11 | `covered_by_discrepancy` | D-07 |
| S-sd-12 | `covered_by_discrepancy` | D-07 |
| S-sd-13 | `covered_by_discrepancy` | D-08, D-14 |
| S-sd-14 | `covered_by_discrepancy` | D-09 |
| S-sd-15 | `covered_by_discrepancy` | D-10 |
| S-sd-16 | `covered_by_discrepancy` | D-11 |
| S-sd-17 | `covered_by_discrepancy` | D-03, D-12 |
| S-sd-18 | `covered_by_discrepancy` | D-13 |
| S-sd-19 | `covered_by_discrepancy` | D-14 |

closure = 19/19；但 S-sd-06 的 disposition 是 `unverified`，仍有 open 证据缺口。

---

## 03. 数值证据

命令（在 `sacm_plus/` 下；只 build 本地模型，不下载、不训练）：

```bash
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_same_state_probes.py'
```

结果见 `03_numeric_probe.json`：

```text
local ckpt:                     checkpoints/dinov3/pytorch_model.bin
ckpt bytes:                     86,407,250
port build with local ckpt:      True
port total params:              24,062,881
timm has get_intermediate_layers: False
timm embed_dim:                 384
DPT head keys/shapes equal:     True / True
DPT head params:                2,475,937
full-model same-state forward:  NOT RUN（官方 hub repo/backbone 不在本地）
```

结论：**G2=unverified**。只有 DPT head 的结构/state 探针，没有“官方完整模型与移植
同权重”的 forward/loss/grad。DPT head 一致不能外推为完整模型一致。

---

## 04. 训练配置审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| epochs | 50（README/parser） | 50 | same |
| batch | parser 8；README 4 | 4 | 按 README |
| input | 256（README 训练） | 256 | same |
| optimizer | AdamW 1e-4/wd1e-4 | 同 | same |
| scheduler | 无 | 无 | same |
| loss | BCEWithLogits | 同 | 代码同 |
| transforms | BGR→RGB、BICUBIC antialias、ImageNet norm；无增强 | PIL BICUBIC、ImageNet norm；无增强 | 后端不同，**未验证** |
| train data | 官方 train split | 18/6 3-shot | 协议适配 |
| val/test | test split 当 val，Dice | 6 图 val，Dice | 协议适配 |
| workers/GPU | 16 / 单卡 | 0 / 单卡 | 不同 |
| seed | 42 | 协议 seed | 不同 |
| backbone | hub DINOv3 + 门控权重 | timm + 本地 DINOv3 权重 | **路径不同** |
| selection | test split Dice | val Dice | 协议适配 |

G3 不能 pass：完整模型/官方权重不可构造，预处理后端没有同图探针，worker/seed 不同。

---

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | BGR→RGB、BICUBIC antialias、ImageNet norm | PIL BICUBIC、ImageNet norm | 后端不同，未验证 |
| checkpoint | 官方 test-Dice best | port val-Dice best | 协议适配 |
| postprocess | sigmoid>0.5 @输入分辨率 | sigmoid→bilinear GT→0.5 | 协议适配 |
| threshold | 0.5 | 0.5 | same |
| metric | 官方 Dice/IoU/HD95 @输入 | 项目 Dice/IoU/clDice/HD95 @GT | 口径不同 |
| export | 原图/输入分辨率 mask | GT bool | 协议适配 |
| 分辨率 | test 默认 1024；README 256 | 256 | 跟随 README |

推理只能确认“移植路径自洽 + 导出协议”，不能确认官方完整模型输出；G4=unverified。

---

## 07. 门禁结论（provisional）

```text
A0 = unverified            # 无 calibration；正式 07 = <blocked>
G0 = pass                  # MIT/LICENSE 本地；上游 URL 未记录
G1 = unverified            # 只有 DPT head 结构；官方完整模型/hub 不在本地
G2 = unverified            # 无完整同权重 forward/loss/grad
G3 = unverified            # 预处理后端、workers/seed、官方权重路径未闭合
G4 = unverified            # 无完整官方模型输出
G5 = pass                  # 18/6、test→val 选择、256/导出均披露
G6a = unverified           # 无 20-step 路径
G6b = N/A                  # 无官方 SegDINO 分割 checkpoint
G6c = unverified           # 无全量训练/评测
main_table = blocked
claim_label = none
official_metric_match = unverified
```

缺失/阻断项：官方 `dinov3` hub repo 与匹配的 DINOv3 backbone 权重；完整同 state 探针；
预处理同图张量探针；worker/seed 披露；G6a/G6c；上游 URL；A0。

## 最终判断

- 只有 **DPT head 结构/state** 与移植一致；完整模型没有构造，不能称 G1/G2 pass。
- 官方训练把 test 当 val，移植改成 6 图 val，这是正确的协议修复；但官方资源缺失使
  训练/推理保真无法数值验证。
- 当前只能给 `G5=pass` 和 head-level 证据；不能进主表，也不能声称官方指标复现。
