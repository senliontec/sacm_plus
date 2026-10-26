# Matcher 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方：`3rd/Matcher`，commit `d59b6c761f427d52b1ba00132a6b3488f169d039`
- 移植：`src/baseline_zoo/models/matcher/{Matcher.py,model.py,UPSTREAM.md}` + `src/baseline_zoo/adapters/matcher.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`training_free_support_based_sam`；BSD-2-Clause `LICENSE`。
- 资产：SAM ViT-H 在本地；**DINOv2 ViT-L/14 权重缺失**
  (`checkpoints/dinov2_vitl14_pretrain.pth` 不存在)。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 只用于定位问题。

## 01. 官方敏感点清单

| id | 官方敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-mat-01 | 官方 commit、DINOv2/SAM 权重 | `UPSTREAM.md:3-20`; `main_oss.py:86-89` | 两套权重必须可共享；缺 DINOv2 无法做 L2 |
| S-mat-02 | DINOv2 ViT-L/14 `img_size=518` + SAM ViT-H | `model.py:45-65` | 同 backbone/config；否则特征空间不同 |
| S-mat-03 | 输入 square resize 518 + ToTensor | `data/dataset.py:29-32` | 同几何/数值；否则 support/target 特征不一致 |
| S-mat-04 | support mask nearest resize、patch avg-pool、空 mask center box | `data/fss.py:39-47`; `Matcher.py:83-102` | 空 mask 不能变成全背景 |
| S-mat-05 | AMG 参数、k-means、点采样范围/迭代 | `main_oss.py:92-119`; `Matcher.py` | 同 mask proposals/prompt 集 |
| S-mat-06 | score 公式 `alpha*emd + beta*purity*coverage**exp` | `Matcher.py:267` | FSS 官方脚本 α=.8 β=.2 exp=1 |
| S-mat-07 | filters、top-k、merge 数 | `main_oss.py:112-119` | EMD/purity/coverage 关闭、topk .7、merge 10 |
| S-mat-08 | n-shot support 语义 | `main_oss.py:79`; `fss.py:91` | 官方 CLI 默认 1 shot；多 shot 需明确 |
| S-mat-09 | target 只用 support similarity，无 GT prompt | `Matcher.py` predict path | 若有 target GT 即泄漏 |
| S-mat-10 | 输出 518 mask | `main_oss.py:45-49` | 官方在 518 算 metric；设备分辨率转换需披露 |
| S-mat-11 | 测试固定随机种子 | `main_oss.py:18-19,25-27` | `random.seed(0)` + `fix_randseed(0)`；否则重复性不同 |
| S-mat-12 | support/donor 来源 | 官方 FSS episode | 项目只允许 train support，不得 test/val |
| S-mat-13 | 方法 checkpoint | 官方无 Matcher 训练权重 | 只加载 DINOv2 + SAM；G6b N/A |

## 02. 移植对照与差异

```text
# 3rd/Matcher/scripts/fss.sh:5-11
python main_oss.py --benchmark fss --max_sample_iterations 30 \
  --sample-range "(4,6)" --multimask_output 0 \
  --alpha 0.8 --beta 0.2 --exp 1. --num_merging_mask 10

# src/baseline_zoo/models/matcher/model.py:91-105
Matcher(..., alpha=1.0, beta=0.0, exp=0.0, ...)

# src/baseline_zoo/models/matcher/UPSTREAM.md:54-55
Official score filters: ... alpha 1.0, beta 0.0, exp 0.0.
```

| id | verdict | category | gate | 官方 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-mat-01 | `differ` | `official_recipe` | G4 | FSS 官方脚本 α=.8/β=.2/exp=1；移植 α=1/β=0/exp=0，且文档误称这是官方值 | **未披露/文档错误**（`UPSTREAM.md:54-55`） | 无法测（DINOv2 缺）；改变 mask 排序与合并，非零 |
| D-mat-02 | `differ` | `bug_fix` / `wrapper_change` | G3/G6a | 官方测试固定 `random.seed(0)` + `fix_randseed(0)`；移植 `run_prediction` 不 seed | **未披露**（adapter 未写） | prompt 点采样走全局 `random`，重复结果不保证；数值未测 |
| D-mat-03 | `unverified` | `missing_asset` | G2/G6a | 官方 DINOv2 ViT-L/14；移植默认同一路径但文件缺失，random-init 仅 smoke | 已披露（`adapter.py:52-54`; `UPSTREAM.md:67-69`） | 无同权重，G2 只能 unverified；不得用 random-init 报结果 |
| D-mat-04 | `intentional_deviation` | `protocol_adaptation` | G4 | 官方返回/评测 518×518；移植 bilinear 上采样到 GT 分辨率并 >0.5 二值化 | 已披露（`adapter.py:40-42`; `UPSTREAM.md:63-64`） | 几何/阈值转换未做数值对照 |
| D-mat-05 | `intentional_deviation` | `protocol_adaptation` | G4/G5 | 官方 FSS CLI 默认 `nshot=1`；移植默认 3 supports | 已披露（`adapter.py:43-45`） | 官方本身支持 n-shot；项目 3-shot 未做逐 shot 对照 |
| D-mat-06 | `unverified` | `bug_fix` | G2/G5 | 官方空 support mask 填 14×14 center box；移植依赖 `select_supports` 过滤空 mask | **未披露** | 项目 support 保证非空时路径不触发；空 mask 路径未测 |
| D-mat-07 | `equivalent` | `wrapper_change` | G2 | 官方 query/support mask nearest 到 518；移植 support mask nearest，target 无 GT | 已披露（`UPSTREAM.md:47-48`） | 几何一致；无数值（缺权重） |
| D-mat-08 | `equivalent` | `wrapper_change` | G1/G2 | 包相对 import、support feature cache、`np.int`→`int` | 已披露（`UPSTREAM.md:27-39`） | cache 理论上只改吞吐；未做 state-dict/forward 对照 |
| D-mat-09 | `unverified` | `missing_asset` | G6b | 官方无 Matcher 方法 checkpoint | 已披露（`UPSTREAM.md:70-71`） | G6b=N/A |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-mat-01 | `unverified` | D-mat-03；DINOv2 缺 |
| S-mat-02 | `covered_by_discrepancy` | D-mat-03, D-mat-08 |
| S-mat-03 | `preserved` | 518 square resize + /255 与官方 ToTensor 路径一致；未数值测 |
| S-mat-04 | `covered_by_discrepancy` | D-mat-06, D-mat-07 |
| S-mat-05 | `preserved` | AMG/kmeans/采样参数逐项相同（`model.py:66-105`） |
| S-mat-06 | `covered_by_discrepancy` | D-mat-01 |
| S-mat-07 | `preserved` | filters/topk/merge 与 `main_oss.py:112-119` 相同 |
| S-mat-08 | `covered_by_discrepancy` | D-mat-05 |
| S-mat-09 | `preserved` | adapter 无 target GT/prompt；`predict_sample` 只用 prepared image |
| S-mat-10 | `covered_by_discrepancy` | D-mat-04 |
| S-mat-11 | `covered_by_discrepancy` | D-mat-02 |
| S-mat-12 | `covered_by_discrepancy` | D-mat-05；support.py 仅 train |
| S-mat-13 | `not_applicable` | D-mat-09；无官方方法权重 |

closure = 13/13；open 实质项是 **D-mat-01（score recipe）** 和 **D-mat-02（seed）**。

## 03. 数值探针结果

未运行 same-state 数值探针：`checkpoints/dinov2_vitl14_pretrain.pth` 不存在，
且规则要求“缺 DINOv2 权重必须 unverified”，random-init 不能用于 parity。

```bash
ls checkpoints/dinov2_vitl14_pretrain.pth
# ls: cannot access ...: No such file or directory
```

因此：

```text
G1 = unverified   # 未生成官方/移植 DINOv2+SAM state_dict 对照
G2 = unverified   # 无同权重固定 support/target 探针
G3 = unverified   # 无 DINOv2 权重，无法验证 prompt 采样确定性
G6a = unverified  # 无同 support/target + 官方权重的 mask/prompt 重复
```

数值修复后至少要记录：support/target 固定图、SAM/DINOv2 SHA、prompt points、每张
mask、3 repeats、以及 α/β/exp 两种配置的差异。

## 04. 训练配置审计

| 项 | 官方 Matcher | 移植 | 判定 |
|---|---|---|---|
| 模式 | training-free | training-free | same |
| loss/optimizer/epochs/lr | 无 | 无 | N/A |
| backbone | DINOv2 ViT-L/14 518 + SAM ViT-H | 同 config；DINOv2 权重缺失 | unverified |
| support | FSS CLI 默认 1 shot | 默认 3 shots；seen 3，unseen/novel donor | intentional，已披露 |
| AMG | points/side 64, batch 64, IoU .88, stability .95 | 同 | preserved（`model.py:66-81`） |
| kmeans/sampling | 8 centers, range (4,6), 30 iters | 同 | preserved（`model.py:91-105`） |
| filters/topk/merge | EMD/purity/coverage off, topk .7, merge 10 | 同 | preserved |
| score | **FSS 脚本 α=.8 β=.2 exp=1** | **α=1 β=0 exp=0** | **differ，未披露** |
| seed | `random.seed(0)` + `fix_randseed(0)` per test | 无 seed | **differ，未披露** |
| input/output | 518 / 518 | 518 / GT | protocol adaptation，已披露 |

G3=unverified：缺权重无法测 prompt/support 确定性，且 D-mat-02 未披露；
G4=fail：D-mat-01 是明确的官方 FSS score recipe 差异，D-mat-02 使评测不可重复。

## 05. 评测/推理审计

| 项 | 官方 | 移植 | 判定 |
|---|---|---|---|
| eval transform | Resize 518 + ToTensor | bilinear 518 + /255 | 几何相同；无权重未数值测 |
| checkpoint | DINOv2 ViT-L/14 + SAM ViT-H | 同 config；DINOv2 缺 | unverified |
| postprocess | AMG/merge/filter 后返回 518 mask | 同后处理；GT 上采样 + >0.5 | protocol adaptation，已披露 |
| score | α=.8 β=.2 exp=1（fss.sh） | α=1 β=0 exp=0 | **differ，未披露** |
| seed | 测试固定 seed 0 | 无 seed | **differ，未披露** |
| metric | FSS mIoU/FB-IoU @518 | Dice/IoU/clDice/HD95 @GT | 项目协议 |
| export | 518 预测 | GT resolution bool | protocol adaptation，已披露 |
| TTA | 无 | 无 | same |

G4=fail：不解释 α/β/exp 和 seed 的差异，不能把项目数字当作官方 Matcher FSS 结果。
G5=pass：support 来自 train（`support.py:27-79`），3-shot/donor 已披露。

## 07. 门禁结论（provisional）

```text
A0  = unverified
G0  = pass                  # repo/commit/LICENSE/SAM 可追溯；DINOv2 缺失已披露
G1  = unverified            # 无 DINOv2/SAM 两侧 state-dict 对照
G2  = unverified            # 缺 DINOv2 权重；random-init 不算 parity
G3  = unverified            # 无权重；prompt 采样未 seed
G4  = fail                  # score α/β/exp 与官方 FSS 脚本不符，且未披露
G5  = pass                  # train-only support/donor；无 target GT
G6a = unverified            # 无同 support/target + 官方权重的重复
G6b = N/A                   # 无官方 Matcher 方法 checkpoint
G6c = unverified            # 未做全量项目训练/评测

main_table = blocked
claim_label = none
official_metric_match = unverified
```

仍缺：补 DINOv2 ViT-L/14 权重；把 score 改成官方 FSS `.8/.2/1` 或在 deviations
中明确披露；恢复测试 seed 0；补同一 support/target 的 3-repeat mask/prompt 探针；
补空 support mask 路径；跑全量 G6c；冻结 A0 calibration pack。
