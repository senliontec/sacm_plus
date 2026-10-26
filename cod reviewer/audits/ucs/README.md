# UCS 移植审计（手工 pilot）

> **修复后（2026-10-26）：** 本报告表格是修复前审计证据；当前代码状态和回归证据见
> [`../POST_FIX.md`](../POST_FIX.md)。A0/G6c 仍未完成，gate 结论不变。

- 官方：`3rd/UCS`，commit `cff2db167ea096648b520e78b3c1c20af6da57eb`
- 论文：UCS: A Universal Model for Curvilinear Structure Segmentation（arXiv:2504.04034）
- 移植：`src/baseline_zoo/models/ucs/**` + `src/baseline_zoo/adapters/ucs.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`supervised_oracle_prompted_sam`；**附录/oracle reference，不是 prompt-free baseline**。
- 资产：SAM ViT-L `sam_vit_l_0b3195.pth`
  SHA256 `3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622`；
  官方无 UCS 训练 checkpoint，snapshot 无 LICENSE。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 只用于定位问题。

## 01. 官方敏感点清单

| id | 敏感点 | 官方位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-ucs-01 | repo/commit/依赖/SAM 权重/许可证 | `UPSTREAM.md:1-8`; 无 LICENSE | 来源可追溯；缺 LICENSE 需披露 |
| S-ucs-02 | paper 版 Sparse Adapter / HQ compression vs HEAD 架构 | `UPSTREAM.md:49-55` | HEAD 与论文不同必须说明 |
| S-ucs-03 | ViT-L + Adapter + PromptGenerator + PCA decoder 结构 | `segment_anything/build_sam.py:54-128` | key/shape/param/buffer 一致 |
| S-ucs-04 | prompt encoder 冻结；encoder 仅 Adapter/neck/prompt_generator 可训；Adam 全部参数 | `train.py:207-227` | 同可训参数集合/优化器 |
| S-ucs-05 | 20 epochs、batch 1、1024、Adam 1e-4、MultiStepLR [5,10] γ.5、iter_point=8 | `train.py:18-44,236` | 官方 20-epoch 配方，不是项目 200 |
| S-ucs-06 | 初始 50/50 box/point、GT box/point、GT-guided refinement、9 次 optimizer step | `train.py:111-204`; `utils.py:239-252` | 迭代 prompt 语义不能丢 |
| S-ucs-07 | 图像 0–255 上 mean/std、train/test transforms、mask 0/1 | `DataLoader.py:196-210,358-385`; `utils.py:188-215` | 同预处理/几何 |
| S-ucs-08 | 无 val；按训练 Dice 选 best；保存全 state dict | `train.py:263-267` | 选择语义 |
| S-ucs-09 | test 默认 boxes_prompt=False；GT point、丢弃 box、8 次 GT-guided refinement | `test.py:23-47,214-239` | **oracle-prompted**，不是 prompt-free |
| S-ucs-10 | postprocess GT pad/crop/resize，sigmoid>0.5，GT 导出 | `test.py:66-83`; `utils.py:269-283` | 同分辨率/阈值 |
| S-ucs-11 | 官方无 seed；test/训练用全局 `random`/`np.random` | `train.py:121,169`; `test.py:214-239` | 随机 prompt；不得声称已固定 |
| S-ucs-12 | 无官方 UCS 权重 | `UPSTREAM.md:63-67` | G6b N/A |
| S-ucs-13 | 18/6 split、无 support/donor；GT prompt 仅作为已披露 oracle | 项目 manifest | 不能当 held-out prompt-free 数字 |
| S-ucs-14 | 官方 metric：iou/dice/f1/pre/rec | `test.py:30-39,245-258` | 项目指标需单独说明 |

## 02. 对照与差异

```text
# 3rd/UCS/train.py:121-126,169-186
if random.random() > 0.5:
    batched_input["point_coords"] = None; flag = "boxes"
else:
    batched_input["boxes"] = None; flag = "point"
init_mask_num = np.random.randint(1, args.iter_point - 1)
...
batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)

# src/baseline_zoo/adapters/ucs.py:155-173
if rng.random() > 0.5:
    boxes = utils.get_boxes_from_mask(mask_np, max_pixel=5).float().unsqueeze(0)
else:
    point_coords, point_labels = utils.init_point_sampling(mask_np, 1)
...
boxes = utils.get_boxes_from_mask(mask_np, max_pixel=0)...  # eval
point_coords, point_labels = utils.init_point_sampling(mask_np, self.point_num)
```

```text
# 3rd/UCS/utils.py:271-278 (save_masks)
preds = torch.sigmoid(preds)
preds[preds > 0.5] = int(1)
preds[preds <= 0.5] = int(0)

# src/baseline_zoo/adapters/ucs.py:304-332
batch['boxes'] = None
for iteration in range(self.iter_point):
    masks, low_res, _ = self._decode(...)
    batch = utils.generate_point(masks, labels, low_res, batch, self.point_num)
...
return (torch.sigmoid(output)[0, 0].cpu().numpy() > 0.5)
```

| id | verdict | category | gate | 官方 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-ucs-01 | `equivalent` | `wrapper_change` | G1 | 官方 HEAD/移植 569 keys、314,716,560 params、262 buffers 全同 | 已披露（`UPSTREAM.md:20-31`） | 无 missing/extra/shape/dtype 差异 |
| D-ucs-02 | `intentional_deviation` | `architecture_drift` | G0/G1 | 论文 Sparse Adapter/HQ compression 已在 HEAD 重构掉；移植用 HEAD | 已披露（`UPSTREAM.md:49-55`） | 与论文不完全同架构；不能称 paper-faithful |
| D-ucs-03 | `equivalent` | `wrapper_change` | G1 | 仅 `weights_only=False`、PCA `.cuda()`→device、batch-1 guard；`utils.py` diff 为空 | 已披露（`UPSTREAM.md:33-42`） | 结构/公式不变；forward 见 03 |
| D-ucs-04 | `intentional_deviation` | `protocol_adaptation` | G5 | 官方 JSON dataset；移植直接消费共享 18/6 split | 已披露（`adapter.py:52-53`） | 数据域/数量不同；无 support/donor |
| D-ucs-05 | `equivalent` | `official_recipe` | G3 | 20 epochs/batch1/1024/Adam1e-4/MultiStep[5,10]/.5/iter8 均保留 | 已披露（`adapter.py:48-49`） | 优化器只收可训参数；wd=0，冻结参数无梯度，等价 |
| D-ucs-06 | `differ` | `reproducibility` | G3/G4/G6a | 官方无 seed；移植训练 seed 全局 RNG，但 eval `run_prediction` 不 seed，`random.Random(0)` 管不到 `init_point_sampling` 的全局 numpy | **UPSTREAM 误称“shared runtime seeds ... prompt sampling”**（`UPSTREAM.md:60-61`） | 4 次 eval 采样得到 4 个不同点；test 指标非确定 |
| D-ucs-07 | `differ` | `wrapper_change` | G3 | 官方 train workers=4；移植 runtime workers=0 | 未在 deviations 明写 | 数据顺序/吞吐不同；样本集合不变 |
| D-ucs-08 | `equivalent` | `official_recipe` | G3 | `FocalDiceloss_IoULoss`、9 次 step、GT-guided `generate_point` 均调用同一 vendored `utils.py` | 已披露（`UPSTREAM.md:24-26`） | `utils.py` 与官方 diff 为空 |
| D-ucs-09 | `intentional_deviation` | `protocol_adaptation` | G4 | 官方无 val、按 train Dice；移植仍按 train Dice，val 只记录 | 已披露（`adapter.py:50-51`） | 选择语义保留 |
| D-ucs-10 | `intentional_deviation` | `protocol_adaptation` | G4 | 官方 iou/dice/f1/pre/rec；移植项目 Dice/IoU/clDice/HD95、GT bool 导出 | 项目协议；adapter deviations 未逐项列 metric | 官方 metric 未复现；`official_metric_match=unverified` |
| D-ucs-11 | `equivalent` | `postprocess` | G4 | 官方 bilinear→1024、GT crop/resize、sigmoid>0.5；移植同 | 已披露（`UPSTREAM.md:24-31`; `adapter.py:320-332`） | 公式逐项相同 |
| D-ucs-12 | `equivalent` | `oracle_prompt` | G5 | 官方 test GT box/point、丢 box、GT-guided 8 次 refinement；移植同 | **已披露**（`adapter.py:45-47`; `configs/models/ucs.yaml:6-9`） | 这是 oracle 设置；不能与 prompt-free 同表比较 |
| D-ucs-13 | `unverified` | `missing_asset` | G6b/G6c | 无官方 UCS 权重；未完成全量训练/评测 | 已披露（`UPSTREAM.md:63-67`） | G6b=N/A；G6c=unverified |
| D-ucs-14 | `unverified` | `missing_asset` | G6a | 未跑官方 20-step 训练对照；本机 1024 ViT-L 全反向 OOM | — | G6a=unverified |
| D-ucs-15 | `equivalent` | `numeric_parity` | G2 | 同 state dict、固定图/point 下 forward/中间层 exact；decoder loss/grad/Adam step 在容差内 | 03 探针 | loss diff 0；grad max 5.96e-8；Adam param max 2.45e-5 |
| D-ucs-16 | `unverified` | `paper_fidelity` | G0/G4 | HEAD 与论文架构不同；paper-faithful 版本只在 git history | 已披露（`UPSTREAM.md:49-55`） | 不能声称复现论文 UCS |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-ucs-01 | `covered_by_discrepancy` | D-ucs-02；无 LICENSE |
| S-ucs-02 | `covered_by_discrepancy` | D-ucs-02, D-ucs-16 |
| S-ucs-03 | `preserved` | D-ucs-01；03 state-dict probe |
| S-ucs-04 | `preserved` | D-ucs-01, D-ucs-05 |
| S-ucs-05 | `preserved` | D-ucs-05 |
| S-ucs-06 | `preserved` | D-ucs-08；同一 `utils.py` |
| S-ucs-07 | `preserved` | D-ucs-04；官方 mean/std/transforms 直接调用 |
| S-ucs-08 | `preserved` | D-ucs-09 |
| S-ucs-09 | `covered_by_discrepancy` | D-ucs-12；oracle 已披露 |
| S-ucs-10 | `preserved` | D-ucs-11 |
| S-ucs-11 | `covered_by_discrepancy` | D-ucs-06 |
| S-ucs-12 | `not_applicable` | D-ucs-13 |
| S-ucs-13 | `covered_by_discrepancy` | D-ucs-04, D-ucs-12 |
| S-ucs-14 | `covered_by_discrepancy` | D-ucs-10 |

closure = 14/14；open 实质项：eval prompt 非确定（D-06）、HEAD/论文漂移（D-02）、
官方 metric 未复现（D-10）、无训练/G6a（D-13/14）。

## 03. 数值探针

```bash
python /tmp/ucs_structure.py
CUDA_VISIBLE_DEVICES=0 /home/dev/anaconda3/envs/tsai/bin/python /tmp/ucs_forward_probe.py
CUDA_VISIBLE_DEVICES=0 /home/dev/anaconda3/envs/tsai/bin/python /tmp/ucs_decoder_grad_probe.py
/home/dev/anaconda3/envs/tsai/bin/python /tmp/ucs_prompt_probe.py
```

```text
state dict: official/port 314,716,560 params; 262 buffers; 569 keys;
            missing 0; extra 0; shape/dtype mismatch 0; exact true
forward (same state, fixed DRIVE 1024 image, point (512,512)):
  image_embedding max|Δ| 0.0; 4 intermediate maps max|Δ| 0.0
  sparse prompt max|Δ| 0.0; dense prompt max|Δ| 0.0
  low_res_masks max|Δ| 0.0; iou_pred max|Δ| 0.0; binary mask diff 0
decoder step (same state, GT mask/point; official vs port):
  loss 3.4985733032226562 both, abs diff 0.0; lowres equal true
  129 grad tensors, grad max|Δ| 5.96e-8, mean|Δ| 1.08e-9
  Adam step (lr 1e-4) parameter max|Δ| 2.45e-5
eval prompt determinism (same image/mask, 4 calls):
  points (313,484), (549,605), (443,742), (564,719); box identical
  all_points_equal false
```

`G2=pass` 的边界：forward/中间层 exact，decoder loss/grad/Adam step 在数值容差内；
encoder Adapter 的完整反向因 1024 ViT-L 在 10.5 GB 卡上 OOM，未单独跑。
无官方 UCS 权重，`G6a=unverified`；`G6b=N/A`；`G6c=unverified`。本模型按
`supervised_oracle_prompted_sam` 审 L1–L6 的可用部分。

## 04. 训练配置

| 项 | 官方 UCS | 移植 | 判定 |
|---|---|---|---|
| epochs | 20 | 20 | same |
| batch | 1 | 1 | same |
| input | 1024 | 1024 | same |
| optimizer | Adam, lr 1e-4, wd 0 | Adam trainable, lr 1e-4 | equivalent |
| scheduler | MultiStepLR [5,10], γ 0.5 | 同 | same |
| loss | FocalDiceloss_IoULoss | 同（vendored） | same |
| iter_point | 8 | 8 | same |
| point_list | [1,3,5,9] | 同 | same |
| 初始 prompt | 50/50 GT box/point | 同（rng 分支） | equivalent |
| refinement | 8 次，GT-guided，末次 `setting_prompt_none` | 同 | preserved |
| transforms | pad/resize nearest + rotate/gauss/crop | 同官方 utils | preserved |
| data | 官方 JSON manifests | 共享 18/6 split | intentional，已披露 |
| workers | 4 | 0 | **differ，未披露** |
| seed | 无 | 协议 seed；utils 仍用全局 numpy | **differ**（D-06） |
| selection | train Dice，无 val | 同；val 只记录 | preserved |
| checkpoint | full model state | full state | same |
| GPU | 单卡 | 单卡 | same |

G3=unverified：无训练后对照；workers/seed 差异和 eval 非确定性未写入 deviations。

## 05. 评测/推理

| 项 | 官方 UCS | 移植 | 判定 |
|---|---|---|---|
| eval transform | test_transforms pad/resize 1024 + mean/std | 同官方 utils | preserved |
| checkpoint | full state dict | full state dict；无官方权重 | same（无官方 checkpoint） |
| prompt | GT boxes max_pixel=0 + GT point；默认丢 box | 同；`predict_sample` 显式 `boxes=None` | preserved（oracle） |
| refinement | 8 次 GT-guided；累加 point | 同 | preserved |
| postprocess | bilinear→1024，pad/crop/resize GT | 同 | preserved |
| threshold | sigmoid>0.5 | 同 | same |
| metric | iou/dice/f1/pre/rec | Dice/IoU/clDice/HD95 | protocol adaptation；未做双口径 |
| export | GT bool PNG | GT bool PNG | same |
| TTA | 无（仅 iterative refinement） | 无 | same |
| determinism | 官方无 seed | 移植 eval 也未 seed；4 次 point 不同 | **differ**（D-06） |

G4=pass（caveat）：transform/postprocess/threshold/export 与官方一致；oracle prompt
与项目 metric 已披露，但 metric 不是官方口径。G5=pass（caveat）：18/6 split、
无 support/donor；**oracle GT prompt 是设计的一部分，UCS 必须单独标为 oracle/reference，
不得与 prompt-free 方法同列比较**。

## 07. 门禁结论（provisional）

```text
A0  = unverified
G0  = pass (caveat)         # commit/URL 可追溯；无 LICENSE；HEAD/论文漂移已披露
G1  = pass                  # 569 keys、314,716,560 params、262 buffers 全同
G2  = pass (caveat)         # forward/中间层 exact；decoder loss/grad/Adam step 在容差
G3  = unverified            # 无训练；workers/seed/eval 非确定性
G4  = pass (caveat)         # eval/postprocess/threshold 同；项目 metric 未官方复现
G5  = pass (caveat)         # 18/6 split；oracle prompt 已披露；非 prompt-free
G6a = unverified            # 未跑官方 20-step；1024 全反向 OOM
G6b = N/A                   # 无官方 UCS checkpoint
G6c = unverified            # 未跑全量训练/评测

main_table = blocked        # 若入表，只能作 oracle/reference 行
claim_label = none
official_metric_match = unverified
```

仍缺：把 eval prompt 的随机性固定或显式声明（并修正 `UPSTREAM.md:60-61` 的
seed 声明）；补 workers/seed/metric 到 deviations；明确 HEAD 与 paper 架构差异；
补官方 20-step/G6a 和全量 G6c；补 LICENSE；冻结 A0 calibration pack。
