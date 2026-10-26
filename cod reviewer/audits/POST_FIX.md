# 修复后状态（代码级，2026-10-26）

> **这不是重新审计。** A0 已通过（pack v3: `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`），但本文件基于 A0 之前的
> 22 份模型报告，G6c 仍未运行，因此所有 gate 结论仍是 provisional，主表仍为 `blocked`。本文件只记录已确认缺陷的代码/文档修复
> 和回归证据；各模型 `README.md` 保留修复前的审计证据。

## 已修复的代码缺陷

| 模型 | 修复前缺陷 | 代码修复 | 回归证据 |
|---|---|---|---|
| `unet` | Dice 按图求均值，官方对 B,H,W 全局归约 | `adapters/unet.py` 改为全局归约 | `tests/test_unet.py`；重跑 `_shared/03_unet_hetero_probe.py`，loss/grad diff 均为 0；结果见 `unet/03_numeric_probe_after_fix.json` |
| `sam_adapter` | 官方 AdamW 只传 `lr`，PyTorch 默认 wd=0.01；移植用 0.0 | `official_weight_decay=0.01`，优化器使用该值；`UPSTREAM.md` 改为可执行配方 | `tests/test_sam_adapter.py` 断言 wd=0.01 |
| `knowsam` | unlabeled pool 只排除 train/val，166/384 是 test 图；缺 largest-CC；argmax/zoom 语义不同 | 排除 manifest 中 train/val/test 全部行；argmax→scipy zoom order=0→largest CC；disclose 384 覆盖和融合差异 | `tests/test_runtime_fixes.py` 泄漏测试 + `_largest_cc` 测试；修复后实际 unlabeled 仅剩 CHASEDB1 4、CREMI 246、CORN 134（DRIVE/DCA1/CrackTree200 无可用的非 test 图） |
| `matcher` | FSS 分数默认 1/0/0，官方 0.8/0.2/1.0；预测未 seed，且 `_prediction_seeded` 未初始化 | `Matcher.__init__`/`build_matcher` 改 0.8/0.2/1.0；首次预测按 `--seed` 播种并初始化标志 | `tests/test_matcher.py` 默认值和 seed 测试 |
| `ucs` | 预测 prompt 采样未 seed，重复 eval 不确定 | 仅最终 prediction 时，在 `prepare_eval_sample` 首次采样前按 `--seed` 播种 Python/NumPy/Torch；训练中的 per-epoch validation 不重置训练 RNG | `tests/test_ucs.py` 三个 seed 测试（初始 point sampling、训练 validation 不重置、预测播种） |
| `nnwnet` | planner/preprocess 看到 6 张 val 图；W&B batch 元数据写 CLI 默认值 | 先只 stage train → planner/preprocess → 再 stage val；日志改记 plans 的实际 batch/patch | `tests/test_runtime_fixes.py::test_nnwnet_planning_stage_excludes_val`；全量 nnU-Net 运行仍需重跑 |
| `persam` | 官方只用归一化 mean 算相似度、把 raw mean 传给 target-semantic prompting；移植两处都用归一化值 | `algorithm.py` 返回 `(similarity, raw)` 两路，adapter 分别传递；`UPSTREAM.md` 记录官方行号 | `tests/test_persam.py` embedding split + raw-prompting 测试 |

## 自检更正

本轮修复之后又做了一次对照自检，发现并修正了两处修复本身的问题：

- `ucs`：第一次只把 seed 放在 `predict_sample`，但初始 point 是在
  `prepare_eval_sample -> _prepare_sample -> init_point_sampling` 中抽取的，
  因此第一个 eval 样本仍然未受 `--seed` 控制。现在在最终 prediction 的首次
  eval 采样前播种；训练中的 per-epoch validation 不重置训练 RNG。
- `knowsam`：第一次用 PyTorch `mode='nearest'` 上采样，但官方是
  `scipy.ndimage.zoom(..., order=0)`；非整数缩放时两者并不同。已改为 scipy
  order=0，并重新核对 largest-CC 与 argmax 路径。
- `knowsam` 的 384 cap 披露也修正了：排除 test 后实际只剩
  CHASEDB1 4、CREMI 246、CORN 134，DRIVE/DCA1/CrackTree200 没有非 test
  的 unlabeled 图。

## 仅补充披露（未改训练/推理逻辑）

- `cs2net`：官方 2-GPU DataParallel、workers=2、无 seed；移植单 GPU/workers=0/shared seed。
- `jtfn`：官方 workers 和全局 RNG 未保留；移植 workers=0 + per-epoch RNG。
- `dscnet`：官方 workers/seed 未保留；移植 workers=0 + shared seed。
- `maskvsc`：官方 ratio=0 仍消耗 RNG；移植提前返回，且 workers=0。
- `sam_med2d`：官方 workers=4、无 seed；移植 workers=0 + shared seed；decoder 初始化差异已列。
- `topolora`：shared runtime 对最后一个 partial accumulation group 重新缩放，官方 Lightning 无等价逻辑；18 图 batch-1×4 的最后一组是 2 step。
- `samed`：官方 `train.py` 默认 SGD、README 用 `--AdamW`；移植按 README 的 AdamW，但保留源默认歧义披露。
- `fob_sam`：官方 test support 是 test fold 内的 held-out volume，且只评测有标注的 query slices；移植用 train/donor support 并评测全部 query。**G5 仍为 fail，不能进主表。**
- `sam_octa`：官方只对 Artery/Vein/LargeVessel 加 clDice；二值 20-dataset 移植统一 0.8 Dice + 0.2 clDice；官方指标集和 seed/workers 差异已披露。
- `sacm_official`：删除“upstream 在 1024 评测”的错误声明；官方 `test_sam.py:260-261` 会 resize 回原图。补充 commit、无 LICENSE、workers/seed；`test_sam.py` 自带 `SegmentationDataset(root_dir)`，没有 split 参数。
- `ucs`：官方 workers=4；移植 workers=0；官方指标集未复现。

## G0 文档修复

以下 `UPSTREAM.md` 的过期 `baselines/...` 路径改为本地 `3rd/...`，并补 commit：
`unet`、`cs2net`（同时把 URL 改为 `iMED-Lab/CS-Net`）、`bcunet`、`maskvsc`、
`nnwnet`、`segdino`、`sam_med2d`、`sam_octa`、`cwsam`、`fob_sam`、`knowsam`、
`sacm_official`、`sam_vit_l_unet`。

## 验证

```bash
/home/dev/anaconda3/envs/tsai/bin/python -m pytest tests
# 199 collected, 196 passed, 3 skipped (slow)
/home/dev/anaconda3/envs/tsai/bin/python 'cod reviewer/audits/_shared/03_unet_hetero_probe.py'
# official_global_dice == port_mean_dice; loss_absdiff=0.0; grad_maxdiff=0.0
```

### Remaining-model P1 fixes（2026-10-26）

- `sacm_official`：prediction 允许 base SAM 缺失（`checkpoint=None`），并持久化 `adapter_dim_ratio`/loss weights；非默认配置可重建。
- `sam_vit_l_unet`：checkpoint config 记录 SAM base 路径与 SHA-256；预测使用不同 encoder 时打印明确 warning。
- `dscnet`：`--dsc-stride > --input-size` 直接报错，不再留下未覆盖条带。
- `sam_med2d`：train loss 按 batch 平均，修复日志约为官方值 1/batch_size 的问题。
- `fob_sam`：save/restore inference config（shots、donor、best-pred idx、SAM type、data root、checkpoint paths）。
- 披露补齐：`cwsam` 的 2-GPU/batch-1/workers=8，`sam2_unet`/`segdino`/`fob_sam` 的 official seed/workers 差异。

### Post-fix smoke evidence

- JTFN HRF 3504×2336 at max-side 1024: peak allocated 3.72 GiB / reserved 4.49 GiB; FIVES 2048²: 5.33 / 6.22 GiB; both fit the 10.8 GiB GPU.
- FoB DeepCrack sample: canonical `DeepCrack`, donor `CrackTree200`, query not cached (one support entry).
- SAM-OCTA two fresh processes with seed 7: all 26 prompt coordinates/labels identical.
- Full suite: 194 collected, 191 passed, 3 skipped.

## Other-model review fixes（2026-10-26）

在 `OTHER_MODELS_REVIEW.md` 的 P0 和同批 P1 范围内修复：

- `fob_sam`：test stem → dataset root 的 canonical 解析，修复 unseen/novel donor 全部落到 DRIVE；`masked_attention.sin_pos` 加入 ignore key；test query 不再写入 `_pair_cache`（避免约 3.5 GiB CPU 缓存）。
- `sam_octa`：最终 prediction 的 prompt 采样在首次 eval 前按 `--seed` 播种；`prepare_train_batch` 对 `batch-size > 1` 明确报错，不再静默丢样本。
- `jtfn`：新增 `--jtfn-eval-max-side`（默认 1024）；超长 eval 图像先降采样，预测再双线性恢复到 GT，避免 11 GiB 下的 full-res OOM；这是已披露的 memory adaptation。
- `bcunet`：prediction / `--stage all` 的 `build_model` 不再请求 ImageNet-22k 下载，只有训练阶段使用预训练初始化。
- `samed`：low-res target 改为 `input_size/4`；poly horizon 改为 `max(epochs, 200)`，最后 epoch 不再 lr=0。
- 回归测试覆盖 donor 解析、sin_pos、cache、SAM-OCTA seed/batch guard、JTFN downscale、BCUNet pretrained 选择、SAMed low-res/poly。

仍未修：DSCNet stride gap、SACM base/ratio 持久化、SAM ViT-L provenance、CWSAM/SAM2-UNet/SegDINO/FoB workers 披露、SAM-Med2D loss logging 等；见 `OTHER_MODELS_REVIEW.md`。

## Post-A0 re-audit（7 个修复模型）

A0 v3 通过后，7 个修复模型做了代码级重审；结果与剩余阻断见
[`A0_REAUDIT.md`](A0_REAUDIT.md) / [`A0_REAUDIT.csv`](A0_REAUDIT.csv)。
`unet` 的 G2 已转为 pass；其余模型仍因缺 G6a/G6c、缺权重或未跑全量而
`main_table=blocked`。这份重审不是新的双审计。

## Code-review P0 修复（2026-10-26）

代码审查后确认并修复的 P0 项：

- `nnwnet`：planning 命令加 `--no_pp`，避免先用 8 workers 预处理 18 例又被第二次 24 例预处理覆盖；stage val 后把 raw `dataset.json` 刷新到 `preprocessed/dataset.json`，删除无效的 `plans['numTraining']` 补丁和死 guard。
- `matcher` 测试：POT 依赖的 import 移入测试并 `pytest.importorskip('ot')`；`setup_baselines.sh` 增加 `matcher` extra，干净安装不再在 pytest collection 阶段失败。
- `runtime`：新增 `BaselineAdapter.begin_prediction()`，在 `run_prediction` 开始时重置 `prediction_mode` 和 `_prediction_seeded`，结束后清 `prediction_mode`；同一 adapter 实例换 seed 再预测不会复用旧 latch。
- `knowsam`：更新 `_stage` 注释，明确排除 train/val/test 全部 manifest 行。
- `baseline_zoo.__main__`：训练/评测 YAML 命令里的裸 `python`/`python3` 现在解析为当前 `sys.executable`；修复 G6c 首次启动误用 base conda 的 `torch` 缺失。回归测试 `test_pipeline_commands_use_current_interpreter`。
- 回归测试：nnWNet `--no_pp` 命令、preprocessed `dataset.json` 同步、两阶段 staging（18 train + 6 disjoint val + `numTraining=24`）；prediction latch 重置测试。

## 仍未解决（阻断主表）

1. **A0**：已通过 v3 双审计；但现有 22 份报告是 A0 之前产出，必须按新资格重审。
2. **G6c**：所有模型都没有全量训练/评测/重算/复跑；没有任何模型满足主表。
3. **G6a**：`unet`、`knowsam`、`matcher`、`nnwnet`、`ucs` 等仍缺官方同状态多步训练对照。
4. **资产**：`matcher` 缺 DINOv2 ViT-L；`sam2_unet` 缺 SAM2 Hiera-L；多个模型无官方 checkpoint，G6b=N/A。
5. **协议差异**：`fob_sam` G5 support 语义仍 fail；`sam_octa` class-conditional clDice 未复现；`topolora` partial-group accumulation 未量化。
6. **重审计**：修复后的 `knowsam`/`nnwnet`/`matcher`/`ucs`/`persam` 需要按 02 重新跑探针，不能把本文件的代码级回归当成 gate pass。
