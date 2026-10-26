# SAM ViT-L + U-Net 移植审计（手工 pilot）

- 参考实现：项目上一版适配器 `baselines/adapters/sam_vit_l_unet.py@196f03f^`
  （git blob `bc7a894fd909564f772867dfd89f1a9ff743f2ae`）；**项目扩展，不是论文 baseline**。
- 移植：`src/baseline_zoo/models/sam_vit_l_unet/**` + `src/baseline_zoo/adapters/sam_vit_l_unet.py`
- 项目 commit：`fc7d8f4711c025d75240581541fa0a067427a011`；日期：2026-10-25
- Profile：`probe_standard_sam`；附录/探针，不进主表。
- 资产：SAM ViT-L `sam_vit_l_0b3195.pth`
  SHA256 `3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622`；
  **无官方 decoder checkpoint**。当前 `baselines/` 目录已删除，旧适配器只剩 git 历史。

> **A0=unverified。** 无冻结 calibration pack；严格按 `cod reviewer/README.md:71-76`
> 整体 blocked。以下 gate 只用于定位问题。

## 01. 官方敏感点清单

| id | 敏感点 | 参考位置 | 期望 / 失败症状 |
|---|---|---|---|
| S-vl-01 | 来源、许可证、旧适配器、SAM 权重 | `UPSTREAM.md:1-8`; `196f03f^:1-40` | 可追溯；否则 G0 不可信 |
| S-vl-02 | 自包含 SAM ViT-L encoder | `src/models/sam_network/build_sam.py:54-72` | 与上游 `segment-anything` ViT-L 同结构/同权重 |
| S-vl-03 | U-Net decoder 结构 | `model.py:25-101` | 与旧 adapter decoder 逐 AST 相同 |
| S-vl-04 | 冻结 encoder、4 中间特征 + neck、fp16 缓存语义 | `model.py:104-122`; 旧 `313-321` | 特征/图像精度与旧版一致 |
| S-vl-05 | 训练输入 bicubic image / nearest mask 1024 | `adapter.py:85-101` | 同几何/二值化 |
| S-vl-06 | 先编码未增强图，再对 feature/image/mask 做同一 flips/rot90 | `adapter.py:113-136` | 编码翻转图会破坏绝对位置编码 |
| S-vl-07 | BCE + Dice loss | `adapter.py:19-23,138-139` | 同公式 |
| S-vl-08 | 200 epochs、batch 4、AdamW decoder 1e-3/wd1e-4、cosine eta1e-6、val Dice | `adapter.py:31-36,76-83`; 旧 `435-439` | 同配方/选择 |
| S-vl-09 | decoder-only checkpoint，兼容 full state dict | `adapter.py:141-156` | 旧版也只存 decoder |
| S-vl-10 | 预测 bicubic 1024、logits 双线性回 GT、sigmoid>0.5、无 TTA | `adapter.py:158-172` | 同推理/阈值 |
| S-vl-11 | 18 train / 6 val；无 support/donor | 旧 `390-391`; 项目 split | 无泄漏 |
| S-vl-12 | 官方方法 checkpoint | 无 | G6b N/A |

## 02. 对照与差异

```text
# git show 196f03f^:baselines/adapters/sam_vit_l_unet.py (313-321, 447-482)
'features': [feature.half().cpu() for feature in features],
'image': image.half().cpu(),
'mask': mask.half().cpu(),
...
model.train(); random.shuffle(order)
loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(...)
improved = metrics['dice'] > best_dice

# src/baseline_zoo/adapters/sam_vit_l_unet.py (119-139, 158-172)
feature_maps = [feature.half().float() for feature in model.encode_features(images)]
...
logits = model.decoder(images, feature_maps)
return nn.BCEWithLogitsLoss()(logits, masks) + dice_loss(logits, masks)
...
logits = F.interpolate(logits, size=(height, width), mode='bilinear')
return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
```

| id | verdict | category | gate | 参考 vs 移植 | 披露 | numeric impact |
|---|---|---|---|---|---|---|
| D-vl-01 | `equivalent` | `wrapper_change` | G1 | 自包含 encoder 与上游 ViT-L 345 keys/全部 tensor 值相同 | 已披露（`UPSTREAM.md:4-5`） | 308,278,272 params、0 buffers；exact true |
| D-vl-02 | `equivalent` | `wrapper_change` | G1 | decoder `DoubleConv/Up/SAMViTLUNetDecoder` 去掉 docstring 后 AST 相同 | 已披露（`UPSTREAM.md:6-7`） | 712,625 decoder params；结构零差异 |
| D-vl-03 | `intentional_deviation` | `wrapper_change` | G3 | 旧版一次缓存全 split 特征；新版每 batch 重算 | 已披露（`adapter.py:40-44`） | 冻结 encoder eval，特征应同；仅吞吐/显存不同 |
| D-vl-04 | `differ` | `precision` | G2/G3 | 旧版缓存 `image.half()`/`mask.half()`；新版 image/mask 保持 float32，仅 feature half→float | 部分披露（只写 feature 量化） | 固定图 logits max diff **6.258e-7**；mask diff 0（随机 decoder 退化，不能当精度结论） |
| D-vl-05 | `differ` | `reproducibility` | G3 | 旧版 `random.shuffle(order)`；新版 DataLoader shuffle + 协议 seed | 未在 deviations 明写 | 训练顺序不同；未训练，无法量化 |
| D-vl-06 | `equivalent` | `wrapper_change` | G2/G4 | 旧版保存 decoder state dict；新版 runtime payload 内仍只存 decoder | 已披露（`adapter.py:45-47`） | 无模型权重差异 |
| D-vl-07 | `unverified` | `missing_asset` | G2/G3/G6b/G6c | 无官方 decoder checkpoint；全量 run 在 `paper_metrics` 失败，`n_images=0` | 已记录（status JSON） | 无训练后数值；G6b=N/A，G6c=unverified |
| D-vl-08 | `intentional_deviation` | `profile` | G0/G6 | 项目扩展，不是论文 baseline；不做 L6、不进主表 | 已披露（`UPSTREAM.md:3`; `adapter.py:37-39`） | 只能作附录/探针 |
| D-vl-09 | `differ` | `missing_provenance` | G0 | 模型目录与 `src/models/sam_network` 无 LICENSE；旧 adapter 已从工作树删除 | `UPSTREAM.md` 未写许可证 | G0 只能 pass(caveat) |
| D-vl-10 | `equivalent` | `postprocess` | G4 | 旧/新均 bicubic 1024、双线性回 GT、sigmoid>0.5 | 已披露（`adapter.py:50`） | 同一路径 |
| D-vl-11 | `equivalent` | `data` | G5 | 旧/新均使用 18 train / 6 val；无 support/donor | 已披露（`UPSTREAM.md:1-8`） | 无 test/val support |
| D-vl-12 | `unverified` | `missing_asset` | G6a | 未做同初始化 ≥20-step 训练对照 | — | G6a=unverified |

### 01→02 closure

| S-id | disposition | 对应 |
|---|---|---|
| S-vl-01 | `covered_by_discrepancy` | D-vl-09；旧 adapter git 可追溯 |
| S-vl-02 | `preserved` | D-vl-01；03 encoder probe |
| S-vl-03 | `preserved` | D-vl-02；AST 对照 |
| S-vl-04 | `covered_by_discrepancy` | D-vl-03, D-vl-04 |
| S-vl-05 | `preserved` | `adapter.py:85-101` 与旧 `load_pair` 同插值/阈值 |
| S-vl-06 | `preserved` | `adapter.py:113-136` 与旧 `cache_split` + `augment` 同语义 |
| S-vl-07 | `preserved` | D-vl-02；公式逐项相同 |
| S-vl-08 | `covered_by_discrepancy` | D-vl-05；配方相同、顺序不同 |
| S-vl-09 | `preserved` | D-vl-06 |
| S-vl-10 | `preserved` | D-vl-10 |
| S-vl-11 | `preserved` | D-vl-11 |
| S-vl-12 | `not_applicable` | D-vl-07；无官方方法 checkpoint |

closure = 12/12；实质 open 项：无 decoder checkpoint/训练、image fp16 语义差异未完全披露。

## 03. 数值证据

```bash
/home/dev/anaconda3/envs/tsai/bin/python /tmp/sam_vit_l_encoder_probe.py
CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/sam_vit_l_smoke.py
CUDA_VISIBLE_DEVICES=1 /home/dev/anaconda3/envs/tsai/bin/python /tmp/sam_vit_l_precision_probe.py
```

```text
encoder vs 3rd/segment-anything vit_l:
  official/port params 308,278,272 / 308,278,272; buffers 0 / 0
  keys 345; missing 0; extra 0; shape mismatch 0; value mismatch 0; exact true
decoder AST vs 196f03f^: DoubleConv/Up/SAMViTLUNetDecoder same without docstrings
smoke: logits (1,1,1024,1024), finite true, BCE(zeros)=0.6319735050,
       5 feature maps (4x1024x64x64 + 1x256x64x64), decoder 712,625 params
precision: new float image vs old fp16-cache image max|Δlogit| = 6.258e-7
           float32 features vs fp16-roundtrip features max|Δlogit| = 2.980e-8
           （随机 decoder 输出全前景，mask diff 0 无判别力）
full run: results/.run_status/sam_vit_l_unet.json status=failed,
          stage=paper_metrics, n_images=0; metrics.csv 只有表头
```

`G2=unverified`：没有官方 decoder 权重，也没有 official-vs-port 同权重输出；
上述 encoder/AST/smoke 只能证明结构与路径，不足以 G2 pass。`G6a=unverified`；
`G6c=unverified`。本模型按 `probe_standard_sam` 只审 L1–L5，不做 L6。

## 04. 训练配置

| 项 | 旧 adapter（参考） | 移植 | 判定 |
|---|---|---|---|
| epochs | 200 | 200 | same |
| batch | 4 | 4 | same |
| optimizer | AdamW decoder, lr 1e-3, wd 1e-4 | 同 | same |
| scheduler | cosine, T_max=epochs, eta 1e-6 | 同 | same |
| loss | BCE + Dice | 同 | preserved |
| input | bicubic 1024 image / nearest mask | 同 | preserved |
| features | 一次缓存，fp16 feature/image/mask | 每 batch 重算；仅 feature fp16 回 float | **differ**（D-vl-03/04） |
| augmentation | 编码未增强图，再对 feature/image/mask 同 flips/rot90 | 同 | preserved |
| data | 18 train / 6 val | 同 | same |
| workers/GPU | 缓存后单进程；单卡 | DataLoader workers=0；单卡 | 实际等价 |
| selection | val Dice | val Dice | same |
| seed | 默认 42，seed 全部 RNG | 协议 seed | 顺序仍不同（D-vl-05） |

G3=unverified：没有训练后 checkpoint；特征重算与 image 精度/数据顺序差异未做
同初始化训练对照。

## 05. 评测/推理

| 项 | 旧 adapter | 移植 | 判定 |
|---|---|---|---|
| eval transform | bicubic 1024, /255 | 同 | same |
| checkpoint | decoder-only | runtime payload 内 decoder-only | equivalent |
| features | 缓存 fp16→float | 现场编码 float；image float | 微小精度差（D-vl-04） |
| postprocess | 双线性回 GT，sigmoid>0.5 | 同 | same |
| metric | Dice/IoU/clDice/HD95 | 同项目指标 | same |
| export | GT bool | GT bool | same |
| resolution | GT | GT | same |
| TTA | 无 | 无 | same |

G4=pass：旧/新推理与项目指标一致，无额外 eval 变换；G5=pass：18/6 split，
无 support/donor，无 test/val 泄漏。

## 07. 门禁结论（provisional）

```text
A0  = unverified
G0  = pass (caveat)         # 旧 adapter 仅存 git 历史；模型目录无 LICENSE
G1  = pass                  # encoder 345 keys 全同；decoder AST 相同
G2  = unverified            # 无官方 decoder checkpoint/同权重输出
G3  = unverified            # 无训练；image fp16/顺序差异
G4  = pass                  # 推理/指标与旧 adapter 一致
G5  = pass                  # 18/6 split；无 support/donor
G6a = unverified            # 未做同初始化 ≥20-step
G6b = N/A                   # 无官方 decoder checkpoint
G6c = unverified            # 全量 run failed，n_images=0

main_table = blocked        # 且 profile 本身为附录/探针
claim_label = none
official_metric_match = unverified
```

仍缺：补 decoder checkpoint 或完成一次可复现训练；把 image/mask fp16 差异和
DataLoader 顺序写入 deviations；补同初始化 G6a；修复 `paper_metrics` 失败并跑
G6c；补模型目录 LICENSE；冻结 A0 calibration pack。
