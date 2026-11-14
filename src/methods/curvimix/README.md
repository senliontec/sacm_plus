# CurviMix-SAM

`CurviMix-SAM` 是独立于 `baseline_zoo` 模型内容的新模型（adapter 在新文件夹，
不改任何已有 baseline 模型文件）。它沿用冻结 SAM ViT-L + DCD 四候选脚手架
和 SPA-SAM 的 mild augmentation，并新增：

1. **结构感知 Copy-Paste（SCP）**：从**训练集**的其它图像中取完整的曲线结构，
   只做翻转、90°旋转和平移，贴到当前训练图；不裁剪、不弹性形变，保持细线
   拓扑。测试图/验证图永远不作为 donor。
2. **多视角 TTA**：推理时对 fixed candidate 3 做 h/v/hv 翻转的 logits 平均；
   可选 `--tta-select` 根据四视图二值 mask 的稳定性选择候选头，无需标签。
3. 部署仍是 fixed candidate 3 + `head_fusion=0`；Copy-Paste 不增加推理参数，
   TTA 是可选推理增强。

## 文件

- `adapter.py`：`CurviMixAdapter(SpaSamAdapter)`，copy-paste 和 TTA。
- `run_one.py`：本地训练/预测入口（不经过 baseline registry）。
- `derive_fixed3.py`：CurviMix 自己的 fixed3 派生。
- `evaluate.py`：先预测、再调用已有指标收集器。
- `configs/models/curvimix.yaml`：默认 50 epochs 的本地配置。

## 当前 seed 0 结果（20 数据集宏平均）

| 设置 | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| SPA-SAM 50（旧 seed 0） | 58.08 | 44.74 | 66.53 | 62.05 |
| CurviMix 50，无 TTA | 59.12 | 45.82 | 67.53 | 62.12 |
| CurviMix 50，4-view TTA | **59.76** | **46.52** | **68.14** | **61.00** |
| CurviMix 200，无 TTA | 59.42 | 45.77 | 67.99 | 61.34 |
| **CurviMix 200，4-view TTA** | **60.50** | **47.01** | **69.24** | **58.97** |

相对 SPA-SAM seed0 的 50 轮结果，200 轮 + TTA 为
`+2.42 / +2.27 / +2.71 / -3.08`；相对 SPA-SAM 4-seed 宏平均
（57.39/43.76/66.34/62.95）为 `+3.11 / +3.25 / +2.90 / -3.98`。
逐角色（200 TTA）：Seen 64.61/49.85/73.15/9.77，
Unseen 73.15/59.38/80.88/38.99，Novel 39.51/27.70/49.80/134.80。

## 已知弱点

- 目前主要是 seed 0；4 seed 结果还在跑。
- TYRE、WIRE 等部分数据集 copy-paste 后变差；需要 donor/密度自适应或更多
  seed 验证。
- HD95 仍受 TopoMortar 等少数数据集驱动；必须逐数据集报告。
- Copy-paste 是训练期数据引擎，不是解码器架构创新；论文需要把“结构保持”
  和 TTA 的 label-free 共识写成可检验的贡献。

## 论文灵感（parsing 中相关论文）

- TRACE：prediction-adaptive connectivity supervision；
- SnakeSAM：Dynamic Snake Adapter / path-field prompting；
- Dual-level Adapter：Adapter-I + Adapter-E 的双层适配；
- PEPA：Target-Conditioned Snake Upsampling + adaptive threshold；
- FineSAM++：不确定区域的 residual expert；
- TopoTTA / Skeleton-Guided TTA：topology-aware test-time adaptation；
- Widest-Path Reachability：connectivity-preserving fields。

当前实现只采用 **intact structure copy-paste + flip TTA**，没有声称实现上述
论文的架构；后续若要冲 ICCV，需要把 topology-aware 训练/推理做成更强的
方法级贡献。

## 运行

```bash
# 50 轮 seed0 训练
PYTHONPATH=src python -m methods.curvimix.run_one --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/runs/curvimix/default/seed_0 \
  --epochs 50 --seed 0 --device cuda:0 --use_wandb 0 \
  --sam-checkpoint checkpoints/sam_vit_l_0b3195.pth \
  --train-aug mild --copy-paste-prob 0.5 --head-select gating \
  --deployment-head 3 --head-fusion 1

# fixed3 派生
PYTHONPATH=src python -m methods.curvimix.derive_fixed3 \
  --input results/runs/curvimix/default/seed_0/best_model.pth \
  --output results/runs/curvimix/default/seed_0/best_model_fixed3.pth \
  --deployment-head 3

# 20 数据集评估（可加 --tta-flip / --tta-select；--tag 分开目录）
PYTHONPATH=src python -m methods.curvimix.evaluate \
  --data-root data/sacm_3shot_seed0 \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
  --workdir results/runs/curvimix/default/seed_0 \
  --checkpoint results/runs/curvimix/default/seed_0/best_model_fixed3.pth \
  --seed 0 --device cuda:0 --use-wandb 0 --tta-flip --tag _tta
```
