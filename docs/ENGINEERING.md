# 工程化手册（Engineering Guide）

本文档是项目的工程规范与操作手册。算法设计背景见 `docs/ARCHITECTURE_CHANGES.html`。

## 1. 项目结构

```
SACM/                              # 项目 = 分割算法平台（框架 + 算法分层）
├── src/                           # 全部源代码（标准 src 布局）
│   ├── core/                  # 通用框架层（算法无关）
│   │   ├── trainer.py             #   统一训练引擎（模型无关，经 ModelSpec）
│   │   ├── evaluator.py           #   统一评测引擎（TTA/选择/全指标/输出）
│   │   ├── data.py / io.py / augmentation.py
│   │   ├── models.py              #   ModelSpec + 模型注册表（通用机制）
│   │   ├── losses/                #   损失注册表（核心 + 14 拓扑损失官方移植）
│   │   └── metrics/               #   指标（16 常规 + 4 可选，官方对齐）
│   ├── models/                # 算法层（每个算法一个模块）
│   │   ├── __init__.py            #   算法注册表（MODEL_REGISTRY）
│   │   ├── sam/                    #   SAM 架构模块（含全部 SACM 改造；已展平：无 modeling/utils 子层）
│   │   └── sacm/                   #   SACM 算法（只是众多算法之一）
│   │       ├── specs.py            #   SAM 家族 ModelSpec
│   │       ├── configs.py          #   SACM 消融预设（14 个）
│   │       └── diagnose.py         #   SACM 诊断（论文 motivation 证据）
│   ├── train/                     # 训练 CLI（trainer.py，--algorithm + --model + --config）
│   ├── eval/                      # 评测 CLI（evaluate.py）
├── scripts/                      # 入口脚本（数据准备/实验调度/结果聚合/官方对拍）
│   └── prepare_data.py / run_experiments.py / aggregate_results.py / verify_against_official.py
├── configs/                       # YAML 配置（presets.yaml，--config 叠加覆盖）
├── datasets/                      # 数据（gitignored）
├── outputs/                       # 输出：结果/日志/checkpoint（gitignored）
├── tests/                         # pytest（指标手算值、损失冒烟、增强、预设、慢速训练一步）
├── docs/                          # 本手册 + 模型管理指南 + 设计书
├── pyproject.toml / Makefile / README.md / .gitignore
└── .github/workflows/ci.yml       # CI：lint + 全测试 + 模型冒烟
```

## 2. 日常命令

```bash
make install      # pip install -e ".[dev]"
make test         # 全部单元测试（CPU，秒级）
make test-slow    # 含"训练一步"端到端测试（CPU，约 1 分钟）
make lint         # ruff 检查
make smoke        # 构建完整模型 + 一次前向（新机器第一件事）
make verify       # 与官方仓库数值对拍（需 ../segment/3rd 下的官方仓库）
make registry     # 打印当前环境可用的拓扑损失清单
```

## 3. 新机器的验证顺序（标准流程）

```bash
make install && make smoke        # 1. 模型能建、张量流通
make test                         # 2. 指标手算值 + 损失冒烟全过
make verify                       # 3. 与官方代码数值对拍（严格一致）
python src/train/trainer.py --preset sacm --data_root <root> --checkpoint <sam.pth>   # 4. 基线复现
```

## 4. 吸收新损失的标准流程

1. **溯源对齐**：在 `losses/` 建家族模块，逐行/逐字移植官方实现；docstring 必写：官方出处（仓库+文件）、对齐方式、全部偏差
2. **统一桥接**：单通道 logits + 浮点掩膜输入、标量输出；官方多通道用 `cat([zeros, logits])`（softmax fg ≡ sigmoid）；概率输入先 sigmoid；性能受限的在适配器统一 256 分辨率
3. **注册**：`register('name')(Adapter)`，硬依赖缺失则跳过注册（satloss 模式）
4. **测试**：在 `tests/test_losses.py` 的参数化冒烟测试中自动覆盖（无需改测试代码）
5. **文档**：README 损失清单 + `ARCHITECTURE_CHANGES.html` 修订记录

## 5. 吸收新指标的标准流程

1. 按家族放入 `metrics/`（pixel/distance/surface/topology/persistence/aggregate）
2. 官方约定（含边界情形：空掩膜、退化除法）必须与官方一致并写入 docstring
3. 在 `tests/test_metrics.py` 加手算值断言
4. 接入 `compute_metrics`（常规或可选 flag）

## 6. 纪律

- **不盲封装**：需要编译库的，写构建指引而非不可测试的封装
- **自带 Dice 的损失用小权重**：作为附加项时 `--topology_loss_weight` ≤ 0.1
- **性能警告进 docstring**：纯 Python 引擎必须注明分辨率与耗时预期
- **约定显式化**：每个非显然约定（阈值、平滑、聚合方式）写进 docstring——本项目的"严格一致"承诺靠此兑现
- **改动必测试**：任何 losses/metrics 改动跑 `make test`；任何官方移植改动跑 `make verify`

## 7. 已知技术债（有意的阶段划分）

| 项 | 状态 | 计划 |
|---|---|---|
| `sacm/` src 布局（segment_anything 入包） | 待做 | 运行验证通过后进行，需全仓 import 重写 |
| yaml 配置系统 | 待做 | configs.py 预设暂为唯一配置源，够用 |
| C 档损失（TopoLoss/DMT/CubicalRipser） | 指引化 | 目标机器确认编译工具链后补严格封装 |
| 训练脚本与 losses/metrics 包的同构化重构 | 部分 | train/test 已接包；更深拆分随 src 布局一起做 |
