# 模型管理与配置（Model Management Guide）

项目通过**模型注册表 + ModelSpec 接口**管理所有可训练/可评测的模型。
训练与评测脚本只认 ModelSpec、不认具体模型——吸纳新架构只需在
`models/` 里新增一个模块，脚本零改动。

## 1. 当前已注册模型

| 名称 | 架构 | 用途 |
|---|---|---|
| `sam_l` | SAM ViT-L + 完整 SACM-v2 架构 | 默认（论文主模型） |
| `sam_b` | SAM ViT-B + 同架构开关 | 调试/快速实验（CPU 友好） |
| `sam_h` | SAM ViT-H + 同架构开关 | 容量上限实验 |

```bash
python src/train/trainer.py --model sam_b --preset stage2 ...   # 换模型 = 换一个参数
```

## 2. ModelSpec 接口（`models/base.py`）

每个模型是一份"自述书"，六个可调用对象：

```python
@dataclass
class ModelSpec:
    name: str
    build: Callable          # (**overrides) -> nn.Module
    freeze: Callable         # (model) -> None           冻结协议
    param_groups: Callable   # (model, args) -> list     差分学习率分组
    preprocess: Callable     # (model, images) -> images 输入前处理
    prompt_builder: Callable # (model, batch) -> (sparse, dense) 提示锚点
    forward: Callable        # (model, images, sparse, dense, return_stage1) -> outputs
    input_size: int = 1024
    description: str = ""
    default_config: dict = {}
```

**约定**：`forward` 返回 SAM 形状的三元组 `(masks_stage2 [B,4,256,256],
iou_pred [B,4], masks_stage1 [B,4,256,256])`（或二元组）。新模型若不
具备多 head/两阶段结构，返回的辅助项置 `None` 并在 spec 里声明
`has_aux = False`，训练循环按声明跳过辅助损失（见 §5 的迁移路线）。

## 3. 吸纳新模型的五步流程

以"吸纳 U-Net 作为基线"为例：

1. **新建 `models/unet.py`**，实现六个函数（build/freeze/param_groups/
   preprocess/prompt_builder/forward——forward 返回单掩膜，其余置 None）
2. **注册**：
   ```python
   register_model('unet')(ModelSpec(name='unet', build=..., ...))
   ```
3. **import 副作用**：在 `models/__init__.py` 加 `from . import unet`
4. **测试**：`tests/` 加该模型的冒烟（build + 一次 forward 形状断言）
5. **文档**：本文件注册表 + `ARCHITECTURE_CHANGES.html` 修订记录

完成后 `--model unet` 立即可用于 train/test 脚本（CLI choices 从注册
表动态生成，无需改脚本）。

## 4. 模型配置的三层结构

```
全局训练配置（src/train/trainer.py 的 argparse）
    └── 模型选择（--model，来自注册表）
    │       └── 架构覆盖（--use_geo_i 等；当前为 SAM 家族专用开关）
    └── 预设（configs.py 的 PRESETS）——SAM 架构消融专用，覆盖架构开关
```

**分层原则**：
- **全局**：epochs/lr/损失权重/增强——与模型无关
- **模型**：spec.default_config 存放该模型的默认架构参数；`spec.build`
  用 `{**defaults, **overrides}` 合并（SAM 家族已示范此模式）
- **预设**：仅对 SAM 家族的架构开关有意义；新模型若有自己的架构开关，
  在 spec.default_config 中声明，configs.py 保持 SAM 专用并注明

**未来**：spec.default_config 迁往 yaml（`models/configs/<name>.yaml`），
CLI 增加 `--model_config path` 覆盖——已列入技术债表，验证后实施。

## 5. 训练循环的模型无关化边界（迁移路线）

当前训练循环（src/train/trainer.py）的损失组装对 SAM 输出形状**有感知**
（深监督用 masks_stage1、IoU 监督用 iou_pred）。完全的模型无关化分两步：

- **阶段 A（本次已做）**：构建/冻结/分组/前处理/提示/前向六件事全部
  经 spec——脚本不再 import 任何具体模型
- **阶段 B（验证后做）**：损失组装也模型化——spec 增加
  `aux_losses(model, outputs, targets) -> (aux_loss, aux_terms)`，
  通用层（BCE/Dice/clDice 作用于主掩膜）留在训练循环，SAM 特有的
  深监督/IoU 移入 `models/sacm/`。届时吸纳单输出模型（U-Net）时
  训练循环真正零改动

## 6. 纪律

1. **新模型必须带 spec**：禁止在 train/test 脚本里写死任何具体模型
2. **forward 契约先行**：新模型的输出形状与 SAM 三元组不一致时，
   先扩展 ModelSpec 协议（阶段 B），不要为单个模型打补丁
3. **测试随行**：每个新模型一个冒烟测试（build + 形状断言 + 可训练
   参数非空）
4. **checkpoint 自描述**：训练保存的 checkpoint 已含 `config` 键，
   新模型应追加 `model` 键（模型名），评测时校验匹配
