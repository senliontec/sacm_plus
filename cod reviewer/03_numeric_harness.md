<task>
写一个 Python 脚本，同时加载官方实现与移植实现，自动对比并输出 JSON 报告。
只输出脚本和 <cannot_cover>，不要解释性文字。
</task>

<context>
必须按复合 profile 选择检查（同时记录 training mode / prompt modality / backbone）：
- `supervised_prompt_free_sam`：state_dict + forward + intermediate + loss + grad + optimizer step；
- `supervised_oracle_prompted_sam`：上述 + GT prompt tensor 对比 + 后处理；
- `supervised_support_based_sam`：state_dict + forward + loss/grad + support/episode 对比；
- `training_free_support_based_sam`：不跑 loss/grad；固定 support/target 下的 mask IoU、
  prompt 差、support 选择、确定性重复；
- `semi_supervised_prompt_free_sam`：模型 + unlabeled pool 审计 + prompt 路径；
- `supervised_standard_non_sam`：state_dict + forward + intermediate + loss + grad + optimizer step；
- `probe_standard_sam`：L1–L5（训练时含 loss/grad）；不做 L6。
</context>

<requirements>
1. 固定 torch/numpy/random 种子；固定输入张量（shape、dtype、生成方式写死）；
   记录 torch/CUDA/cuDNN/库版本和 deterministic 设置。
2. 隔离包名冲突：首选 subprocess 分别在官方仓库根目录和项目根目录执行探针，
   两侧通过临时 .pt/.npz 交换张量，主进程做对比。禁止用 importlib 单文件加载
   含包内相对导入的模块，除非在 <cannot_cover> 说明风险。
2b. **同一权重**：任何 forward/loss/grad 对比前，两侧必须加载同一份权重。
   首选官方 checkpoint 映射到两侧；次选把同一 state_dict 复制到两侧。
   若两侧只能各自独立随机初始化（不能共享同一 state_dict），L2 必须
   `skipped`/`unverified`；共享同一随机 state_dict 可以做 L2 实现等价性对比，
   但不能替代官方权重对比；G6 仍需官方 checkpoint 或短程 loss/grad parity。
   记录 `weights.match`（`full|partial|none`）、权重来源、sha256、成功映射的 key 数。
   `full`=共享键全部映射且无 unexpected；`partial`=只映射共享模块（如类数/输出空间改变）；
   `none`=无可共享权重，L2 必须 skipped/unverified。
3. state_dict 检查（不是简单键集合相等）：
   - 提供 key_mapping（official key → ported key）；
   - 区分 shared_keys、expected_missing、expected_extra、unexpected_missing、
     unexpected_extra、shape_mismatch；
   - 遍历 named_parameters 和 named_buffers；BN 的 running_mean/var/num_batches_tracked
     属于 buffer，单独打印 track_running_stats；
   - 若提供官方权重，映射后比较 tensor 的 max_abs_diff，而不是只比 key；
   - 对 LoRA/Adapter/wrapper，比较 base weight 的映射，不要期待 wrapper key 同名。
4. eval 前向：输出 shape/dtype、mean/std/min/max、逐元素最大绝对差、allclose；
   对 class head 改变的模型，优先用官方代码的同一 num_classes 配置；否则只比共享模块，
   class-head/output parity 明确 skip 并写 unverified 原因。
5. 中间层：用 forward hook 抓取 <intermediate_layer_mapping> 中的层；
   官方名与移植名必须映射；比较统计量和最大绝对差。
6. train 模式：同一输入/标签下比较 loss 标量、各 loss 分量、反传后指定参数梯度范数、
   一次 optimizer.step 后参数差、scheduler 一步后的 lr/状态。
7. training-free：固定 support/target；重复 N 次（至少 3）比较 mask IoU、prompt 坐标差、
   support 选择是否一致；若官方入口只能跑 CLI，写 wrapper 并记录。
8. 协议探针：
   - transform 输出：固定图像/mask，输出 tensor hash 或逐元素差；
   - normalize：mean/std/极值；
   - metric：固定 pred/gt 的 Dice/IoU/clDice/HD95 向量；
   - export：mask shape、dtype、unique values、GT 分辨率。
9. 非确定性算子（k-means/AMG/PCA/DSConv/CUDA）：
   记录是否 deterministic；重复运行并记录方差；不得仅凭单次运行判 fail。
   若重复后差异仍稳定超差，必须判 fail；只有差异随运行波动且无法收敛时才标 skipped/unverified，
   并写清掩盖了哪类差异。
10. 容差为顶部常量：float32 forward atol=1e-5、rtol=1e-4；
    自定义算子 atol=1e-3；mask 要求 IoU ≥ 0.9999 或逐像素一致；
    loss/grad 建议 atol=1e-4。可按 baseline 覆盖，但必须记录。
11. 任一项超差：打印 key、两侧数值、差值，继续跑完，最后非零退出。
12. 输出 JSON 到 stdout，字段严格遵循下方 schema，便于批量汇总。
</requirements>

<json_schema>
{
  "baseline": "<str>",
  "profile": "<str>",
  "method": {"training_mode": "<str>", "prompt_modality": "<str>",
             "backbone": "<str>"},
  "environment": {"torch": "<str>", "cuda": "<str>", "cudnn": "<str>",
                  "numpy": "<str>", "device": "<str>", "deterministic": true,
                  "library_versions": {"<name>": "<version>"}},
  "context": {"official_commit": "<str>", "ported_ref": "<str>",
              "official_files": ["..."], "ported_files": ["..."]},
  "weights": {"match": "full|partial|none", "source": "<official_checkpoint|state_dict_copy|random_init>",
              "sha256": "<str|null>", "mapped_keys": 0,
              "unmapped_expected": ["..."]},
  "state_dict": {
    "key_mapping": [{"official": "<str>", "ported": "<str>"}],
    "shared_keys": 0,
    "expected_missing": ["..."],
    "expected_extra": ["..."],
    "unexpected_missing": ["..."],
    "unexpected_extra": ["..."],
    "shape_mismatch": [{"key": "<str>", "official": [0], "ported": [0]}],
    "param_count": {"official": 0, "ported": 0},
    "buffer_count": {"official": 0, "ported": 0},
    "loaded_weight_diffs": [{"official_key": "<str>", "ported_key": "<str>",
                             "max_abs_diff": 0.0}],
    "loaded_weight_max_abs_diff": null
  },
  "forward_eval": {
    "shape": [0], "dtype": "<str>",
    "official_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "ported_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "max_abs_diff": 0.0, "allclose": true, "skipped_reason": null,
    "outputs": {"<name>": {"shape": [0], "max_abs_diff": 0.0, "allclose": true}}
  },
  "intermediate": [
    {"name": "<str>", "mapping": "<str>", "max_abs_diff": 0.0,
     "allclose": true, "skipped_reason": null}
  ],
  "train": {
    "applicable": true,
    "skip_reason": null,
    "loss": {"official": 0.0, "ported": 0.0, "max_abs_diff": 0.0},
    "components": {"<name>": {"official": 0.0, "ported": 0.0}},
    "grad_norm": [{"param": "<str>", "official": 0.0, "ported": 0.0}],
    "optimizer_step_max_abs_diff": 0.0,
    "scheduler": {"official_lr": 0.0, "ported_lr": 0.0,
                  "official_step": 0, "ported_step": 0}
  },
  "training_free": {
    "applicable": false,
    "support_set": "<str>", "target": "<str>",
    "mask_iou": null, "prompt_max_abs_diff": null,
    "determinism_repeats": 0
  },
  "protocol_probes": {
    "transform_max_abs_diff": null,
    "normalization": {"official": [0.0], "ported": [0.0]},
    "metrics": {"official": [0.0], "ported": [0.0]},
    "split_manifest_sha256": {"official": "<str|null>", "ported": "<str|null>"},
    "support_selection": {"official": ["..."], "ported": ["..."]},
    "export": {"official_shape": [0], "ported_shape": [0],
               "official_unique": [0], "ported_unique": [0]}
  },
  "tolerance": {"atol": 1e-5, "rtol": 1e-4,
                "grad_atol": 1e-4, "optimizer_step_atol": 1e-4,
                "mask_iou": 0.9999},
  "skipped": [{"check": "<str>", "reason": "<str>"}],
  "passed": true,
  "failures": ["..."]
}
</json_schema>

<output_format>
<script>完整脚本</script>
<cannot_cover>
每项：无法验证什么、为什么、应改用什么方式验证。至少逐项检查：
- 官方 dataset/transform 入口不可运行 → 用固定 tensor 直接调 transform；
- 官方 checkpoint 缺失 → 用同一随机 state_dict 做 L2 实现等价性对比；G2 可 pass，
  G6 标 unverified（除非官方代码+同一数据短程 parity）；
- DDP/SyncBN → 单卡无法验证，改用单卡等价 batch 或标 unverified；
- EMA → 单独比较 EMA 权重/更新公式；
- 全量训练 → 同一份数据（官方或项目 18/6）+同一初始化/state_dict（能共享时）短程 N≥20 steps loss/grad 对比；
  短程只证明训练路径一致，最终精度仍需全量训练；
- 非确定性算子 → 重复运行并记录方差；稳定超差仍判 fail，只有波动且无法收敛时标 skipped/unverified；
- 目标 GPU 不可得 → 标 unverified，不用 CPU 结果代替。
</cannot_cover>
</output_format>
