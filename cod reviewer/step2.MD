<task>
写一个 Python 脚本，同时加载官方实现与移植实现，自动对比并输出 JSON 报告。不要输出解释性文字，只输出脚本。
</task>

<requirements>
1. 固定 torch/numpy/random 种子；固定输入张量（shape 与生成方式写死在脚本里）。
2. 依次检查并 assert：
   - state_dict：键集合相等、每个键 shape 相等、参数量相等。必须遍历 named_parameters 与 named_buffers（BN 的 running_mean/var/num_batches_tracked 属于 buffer）。
   - eval 模式前向：输出 shape/dtype、mean/std/min/max、两实现逐元素最大绝对差。
   - 关键中间层：用 forward hook 抓取 {{中间层名称列表}}，同样对比统计量。
   - train 模式：同一输入与标签下 loss 标量对比、loss 各项分量对比、反传后指定参数梯度范数对比。
3. 容差作为脚本顶部常量，默认 atol=1e-5（float32 前向）/ 1e-3（含自定义算子时）。
4. 官方与移植实现的加载必须隔离包名冲突：
   - 首选：subprocess 分别在官方仓库根目录和项目根目录下执行探针，各自 import 本地包，两侧通过临时 .pt/.npz 交换张量，主进程做数值对比。
   - 次选：sys.path 手工插入，并在每次 import 后立即从 sys.modules 清除同名包。
   - 禁止用 importlib 单文件加载含包内相对导入的模块，除非在 <cannot_cover> 中说明该风险。
5. BN 在 eval 前显式切换并打印 track_running_stats 状态。
6. 任一项超差：打印该 key、两侧数值、差值，继续跑完所有项，最后以非零码退出。
7. 输出 JSON 到 stdout，字段名必须严格遵循下方 <json_schema>，便于批量汇总 20 个 baseline。
</requirements>

<json_schema>
{
  "baseline": "<str>",
  "state_dict": {"keys_match": bool, "shape_mismatch": [{"key": str, "official": [int], "ported": [int]}], "param_count": {"official": int, "ported": int}},
  "forward_eval": {"shape": [int], "dtype": str, "official_stats": {"mean": float, "std": float, "min": float, "max": float}, "max_abs_diff": float},
  "intermediate": [{"name": str, "max_abs_diff": float}],
  "train_loss": {"official": float, "ported": float, "components": {}},
  "grad_norm": [{"param": str, "official": float, "ported": float}],
  "tolerance": {"atol": float, "rtol": float},
  "passed": bool,
  "failures": [str]
}
</json_schema>

<official_impl>
  <file path="..." role="model_def">...全文...</file>
  <file path="..." role="dependency">...全文...</file>
</official_impl>

<ported_impl>
  <file path="..." role="model_def">...全文...</file>
</ported_impl>

<env>torch {{}} / cuda {{}}</env>

<output_format>
先输出 <script> 完整脚本，再输出 <cannot_cover>：列出脚本无法验证的项（评测协议、数据增强、权重加载路径等），每项给出应改用什么方式验证。
</output_format>