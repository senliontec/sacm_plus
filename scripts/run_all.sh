#!/bin/bash
# 服务器一键流程: 环境 → 数据 → 训练 → 评测 → 报告
#
# 用法:
#   bash scripts/run_all.sh                        # 全流程
#   bash scripts/run_all.sh --skip-setup --skip-data   # 只训练+评测
#   bash scripts/run_all.sh --skip-train --skip-eval   # 只准备环境+数据
#   PRESET=stage2 EPOCHS=30 bash scripts/run_all.sh    # 环境变量覆盖配置
#
# 前提:
#   - 本机有 conda;SAM checkpoint 在 checkpoints/sam_vit_l_0b3195.pth
#   - 原始数据集在 $RAW_DATA_ROOT(每数据集一个子目录;layout 由 organize_datasets.py 识别)
#   - wandb: 首次运行前执行 wandb login --host http://172.16.1.7:8080(失败不致命)

set -euo pipefail

# 默认后台运行(BACKGROUND=true):nohup 重入自身,SSH 断开也不中断;
# 前台调试用 BACKGROUND=false
if [ "${BACKGROUND:-true}" = "true" ] && [ -z "${_BG_REENTRY:-}" ]; then
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  LOGF="run_$(date +%Y%m%d_%H%M%S).log"
  nohup env _BG_REENTRY=1 bash "$SELF" "$@" > "$LOGF" 2>&1 &
  echo "已在后台启动 (pid $!),日志: $LOGF"
  echo "实时查看: tail -f $LOGF"
  exit 0
fi

# ---------------- 配置(仍可用环境变量覆盖,默认值 = 当前实验计划) ----------------
ENV_NAME="${ENV_NAME:-tsai}"
PRESET="${PRESET:-full}"
EPOCHS="${EPOCHS:-20}"                   # 试水轮数;确认收敛后 EPOCHS=50 跑消融矩阵
VAL_INTERVAL="${VAL_INTERVAL:-1}"        # 每 epoch 验证(引擎指标已节流至 50 张,~3-5 分钟/次)
SHOTS="${SHOTS:-3}"                      # 仅 FULL_DATA=false(3-shot 协议层)时生效
VAL_SHOTS="${VAL_SHOTS:-1}"
SEED="${SEED:-42}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-datasets}"   # 原始数据集目录(仓库内,与 up2server 上传布局一致)
SPLIT_ROOT="${SPLIT_ROOT:-data/dis5k_full}"
TRAIN_SOURCES="${TRAIN_SOURCES:-datasets/DIS5K_train}"   # 架构研究层:DIS5K 全量
TEST_DATASETS="${TEST_DATASETS:-datasets/DRIVE_test datasets/DIS5K_test datasets/ThinObject5K}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
USE_WANDB="${USE_WANDB:-true}"
TOPOLOGY_LOSS="${TOPOLOGY_LOSS:-topograph}"   # 默认 topograph:针对 β 碎片化瓶颈;none 关闭
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.002}"   # 0.002: topograph 原始量级~200,需对齐主损失 ~0.7(实测 0.1 时拓扑项占 97%)
TOPOLOGY_RESOLUTION="${TOPOLOGY_RESOLUTION:-256}"   # 引擎损失分辨率;训练太慢降 128
FULL_DATA="${FULL_DATA:-true}"           # 架构研究层默认全量;3-shot 协议层设 FULL_DATA=false
VAL_RATIO="${VAL_RATIO:-0.2}"
DDP="${DDP:-true}"                       # true = torchrun 8 卡 DDP(全量训练);false = 单卡
BATCH_SIZE="${BATCH_SIZE:-1}"            # 每卡 batch:3090 24GB 实测 2 会 OOM(ViT-L 1024²);
                                         # DDP 有效 batch = 卡数 × 此值

cd "$(dirname "$0")/.."   # 仓库根
ROOT="$(pwd)"

SKIP_SETUP=0; SKIP_DATA=0; SKIP_TRAIN=0; SKIP_EVAL=0; NO_KILL=0
for a in "$@"; do case "$a" in
  --skip-setup) SKIP_SETUP=1 ;;
  --skip-data)  SKIP_DATA=1 ;;
  --skip-train) SKIP_TRAIN=1 ;;
  --skip-eval)  SKIP_EVAL=1 ;;
  --no-kill)    NO_KILL=1 ;;
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) echo "未知参数: $a"; exit 1 ;;
esac; done

# 重跑前清理上一次的残留训练进程与 GPU(参考 fencing-algs;--no-kill 跳过)
if [ "$NO_KILL" -eq 0 ] && [ "$SKIP_TRAIN" -eq 0 ]; then
  bash scripts/kill_runs.sh
fi

step() { echo; echo "════════════════════════ $* ════════════════════════"; }
fail() { echo "❌ $*" >&2; exit 1; }

# ---------------- 1. 环境 ----------------
if [ "$SKIP_SETUP" -eq 0 ]; then
  step "1/5 环境: conda $ENV_NAME + 项目依赖"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10 || fail "conda create 失败"
  fi
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  python -c "import torch" 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -e ".[dev,topology]" || fail "pip install 失败"
  echo "✅ 环境就绪 ($(python --version) @ $(which python))"
else
  eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME" 2>/dev/null || true
fi

# ---------------- 2. 数据 ----------------
build_split() {
  [ -d "$RAW_DATA_ROOT" ] || fail "原始数据集目录不存在: $RAW_DATA_ROOT"
  python scripts/organize_datasets.py --src "$RAW_DATA_ROOT" --out datasets || true
  if [ "$FULL_DATA" = "true" ]; then
    python scripts/prepare_data.py --train_dirs $TRAIN_SOURCES --use_all \
        --val_ratio "$VAL_RATIO" --out_dir "$SPLIT_ROOT" --seed "$SEED" \
        || fail "prepare_data(全量) 失败"
  else
    python scripts/prepare_data.py --train_dirs $TRAIN_SOURCES \
        --shots "$SHOTS" --val_shots "$VAL_SHOTS" --out_dir "$SPLIT_ROOT" --seed "$SEED" \
        || fail "prepare_data 失败"
  fi
}

if [ -f "$SPLIT_ROOT/split_manifest.csv" ]; then
  echo "⚠️  $SPLIT_ROOT 已存在,跳过采样(如需重建请删除该目录)"
else
  step "2/5 数据: 整理原始数据集 + 构建训练划分"
  # shellcheck disable=SC2086
  build_split
  echo "✅ 数据就绪"
fi

# ---------------- 3. 训练 ----------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  step "3/5 训练: preset=$PRESET epochs=$EPOCHS ddp=$DDP"
  [ -f "$CKPT" ] || fail "checkpoint 不存在: $CKPT(先下载 sam_vit_l_0b3195.pth 放入 checkpoints/)"
  TRAIN_ARGS=(--preset "$PRESET" --data_root "$SPLIT_ROOT" \
      --checkpoint "$CKPT" --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --topology_loss "$TOPOLOGY_LOSS" --topology_loss_weight "$TOPOLOGY_LOSS_WEIGHT" \
      --topology_resolution "$TOPOLOGY_RESOLUTION" \
      --use_wandb "$USE_WANDB" --save_path "$RESULTS_ROOT/$PRESET/best_model.pth")
  if [ "$DDP" = "true" ]; then
    NG="$(nvidia-smi -L 2>/dev/null | wc -l)"
    [ "$NG" -gt 0 ] || fail "nvidia-smi 不可用,无法 DDP"
    # 参考 fencing-algs:显式 localhost + 固定端口(避免 torchrun 用主机名/错误网卡导致 NCCL 秒崩);
    # NCCL_P2P_DISABLE 默认 1(3090 多卡节点常见必须;要恢复直连设 NCCL_P2P_DISABLE=0)
    mkdir -p "$RESULTS_ROOT/torchrun_logs"
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
    torchrun --nproc_per_node="$NG" --log-dir "$RESULTS_ROOT/torchrun_logs" \
      src/train/trainer.py "${TRAIN_ARGS[@]}" \
      || fail "训练失败(DDP, $NG 卡)——逐 rank 日志在 $RESULTS_ROOT/torchrun_logs/"
  else
    python src/train/trainer.py "${TRAIN_ARGS[@]}" || fail "训练失败"
  fi
  echo "✅ 训练完成"
else
  [ -f "$RESULTS_ROOT/$PRESET/best_model.pth" ] || fail "--skip-train 但 checkpoint 不存在"
fi

# ---------------- 4. 评测 ----------------
if [ "$SKIP_EVAL" -eq 0 ]; then
  step "4/5 评测: $TEST_DATASETS"
  for ds in $TEST_DATASETS; do
    [ -d "$ds" ] || { echo "⚠️  跳过(不存在): $ds"; continue; }
    name="$(basename "$ds")"
    echo "── 评测 $name"
    python src/eval/evaluate.py --preset "$PRESET" --data_root "$ds" \
        --trained_weights "$RESULTS_ROOT/$PRESET/best_model.pth" \
        --output_dir "$RESULTS_ROOT/$PRESET/$name" --selection iou \
        --use_wandb "$USE_WANDB" \
        || fail "$name 评测失败"
  done
  echo "✅ 评测完成"
fi

# ---------------- 5. 报告 ----------------
step "5/5 报告: 汇总 + AI 诊断"
python scripts/aggregate_results.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables" || true
python scripts/metrics_report.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables/report.md" || true

echo
echo "✅ 全流程完成"
echo "   结果:   $RESULTS_ROOT/<preset>/<dataset>/(metrics.csv, results_summary.txt)"
echo "   汇总:   $RESULTS_ROOT/tables/(all_metrics.csv, table_*.tex)"
echo "   诊断:   $RESULTS_ROOT/tables/report.md"
