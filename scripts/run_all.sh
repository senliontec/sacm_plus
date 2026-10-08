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

# ---------------- 配置(全部可用环境变量覆盖) ----------------
ENV_NAME="${ENV_NAME:-tsai}"
PRESET="${PRESET:-full}"
EPOCHS="${EPOCHS:-50}"
SHOTS="${SHOTS:-3}"
VAL_SHOTS="${VAL_SHOTS:-1}"
SEED="${SEED:-42}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-datasets}"   # 原始数据集目录(仓库内,与 up2server 上传布局一致;如需仓库外数据用环境变量覆盖)
SPLIT_ROOT="${SPLIT_ROOT:-data/sacm_${SHOTS}shot}"
TRAIN_SOURCES="${TRAIN_SOURCES:-datasets/DIS5K_train datasets/DRIVE_train datasets/ThinObject5K}"
TEST_DATASETS="${TEST_DATASETS:-datasets/DRIVE_test datasets/DIS5K_test datasets/ThinObject5K}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
USE_WANDB="${USE_WANDB:-true}"

cd "$(dirname "$0")/.."   # 仓库根
ROOT="$(pwd)"

SKIP_SETUP=0; SKIP_DATA=0; SKIP_TRAIN=0; SKIP_EVAL=0
for a in "$@"; do case "$a" in
  --skip-setup) SKIP_SETUP=1 ;;
  --skip-data)  SKIP_DATA=1 ;;
  --skip-train) SKIP_TRAIN=1 ;;
  --skip-eval)  SKIP_EVAL=1 ;;
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) echo "未知参数: $a"; exit 1 ;;
esac; done

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
if [ "$SKIP_DATA" -eq 0 ]; then
  step "2/5 数据: 整理原始数据集 + 构建 ${SHOTS}-shot 训练划分"
  [ -d "$RAW_DATA_ROOT" ] || fail "原始数据集目录不存在: $RAW_DATA_ROOT"
  python scripts/organize_datasets.py --src "$RAW_DATA_ROOT" --out datasets || true
  if [ -f "$SPLIT_ROOT/split_manifest.csv" ]; then
    echo "⚠️  $SPLIT_ROOT 已存在,跳过采样(如需重建请删除该目录)"
  else
    # shellcheck disable=SC2086
    python scripts/prepare_data.py --train_dirs $TRAIN_SOURCES \
        --shots "$SHOTS" --val_shots "$VAL_SHOTS" --out_dir "$SPLIT_ROOT" --seed "$SEED" \
        || fail "prepare_data 失败"
  fi
  echo "✅ 数据就绪"
else
  [ -f "$SPLIT_ROOT/split_manifest.csv" ] || fail "--skip-data 但 $SPLIT_ROOT 不存在"
fi

# ---------------- 3. 训练 ----------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  step "3/5 训练: preset=$PRESET epochs=$EPOCHS"
  [ -f "$CKPT" ] || fail "checkpoint 不存在: $CKPT(先下载 sam_vit_l_0b3195.pth 放入 checkpoints/)"
  python src/train/trainer.py --preset "$PRESET" --data_root "$SPLIT_ROOT" \
      --checkpoint "$CKPT" --epochs "$EPOCHS" --seed "$SEED" \
      --use_wandb "$USE_WANDB" --save_path "$RESULTS_ROOT/$PRESET/best_model.pth" \
      || fail "训练失败"
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
