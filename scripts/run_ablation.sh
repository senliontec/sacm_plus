#!/bin/bash
# 一键消融矩阵: 准备 3-shot 数据 → 8 卡并行跑全部 14 个预设 × 测试集 → 汇总 + AI 诊断
#
# 用法:
#   bash scripts/run_ablation.sh                    # 全矩阵,全 GPU
#   GPUS="0 1 2 3" bash scripts/run_ablation.sh     # 只用前 4 卡
#   PRESETS="sacm full" EPOCHS=30 bash scripts/run_ablation.sh   # 子集快跑
#   bash scripts/run_ablation.sh --skip-train       # 已有权重,只补评测
#
# 输出布局: results/<preset>/<dataset>/(best_model.pth, metrics.csv, ...)
#           results/tables/(all_metrics.csv, table_*.tex, report.md)

set -euo pipefail

# ---------------- 配置(可用环境变量覆盖) ----------------
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-datasets}"   # 仓库内(与 up2server 上传布局一致)
SPLIT_ROOT="${SPLIT_ROOT:-data/sacm_3shot}"
TRAIN_SOURCES="${TRAIN_SOURCES:-datasets/DIS5K_train datasets/DRIVE_train datasets/ThinObject5K}"
TEST_DATASETS="${TEST_DATASETS:-datasets/DRIVE_test datasets/DIS5K_test datasets/ThinObject5K}"
PRESETS="${PRESETS:-sacm stage1 stage2 full no_geo_i no_geo_e no_c2f no_fusion_v2 no_multi_depth no_cl no_ds no_iou geo_e_shallow geo_e_deep}"
EPOCHS="${EPOCHS:-50}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"   # 每 epoch 显示全指标;论文协议口径 10,最终投稿跑 10
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
TTA="${TTA:-}"

cd "$(dirname "$0")/.."

SKIP_TRAIN=0
NO_KILL=0
for a in "$@"; do case "$a" in
  --skip-train) SKIP_TRAIN=1 ;;
  --no-kill)    NO_KILL=1 ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "未知参数: $a"; exit 1 ;;
esac; done

# 重跑前清理上一次的残留进程与 GPU(参考 fencing-algs;--no-kill 跳过)
if [ "$NO_KILL" -eq 0 ]; then
  bash scripts/kill_runs.sh
fi

echo "══════════ 1/4 数据: 组织 + 3-shot 划分 ══════════"
[ -d "$RAW_DATA_ROOT" ] || { echo "❌ 原始数据集目录不存在: $RAW_DATA_ROOT"; exit 1; }
python scripts/organize_datasets.py --src "$RAW_DATA_ROOT" --out datasets || true
if [ -f "$SPLIT_ROOT/split_manifest.csv" ]; then
  echo "⚠️  $SPLIT_ROOT 已存在,跳过采样"
else
  # shellcheck disable=SC2086
  python scripts/prepare_data.py --train_dirs $TRAIN_SOURCES \
      --shots 3 --val_shots 1 --out_dir "$SPLIT_ROOT" --seed "$SEED"
fi

echo "══════════ 2/4 多卡消融调度 ══════════"
TEST_SPECS=""
for ds in $TEST_DATASETS; do
  TEST_SPECS="$TEST_SPECS $(basename "$ds"):$ds"
done

EXTRA=""
[ "$SKIP_TRAIN" -eq 1 ] && EXTRA="--skip_train"
[ -n "$TTA" ] && EXTRA="$EXTRA --tta"

# shellcheck disable=SC2086
python scripts/run_experiments.py \
    --gpus $GPUS \
    --checkpoint "$CKPT" \
    --train_root "$SPLIT_ROOT" \
    --test_datasets $TEST_SPECS \
    --presets $PRESETS \
    --output_root "$RESULTS_ROOT" \
    --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
    $EXTRA

echo "══════════ 3/4 汇总: CSV + LaTeX ══════════"
python scripts/aggregate_results.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables"

echo "══════════ 4/4 AI 诊断: 组件贡献排名 ══════════"
python scripts/metrics_report.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables/report.md"

echo
echo "✅ 消融矩阵完成"
echo "   结果:   $RESULTS_ROOT/<preset>/<dataset>/"
echo "   汇总:   $RESULTS_ROOT/tables/(all_metrics.csv, table_dice/table_iou/table_cldice/table_hd95.tex)"
echo "   诊断:   $RESULTS_ROOT/tables/report.md  ← 先看这个:每个组件的贡献排名"
