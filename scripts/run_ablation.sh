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

# ---------------- 配置(仍可用环境变量覆盖,默认值 = 当前实验计划) ----------------
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-../datasets}"   # canonical raw library: project-root datasets/, sibling of sacm_plus/
SPLIT_ROOT="${SPLIT_ROOT:-data/sacm_3shot}"
TRAIN_SOURCES="${TRAIN_SOURCES:-datasets/seen/DRIVE datasets/seen/CHASEDB1 datasets/seen/DCA1 datasets/seen/CREMI datasets/seen/CrackTree200 CORN:datasets/seen/CORN/train}"   # SACM protocol domains (6 x 3-shot = 18)
VAL_SOURCES="${VAL_SOURCES:-CORN:datasets/seen/CORN/val}"   # official val split (CORN batch 9)
TEST_SOURCES="${TEST_SOURCES:-CORN:datasets/seen/CORN/test}"     # official test split (CORN batches 10-11)
DRIVE_TEST_GT="${DRIVE_TEST_GT:-}"       # 可选:已核实的 DRIVE test 血管 GT 目录;官方 test/mask 是 FOV
TEST_DATASETS="${TEST_DATASETS:-datasets/unseen/FIVES datasets/unseen/STARE datasets/unseen/XCAD datasets/unseen/DeepCrack datasets/unseen/HRF datasets/unseen/LES-AV datasets/unseen/CHCrack5k datasets/unseen/OCTA-500 datasets/novel/ROAD datasets/novel/LEAF datasets/novel/TYRE datasets/novel/WIRE datasets/novel/DeepGlobe datasets/novel/TopoMortar}"
# DRIVE_test 默认不加入:官方 test/mask 是 FOV,不是血管 GT。
# 用 seen/DRIVE 的 3/17 留出划分,或提供 DRIVE_TEST_GT 后再显式加入 TEST_DATASETS。
PRESETS="${PRESETS:-sacm stage1 stage2 full full_wta full_allhead no_geo_i no_geo_e no_c2f no_fusion_v2 no_multi_depth no_cl no_ds no_iou geo_e_shallow geo_e_deep}"
EPOCHS="${EPOCHS:-50}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
TTA="${TTA:-}"
DDP="${DDP:-true}"   # true = 每个预设用全部 GPU 做 DDP 顺序跑(全量数据推荐);
                    # false = run_experiments 任务级并行(每卡一个 job,3-shot 协议用)
BATCH_SIZE="${BATCH_SIZE:-1}"   # 每卡 batch:3090 24GB 实测 2 会 OOM;DDP 有效 batch = 卡数 × 此值
SELECTION="${SELECTION:-auto}"           # auto: iou only when iou_loss_weight>0, else gating
PRED_THRESHOLD="${PRED_THRESHOLD:-0.0}"  # logits; SACM fidelity uses 0.5
GROUP_REGEX="${GROUP_REGEX:-}"           # space-separated NAME:REGEX; patient/volume grouping
CLEAN_SPLIT="${CLEAN_SPLIT:-true}"       # clear train/val before writing the split
MAKE_TEST="${MAKE_TEST:-true}"           # also write group-disjoint test/ for Seen

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
if [ -n "$DRIVE_TEST_GT" ]; then
  python scripts/organize_datasets.py --src "$RAW_DATA_ROOT" --out datasets \
      --drive_test_gt "$DRIVE_TEST_GT" \
      || { echo "❌ organize_datasets 失败"; exit 1; }
else
  python scripts/organize_datasets.py --src "$RAW_DATA_ROOT" --out datasets \
      || { echo "❌ organize_datasets 失败"; exit 1; }
fi
python scripts/group_datasets.py --root datasets \
    || { echo "❌ group_datasets(seen/unseen/novel) 失败"; exit 1; }
# Author-provided official SACM datasets: rebuild prepared pools.
python scripts/datasets/prepare_CORN.py --overwrite \
    || { echo "❌ prepare_CORN 失败"; exit 1; }
python scripts/datasets/prepare_leaf.py --overwrite \
    || { echo "❌ prepare_leaf 失败"; exit 1; }
python scripts/datasets/prepare_OCTA-500.py --overwrite \
    --src "$RAW_DATA_ROOT/OCTA-500" \
    || { echo "❌ prepare_OCTA-500 失败"; exit 1; }
python scripts/datasets/prepare_TopoMortar.py --overwrite \
    --src "$RAW_DATA_ROOT/TopoMortar" \
    || { echo "❌ prepare_TopoMortar 失败"; exit 1; }
python scripts/datasets/prepare_DeepGlobe.py --overwrite \
    --src "$RAW_DATA_ROOT/DeepGlobe" \
    || { echo "❌ prepare_DeepGlobe 失败"; exit 1; }
if [ -f "$SPLIT_ROOT/split_manifest.csv" ]; then
  echo "⚠️  $SPLIT_ROOT 已存在,跳过采样"
else
  read -ra GROUP_SPECS <<< "$GROUP_REGEX"
  GROUP_ARGS=()
  for spec in "${GROUP_SPECS[@]}"; do
    [ -n "$spec" ] && GROUP_ARGS+=(--group_regex "$spec")
  done
  CLEAN_ARG=()
  [ "$CLEAN_SPLIT" = "true" ] && CLEAN_ARG=(--clean)
  TEST_ARG=()
  [ "$MAKE_TEST" = "true" ] && TEST_ARG=(--make_test)
  VAL_ARGS=()
  [ -n "$VAL_SOURCES" ] && VAL_ARGS=(--val_dirs $VAL_SOURCES)
  TEST_SOURCES_ARGS=()
  [ -n "$TEST_SOURCES" ] && TEST_SOURCES_ARGS=(--test_dirs $TEST_SOURCES)
  # shellcheck disable=SC2086
  python scripts/prepare_data.py --train_dirs $TRAIN_SOURCES \
      "${VAL_ARGS[@]}" "${TEST_SOURCES_ARGS[@]}" \
      --shots 3 --val_shots 1 --out_dir "$SPLIT_ROOT" --seed "$SEED" \
      "${GROUP_ARGS[@]}" "${CLEAN_ARG[@]}" "${TEST_ARG[@]}"
fi
if [ "$MAKE_TEST" = "true" ]; then
  python scripts/build_protocol_manifest.py \
      --split-root "$SPLIT_ROOT" --datasets-root datasets \
      --seed "$SEED" \
      || { echo "❌ protocol manifest 校验/生成失败"; exit 1; }
fi
if [ "$MAKE_TEST" = "true" ]; then
  for td in "$SPLIT_ROOT"/test/*; do
    [ -d "$td/masks" ] && TEST_DATASETS="$TEST_DATASETS $td"
  done
fi

echo "══════════ 2/4 多卡消融调度 ══════════"
if [ "$DDP" = "true" ]; then
  # 每个预设独占全部 GPU(DDP)顺序跑:全量数据下单个训练就吃满 8 卡
  NG="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [ "$NG" -gt 0 ] || { echo "❌ nvidia-smi 不可用,无法 DDP"; exit 1; }
  for p in $PRESETS; do
    echo "── DDP 训练 preset=$p ($NG 卡)"
    if [ "$SKIP_TRAIN" -eq 0 ]; then
      mkdir -p "$RESULTS_ROOT/torchrun_logs"
      MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
      NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
      torchrun --nproc_per_node="$NG" --log-dir "$RESULTS_ROOT/torchrun_logs" \
          src/train/trainer.py \
          --preset "$p" --data_root "$SPLIT_ROOT" --checkpoint "$CKPT" \
          --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
          --batch_size "$BATCH_SIZE" \
          --val_selection "$SELECTION" --pred_threshold "$PRED_THRESHOLD" \
          --use_wandb true --save_path "$RESULTS_ROOT/$p/best_model.pth"
    fi
    TTA_ARG=()
    [ -n "$TTA" ] && TTA_ARG=(--tta)
    for ds in $TEST_DATASETS; do
      [ -d "$ds" ] || { echo "❌ 测试集不存在: $ds"; exit 1; }
      if [ ! -d "$ds/masks" ]; then
        echo "❌ 测试集缺少 masks/: $ds (DRIVE 官方 test/mask 是 FOV,不能当 GT)"
        exit 1
      fi
      echo "── 评测 $p @ $(basename "$ds")"
      python src/eval/evaluate.py --preset "$p" --data_root "$ds" \
          --trained_weights "$RESULTS_ROOT/$p/best_model.pth" \
          --output_dir "$RESULTS_ROOT/$p/$(basename "$ds")" \
          --selection "$SELECTION" --pred_threshold "$PRED_THRESHOLD" \
          "${TTA_ARG[@]}" --use_wandb true
    done
  done
else
  TEST_SPECS=""
  for ds in $TEST_DATASETS; do
    [ -d "$ds" ] || { echo "❌ 测试集不存在: $ds"; exit 1; }
    if [ ! -d "$ds/masks" ]; then
      echo "❌ 测试集缺少 masks/: $ds (DRIVE 官方 test/mask 是 FOV,不能当 GT)"
      exit 1
    fi
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
      --selection "$SELECTION" --pred_threshold "$PRED_THRESHOLD" \
      $EXTRA
fi

echo "══════════ 3/4 汇总: CSV + LaTeX ══════════"
python scripts/aggregate_results.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables" --strict-protocol

echo "══════════ 4/4 AI 诊断: 组件贡献排名 ══════════"
python scripts/metrics_report.py --root "$RESULTS_ROOT" --out "$RESULTS_ROOT/tables/report.md"

echo
echo "✅ 消融矩阵完成"
echo "   结果:   $RESULTS_ROOT/<preset>/<dataset>/"
echo "   汇总:   $RESULTS_ROOT/tables/(all_metrics.csv, table_dice/table_iou/table_cldice/table_hd95.tex)"
echo "   诊断:   $RESULTS_ROOT/tables/report.md  ← 先看这个:每个组件的贡献排名"
