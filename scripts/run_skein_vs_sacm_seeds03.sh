#!/usr/bin/env bash
# One command: train + evaluate Skein and official SACM on seeds 0-3, then
# build the 20-dataset paper Table 2.
#
#   bash scripts/run_skein_vs_sacm_seeds03.sh
#
# Skein uses GPUs 0-3 and official SACM uses GPUs 4-7, so the two runs go in
# parallel.  Both child scripts train every seed, evaluate all 20 protocol
# test datasets with Dice/IoU/clDice/HD95, and write per-dataset records.
# The Table 2 builder runs only after both children succeed.
#
# Env overrides are inherited by both children, e.g.:
#   RESUME=1               skip completed jobs
#   SKIP_TRAIN=1           evaluate existing checkpoints only
#   SEEDS="0 1"            run a subset
#   SKEIN_GPUS=0,1,2,3     GPU set for Skein
#   SACM_GPUS=4,5,6,7      GPU set for official SACM
#
# Outputs:
#   results/reports/paper_table2/table2.tex
#   results/reports/paper_table2/table2_with_std.tex
#   results/reports/paper_table2/table2_means.md
#   results/logs/table2_<timestamp>/{skein,sacm}.log
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

LOG_DIR="${LOG_DIR:-results/logs/table2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKEIN_GPUS="${SKEIN_GPUS:-0,1,2,3}"
SACM_GPUS="${SACM_GPUS:-4,5,6,7}"

echo "============================================================"
echo " Skein vs official SACM, seeds ${SEEDS:-0 1 2 3}"
echo " Skein GPUs:  $SKEIN_GPUS"
echo " SACM  GPUs:  $SACM_GPUS"
echo " logs:        $LOG_DIR"
echo "============================================================"

GPU_LIST="$SKEIN_GPUS" bash scripts/run_skein_seeds03_gpu0123.sh \
  > >(tee "$LOG_DIR/skein.log") 2>&1 &
skein_pid=$!

GPU_LIST="$SACM_GPUS" bash scripts/run_sacm_official_seeds03_gpu4567.sh \
  > >(tee "$LOG_DIR/sacm.log") 2>&1 &
sacm_pid=$!

status_skein=0
status_sacm=0
wait "$skein_pid" || status_skein=$?
wait "$sacm_pid" || status_sacm=$?

echo "============================================================"
echo " Skein exit: $status_skein"
echo " SACM  exit: $status_sacm"
echo "============================================================"
if [ "$status_skein" -ne 0 ] || [ "$status_sacm" -ne 0 ]; then
  echo "ERROR: at least one run failed; not building the paper table." >&2
  echo "Inspect $LOG_DIR/skein.log and $LOG_DIR/sacm.log." >&2
  exit 1
fi

if [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=1: no evaluation records; not building the paper table." >&2
  exit 0
fi

bash scripts/build_paper_table.sh

echo "============================================================"
echo " Done. Paper table: results/reports/paper_table2/table2.tex"
echo " Preview:         results/reports/paper_table2/table2_means.md"
echo " Combined logs:   $LOG_DIR"
echo "============================================================"
