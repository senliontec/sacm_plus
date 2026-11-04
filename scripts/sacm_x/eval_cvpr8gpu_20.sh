#!/usr/bin/env bash
# One-click 20-dataset evaluation of the previous 8-GPU screen.
#
# The eight runs are evaluated concurrently, one run per GPU:
#
#   GPU 0: full_fixed2_aux          seed 0
#   GPU 1: full_fixed2_evp          seed 0
#   GPU 2: full_fixed2_convlora     seed 0
#   GPU 3: full_fixed2_lora_evp     seed 0
#   GPU 4: full_fixed2_peft_lora_evp seed 0
#   GPU 5: full_fixed2_aux          seed 1
#   GPU 6: full_fixed2_evp          seed 1
#   GPU 7: full_fixed2_lora_evp     seed 1
#
# These are the encoder-side comparison runs, not the newer
# SCNP/ResMoE/hybrid screen.  Keep their results in a separate table.
#
# Usage:
#   bash scripts/sacm_x/eval_cvpr8gpu_20.sh
#   SCREEN_ROOT=results/runs/cvpr8gpu_screen_old \
#     bash scripts/sacm_x/eval_cvpr8gpu_20.sh
#   FORCE=1 bash scripts/sacm_x/eval_cvpr8gpu_20.sh
#
# Defaults match the old screen geometry: resize evaluation, no empty guard,
# fixed Stage-2 head 2.  Do not change these when comparing against the old
# training/validation numbers.
#
# Each run is searched under:
#   $SCREEN_ROOT/sacm_x/<preset>/seed_<seed>
#   $SCREEN_ROOT/<preset>/seed_<seed>
#   results/runs/sacm_x/<preset>/seed_<seed>
# Set SCREEN_ROOT if the old batch lives somewhere else.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PY=python
  fi
fi
export PYTHON="$PY"

SCREEN_ROOT="${SCREEN_ROOT:-results/runs/cvpr8gpu_screen}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Old-screen evaluation point.  These must remain fixed for the comparison.
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
EMPTY_GUARD="${EMPTY_GUARD:-false}"
FIXED_HEAD="${FIXED_HEAD:-2}"
SELECTION="${SELECTION:-}"
USE_TTA="${USE_TTA:-false}"
USE_WANDB="${USE_WANDB:-false}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REUSE_EVAL_MODEL="${REUSE_EVAL_MODEL:-1}"
PAPER_METRICS_ONLY="${PAPER_METRICS_ONLY:-0}"
COMPONENT_METRICS="${COMPONENT_METRICS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"

# gpu preset seed
JOB_SPECS=(
  "0 full_fixed2_aux 0"
  "1 full_fixed2_evp 0"
  "2 full_fixed2_convlora 0"
  "3 full_fixed2_lora_evp 0"
  "4 full_fixed2_peft_lora_evp 0"
  "5 full_fixed2_aux 1"
  "6 full_fixed2_evp 1"
  "7 full_fixed2_lora_evp 1"
)

resolve_ckpt() {
  local root="$1" preset="$2"
  local cand
  for cand in \
    "$root/best_model.pth" \
    "$root/best_model" \
    "$root/$preset/best_model.pth" \
    "$root/train/best_model.pth" \
    "$root/$preset/best_model"; do
    if [ -f "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

resolve_run_root() {
  local preset="$1" seed="$2"
  local base layout cand ckpt
  for base in "$SCREEN_ROOT" "results/runs/cvpr8gpu_screen" "results/runs"; do
    [ -n "$base" ] || continue
    for layout in \
      "sacm_x/$preset/seed_$seed" \
      "sacm_x/$preset/seed$seed" \
      "$preset/seed_$seed" \
      "$preset/seed$seed"; do
      cand="$base/$layout"
      [ -d "$cand" ] || continue
      if ckpt="$(resolve_ckpt "$cand" "$preset")"; then
        printf '%s\t%s\n' "$cand" "$ckpt"
        return 0
      fi
    done
  done
  return 1
}

JOB_GPU=()
JOB_PRESET=()
JOB_SEED=()
JOB_RUN=()
JOB_FINAL=()
JOB_CKPT=()
JOB_ACTION=()
JOB_LOG=()
JOB_PID=()
MISSING=()

for spec in "${JOB_SPECS[@]}"; do
  read -r gpu preset seed <<< "$spec"
  idx="${#JOB_GPU[@]}"
  JOB_GPU+=("$gpu")
  JOB_PRESET+=("$preset")
  JOB_SEED+=("$seed")
  JOB_ACTION+=("run")
  JOB_LOG+=("")
  JOB_PID+=("")
  JOB_FINAL+=("$SCREEN_ROOT/sacm_x/$preset/seed_$seed")

  if resolved="$(resolve_run_root "$preset" "$seed")"; then
    run_root="${resolved%%$'\t'*}"
    ckpt="${resolved#*$'\t'}"
    JOB_RUN+=("$run_root")
    JOB_CKPT+=("$ckpt")
  else
    JOB_RUN+=("")
    JOB_CKPT+=("")
    MISSING+=("$preset seed $seed: no checkpoint under $SCREEN_ROOT or results/runs")
    continue
  fi

  if [ "$FORCE" != "1" ] && [ "$SKIP_EXISTING" = "1" ] \
      && [ -f "$run_root/eval/tables/macro_metrics.csv" ]; then
    JOB_ACTION[$idx]="skip"
  fi
done

# Required data/protocol and model checkpoint.
for f in "$SAM_CKPT" \
         data/sacm_3shot_seed0/protocol_manifest.csv \
         data/sacm_3shot_seed1/protocol_manifest.csv \
         scripts/sacm_x/eval_sacm_x_20.sh; do
  [ -f "$f" ] || MISSING+=("$f")
done

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "ERROR: missing prerequisites for the old-screen evaluation:" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  echo >&2
  echo "If the old batch is in another directory, set SCREEN_ROOT explicitly:" >&2
  echo "  SCREEN_ROOT=results/runs/cvpr8gpu_screen_old \\" >&2
  echo "    bash scripts/sacm_x/eval_cvpr8gpu_20.sh" >&2
  exit 2
fi

# Check only GPUs that will actually launch an evaluation.
RUN_COUNT=0
for action in "${JOB_ACTION[@]}"; do
  [ "$action" = "run" ] && RUN_COUNT=$((RUN_COUNT + 1))
done
if [ "$DRY_RUN" = "0" ] && [ "$RUN_COUNT" -gt 0 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found" >&2
    exit 2
  fi
  visible="$(nvidia-smi -L | wc -l)"
  for i in "${!JOB_GPU[@]}"; do
    [ "${JOB_ACTION[$i]}" = "run" ] || continue
    gpu="${JOB_GPU[$i]}"
    if [ "$visible" -lt "$((gpu + 1))" ]; then
      echo "ERROR: GPU $gpu is not visible (found $visible GPUs)" >&2
      exit 2
    fi
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 2000 ]; then
      echo "ERROR: GPU $gpu already uses ${used} MiB; free it first." >&2
      exit 2
    fi
  done
fi

echo "============================================================"
echo " Old 8-GPU screen: 20-dataset evaluation"
echo " screen root: $SCREEN_ROOT"
echo " python:      $PY"
echo " geometry:    eval=$EVAL_INPUT_MODE empty_guard=$EMPTY_GUARD"
echo " selection:   fixed head $FIXED_HEAD"
echo " workers:     $NUM_WORKERS"
echo " reuse model: $REUSE_EVAL_MODEL"
echo " skip done:   $SKIP_EXISTING (FORCE=$FORCE)"
echo "============================================================"
printf '%-3s %-26s %-4s %-4s %s\n' GPU PRESET SEED ACT RUN_ROOT
for i in "${!JOB_GPU[@]}"; do
  printf '%-3s %-26s %-4s %-4s %s\n' \
    "${JOB_GPU[$i]}" "${JOB_PRESET[$i]}" "${JOB_SEED[$i]}" \
    "${JOB_ACTION[$i]}" "${JOB_RUN[$i]}"
done
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: preflight only; no evaluation launched."
  exit 0
fi

for i in "${!JOB_GPU[@]}"; do
  [ "${JOB_ACTION[$i]}" = "run" ] || continue
  gpu="${JOB_GPU[$i]}"
  preset="${JOB_PRESET[$i]}"
  seed="${JOB_SEED[$i]}"
  run_root="${JOB_RUN[$i]}"
  log="$LOG_DIR/cvpr8gpu_eval20_${preset}_seed${seed}_${STAMP}.log"
  JOB_LOG[$i]="$log"
  echo "LAUNCH GPU $gpu -> $preset / seed $seed"
  GPU="$gpu" PRESET="$preset" SEED="$seed" \
  TRAIN_ROOT="data/sacm_3shot_seed${seed}" \
  TRAIN_OUT="$run_root" OUTPUT_ROOT="$run_root/eval" CKPT="${JOB_CKPT[$i]}" \
  CANONICAL_ROOT="$SCREEN_ROOT" SAM_CKPT="$SAM_CKPT" \
  EVAL_INPUT_MODE="$EVAL_INPUT_MODE" EMPTY_GUARD="$EMPTY_GUARD" \
  USE_TTA="$USE_TTA" FIXED_HEAD="$FIXED_HEAD" SELECTION= \
  USE_WANDB="$USE_WANDB" NUM_WORKERS="$NUM_WORKERS" \
  REUSE_EVAL_MODEL="$REUSE_EVAL_MODEL" \
  PAPER_METRICS_ONLY="$PAPER_METRICS_ONLY" \
  COMPONENT_METRICS="$COMPONENT_METRICS" \
  SKIP_EVAL=false \
    bash scripts/sacm_x/eval_sacm_x_20.sh > "$log" 2>&1 &
  JOB_PID[$i]="$!"
done

RC=0
for i in "${!JOB_GPU[@]}"; do
  if [ "${JOB_ACTION[$i]}" = "skip" ]; then
    echo "SKIP  ${JOB_PRESET[$i]}/seed${JOB_SEED[$i]}  (existing tables)"
    continue
  fi
  if [ -z "${JOB_PID[$i]}" ]; then
    continue
  fi
  if wait "${JOB_PID[$i]}"; then
    echo "DONE  ${JOB_PRESET[$i]}/seed${JOB_SEED[$i]}  (${JOB_LOG[$i]})"
  else
    echo "FAIL  ${JOB_PRESET[$i]}/seed${JOB_SEED[$i]}  (${JOB_LOG[$i]})"
    tail -n 30 "${JOB_LOG[$i]}" || true
    RC=1
  fi
done

# Pivot the per-run macro CSVs into one wide table for quick inspection.
SUMMARY_PATHS="$LOG_DIR/cvpr8gpu_eval20_${STAMP}_paths.tsv"
: > "$SUMMARY_PATHS"
for i in "${!JOB_GPU[@]}"; do
  [ -n "${JOB_RUN[$i]}" ] || continue
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${JOB_PRESET[$i]}" "${JOB_SEED[$i]}" "${JOB_GPU[$i]}" \
    "${JOB_RUN[$i]}" "${JOB_FINAL[$i]}" >> "$SUMMARY_PATHS"
done

SUMMARY_CSV="$SCREEN_ROOT/eval20_summary.csv"
"$PY" - "$SUMMARY_PATHS" "$SUMMARY_CSV" <<'PY'
import csv
import pathlib
import sys

paths = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
rows = []
for line in paths.read_text().splitlines():
    if not line.strip():
        continue
    preset, seed, gpu, run_root, final_root = line.split('\t')
    macro = None
    for root in (pathlib.Path(run_root), pathlib.Path(final_root)):
        candidate = root / 'eval' / 'tables' / 'macro_metrics.csv'
        if candidate.is_file():
            macro = candidate
            break
    if macro is None:
        continue
    by_role = {}
    try:
        with macro.open(newline='') as handle:
            for row in csv.DictReader(handle):
                variant = row.get('variant') or row.get('preset') or ''
                if variant and variant != preset:
                    continue
                role = row.get('role') or 'All'
                metric = row.get('metric') or ''
                by_role[(role, metric)] = row.get('mean', '')
    except OSError:
        continue
    roles = [role for role in ('Seen', 'Unseen', 'Novel', 'All')
             if any(key[0] == role for key in by_role)]
    for role in roles:
        item = {'preset': preset, 'seed': seed, 'gpu': gpu, 'role': role}
        for metric in ('dice', 'iou', 'cldice', 'hd95'):
            item[metric] = by_role.get((role, metric), '')
        item['source'] = str(macro)
        rows.append(item)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as handle:
    fields = ['preset', 'seed', 'gpu', 'role',
              'dice', 'iou', 'cldice', 'hd95', 'source']
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

print()
print('=== macro summary (All rows) ===')
all_rows = [row for row in rows if row['role'] == 'All'] or rows
if not all_rows:
    print('(no completed macro_metrics.csv yet)')
else:
    for row in all_rows:
        print(f"{row['preset']:<28} seed{row['seed']} {row['role']:<7} "
              f"Dice={row['dice']} clDice={row['cldice']} "
              f"HD95={row['hd95']}")
print(f'summary: {out}')
PY

echo
echo "============================================================"
echo " Evaluation finished (rc=$RC)"
echo " Per-run eval roots:"
for i in "${!JOB_RUN[@]}"; do
  [ -n "${JOB_RUN[$i]}" ] || continue
  echo "   ${JOB_PRESET[$i]}/seed${JOB_SEED[$i]}: ${JOB_RUN[$i]}/eval"
done
echo " Summary: $SUMMARY_CSV"
echo " Logs:    $LOG_DIR/cvpr8gpu_eval20_*_${STAMP}.log"
if [ "$RC" -eq 0 ]; then
  echo
  echo "These are the old encoder-side controls.  Report them separately"
  echo "from the new SCNP/ResMoE/hybrid screen."
fi
echo "============================================================"
exit "$RC"
