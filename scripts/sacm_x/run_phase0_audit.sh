#!/usr/bin/env bash
# One-click Phase-0 validation from a clean results/ directory.
#
# Does NOT train. It creates every eval/audit it needs:
#   - fixed2 aspect_pad full20
#   - fixed2 resize full20 (3 GPU shards)
#   - iou six-dataset control
#   - fixed head 0/1/3 six-dataset
#   - six-dataset candidate generation
#   - optional typed h3/h0 if a full_typed checkpoint exists
# Then audits, builds head comparison, runs topology ranker, packages.
#
# Default GPU allocation (override any GPU_* variable):
#   GPU_EVAL_1..3 = 0,1,2   resize shards
#   GPU_CAND      = 3       candidate generation
#   GPU_HEAD0/1/3 = 4,5,6   fixed heads
#   GPU_ASPECT    = 7       fixed2 aspect_pad full20
#   GPU_IOU       = 7       iou six-dataset control (after aspect)
#   GPU_TYPED     = 7       optional typed h3/h0 (after iou)
#
# Usage:
#   bash scripts/sacm_x/run_phase0_audit.sh
#
# FORCE=1 reruns even completed evals.
set -uo pipefail

REPO="${REPO:-/home/zkyd/code/sacm-plus}"
cd "$REPO" || exit 1

PYTHON="${PYTHON:-python}"
GPU_EVAL_1="${GPU_EVAL_1:-0}"
GPU_EVAL_2="${GPU_EVAL_2:-1}"
GPU_EVAL_3="${GPU_EVAL_3:-2}"
GPU_CAND="${GPU_CAND:-3}"
GPU_HEAD0="${GPU_HEAD0:-4}"
GPU_HEAD1="${GPU_HEAD1:-5}"
GPU_HEAD3="${GPU_HEAD3:-6}"
GPU_ASPECT="${GPU_ASPECT:-7}"
GPU_IOU="${GPU_IOU:-7}"
GPU_TYPED="${GPU_TYPED:-7}"

BASE="${BASE:-$REPO/results/runs/sacm_x/full/seed_0}"
AUDIT_ROOT="${AUDIT_ROOT:-$REPO/results/selector_audit_clean}"
TOPO_ROOT="${TOPO_ROOT:-$REPO/results/topo_ranker_candidates}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed0}"
FORCE="${FORCE:-0}"
PACKAGE="${PACKAGE:-0}"

SHARD1="CHASEDB1 CREMI DRIVE DeepGlobe LES-AV STARE WIRE"
SHARD2="CHCrack5k CrackTree200 DeepCrack FIVES OCTA-500 TYRE XCAD"
SHARD3="CORN DCA1 HRF LEAF ROAD TopoMortar"
ALL20="$SHARD1 $SHARD2 $SHARD3"
SIX="CREMI LEAF OCTA-500 TYRE TopoMortar WIRE"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$AUDIT_ROOT/logs_$STAMP"
mkdir -p "$LOG_ROOT"
fail=0

need() { [ -e "$1" ] || { echo "ERROR: missing $1" >&2; exit 2; }; }

echo "repo:      $REPO"
echo "ckpt:      $BASE/best_model.pth"
echo "GPUs: eval=$GPU_EVAL_1,$GPU_EVAL_2,$GPU_EVAL_3 cand=$GPU_CAND heads=$GPU_HEAD0,$GPU_HEAD1,$GPU_HEAD3 aspect=$GPU_ASPECT iou=$GPU_IOU typed=$GPU_TYPED"
echo "logs:      $LOG_ROOT"

need "$BASE/best_model.pth"
need "$TRAIN_ROOT/protocol_manifest.csv"
need "scripts/sacm_x/eval_sacm_x_20.sh"
need "scripts/sacm_x/rank_candidates_topo.py"
need "scripts/pipeline/audit_predictions.py"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
pred_dir_for() {
  local out_root="$1" preset="$2"
  local pred="$out_root/canonical/sacm_x/$preset/seed_0/eval"
  [ -d "$pred" ] || pred="$out_root/eval/$preset"
  printf '%s\n' "$pred"
}

count_metrics() {
  local pred
  pred="$(pred_dir_for "$1" "$2")"
  find "$pred" -maxdepth 2 -name metrics.csv 2>/dev/null | wc -l
}

run_eval() {
  local tag="$1" out_root="$2" gpu="$3" input_mode="$4"
  local empty_guard="$5" datasets="$6" save_candidates="$7" log="$8"
  local preset="$9" train_out="${10}" ckpt="${11}"
  local fixed_head="${12}" selection="${13:-}"
  local save_masks=false save_probs=false audit_heads=false
  if [ "$save_candidates" = "1" ]; then
    save_masks=true; save_probs=true; audit_heads=true
  fi

  mkdir -p "$out_root"
  echo "=== eval: $tag (GPU $gpu, $preset, $input_mode, fixed=$fixed_head, selection=${selection:-none}) ==="
  env \
    GPU="$gpu" SEED=0 PRESET="$preset" \
    TRAIN_OUT="$train_out" CKPT="$ckpt" \
    TRAIN_ROOT="$TRAIN_ROOT" \
    OUTPUT_ROOT="$out_root/eval" \
    CANONICAL_ROOT="$out_root/canonical" \
    FIXED_HEAD="$fixed_head" SELECTION="$selection" \
    EMPTY_GUARD="$empty_guard" EVAL_INPUT_MODE="$input_mode" \
    USE_WANDB=false DATASETS="$datasets" \
    REUSE_EVAL_MODEL=1 PAPER_METRICS_ONLY=1 \
    SAVE_CANDIDATE_MASKS="$save_masks" \
    SAVE_CANDIDATE_PROBS="$save_probs" \
    AUDIT_HEADS="$audit_heads" \
    bash scripts/sacm_x/eval_sacm_x_20.sh >"$log" 2>&1
}

audit_pred() {
  local tag="$1" pred="$2" out="$3" datasets="${4:-}" log="$5"
  [ -d "$pred" ] || { echo "ERROR: no prediction tree for $tag: $pred" >&2; return 1; }
  local args=(--pred-root "$pred" --protocol "$TRAIN_ROOT/protocol_manifest.csv" --out "$out" --workers 8)
  [ -n "$datasets" ] && args+=(--datasets $datasets)
  echo "=== audit: $tag ==="
  "$PYTHON" scripts/pipeline/audit_predictions.py "${args[@]}" >"$log" 2>&1
}

launch() {
  local tag="$1" out_root="$2" gpu="$3" input_mode="$4"
  local empty_guard="$5" datasets="$6" save_candidates="$7"
  local preset="$8" train_out="$9" ckpt="${10}" fixed_head="${11}"
  local selection="${12:-}" log="$LOG_ROOT/${tag}.log"
  run_eval "$tag" "$out_root" "$gpu" "$input_mode" "$empty_guard" \
    "$datasets" "$save_candidates" "$log" "$preset" "$train_out" \
    "$ckpt" "$fixed_head" "$selection" &
  pids+=("$!"); names+=("$tag")
}

# ---------------------------------------------------------------------------
# 1. Parallel evals
# ---------------------------------------------------------------------------
RESIZE_ROOT="$AUDIT_ROOT/full_fixed2_resize_20"
R1="$RESIZE_ROOT/shard1/seed_0"
R2="$RESIZE_ROOT/shard2/seed_0"
R3="$RESIZE_ROOT/shard3/seed_0"
TOPO_OUT="$TOPO_ROOT/full_fixed2_6/seed_0"
H0="$AUDIT_ROOT/full_fixed0_6/seed_0"
H1="$AUDIT_ROOT/full_fixed1_6/seed_0"
H3="$AUDIT_ROOT/full_fixed3_6/seed_0"

pids=(); names=()

if [ "$FORCE" = "1" ] || [ "$(count_metrics "$R1" full)" -ne 7 ]; then
  launch resize_s1 "$R1" "$GPU_EVAL_1" resize false "$SHARD1" 0 full "$BASE" "$BASE/best_model.pth" 2 ""
fi
if [ "$FORCE" = "1" ] || [ "$(count_metrics "$R2" full)" -ne 7 ]; then
  launch resize_s2 "$R2" "$GPU_EVAL_2" resize false "$SHARD2" 0 full "$BASE" "$BASE/best_model.pth" 2 ""
fi
if [ "$FORCE" = "1" ] || [ "$(count_metrics "$R3" full)" -ne 6 ]; then
  launch resize_s3 "$R3" "$GPU_EVAL_3" resize false "$SHARD3" 0 full "$BASE" "$BASE/best_model.pth" 2 ""
fi

cand_done=0
if [ "$FORCE" != "1" ] \
    && [ -d "$TOPO_OUT/canonical/sacm_x/full/seed_0/eval/CREMI/candidates" ] \
    && [ -d "$TOPO_OUT/canonical/sacm_x/full/seed_0/eval/WIRE/candidates" ]; then
  cand_done=1
fi
if [ "$cand_done" = "0" ]; then
  launch candidates_6 "$TOPO_OUT" "$GPU_CAND" aspect_pad true "$SIX" 1 full "$BASE" "$BASE/best_model.pth" 2 ""
fi

if [ "$FORCE" = "1" ] || [ "$(count_metrics "$H0" full)" -ne 6 ]; then
  launch head0_6 "$H0" "$GPU_HEAD0" aspect_pad true "$SIX" 0 full "$BASE" "$BASE/best_model.pth" 0 ""
fi
if [ "$FORCE" = "1" ] || [ "$(count_metrics "$H1" full)" -ne 6 ]; then
  launch head1_6 "$H1" "$GPU_HEAD1" aspect_pad true "$SIX" 0 full "$BASE" "$BASE/best_model.pth" 1 ""
fi
if [ "$FORCE" = "1" ] || [ "$(count_metrics "$H3" full)" -ne 6 ]; then
  launch head3_6 "$H3" "$GPU_HEAD3" aspect_pad true "$SIX" 0 full "$BASE" "$BASE/best_model.pth" 3 ""
fi

# Aspect_pad fixed2 full20, then iou six, on GPU_ASPECT/GPU_IOU sequentially.
ASPECT_ROOT="$AUDIT_ROOT/full_fixed2_20/seed_0"
IOU_ROOT="$AUDIT_ROOT/full_iou_6/seed_0"

if [ "$FORCE" = "1" ] || [ "$(count_metrics "$ASPECT_ROOT" full)" -ne 20 ]; then
  run_eval fixed2_20 "$ASPECT_ROOT" "$GPU_ASPECT" aspect_pad true "" 0 \
    "$LOG_ROOT/fixed2_20.log" full "$BASE" "$BASE/best_model.pth" 2 "" \
    || { echo "ERROR: fixed2_20 failed" >&2; fail=1; }
fi

if [ "$FORCE" = "1" ] || [ "$(count_metrics "$IOU_ROOT" full)" -ne 6 ]; then
  run_eval iou_6 "$IOU_ROOT" "$GPU_IOU" aspect_pad true "$SIX" 0 \
    "$LOG_ROOT/iou_6.log" full "$BASE" "$BASE/best_model.pth" "" iou \
    || { echo "ERROR: iou_6 failed" >&2; fail=1; }
fi

for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "${names[$i]}: done"
  else
    echo "ERROR: ${names[$i]} failed; see $LOG_ROOT/${names[$i]}.log" >&2
    fail=1
  fi
done

# ---------------------------------------------------------------------------
# 2. Merge resize shards
# ---------------------------------------------------------------------------
MERGED="$RESIZE_ROOT/merged/eval"
mkdir -p "$MERGED"

merge_shard() {
  local root="$1" dslist="$2" src
  src="$(pred_dir_for "$root" full)"
  for ds in $dslist; do
    if [ -d "$src/$ds" ]; then
      ln -sfn "$src/$ds" "$MERGED/$ds"
    else
      echo "WARNING: missing shard dataset $ds in $src" >&2
      fail=1
    fi
  done
}
merge_shard "$R1" "$SHARD1"
merge_shard "$R2" "$SHARD2"
merge_shard "$R3" "$SHARD3"

merged_n=0
for ds in $ALL20; do [ -e "$MERGED/$ds" ] && merged_n=$((merged_n + 1)); done
if [ "$merged_n" -eq 20 ]; then
  audit_pred fixed2_resize_20 "$MERGED" "$RESIZE_ROOT/audit" "" \
    "$LOG_ROOT/audit_fixed2_resize_20.log" \
    || { echo "ERROR: resize audit failed" >&2; fail=1; }
else
  echo "ERROR: merged resize tree has $merged_n/20 datasets" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# 3. Audits
# ---------------------------------------------------------------------------
ASPECT_PRED="$(pred_dir_for "$ASPECT_ROOT" full)"
F2_AUDIT="$AUDIT_ROOT/full_fixed2_6/audit"
IOU_AUDIT="$IOU_ROOT/audit"

audit_pred fixed2_20_aspectpad "$ASPECT_PRED" \
  "$ASPECT_ROOT/audit" "" "$LOG_ROOT/audit_fixed2_20.log" \
  || { echo "ERROR: fixed2_20 audit failed" >&2; fail=1; }

audit_pred fixed2_6 "$ASPECT_PRED" "$F2_AUDIT" "$SIX" \
  "$LOG_ROOT/audit_fixed2_6.log" \
  || { echo "ERROR: fixed2_6 audit failed" >&2; fail=1; }

IOU_PRED="$(pred_dir_for "$IOU_ROOT" full)"
audit_pred iou_6 "$IOU_PRED" "$IOU_AUDIT" "$SIX" \
  "$LOG_ROOT/audit_iou_6.log" \
  || { echo "ERROR: iou_6 audit failed" >&2; fail=1; }

audit_six() {
  local tag="$1" root="$2" preset="${3:-full}" pred
  pred="$(pred_dir_for "$root" "$preset")"
  [ -d "$pred" ] || { echo "WARNING: no $tag predictions" >&2; return 0; }
  audit_pred "$tag" "$pred" "$root/audit" "$SIX" \
    "$LOG_ROOT/audit_${tag}.log" \
    || { echo "ERROR: $tag audit failed" >&2; fail=1; }
}
audit_six head0_6 "$H0"
audit_six head1_6 "$H1"
audit_six head3_6 "$H3"

CMP="$AUDIT_ROOT/head_selection_6"
if [ -f scripts/pipeline/compare_audits.py ] \
    && [ -f "$F2_AUDIT/per_dataset.csv" ] \
    && [ -f "$IOU_AUDIT/per_dataset.csv" ] \
    && [ -f "$H0/audit/per_dataset.csv" ] \
    && [ -f "$H1/audit/per_dataset.csv" ] \
    && [ -f "$H3/audit/per_dataset.csv" ]; then
  "$PYTHON" scripts/pipeline/compare_audits.py \
    --audit iou="$IOU_AUDIT" \
    --audit fixed0="$H0/audit" \
    --audit fixed1="$H1/audit" \
    --audit fixed2="$F2_AUDIT" \
    --audit fixed3="$H3/audit" \
    --primary fixed2 --out "$CMP" \
    >"$LOG_ROOT/head_selection_6.log" 2>&1 \
    || { echo "ERROR: head selection compare failed" >&2; fail=1; }
else
  echo "WARNING: compare_audits.py missing or audits incomplete; skip head comparison" >&2
fi

# ---------------------------------------------------------------------------
# 4. Optional typed h3/h0
# ---------------------------------------------------------------------------
TYPED_BASE="${TYPED_BASE:-}"
if [ -z "$TYPED_BASE" ]; then
  typed_ckpt="$(find "$REPO/results" -maxdepth 6 -path '*full_typed*' -name best_model.pth 2>/dev/null | head -1)"
  [ -n "$typed_ckpt" ] && TYPED_BASE="$(dirname "$typed_ckpt")"
fi

if [ -n "$TYPED_BASE" ] && [ -f "$TYPED_BASE/best_model.pth" ]; then
  TH3="$AUDIT_ROOT/typed_h3_6/seed_0"
  TH0="$AUDIT_ROOT/typed_h0_6/seed_0"
  echo "typed checkpoint: $TYPED_BASE/best_model.pth"
  run_eval typed_h3 "$TH3" "$GPU_TYPED" aspect_pad true "$SIX" 0 \
    "$LOG_ROOT/typed_h3.log" full_typed "$TYPED_BASE" \
    "$TYPED_BASE/best_model.pth" 3 "" \
    || { echo "ERROR: typed_h3 failed" >&2; fail=1; }
  run_eval typed_h0 "$TH0" "$GPU_TYPED" aspect_pad true "$SIX" 0 \
    "$LOG_ROOT/typed_h0.log" full_typed "$TYPED_BASE" \
    "$TYPED_BASE/best_model.pth" 0 "" \
    || { echo "ERROR: typed_h0 failed" >&2; fail=1; }
  audit_six typed_h3_6 "$TH3" full_typed
  audit_six typed_h0_6 "$TH0" full_typed
else
  echo "skip typed: no full_typed checkpoint found"
fi

# ---------------------------------------------------------------------------
# 5. Topology ranker
# ---------------------------------------------------------------------------
PRED="$(pred_dir_for "$TOPO_OUT" full)"
missing=0
for ds in $SIX; do
  [ -d "$PRED/$ds/candidates" ] || { echo "WARNING: missing candidates: $ds" >&2; missing=1; }
done

if [ "$missing" = "1" ]; then
  echo "ERROR: candidates missing; ranker skipped" >&2
  fail=1
else
  mkdir -p "$TOPO_OUT/ranker"
  echo "=== ranker smoke: CREMI 10 images ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" scripts/sacm_x/rank_candidates_topo.py \
    --candidates-dir "$PRED/CREMI/candidates" \
    --probs-dir "$PRED/CREMI/candidates" \
    --images-dir data/sacm_3shot_seed0/test/CREMI/images \
    --masks-dir data/sacm_3shot_seed0/test/CREMI/masks \
    --head-audit "$PRED/CREMI/head_audit.csv" \
    --output-dir "$TOPO_OUT/ranker/CREMI_smoke" --limit 10 \
    >"$LOG_ROOT/ranker_CREMI_smoke.log" 2>&1

  if [ "$?" -ne 0 ]; then
    echo "ERROR: ranker smoke failed" >&2
    fail=1
  else
    rpids=(); rnames=()
    for ds in $SIX; do
      case "$ds" in
        CREMI)      ROOT=data/sacm_3shot_seed0/test/CREMI ;;
        LEAF)       ROOT=datasets/novel/LEAF ;;
        OCTA-500)   ROOT=datasets/unseen/OCTA-500 ;;
        TYRE)       ROOT=datasets/novel/TYRE ;;
        TopoMortar) ROOT=datasets/novel/TopoMortar ;;
        WIRE)       ROOT=datasets/novel/WIRE ;;
      esac
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON" scripts/sacm_x/rank_candidates_topo.py \
        --candidates-dir "$PRED/$ds/candidates" \
        --probs-dir "$PRED/$ds/candidates" \
        --images-dir "$ROOT/images" \
        --masks-dir "$ROOT/masks" \
        --head-audit "$PRED/$ds/head_audit.csv" \
        --output-dir "$TOPO_OUT/ranker/$ds" \
        >"$LOG_ROOT/ranker_${ds}.log" 2>&1 &
      rpids+=("$!"); rnames+=("$ds")
    done
    for i in "${!rpids[@]}"; do
      wait "${rpids[$i]}" || { echo "ERROR: ranker ${rnames[$i]} failed" >&2; fail=1; }
    done
  fi
fi

# ---------------------------------------------------------------------------
# 6. Optional package
# ---------------------------------------------------------------------------
if [ "$PACKAGE" = "1" ]; then
  PACK_FILES=()
  for rel in \
    "selector_audit_clean/full_fixed2_20/seed_0/audit" \
    "selector_audit_clean/full_fixed2_resize_20/audit" \
    "selector_audit_clean/full_fixed2_6/audit" \
    "selector_audit_clean/full_iou_6/seed_0/audit" \
    "selector_audit_clean/full_fixed0_6/seed_0/audit" \
    "selector_audit_clean/full_fixed1_6/seed_0/audit" \
    "selector_audit_clean/full_fixed3_6/seed_0/audit" \
    "selector_audit_clean/head_selection_6" \
    "selector_audit_clean/typed_h3_6/seed_0/audit" \
    "selector_audit_clean/typed_h0_6/seed_0/audit" \
    "topo_ranker_candidates/full_fixed2_6/seed_0/ranker"; do
    [ -e "$REPO/results/$rel" ] && PACK_FILES+=("$rel")
  done
  if [ "${#PACK_FILES[@]}" -gt 0 ]; then
    TAR="/tmp/sacm_x_phase0_${STAMP}.tar.gz"
    tar -czf "$TAR" -C "$REPO/results" "${PACK_FILES[@]}"
    echo "packaged: $TAR"
  fi
else
  echo "packaging skipped (PACKAGE=0)"
  echo "results: $AUDIT_ROOT"
  echo "results: $TOPO_ROOT"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "Phase-0 finished with errors. Logs: $LOG_ROOT" >&2
  exit 1
fi
echo "Phase-0 done."
echo "scp the result directories directly; no tar needed."
