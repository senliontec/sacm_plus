#!/usr/bin/env bash
# One-click Optuna HPO for the paper baselines on GPUs 0-6.
#
# GPU 7 is reserved for SACM-X: it is never checked, masked in, or used here.
# The runner dynamically assigns trials to the free GPUs in the selected pool.
#
# Usage (after uploading the code to the server):
#   bash scripts/tools/run_optuna_paper.sh
#   bash scripts/tools/run_optuna_paper.sh --models dscnet,unet --seeds 0
#   bash scripts/tools/run_optuna_paper.sh --trials 8 --detach
#   bash scripts/tools/run_optuna_paper.sh --dry-run
#
# Defaults: all 22 paper baselines, seed 0, trials=8 per (model, seed),
# storage=results/tuning/optuna.db.  Four models are not tunable and are
# reported as skipped by the Python runner: nnwnet/knowsam (custom pipelines)
# and matcher/persam (training-free).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-all}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
TRIALS="${TRIALS:-8}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-${EPOCHS:-}}"
STORAGE="${STORAGE:-results/tuning/optuna.db}"
PROTOCOL="${PROTOCOL:-seeded5}"
PYTHON="${PYTHON:-}"
read -r -a SEEDS <<<"${SEEDS:-0}"
DRY_RUN=0
NO_INSTALL=0
ALLOW_BUSY=0
KEEP_CHECKPOINTS=0
DETACH=0
ORIGINAL_ARGS=("$@")

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next }
         NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) shift; MODELS="${1:-}"; shift ;;
        --gpus) shift; GPUS="${1:-}"; shift ;;
        --seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [[ "$1" == *,* ]]; then
                    IFS=',' read -r -a seed_parts <<<"$1"
                    SEEDS+=("${seed_parts[@]}")
                else
                    SEEDS+=("$1")
                fi
                shift
            done
            ;;
        --trials) shift; TRIALS="${1:-}"; shift ;;
        --trial-timeout) shift; TRIAL_TIMEOUT="${1:-}"; shift ;;
        --epochs) shift; SEARCH_EPOCHS="${1:-}"; shift ;;
        --storage) shift; STORAGE="${1:-}"; shift ;;
        --protocol) shift; PROTOCOL="${1:-}"; shift ;;
        --python) shift; PYTHON="${1:-}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --no-install) NO_INSTALL=1; shift ;;
        --allow-busy) ALLOW_BUSY=1; shift ;;
        --keep-trial-checkpoints) KEEP_CHECKPOINTS=1; shift ;;
        --detach) DETACH=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$DETACH" == 1 && "${OPTUNA_DETACHED:-0}" != 1 ]]; then
    mkdir -p results/tuning/logs
    LOG="results/tuning/logs/optuna_paper_$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="results/tuning/logs/optuna_paper.pid"
    export OPTUNA_LOG="$LOG"
    OPTUNA_DETACHED=1 nohup bash "$0" "${ORIGINAL_ARGS[@]}" >"$LOG" 2>&1 &
    echo $! >"$PID_FILE"
    echo "started Optuna HPO in background"
    echo "  pid:  $(cat "$PID_FILE")"
    echo "  log:  $ROOT/$LOG"
    echo "  stop: kill \$(cat $ROOT/$PID_FILE)"
    exit 0
fi

if [[ -z "$PYTHON" ]]; then
    for candidate in \
        /home/zkyd/anaconda3/envs/tsai/bin/python \
        /home/dev/anaconda3/envs/tsai/bin/python \
        "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "python interpreter not found; set PYTHON=/path/to/python" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "python has no torch: $PYTHON" >&2
    echo "use PYTHON=/path/to/tsai/python or activate the tsai env" >&2
    exit 1
fi

IFS=',' read -r -a PHYSICAL_GPUS <<<"$GPUS"
if [[ ${#PHYSICAL_GPUS[@]} -eq 0 ]]; then
    echo "GPUS is empty" >&2
    exit 2
fi
PHYSICAL_SPEC=""
LOGICAL_GPUS=""
for index in "${!PHYSICAL_GPUS[@]}"; do
    gpu="${PHYSICAL_GPUS[$index]}"
    gpu="${gpu//[[:space:]]/}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "invalid GPU id: $gpu" >&2
        exit 2
    fi
    if [[ ",$PHYSICAL_SPEC," == *",$gpu,"* ]]; then
        echo "duplicate GPU id: $gpu" >&2
        exit 2
    fi
    PHYSICAL_GPUS[$index]="$gpu"
    if [[ -n "$PHYSICAL_SPEC" ]]; then
        PHYSICAL_SPEC+=","
        LOGICAL_GPUS+=","
    fi
    PHYSICAL_SPEC+="$gpu"
    LOGICAL_GPUS+="$index"
done

if [[ ",$PHYSICAL_SPEC," == *",7,"* && "${ALLOW_GPU7:-0}" != 1 ]]; then
    echo "GPU 7 is reserved for SACM-X; refusing to use it." >&2
    echo "Set ALLOW_GPU7=1 only if you intentionally want to share it." >&2
    exit 2
fi

if [[ "$DRY_RUN" != 1 ]]; then
    if [[ "$ALLOW_BUSY" != 1 ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi not found; cannot verify selected GPUs" >&2
            exit 1
        fi
        GPU_STATUS="$(
            nvidia-smi -i "$PHYSICAL_SPEC" \
                --query-gpu=index,memory.used \
                --format=csv,noheader,nounits 2>/dev/null
        )" || {
            echo "nvidia-smi could not query GPUs: $PHYSICAL_SPEC" >&2
            exit 1
        }
        BUSY_LIMIT_MIB="${BUSY_LIMIT_MIB:-512}"
        BUSY=()
        while IFS=',' read -r gpu_index memory_mib; do
            gpu_index="${gpu_index//[[:space:]]/}"
            memory_mib="${memory_mib//[[:space:]]/}"
            [[ -z "$gpu_index" ]] && continue
            if [[ ! "$memory_mib" =~ ^[0-9]+$ ]]; then
                echo "unexpected nvidia-smi memory value for GPU $gpu_index:" \
                    "'$memory_mib'" >&2
                exit 1
            fi
            if (( memory_mib > BUSY_LIMIT_MIB )); then
                BUSY+=("GPU${gpu_index}:${memory_mib}MiB")
            fi
        done <<<"$GPU_STATUS"
        if [[ ${#BUSY[@]} -gt 0 ]]; then
            echo "selected GPUs are busy: ${BUSY[*]}" >&2
            echo "GPU 7 is intentionally not checked. Free only the selected" >&2
            echo "GPUs (or pass --allow-busy) and rerun." >&2
            exit 1
        fi
        echo "GPU preflight OK: ${PHYSICAL_SPEC} (GPU 7 untouched)"
    fi

    if ! "$PYTHON" -c 'import optuna' >/dev/null 2>&1; then
        if [[ "$NO_INSTALL" == 1 ]]; then
            echo "optuna is not installed in $PYTHON" >&2
            echo "run: $PYTHON -m pip install optuna" >&2
            exit 1
        fi
        echo "installing optuna into $PYTHON ..."
        "$PYTHON" -m pip install optuna
    fi
fi

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "SEEDS is empty" >&2
    exit 2
fi

mkdir -p results/tuning/logs
if [[ -n "${OPTUNA_LOG:-}" ]]; then
    # The detached parent already redirected this process to OPTUNA_LOG.
    # Appending directly avoids tee writing the same file twice.
    LOG="$OPTUNA_LOG"
    exec >>"$LOG" 2>&1
else
    LOG="results/tuning/logs/optuna_paper_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG") 2>&1
fi

echo "=== Optuna baseline HPO ==="
echo "repo:       $ROOT"
echo "python:     $PYTHON"
echo "models:     $MODELS"
echo "seeds:      ${SEEDS[*]}"
echo "physical:   $PHYSICAL_SPEC"
echo "logical:    $LOGICAL_GPUS"
echo "trials:     $TRIALS per (model, seed)"
echo "timeout:    ${TRIAL_TIMEOUT:-none} seconds per trial"
echo "epochs:     ${SEARCH_EPOCHS:-model defaults}"
if [[ "$KEEP_CHECKPOINTS" == 1 ]]; then
    echo "checkpoints: keep per-trial weights"
else
    echo "checkpoints: delete per-trial weights after scoring"
fi
echo "storage:    $STORAGE"
echo "protocol:   $PROTOCOL"
echo "log:        $LOG"
echo "GPU 7 is reserved for SACM-X and is not part of this run."

CMD=(
    "$PYTHON" -u scripts/tools/tune_baselines.py
    --models "$MODELS"
    --seeds "${SEEDS[@]}"
    --gpus "$LOGICAL_GPUS"
    --trials "$TRIALS"
    --protocol "$PROTOCOL"
    --storage "$STORAGE"
)
if [[ -n "$SEARCH_EPOCHS" ]]; then
    CMD+=(--epochs "$SEARCH_EPOCHS")
fi
if [[ -n "$TRIAL_TIMEOUT" ]]; then
    CMD+=(--trial-timeout "$TRIAL_TIMEOUT")
fi
if [[ "$KEEP_CHECKPOINTS" == 1 ]]; then
    CMD+=(--keep-trial-checkpoints)
fi
if [[ "$DRY_RUN" == 1 ]]; then
    CMD+=(--dry-run)
fi

# The runner passes logical indices to --device; the mask keeps GPU 7 out.
export CUDA_VISIBLE_DEVICES="$PHYSICAL_SPEC"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${CMD[@]}"
