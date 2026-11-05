#!/usr/bin/env bash
# ONE-CLICK pull of the latest final SACM-X run from the server.
#
# Just run it from the repo root:
#
#   ./scripts/tools/pull_run_results.sh
#
# It automatically:
#   1. finds a reachable server (zkyd@zkyd, then zkyd@172.16.1.7),
#   2. finds the newest results/runs/final_gpu0_4_* run on that server,
#   3. pulls gate/component metrics + logs into a timestamped local directory,
#   4. updates results/server_pull/latest,
#   5. prints the gate/component All summaries.
#
# No arguments are required.  Optional overrides:
#   REMOTE=user@host
#   REMOTE_RUN=~/code/sacm-plus/results/runs/final_gpu0_4_...
#   DEST=/path/to/output
#   MODE=metrics|all
#   DRY_RUN=1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REMOTE_ENV="${REMOTE:-}"
REMOTE="${REMOTE_ENV:-zkyd@zkyd}"
REMOTE_EXPLICIT=0
[ -n "$REMOTE_ENV" ] && REMOTE_EXPLICIT=1
REMOTE_RUN="${REMOTE_RUN:-}"
DEST="${DEST:-}"
MODE="${MODE:-metrics}"
SSH_PORT="${SSH_PORT:-22}"
SSH_CIPHER="${SSH_CIPHER:-aes128-gcm@openssh.com}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
DRY_RUN="${DRY_RUN:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE="$2"; REMOTE_EXPLICIT=1; shift 2 ;;
    --remote-run) REMOTE_RUN="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --ssh-port) SSH_PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DEST" ]; then
  DEST="results/server_pull/run_$(date +%Y%m%d_%H%M%S)"
fi

case "$MODE" in
  metrics)
    FILTERS=(
      --exclude='*.pth'
      --exclude='*.pt'
      --exclude='*.ckpt'
      --exclude='*.safetensors'
      --exclude='checkpoints/'
      --exclude='predictions/'
      --exclude='visualizations/'
      --exclude='*.png'
      --exclude='*.jpg'
      --exclude='*.jpeg'
      --exclude='*.bmp'
      --exclude='*.tif'
      --exclude='*.tiff'
      --exclude='*.npy'
      --exclude='*.npz'
      --exclude='wandb/'
      --exclude='*.tfevents*'
    )
    ;;
  all)
    FILTERS=()
    ;;
  *)
    echo "ERROR: MODE must be metrics | all (got '$MODE')" >&2
    exit 2
    ;;
esac

if ! command -v "$RSYNC_BIN" >/dev/null 2>&1; then
  echo "ERROR: rsync not found; install rsync or set RSYNC_BIN." >&2
  exit 2
fi

SSH_CMD="ssh -T -p ${SSH_PORT} -o Compression=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
if [ -n "$SSH_CIPHER" ]; then
  SSH_CMD+=" -c ${SSH_CIPHER}"
fi

RSYNC_OPTS=(
  -a
  --no-owner
  --no-group
  --partial
  --append-verify
  --info=progress2
  --human-readable
  --timeout=120
  --no-motd
)
if [ "$DRY_RUN" = "1" ]; then
  RSYNC_OPTS+=(--dry-run)
fi

ssh_candidate() {
  local host="$1"
  ssh -T -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=6 \
    -o StrictHostKeyChecking=accept-new \
    "$host" true >/dev/null 2>&1
}

choose_remote() {
  if [ "$REMOTE_EXPLICIT" = "1" ]; then
    if ! ssh -T -p "$SSH_PORT" -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new "$REMOTE" true; then
      echo "ERROR: cannot SSH to $REMOTE" >&2
      exit 3
    fi
    return 0
  fi
  local host
  for host in "zkyd@zkyd" "zkyd@172.16.1.7"; do
    if ssh_candidate "$host"; then
      REMOTE="$host"
      return 0
    fi
  done
  # Last resort: allow an interactive password prompt once.
  echo "No key-based login found; trying interactive SSH..." >&2
  for host in "zkyd@zkyd" "zkyd@172.16.1.7"; do
    if ssh -T -p "$SSH_PORT" -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new "$host" true; then
      REMOTE="$host"
      return 0
    fi
  done
  return 1
}

echo "============================================================"
echo " ONE-CLICK pull: latest final SACM-X run"
echo "============================================================"

if ! choose_remote; then
  echo "ERROR: no reachable server.  Set REMOTE=user@host and retry." >&2
  exit 3
fi
echo "server: $REMOTE"

if [ -z "$REMOTE_RUN" ]; then
  echo "looking for the newest final_gpu0_4_* run..."
  REMOTE_RUN="$(ssh -T -p "$SSH_PORT" "$REMOTE" \
    'ls -dt ~/code/sacm-plus/results/runs/final_gpu0_4_* 2>/dev/null | head -1')"
fi
if [ -z "$REMOTE_RUN" ]; then
  echo "WARNING: no final_gpu0_4_* run found; trying newest results/runs entry." >&2
  REMOTE_RUN="$(ssh -T -p "$SSH_PORT" "$REMOTE" \
    'ls -dt ~/code/sacm-plus/results/runs/* 2>/dev/null | head -1')"
fi
if [ -z "$REMOTE_RUN" ]; then
  echo "ERROR: no remote run directory found under ~/code/sacm-plus/results/runs." >&2
  exit 3
fi
echo "run:    $REMOTE_RUN"
echo "dest:   $DEST"
echo "mode:   $MODE"
echo "============================================================"

mkdir -p "$DEST"
"$RSYNC_BIN" "${RSYNC_OPTS[@]}" "${FILTERS[@]}" \
  -e "$SSH_CMD" \
  "$REMOTE:$REMOTE_RUN/" \
  "$DEST/"

cat > "$DEST/PULLED.txt" <<EOF
remote=$REMOTE
remote_run=$REMOTE_RUN
pulled_at=$(date -Is)
mode=$MODE
EOF

mkdir -p results/server_pull
if realpath --relative-to=results/server_pull "$DEST" >/dev/null 2>&1; then
  ln -sfn "$(realpath --relative-to=results/server_pull "$DEST")" \
    results/server_pull/latest
else
  ln -sfn "$(cd "$DEST" && pwd)" results/server_pull/latest
fi

echo
echo "============================================================"
echo " Done"
echo " files: $(find "$DEST" -type f | wc -l)"
echo " size:  $(du -sh "$DEST" | awk '{print $1}')"
echo " dest:  $DEST"
echo " link:  results/server_pull/latest"
echo "============================================================"

if command -v python3 >/dev/null 2>&1; then
  python3 - "$DEST" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])

def fnum(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'nan'
    return f'{value:.{digits}f}' if math.isfinite(value) else 'nan'

def mean(rows, key):
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, 'nan'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else float('nan')

def rows_for_role(rows, role):
    return rows if role == 'All' else [
        row for row in rows if row.get('role') == role]

def print_gate(rows, indent='  '):
    for role in ('Seen', 'Unseen', 'Novel', 'All'):
        sub = rows_for_role(rows, role)
        if not sub:
            continue
        print(f'{indent}{role:6s} n={len(sub):2d} '
              f'AUC {fnum(mean(sub, "gate_auc"))} '
              f'AP {fnum(mean(sub, "gate_auprc"))} '
              f'F1@.5 {fnum(mean(sub, "gate_f1_0_5"))} '
              f'topoAUC {fnum(mean(sub, "gate_topo_auc"))} '
              f'topoAP {fnum(mean(sub, "gate_topo_auprc"))}')

def comp_value(row, summary_key, log_key):
    value = row.get(summary_key)
    return value if value is not None else row.get(log_key)

def comp_mean(rows, summary_key, log_key):
    vals = []
    for row in rows:
        try:
            value = float(comp_value(row, summary_key, log_key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else float('nan')

def print_comp(rows, indent='  '):
    for role in ('Seen', 'Unseen', 'Novel', 'All'):
        sub = rows_for_role(rows, role)
        if not sub:
            continue
        print(f'{indent}{role:6s} n={len(sub):2d} '
              f'Dice {fnum(comp_mean(sub, "coarse_dice", "test_coarse_dice"))}->'
              f'{fnum(comp_mean(sub, "prune_dice", "test_prune_dice"))} '
              f'IoU {fnum(comp_mean(sub, "coarse_iou", "test_coarse_iou"))}->'
              f'{fnum(comp_mean(sub, "prune_iou", "test_prune_iou"))} '
              f'clDice {fnum(comp_mean(sub, "coarse_cldice", "test_coarse_cldice"))}->'
              f'{fnum(comp_mean(sub, "prune_cldice", "test_prune_cldice"))} '
              f'HD95 {fnum(comp_mean(sub, "coarse_hd95", "test_coarse_hd95"), 2)}->'
              f'{fnum(comp_mean(sub, "prune_hd95", "test_prune_hd95"), 2)}')

def summary_rows(path):
    data = json.loads(path.read_text())
    rows = []
    for role, metrics in data.items():
        if isinstance(metrics, dict):
            rows.append({'role': role, **metrics})
    return rows

for directory in sorted(root.glob('gate_localization_all20_*')):
    print(f'\nGATE {directory.name}')
    for seed_dir in sorted(directory.glob('seed_*')):
        summary = seed_dir / 'summary.json'
        per_dataset = seed_dir / 'per_dataset.csv'
        try:
            if summary.exists():
                rows = summary_rows(summary)
                print(f'  {seed_dir.name}: summary.json')
                print_gate(rows)
            elif per_dataset.exists():
                rows = list(csv.DictReader(per_dataset.open(newline='')))
                print(f'  {seed_dir.name}: recovered per_dataset.csv '
                      f'({len(rows)} datasets)')
                print_gate(rows)
            else:
                print(f'  {seed_dir.name}: no summary.json / per_dataset.csv')
        except Exception as exc:
            print(f'  {seed_dir.name}: unreadable ({exc})')

for directory in sorted(root.glob('component_eval_all20_*')):
    print(f'\nCOMP {directory.name}')
    for seed_dir in sorted(directory.glob('seed_*')):
        summary = seed_dir / 'summary.json'
        per_dataset = seed_dir / 'per_dataset.csv'
        seed_name = seed_dir.name.replace('seed_', '')
        log_path = directory / 'logs' / f'eval_seed{seed_name}.log'
        try:
            if summary.exists():
                rows = summary_rows(summary)
                print(f'  {seed_dir.name}: summary.json')
                print_comp(rows)
            elif per_dataset.exists():
                rows = list(csv.DictReader(per_dataset.open(newline='')))
                print(f'  {seed_dir.name}: recovered per_dataset.csv '
                      f'({len(rows)} datasets)')
                print_comp(rows)
            elif log_path.exists():
                rows = []
                for line in log_path.open(errors='ignore'):
                    line = line.strip()
                    if not line.startswith('{'):
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if rows:
                    print(f'  {seed_dir.name}: PARTIAL from log '
                          f'({len(rows)} datasets; no per_dataset.csv)')
                    print_comp(rows)
                else:
                    print(f'  {seed_dir.name}: logs only, no result rows')
            else:
                print(f'  {seed_dir.name}: no summary / per_dataset / log')
        except Exception as exc:
            print(f'  {seed_dir.name}: unreadable ({exc})')
PY
fi

echo
echo "Open this directory: $DEST"
