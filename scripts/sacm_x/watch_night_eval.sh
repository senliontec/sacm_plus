#!/usr/bin/env bash
# Wait for both night_final runs to finish, then launch the 20-test eval.
# Polls canonical metrics_history.json (created by finalize_run after training).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
ROOT="${NIGHT_ROOT:-results/runs/night_final}"
PY="${PYTHON:-/home/dev/anaconda3/envs/tsai/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${NIGHT_WATCH_LOG:-results/ablation_logs/night_watch_${STAMP}.log}"

done_run() {
  local hist="$1"
  [ -f "$hist" ] || return 1
  "$PY" - "$hist" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    sys.exit(1)
h = d.get('history') or []
last = int(h[-1].get('epoch', 0)) if h else 0
sys.exit(0 if d.get('status') == 'done' and last >= 800 else 1)
PY
}

V3="$ROOT/sacm_x/full_v3/seed_0/metrics_history.json"
HY="$ROOT/sacm_x/full_v2_skel_aux/seed_0/metrics_history.json"
echo "watching: $V3"
echo "          $HY"
for _ in $(seq 1 600); do
  if done_run "$V3" && done_run "$HY"; then
    echo "both runs done; starting 20-test eval at $(date)"
    NIGHT_ROOT="$ROOT" PYTHON="$PY" bash scripts/sacm_x/run_night_eval.sh
    exit $?
  fi
  sleep 60
done
echo "timeout waiting for night_final (10h)" >&2
exit 1
