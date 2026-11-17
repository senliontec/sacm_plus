#!/usr/bin/env bash
# Wait for the two seed-0 pilots, then screen mean fusion and fixed head 0 on
# the worst-variance TopoMortar seeds (1 and 3) using the canonical checkpoints.
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts/pipeline
export TMPDIR=/home/dev/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/dev/anaconda3/envs/tsai/bin/python

wait_pid() {
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
}

run_eval() {
  local seed=$1 gpu=$2 mode=$3
  local workdir="results/runs/topostrand/final200_pa/seed_${seed}"
  local ckpt="${workdir}/best_model_fixed3.pth"
  local tag="_mean_tm"
  local extra=(--deploy-fusion mean)
  if [ "$mode" = "fixed0" ]; then
    ckpt="${workdir}/best_model_fixed0.pth"
    tag="_fixed0_tm"
    extra=()
  fi
  $PY -m methods.topostrand.evaluate_routed \
    --data-root data/sacm_3shot_seed0 \
    --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
    --workdir "$workdir" --checkpoint "$ckpt" --seed "$seed" \
    --device "cuda:${gpu}" --use-wandb 0 --tta-flip --tile 1024 \
    --stride 896 --routing-skel-ratio 1.0 --routing-area-ratio 0.7 \
    "${extra[@]}" --tag "$tag" --datasets TopoMortar \
    > "results/logs/topostrand_seed${seed}${tag}.log" 2>&1
}

wait_pid 3948343   # fixed3 canonical pilot (GPU1)
run_eval 1 1 mean &
run_eval 3 1 mean &
wait
wait_pid 3975882   # fixed3 no-WCD pilot (GPU0)
run_eval 1 0 fixed0 &
run_eval 3 0 fixed0 &
wait

$PY - <<'PY'
import csv
for seed in (1, 3):
    print(f'seed {seed}')
    for tag, root in [('base', 'eval_routed'), ('mean', 'eval_mean_tm'),
                      ('fixed0', 'eval_fixed0_tm')]:
        path = (f'results/runs/topostrand/final200_pa/seed_{seed}/'
                f'{root}/TopoMortar/summary.csv')
        try:
            row = next(csv.DictReader(open(path)))
        except FileNotFoundError:
            print(f'  {tag}: missing')
            continue
        print(f'  {tag:7s} dice={float(row["dice_mean"])*100:5.2f} '
              f'iou={float(row["iou_mean"])*100:5.2f} '
              f'cld={float(row["cldice_mean"])*100:5.2f} '
              f'hd95={float(row["hd95_mean"]):6.2f}')
PY
