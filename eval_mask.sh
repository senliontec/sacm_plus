#!/bin/bash

python scripts/pipeline/audit_predictions.py \
    --pred-root /home/dev/Downloads/runs/sacm_x/full/seed_0/eval \
    --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
    --out /home/dev/Downloads/runs/audit/sacm_x_full_seed0