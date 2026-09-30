"""Multi-GPU launcher for the ablation matrix (train + test per job).

Each job = (preset, dataset): train with the preset on the few-shot
split, then evaluate on the dataset. Jobs are distributed over the given
GPUs (one job per GPU). Per-job checkpoints and results are isolated in
<output_root>/<preset>/<dataset>/.

Usage:
    python run_experiments.py \
        --gpus 0 1 2 3 4 5 6 7 \
        --checkpoint /path/sam_vit_l_0b3195.pth \
        --train_root data/sacm_3shot \
        --test_datasets DRIVE:/path/DRIVE CHASEDB1:/path/CHASEDB1 \
        --presets sacm stage1 stage2 full no_geo_i no_geo_e no_c2f \
        --output_root results --epochs 50 --tta

Options:
    --skip_train   only run the test step (checkpoints must exist)
    --skip_test    only run the training step
    --selection    mask selection; default: 'gating' for the sacm preset
                   (original behavior), 'iou' otherwise
"""

import argparse
import csv
import os
import subprocess
import sys
import time

# Make the src packages importable without an editable install, and
# locate the train/eval CLIs by absolute path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from models.sacm.configs import PRESETS


def build_commands(args, preset, dataset, test_dir, job_dir):
    """Return (train_cmd, test_cmd, env_base) for a job."""
    ckpt_path = os.path.join(job_dir, 'best_model.pth')
    train_cmd = [
        sys.executable, os.path.join(SRC_DIR, 'train', 'trainer.py'),
        '--preset', preset,
        '--data_root', args.train_root,
        '--checkpoint', args.checkpoint,
        '--epochs', str(args.epochs),
        '--val_interval', str(args.val_interval),
        '--seed', str(args.seed),
        '--save_path', ckpt_path,
    ]
    if args.extra_train_args:
        train_cmd += args.extra_train_args.split()

    sel = args.selection or ('gating' if preset == 'sacm' else 'iou')
    test_cmd = [
        sys.executable, os.path.join(SRC_DIR, 'eval', 'evaluate.py'),
        '--preset', preset,
        '--data_root', test_dir,
        '--trained_weights', ckpt_path,
        '--output_dir', job_dir,
        '--selection', sel,
    ]
    if args.tta:
        test_cmd.append('--tta')
    return train_cmd, test_cmd, ckpt_path


def start_job(gpu, preset, dataset, test_dir, args):
    """Start the train phase of a job; returns the slot dict."""
    job_dir = os.path.join(args.output_root, preset, dataset)
    os.makedirs(job_dir, exist_ok=True)
    train_cmd, test_cmd, ckpt_path = build_commands(args, preset, dataset, test_dir, job_dir)

    log_fh = open(os.path.join(job_dir, 'run.log'), 'w')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    slot = {
        'gpu': gpu, 'preset': preset, 'dataset': dataset,
        'log': log_fh, 'env': env, 'test_cmd': test_cmd,
        'phase': 'train', 'proc': None, 'status': None,
    }

    log_fh.write(f"=== train ===\n{' '.join(train_cmd)}\n")
    log_fh.flush()
    if args.skip_train:
        print(f"[gpu {gpu}] {preset}/{dataset}: train skipped")
    else:
        slot['proc'] = subprocess.Popen(train_cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        print(f"[gpu {gpu}] {preset}/{dataset}: train started")
    return slot


def advance(slot, args):
    """Handle a finished phase; returns True when the whole job is done."""
    preset, dataset = slot['preset'], slot['dataset']
    rc = 0 if slot['proc'] is None else slot['proc'].returncode
    gpu = slot['gpu']

    if slot['phase'] == 'train':
        if rc != 0:
            print(f"[gpu {gpu}] {preset}/{dataset}: TRAIN FAILED (rc={rc})")
            slot['status'] = 'train_failed'
            slot['log'].close()
            return True
        if args.skip_test:
            print(f"[gpu {gpu}] {preset}/{dataset}: train done (test skipped)")
            slot['status'] = 'train_only'
            slot['log'].close()
            return True
        slot['log'].write(f"=== test ===\n{' '.join(slot['test_cmd'])}\n")
        slot['log'].flush()
        slot['proc'] = subprocess.Popen(
            slot['test_cmd'], stdout=slot['log'], stderr=subprocess.STDOUT, env=slot['env']
        )
        slot['phase'] = 'test'
        print(f"[gpu {gpu}] {preset}/{dataset}: test started")
        return False
    else:  # test phase finished
        if rc == 0:
            print(f"[gpu {gpu}] {preset}/{dataset}: done")
            slot['status'] = 'done'
        else:
            print(f"[gpu {gpu}] {preset}/{dataset}: TEST FAILED (rc={rc})")
            slot['status'] = f'test_failed(rc={rc})'
        slot['log'].close()
        return True


def main():
    parser = argparse.ArgumentParser(description='Run the ablation matrix on multiple GPUs')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='GPU ids to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Official SAM ViT-L checkpoint')
    parser.add_argument('--train_root', type=str, required=True, help='Few-shot train/val split root')
    parser.add_argument('--test_datasets', nargs='+', required=True,
                        help='Name:dir pairs, e.g. DRIVE:/data/DRIVE')
    parser.add_argument('--presets', nargs='+', default=['sacm', 'stage1', 'stage2', 'full'],
                        choices=[p for p in PRESETS if p != 'none'])
    parser.add_argument('--output_root', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection', type=str, default=None,
                        help="Override mask selection ('iou' or 'gating')")
    parser.add_argument('--tta', action='store_true', help='Enable four-way flip TTA at test time')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--extra_train_args', type=str, default='',
                        help='Extra flags appended to the train command')
    args = parser.parse_args()

    # Parse name:dir pairs
    test_datasets = []
    for spec in args.test_datasets:
        name, _, path = spec.partition(':')
        if not path:
            raise ValueError(f"--test_datasets entries must be NAME:DIR, got {spec!r}")
        test_datasets.append((name, path))

    jobs = []
    for preset in args.presets:
        for name, path in test_datasets:
            jobs.append((preset, name, path))
    print(f"Total jobs: {len(jobs)} on GPUs {args.gpus}")

    os.makedirs(args.output_root, exist_ok=True)
    status_rows = []
    slots = {g: None for g in args.gpus}
    queue = list(jobs)

    # Initial fill
    for gpu in args.gpus:
        if queue and slots[gpu] is None:
            preset, dataset, test_dir = queue.pop(0)
            slots[gpu] = start_job(gpu, preset, dataset, test_dir, args)

    while queue or any(s is not None for s in slots.values()):
        for gpu in args.gpus:
            slot = slots[gpu]
            if slot is None:
                continue
            if slot['proc'] is not None and slot['proc'].poll() is None:
                continue
            if advance(slot, args):
                status_rows.append({
                    'preset': slot['preset'],
                    'dataset': slot['dataset'],
                    'gpu': gpu,
                    'status': slot['status'],
                })
                slots[gpu] = None
        # Fill free slots from the queue
        for gpu in args.gpus:
            if queue and slots[gpu] is None:
                preset, dataset, test_dir = queue.pop(0)
                slots[gpu] = start_job(gpu, preset, dataset, test_dir, args)
        time.sleep(2)

    with open(os.path.join(args.output_root, 'jobs.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['preset', 'dataset', 'gpu', 'status'])
        writer.writeheader()
        writer.writerows(status_rows)
    print(f"All jobs finished. Status table: {os.path.join(args.output_root, 'jobs.csv')}")


if __name__ == '__main__':
    main()
