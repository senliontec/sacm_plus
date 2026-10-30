"""Multi-GPU launcher: train once per preset, then evaluate every dataset.

Each preset is trained ONCE on ``--train_root``.  The checkpoint is written
to ``<output_root>/<preset>/best_model.pth`` and then evaluated on every
``NAME:DIR`` entry in ``--test_datasets`` (sequentially on the same GPU).
Results and logs live under ``<output_root>/<preset>/``.

This replaces the old job model, which retrained the same preset once per
test dataset (the training root never depended on the dataset).

Usage:
    python run_experiments.py \
        --gpus 0 1 2 3 4 5 6 7 \
        --checkpoint /path/sam_vit_l_0b3195.pth \
        --train_root data/sacm_3shot \
        --test_datasets DRIVE:data/sacm_3shot/test/DRIVE \
                        CHASEDB1:data/sacm_3shot/test/CHASEDB1 \
        --presets sacm stage1 stage2 full \
        --output_root results --epochs 50 --tta

Options:
    --skip_train   only run evaluation; checkpoints must already exist
    --skip_test    only run training
    --selection    mask selection; default: 'iou' iff the preset trains the
                   IoU head, else gating (original SACM behavior)
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time

# Make the src packages importable without an editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from models.sacm_x.configs import PRESETS  # noqa: E402


def resolve_selection(args, preset):
    """Resolve the mask-selection policy for one preset."""
    iou_weight = PRESETS.get(preset, {}).get('iou_loss_weight', 0.0)
    return args.selection or (
        'iou' if float(iou_weight or 0.0) > 0 else 'gating')


def train_command(args, preset, ckpt_path, selection):
    cmd = [
        sys.executable, os.path.join(SRC_DIR, 'train', 'trainer.py'),
        '--preset', preset,
        '--data_root', args.train_root,
        '--checkpoint', args.checkpoint,
        '--epochs', str(args.epochs),
        '--val_interval', str(args.val_interval),
        '--seed', str(args.seed),
        '--save_path', ckpt_path,
        '--val_selection', selection,
        '--pred_threshold', str(args.pred_threshold),
    ]
    if args.extra_train_args:
        cmd += args.extra_train_args.split()
    return cmd


def test_command(args, preset, dataset, test_dir, ckpt_path, selection):
    cmd = [
        sys.executable, os.path.join(SRC_DIR, 'eval', 'evaluate.py'),
        '--preset', preset,
        '--data_root', test_dir,
        '--trained_weights', ckpt_path,
        '--output_dir', os.path.join(args.output_root, preset, dataset),
        '--selection', selection,
        '--pred_threshold', str(args.pred_threshold),
    ]
    if args.tta:
        cmd.append('--tta')
    if getattr(args, 'extra_eval_args', ''):
        cmd += args.extra_eval_args.split()
    return cmd


def status_rows(slot, status):
    """One jobs.csv row per (preset, dataset) for a terminal state."""
    datasets = slot['datasets'] or ['']
    return [
        {'preset': slot['preset'], 'dataset': dataset,
         'gpu': slot['gpu'], 'status': status}
        for dataset, _test_dir in datasets
    ]


def _pump_output(proc, log_fh, prefix=''):
    """Relay child stdout to the run log and the parent terminal.

    The log file gets the raw line (preserving the one-line-per-epoch table);
    ``prefix`` is only added to terminal output so multi-GPU jobs remain
    distinguishable.
    """
    try:
        for line in iter(proc.stdout.readline, ''):
            # Keep draining the child pipe even if the terminal disconnects
            # or the log file becomes unwritable; otherwise the child blocks
            # on a full stdout pipe and the run appears hung.
            try:
                log_fh.write(line)
                log_fh.flush()
            except Exception:
                pass
            try:
                sys.stdout.write(prefix + line if prefix else line)
                sys.stdout.flush()
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def _launch(slot, cmd):
    """Start a child process with stdout relayed to log + terminal."""
    prefix = f"[gpu {slot['gpu']}] " if slot.get('multi_gpu') else ''
    slot['proc'] = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=slot['env'], text=True, bufsize=1)
    slot['reader'] = threading.Thread(
        target=_pump_output,
        args=(slot['proc'], slot['log'], prefix),
        daemon=True,
    )
    slot['reader'].start()


def _join_reader(slot):
    """Wait for the output relay before closing the shared log file."""
    reader = slot.get('reader')
    if reader is not None:
        reader.join(timeout=10)
        slot['reader'] = None


def start_next_test(slot, args):
    """Launch the next test command, or close the job when the queue is empty."""
    if not slot['test_queue']:
        slot['phase'] = 'done'
        if slot['status'] == 'running':
            statuses = [row['status'] for row in slot['rows']]
            slot['status'] = (
                'done' if statuses and all(s == 'done' for s in statuses)
                else 'done_with_failures')
        slot['log'].close()
        return

    dataset, test_dir = slot['test_queue'].pop(0)
    slot['current_dataset'] = dataset
    slot['phase'] = 'test'
    cmd = test_command(
        args, slot['preset'], dataset, test_dir,
        slot['ckpt'], slot['selection'])
    slot['log'].write(f"=== test {dataset} ===\n{' '.join(cmd)}\n")
    slot['log'].flush()
    _launch(slot, cmd)
    print(f"[gpu {slot['gpu']}] {slot['preset']}/{dataset}: test started",
          flush=True)


def start_job(gpu, preset, test_datasets, args):
    """Start a preset job (train first, then all test datasets)."""
    preset_dir = os.path.join(args.output_root, preset)
    os.makedirs(preset_dir, exist_ok=True)
    ckpt_path = os.path.join(preset_dir, 'best_model.pth')
    log_fh = open(os.path.join(preset_dir, 'run.log'), 'w')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    selection = resolve_selection(args, preset)

    slot = {
        'gpu': gpu, 'preset': preset, 'datasets': list(test_datasets),
        'test_queue': list(test_datasets), 'ckpt': ckpt_path,
        'selection': selection, 'log': log_fh, 'env': env,
        'phase': 'train', 'proc': None, 'status': 'running',
        'current_dataset': None, 'rows': [],
        'reader': None, 'multi_gpu': len(args.gpus) > 1,
    }

    if args.skip_train:
        if not os.path.isfile(ckpt_path):
            print(f"[gpu {gpu}] {preset}: MISSING checkpoint {ckpt_path}")
            slot['status'] = 'missing_checkpoint'
            slot['rows'] = status_rows(slot, 'missing_checkpoint')
            slot['phase'] = 'done'
            log_fh.close()
            return slot
        print(f"[gpu {gpu}] {preset}: train skipped")
        start_next_test(slot, args)
        return slot

    cmd = train_command(args, preset, ckpt_path, selection)
    log_fh.write(f"=== train ===\n{' '.join(cmd)}\n")
    log_fh.flush()
    _launch(slot, cmd)
    print(f"[gpu {gpu}] {preset}: train started", flush=True)
    return slot


def advance(slot, args):
    """Handle a finished phase. Returns True when the job is terminal."""
    if slot['phase'] == 'done':
        return True

    _join_reader(slot)
    preset, gpu = slot['preset'], slot['gpu']
    rc = 0 if slot['proc'] is None else slot['proc'].returncode

    if slot['phase'] == 'train':
        if rc != 0:
            print(f"[gpu {gpu}] {preset}: TRAIN FAILED (rc={rc})")
            slot['status'] = 'train_failed'
            slot['rows'] = status_rows(slot, 'train_failed')
            slot['phase'] = 'done'
            slot['log'].close()
            return True
        if args.skip_test:
            print(f"[gpu {gpu}] {preset}: train done (test skipped)")
            slot['status'] = 'train_only'
            slot['rows'] = status_rows(slot, 'train_only')
            slot['phase'] = 'done'
            slot['log'].close()
            return True
        start_next_test(slot, args)
        return slot['phase'] == 'done'

    dataset = slot['current_dataset']
    if rc == 0:
        print(f"[gpu {gpu}] {preset}/{dataset}: done")
        status = 'done'
    else:
        print(f"[gpu {gpu}] {preset}/{dataset}: TEST FAILED (rc={rc})")
        status = f'test_failed(rc={rc})'
    slot['rows'].append({
        'preset': preset, 'dataset': dataset, 'gpu': gpu, 'status': status,
    })
    start_next_test(slot, args)
    return slot['phase'] == 'done'


def main():
    parser = argparse.ArgumentParser(
        description='Train once per preset, then evaluate all test datasets')
    parser.add_argument('--gpus', nargs='+', type=int, required=True,
                        help='GPU ids to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Official SAM ViT-L checkpoint')
    parser.add_argument('--train_root', type=str, required=True,
                        help='Few-shot train/val split root')
    parser.add_argument('--test_datasets', nargs='+', required=True,
                        help='Name:dir pairs, e.g. DRIVE:data/.../test/DRIVE')
    parser.add_argument('--presets', nargs='+',
                        default=['sacm', 'stage1', 'stage2', 'full'],
                        choices=[p for p in PRESETS if p != 'none'])
    parser.add_argument('--output_root', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection', type=str, default=None,
                        help="Override mask selection ('iou' or 'gating'); "
                             "default: iou iff preset trains the IoU head")
    parser.add_argument('--pred_threshold', type=float, default=0.0,
                        help='Logit binarization threshold; SACM fidelity uses 0.5')
    parser.add_argument('--tta', action='store_true',
                        help='Enable four-way flip TTA at test time')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--extra_eval_args', type=str, default='',
                        help='extra args appended to every evaluate.py command')
    parser.add_argument('--extra_train_args', type=str, default='',
                        help='Extra flags appended to the train command')
    args = parser.parse_args()

    if args.skip_train and args.skip_test:
        parser.error('--skip_train and --skip_test cannot be combined')

    test_datasets = []
    for spec in args.test_datasets:
        name, _, path = spec.partition(':')
        if not path:
            raise ValueError(
                f"--test_datasets entries must be NAME:DIR, got {spec!r}")
        test_datasets.append((name, path))

    jobs = [(preset, test_datasets) for preset in args.presets]
    print(f"Total presets: {len(jobs)}; datasets per preset: "
          f"{len(test_datasets)}; GPUs {args.gpus}")

    os.makedirs(args.output_root, exist_ok=True)
    slots = {gpu: None for gpu in args.gpus}
    queue = list(jobs)
    status_rows_all = []

    for gpu in args.gpus:
        if queue:
            preset, datasets = queue.pop(0)
            slots[gpu] = start_job(gpu, preset, datasets, args)

    while queue or any(slot is not None for slot in slots.values()):
        for gpu in args.gpus:
            slot = slots[gpu]
            if slot is None:
                continue
            if slot['proc'] is not None and slot['proc'].poll() is None:
                continue
            if advance(slot, args):
                status_rows_all.extend(slot['rows'])
                slots[gpu] = None
        for gpu in args.gpus:
            if queue and slots[gpu] is None:
                preset, datasets = queue.pop(0)
                slots[gpu] = start_job(gpu, preset, datasets, args)
        time.sleep(2)

    with open(os.path.join(args.output_root, 'jobs.csv'), 'w',
              newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['preset', 'dataset', 'gpu', 'status'])
        writer.writeheader()
        writer.writerows(status_rows_all)
    failed = [row for row in status_rows_all
              if row['status'] not in ('done', 'train_only')]
    if failed:
        print(f"FAILED jobs ({len(failed)}):")
        for row in failed:
            print(f"  {row['preset']}/{row['dataset']}: {row['status']}")
        print(f"Status table: {os.path.join(args.output_root, 'jobs.csv')}")
        raise SystemExit(1)
    print(f"All jobs finished. Status table: "
          f"{os.path.join(args.output_root, 'jobs.csv')}")


if __name__ == '__main__':
    main()
