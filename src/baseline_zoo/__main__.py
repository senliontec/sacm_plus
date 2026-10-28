"""CLI for the unified paper pipeline.

For now this provides the model/protocol registry and a dry-run planner.
Training/evaluation commands are printed and can be executed by the thin
shell wrappers once the runner layer is added.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .protocols import ProtocolConfig, load_protocol, load_protocols
from .registry import (
    REPO_ROOT,
    ModelConfig,
    load_models,
    resolve_models,
)
from .report import (
    check_protocol_seeds,
    load_runs,
    resolve_result_root,
    write_report,
)

# Every baseline eval entrypoint writes <output_root>/paper_metrics.csv.
# SACM-X aggregates into <output_root>/tables/all_metrics.csv instead.
EVAL_ARTIFACTS = {'sacm_x': Path('tables') / 'all_metrics.csv'}
DEFAULT_EVAL_ARTIFACT = Path('paper_metrics.csv')


def _eval_artifact(model: ModelConfig, output_root: Path) -> Path:
    return output_root / EVAL_ARTIFACTS.get(model.id, DEFAULT_EVAL_ARTIFACT)


def _artifact_ready(path: Path) -> bool:
    """A non-empty file is the minimum proof that eval wrote its tables."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_seeds(values: list[str] | None, protocol: ProtocolConfig) -> list[int]:
    if not values or (len(values) == 1 and values[0].lower() == 'all'):
        return list(protocol.seeds)
    try:
        seeds = [int(value) for value in values]
    except ValueError:
        raise ValueError(
            f'--seeds must be integers or "all", got {values}') from None
    # Duplicate seeds would create two jobs writing the same output root.
    seeds = list(dict.fromkeys(seeds))
    return protocol.select_seeds(seeds)


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(',') if item.strip()]
    if not gpus:
        raise ValueError('--gpus cannot be empty')
    return gpus


def _job_context(model: ModelConfig, protocol: ProtocolConfig,
                 seed: int, gpu: str, output_root: str) -> dict[str, str]:
    split = protocol.split_for(seed)
    defaults = model.defaults
    preset = str(defaults.get('preset', 'full'))
    context = {
        'gpu': gpu,
        'seed': str(seed),
        'split_root': _repo_path(split.split_root),
        'train_manifest': _repo_path(split.train_manifest),
        'test_manifest': _repo_path(split.test_manifest),
        'output_root': output_root,
        'preset': preset,
        'monitor': str(defaults.get('monitor', 'official')),
        'epochs': str(defaults.get('epochs', 200)),
        'init_stage2_from_stage1': str(
            defaults.get('init_stage2_from_stage1', False)).lower(),
        'sam_checkpoint': 'checkpoints/sam_vit_l_0b3195.pth',
        'sam2_checkpoint': 'checkpoints/sam2_hiera_large.pt',
    }
    if model.checkpoint_template:
        context['checkpoint'] = model.checkpoint_template.format(**context)
    else:
        context['checkpoint'] = f'{output_root}/best_model.pth'
    return context


def _format_template(model: ModelConfig, template: str | None,
                     context: dict[str, str]) -> str:
    if not template:
        raise ValueError(f'{model.id}: missing command template')
    try:
        return template.format(**context)
    except KeyError as exc:
        raise ValueError(
            f'{model.id}: unknown command placeholder {{{exc.args[0]}}}') from None


def _pythonize_command(command: str) -> str:
    """Resolve a leading `python`/`python3` to the running interpreter.

    Model YAML templates are executed through a shell.  A bare `python`
    would select the shell PATH (often base conda), not the interpreter that
    launched `baseline_zoo train/eval`.  Keep non-python commands unchanged.
    """
    parts = command.lstrip().split(None, 1)
    if not parts or parts[0] not in ('python', 'python3'):
        return command
    suffix = f' {parts[1]}' if len(parts) > 1 else ''
    return shlex.quote(sys.executable) + suffix


def _train_command(model: ModelConfig, protocol: ProtocolConfig,
                   seed: int, gpu: str, output_root: str) -> str:
    context = _job_context(model, protocol, seed, gpu, output_root)
    return _pythonize_command(
        _format_template(model, model.train_entrypoint, context))


def _eval_command(model: ModelConfig, protocol: ProtocolConfig,
                  seed: int, gpu: str, output_root: str) -> str:
    context = _job_context(model, protocol, seed, gpu, output_root)
    return _pythonize_command(
        _format_template(model, model.eval_entrypoint, context))


def _output_root(model: ModelConfig, seed: int) -> Path:
    return REPO_ROOT / 'results' / 'paper' / model.id / f'seed_{seed}'


def _run(command: str, log_path: Path, prefix: str) -> bool:
    print(f'{prefix}+ {command}', flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(
                command, shell=True, cwd=REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=os.environ.copy())
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(prefix + line, end='', flush=True)
            return proc.wait() == 0
    except OSError as exc:
        print(f'{prefix}[error] {exc}', file=sys.stderr)
        return False


def _build_jobs(args: argparse.Namespace,
                stage: str) -> tuple[list[dict], list[str], ProtocolConfig,
                                     list[int]]:
    models = resolve_models(args.model, load_models())
    protocol = load_protocol(args.protocol)
    seeds = _parse_seeds(args.seeds, protocol)
    gpus = _parse_gpus(args.gpus)
    jobs: list[dict] = []
    job_index = 0
    for model in models:
        for seed in seeds:
            gpu = gpus[job_index % len(gpus)]
            output_root = _output_root(model, seed)
            context = _job_context(
                model, protocol, seed, gpu, _repo_path(output_root))
            command = _pythonize_command(_format_template(
                model, model.train_entrypoint if stage == 'train'
                else model.eval_entrypoint, context))
            checkpoint = Path(context['checkpoint'])
            if not checkpoint.is_absolute():
                checkpoint = REPO_ROOT / checkpoint
            jobs.append({
                'model': model,
                'seed': seed,
                'gpu': gpu,
                'output_root': output_root,
                'marker': output_root / f'.{stage}_done',
                'command': command,
                'checkpoint': checkpoint,
                'eval_artifact': _eval_artifact(model, output_root),
            })
            job_index += 1
    return jobs, gpus, protocol, seeds


def _marker_matches(job: dict) -> bool:
    """Return True when a done-marker still describes the current file."""
    marker = job['marker']
    checkpoint = job['checkpoint']
    if not marker.is_file() or not checkpoint.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
        stat = checkpoint.stat()
        return (data.get('checkpoint') == str(checkpoint)
                and int(data.get('size', -1)) == stat.st_size
                and int(data.get('mtime_ns', -1)) == stat.st_mtime_ns)
    except (OSError, ValueError, TypeError):
        # Legacy/empty marker: force a re-run rather than trusting it.
        return False


def _write_marker(job: dict) -> None:
    checkpoint = job['checkpoint']
    stat = checkpoint.stat()
    job['marker'].write_text(json.dumps({
        'checkpoint': str(checkpoint),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }) + '\n')


def _run_job(job: dict, stage: str, resume: bool, dry_run: bool) -> bool:
    model = job['model']
    seed = job['seed']
    gpu = job['gpu']
    output_root = job['output_root']
    print(f"\n[{model.id}] seed={seed} gpu={gpu}", flush=True)
    if resume and _marker_matches(job):
        # A stale eval marker must not hide deleted predictions/tables:
        # report would otherwise fail while resume keeps skipping eval.
        if stage == 'eval' and not _artifact_ready(job['eval_artifact']):
            print(f"rerun: missing eval artifact {job['eval_artifact']}",
                  flush=True)
            if not dry_run:
                try:
                    job['marker'].unlink()
                except OSError:
                    pass
        else:
            print(f"skip: {job['marker']} matches the checkpoint", flush=True)
            return True
    if not dry_run and not job['checkpoint'].is_file():
        # A stage marker must never hide a missing/renamed checkpoint.
        if stage == 'eval' or job['marker'].is_file():
            print(f"[error] missing checkpoint: {job['checkpoint']}",
                  file=sys.stderr)
            return False
    if dry_run:
        print(f"dry-run: {job['command']}", flush=True)
        return True
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f'{stage}.log'
    prefix = f'[{model.id} seed={seed} gpu={gpu}] '
    print(f"{prefix}start (log: {log_path})", flush=True)
    ok = _run(job['command'], log_path, prefix)
    if ok and job['checkpoint'].is_file():
        _write_marker(job)
        print(f"[{model.id}] seed={seed} gpu={gpu} done", flush=True)
    elif ok:
        print(f"[error] job succeeded but checkpoint is missing: "
              f"{job['checkpoint']}", file=sys.stderr)
        ok = False
    else:
        print(f"[{model.id}] seed={seed} gpu={gpu} failed", flush=True)
    return ok


def _run_gpu(gpu: str, jobs: list[dict], stage: str,
             resume: bool, dry_run: bool) -> bool:
    print(f'[gpu {gpu}] worker: {len(jobs)} job(s)', flush=True)
    ok = True
    for job in jobs:
        ok = _run_job(job, stage, resume, dry_run) and ok
    return ok


def _execute(args: argparse.Namespace, stage: str) -> int:
    jobs, gpus, protocol, seeds = _build_jobs(args, stage)
    mode = 'dry-run' if args.dry_run else (
        'parallel' if len(gpus) > 1 else 'sequential')
    model_ids = sorted({job['model'].id for job in jobs})
    print(f'{stage}: protocol={protocol.name} seeds={seeds} '
          f'models={model_ids} gpus={gpus} ({mode})')

    if args.dry_run or len(gpus) == 1:
        ok = True
        for job in jobs:
            ok = _run_job(job, stage, args.resume, args.dry_run) and ok
        return 0 if ok else 1

    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_gpu[job['gpu']].append(job)

    ok = True
    with ThreadPoolExecutor(max_workers=len(by_gpu)) as pool:
        futures = [
            pool.submit(_run_gpu, gpu, items, stage, args.resume, False)
            for gpu, items in by_gpu.items()
        ]
        for future in as_completed(futures):
            ok = future.result() and ok
    return 0 if ok else 1


def cmd_list_models(args: argparse.Namespace) -> int:
    models = load_models()
    for model_id in sorted(models):
        model = models[model_id]
        note = f'  # {model.note}' if model.note else ''
        print(f'{model_id:16s} {model.status:14s} family={model.family:8s} '
              f'checkpoint={model.checkpoint_template or "-"}{note}')
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    models = load_models()
    protocols = load_protocols()
    available = [m for m in models.values() if m.available]
    reference = [m for m in models.values() if m.status == 'reference_only']
    print(f'models: {len(models)}  available: {len(available)}  '
          f'reference_only: {len(reference)}')
    for model in sorted(reference, key=lambda item: item.id):
        print(f'  reference_only {model.id:16s} {model.note or ""}')
    for protocol in sorted(protocols.values(), key=lambda item: item.name):
        print(f'protocol {protocol.name}: seeds={list(protocol.seeds)}')
        for seed in protocol.seeds:
            split = protocol.split_for(seed)
            state = 'OK' if split.test_manifest.is_file() else 'MISSING'
            print(f'  seed {seed}: {state} {split.test_manifest}')

    pvt_candidates = [
        REPO_ROOT / 'checkpoints' / 'pvt_v2_b2.in1k.safetensors',
        REPO_ROOT / 'checkpoints' / 'pvt_v2_b2.in1k.pth',
        REPO_ROOT / 'checkpoints' / 'pvt_v2_b2.in1k.bin',
        REPO_ROOT / 'checkpoints' / 'pvt_v2_b2.in1k.pt',
    ]
    pvt_weights = next(
        (path for path in pvt_candidates if path.is_file()),
        pvt_candidates[0])

    checks = [
        ('PVTv2-B2 ImageNet', pvt_weights, True),
        ('SAM ViT-L', REPO_ROOT / 'checkpoints' / 'sam_vit_l_0b3195.pth',
         True),
        ('SAM ViT-B', REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth',
         True),
        ('SAM ViT-H', REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth',
         True),
        # SAM2-UNet is reference_only, so its missing Hiera checkpoint is
        # informational rather than a blocker for the paper zoo.
        ('SAM2 Hiera-L', REPO_ROOT / 'checkpoints' / 'sam2_hiera_large.pt',
         False),
        # FSSAM's official default is SAM 2-S (46M); Hiera-L is the local
        # smoke/ablation alternative.
        ('SAM2 Hiera-S', REPO_ROOT / 'checkpoints' / 'sam2_hiera_small.pt',
         False),
        # C2P is a zero-shot foundation baseline; its official checkpoint is
        # required to evaluate it but not to run the rest of the zoo.
        ('C2P checkpoint', REPO_ROOT / 'checkpoints' / 'c2p' / '1.pth',
         False),
        ('Segmenter-B ViT-B/16', REPO_ROOT / 'checkpoints' /
         'segmenter_vit_b16_384.pth', False),
        ('DconnNet ResNet-34', REPO_ROOT / 'checkpoints' /
         'resnet34-333f7ec4.pth', False),
        # Matcher needs the public DINOv2 ViT-L/14 checkpoint; report it as
        # an informational asset like the SAM2 appendix weight.
        ('DINOv2 ViT-L/14', REPO_ROOT / 'checkpoints' /
         'dinov2_vitl14_pretrain.pth', False),
    ]
    failures = 0
    for label, path, required in checks:
        state = 'OK' if path.is_file() else 'MISSING'
        if not path.is_file() and required:
            failures += 1
        print(f'asset {label:18s} {state:7s} {path}')
    print('SAM2 Hiera-L download: '
          'https://dl.fbaipublicfiles.com/segment_anything_2/072824/'
          'sam2_hiera_large.pt')
    print('DINOv2 ViT-L/14 download: '
          'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/'
          'dinov2_vitl14_pretrain.pth')
    print('MedCAGD PVTv2-B2 download: '
          'https://huggingface.co/timm/pvt_v2_b2.in1k/resolve/main/'
          'model.safetensors')
    print('Segmenter-B ViT-B/16 download: '
          'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
          'segmenter/vit_base_p16_384_20220308-96dfe169.pth')
    print('DconnNet ResNet-34 download: '
          'https://download.pytorch.org/models/resnet34-333f7ec4.pth')
    for label, module in (
            ('SAM2 deps', 'hydra'),
            ('SAM2 deps', 'omegaconf'),
            ('UCS deps', 'sklearn'),
            ('UCS deps', 'albumentations'),
            ('MaskVSC deps', 'skan'),
            ('MaskVSC deps', 'kornia'),
            ('nnWNet dep', 'nnunetv2'),
            ('KnowSAM dep', 'medpy'),
            ('KnowSAM dep', 'hausdorff'),
            ('SegDINO dep', 'timm'),
            ('FlowSDF dep', 'scipy'),
            ('A2SAM dep', 'skimage'),
            ('DconnNet dep', 'cv2')):
        state = 'OK' if importlib.util.find_spec(module) else 'MISSING'
        if not importlib.util.find_spec(module):
            failures += 1
        print(f'dep   {label:18s} {state:7s} {module}')
    try:
        import timm
        dino = bool(timm.list_models('*dinov3*'))
    except Exception:
        dino = False
    print(f'dep   {"SegDINO DINOv3":18s} '
          f'{"OK" if dino else "MISSING":7s} timm DINOv3 architectures')
    print(f'doctor: {failures} missing asset/dependency(ies)')
    return 0 if failures == 0 else 1


def cmd_list_protocols(args: argparse.Namespace) -> int:
    protocols = load_protocols()
    for name in sorted(protocols):
        protocol = protocols[name]
        print(f'{name:16s} seeds={list(protocol.seeds)}  {protocol.description}')
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    models = resolve_models(args.model, load_models())
    protocol = load_protocol(args.protocol)
    seeds = _parse_seeds(args.seeds, protocol)
    gpus = _parse_gpus(args.gpus)

    print(f'protocol: {protocol.name}  seeds={seeds}  models={[m.id for m in models]}')
    job_index = 0
    warnings = []
    for model in models:
        for seed in seeds:
            split = protocol.split_for(seed)
            gpu = gpus[job_index % len(gpus)]
            output_root = f'results/paper/{model.id}/seed_{seed}'
            print()
            print(f'[{model.id}] seed={seed} gpu={gpu}')
            print(f'  split_root:   {_repo_path(split.split_root)}')
            print(f'  train_manifest: {_repo_path(split.train_manifest)}')
            print(f'  test_manifest:  {_repo_path(split.test_manifest)}')
            print(f'  output_root:  {output_root}')
            print(f'  train: {_train_command(model, protocol, seed, gpu, output_root)}')
            print(f'  eval:  {_eval_command(model, protocol, seed, gpu, output_root)}')
            for label, path in (('train_manifest', split.train_manifest),
                                ('test_manifest', split.test_manifest)):
                if not path.is_file():
                    warnings.append(f'{model.id} seed={seed}: missing {label} {path}')
            job_index += 1

    if warnings:
        print('\nWARNINGS:')
        for warning in warnings:
            print(f'  - {warning}')
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    return _execute(args, 'train')


def cmd_eval(args: argparse.Namespace) -> int:
    return _execute(args, 'eval')


def cmd_report(args: argparse.Namespace) -> int:
    models = resolve_models(args.model, load_models())
    protocol = load_protocol(args.protocol)
    seeds = _parse_seeds(args.seeds, protocol)
    check_protocol_seeds(protocol, seeds)
    result_root = resolve_result_root(args.result_root)
    rows, missing = load_runs(
        [model.id for model in models], seeds, result_root,
        preset=getattr(args, 'preset', None))

    print(f'report: models={[model.id for model in models]} seeds={seeds} '
          f'root={_repo_path(result_root)} rows={len(rows)}')
    if missing:
        print('missing runs (skipped):')
        for path in missing:
            print(f'  - {path}')
        if (not getattr(args, 'allow_missing', False)
                and not getattr(args, 'dry_run', False)):
            print('[error] refusing to write a partial report; rerun missing '
                  'jobs or pass --allow-missing', file=sys.stderr)
            return 2
    if not rows:
        if args.dry_run:
            print('dry-run: no metric rows yet; report would be empty')
            return 0
        print('[error] no metric rows found', file=sys.stderr)
        return 2
    if args.dry_run:
        print(f'dry-run: would write to {args.out_dir}')
        return 0

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    paths = write_report(rows, out_dir)
    for label, path in paths.items():
        print(f'{label}: {_repo_path(path)}')
    return 0


def _add_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--protocol', default='seeded5')
    parser.add_argument('--model', default='all',
                        help='model id, comma-separated list, or all')
    parser.add_argument('--seeds', nargs='*', default=None,
                        help='seed integers or all (default: all)')
    parser.add_argument('--gpus', default='0',
                        help='comma-separated GPU ids, assigned round-robin')
    parser.add_argument('--resume', action='store_true',
                        help='skip jobs whose .<stage>_done marker exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='print commands without executing them')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='baseline-zoo', description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_models = subparsers.add_parser('list-models', help='list model configs')
    list_models.set_defaults(func=cmd_list_models)

    list_protocols = subparsers.add_parser('list-protocols',
                                           help='list protocol configs')
    list_protocols.set_defaults(func=cmd_list_protocols)

    doctor = subparsers.add_parser('doctor', help='check configs/assets/data')
    doctor.set_defaults(func=cmd_doctor)

    plan = subparsers.add_parser('plan', help='print the (model, seed) job matrix')
    _add_job_args(plan)
    plan.set_defaults(func=cmd_plan)

    train = subparsers.add_parser('train', help='run training jobs')
    _add_job_args(train)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser('eval', help='run evaluation jobs')
    _add_job_args(evaluate)
    evaluate.set_defaults(func=cmd_eval)

    report = subparsers.add_parser('report', help='aggregate existing metrics')
    report.add_argument('--protocol', default='seeded5')
    report.add_argument('--model', default='all',
                        help='model id, comma-separated list, or all')
    report.add_argument('--seeds', nargs='*', default=None,
                        help='seed integers or all (default: all)')
    report.add_argument('--result-root', default='results/paper')
    report.add_argument('--out-dir', default='results/paper/tables')
    report.add_argument('--preset', default='full',
                        help='SACM-X preset table to read')
    report.add_argument('--allow-missing', action='store_true',
                        help='write a report even when some model/seed '
                             'runs are missing')
    report.add_argument('--gpus', default='', help=argparse.SUPPRESS)
    report.add_argument('--resume', action='store_true',
                        help=argparse.SUPPRESS)
    report.add_argument('--dry-run', action='store_true')
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
