"""Unified runtime for baseline adapters.

The runtime owns everything that must be identical across models:

* the 18/6-shot protocol data source and prediction format,
* the epoch loop, validation, checkpointing, history files and W&B logging,
* the terminal table and run-status protocol,
* the CLI surface and checkpoint schema.

Each model adapter owns only its official implementation details: model
construction, official preprocessing/augmentation, official loss, optimizer
and scheduler, official inference postprocessing, and the list of documented
deviations from upstream needed for this protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from core.console import console
from core.reproducibility import configure_determinism, seed_worker
from core.metrics import (
    binary_metrics as canonical_binary_metrics,
)
from core.run_artifacts import (
    ARTIFACT_VERSION,
    capture_generator_state,
    capture_rng_state,
    prepare_training_artifacts,
    read_json,
    restore_generator_state,
    restore_rng_state,
    write_json_atomic,
    write_prediction_index,
    write_wandb_info,
)
from core.run_status import RunStatus
from core.wandb_utils import (
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

from .data import RawPairDataset, Sample, SampleListDataset, load_mask, load_split_samples

REPO_ROOT = Path(__file__).resolve().parents[2]


def _select_cuda_device(device: torch.device) -> None:
    """Make index-less ``.cuda()`` calls target the job's GPU.

    Several vendored official implementations (FSSAM's SAM 2 helpers and
    SCRWKV's ``bi_wkv`` wrapper) call ``tensor.cuda()`` internally.  Without
    an explicit current device that always means GPU 0, which breaks jobs
    assigned to ``cuda:1``..``cuda:7`` by the static multi-GPU runner.
    """
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device.index)


class BaselineAdapter(ABC):
    """Official model logic behind a uniform protocol interface.

    Subclasses must set the class metadata and implement the abstract hooks.
    ``deviations`` is mandatory documentation: every place where this port
    differs from the upstream training/inference code belongs there.
    """

    id: str
    display_name: str
    official_source: str
    official_entrypoint: str
    input_size: int
    official_lr: float
    official_weight_decay: float = 0.0
    train_batch_size: int = 1
    accumulation_steps: int = 1
    default_grad_clip: float = 0.0
    selection: str = 'dice'
    # Metric-based schedulers (ReduceLROnPlateau) are stepped by the main
    # loop after validation, where the selected metric is available.
    scheduler_after_validation: bool = False
    deviations: tuple[str, ...] = ()
    custom_pipeline: bool = False
    training_free: bool = False
    # Training-free adapters may still need the support split (PerSAM,
    # Matcher, RSRM).  A zero-shot adapter can opt out.
    needs_split: bool = True

    def __init__(self):
        self.args: argparse.Namespace | None = None
        self.device = torch.device('cpu')
        self.checkpoint_config: dict[str, Any] = {}
        self.prediction_mode = False
        self._prediction_seeded = False

    def begin_prediction(self) -> None:
        """Reset per-run prediction state before a new eval pass."""
        self.prediction_mode = True
        self._prediction_seeded = False

    def begin_validation(self) -> None:
        """Reset per-validation sampling state before a validation pass.

        The default is a no-op.  Adapters whose official evaluation samples
        prompts randomly can override this to make per-epoch validation
        deterministic, so checkpoint selection compares epochs on the same
        prompts instead of prompt noise.
        """

    # ---- construction -------------------------------------------------
    @abstractmethod
    def build_model(self, args: argparse.Namespace) -> nn.Module:
        """Instantiate the official architecture."""

    def load_pretrained(self, model: nn.Module, args: argparse.Namespace) -> None:
        """Load official pretrained weights before training."""

    def configure_trainable(self, model: nn.Module) -> None:
        """Apply the official frozen/trainable parameter policy."""

    def make_optimizer(self, model: nn.Module,
                       args: argparse.Namespace) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    def make_scheduler(self, optimizer: torch.optim.Optimizer,
                       args: argparse.Namespace):
        return None

    def make_train_sampler(self, samples: Sequence[Sample],
                           args: argparse.Namespace):
        """Optional adapter-specific sampler; None keeps shuffled sampling."""
        return None

    def make_train_dataset(self, samples: Sequence[Sample],
                           args: argparse.Namespace):
        """Optional adapter-specific training Dataset.

        Returning None keeps the shared ``SampleListDataset``.  An adapter
        can return a deterministic expanded dataset (for example, offline
        augmentation) while validation still uses the original samples.
        """
        del samples, args
        return None

    # ---- official data/loss logic ------------------------------------
    @abstractmethod
    def prepare_train_batch(self, samples: Sequence[Sample],
                            rng: random.Random) -> Any:
        """Official preprocessing, augmentation and device transfer."""

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Official loss for one batch."""

    @abstractmethod
    def prepare_eval_sample(self, sample: Sample, device: torch.device) -> Any:
        """Official inference preprocessing for one sample."""

    @abstractmethod
    def predict_sample(self, model: nn.Module, prepared: Any,
                       sample: Sample) -> np.ndarray:
        """Official postprocessing; returns a bool mask at GT resolution."""

    # ---- shared hooks -------------------------------------------------
    def selection_score(self, metrics: dict[str, float],
                        per_dataset: dict[str, dict[str, float]]) -> float:
        key = self.args.selection if self.args is not None else self.selection
        value = metrics[key]
        return -value if key == 'hd95' else value

    def checkpoint_eligible(self, epoch: int) -> bool:
        """Whether this epoch may update the selected best checkpoint.

        Stage-based adapters can return False for prior/warm-up epochs so a
        randomly initialized task model never becomes ``best_model.pth``.
        """
        del epoch
        return True

    def display_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """LR to show in the epoch table for the active training stage."""
        return float(optimizer.param_groups[0]['lr'])

    def step_scheduler(self, scheduler, metrics: dict[str, float]) -> None:
        """Step the epoch scheduler after validation.

        The default keeps the historical no-argument ``step()`` used by
        epoch-based schedules.  Metric-based schedulers (for example
        ReduceLROnPlateau) override this and consume ``metrics``.
        """
        if scheduler is not None:
            scheduler.step()

    def train_epoch(self, model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler,
                    args: argparse.Namespace) -> float:
        model.train()
        rng = random.Random(args.seed + getattr(self, '_epoch', 0))
        accum = max(int(args.accum_steps), 1)
        running = 0.0
        seen = 0
        pending = 0
        optimizer.zero_grad(set_to_none=True)
        for samples in loader:
            batch = self.prepare_train_batch(samples, rng)
            loss = self.train_step(model, batch)
            (loss / accum).backward()
            pending += 1
            if pending == accum:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            running += float(loss.detach()) * len(samples)
            seen += len(samples)
        if pending:
            # The final partial group was divided by ``accum``, so scale it
            # back to the group mean before stepping; otherwise its update
            # is under-weighted by pending/accum.
            if pending < accum:
                scale = accum / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and not self.scheduler_after_validation:
            scheduler.step()
        return running / max(seen, 1)

    def state_dict(self, model: nn.Module) -> dict[str, Any]:
        return model.state_dict()

    def load_state_dict(self, model: nn.Module, state: dict[str, Any]) -> None:
        model.load_state_dict(state)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Model-specific CLI flags."""

    def save_config(self, args: argparse.Namespace) -> dict[str, Any]:
        """Adapter-specific construction args stored in the checkpoint."""
        return {}

    def restore_config(self, args: argparse.Namespace,
                       config: dict[str, Any]) -> None:
        """Restore construction args before building a prediction model."""

    def run_training(self, args: argparse.Namespace) -> Path:
        """Override for frameworks with their own official training loop."""
        raise NotImplementedError

    def run_prediction(self, args: argparse.Namespace, checkpoint: Path) -> None:
        """Override for frameworks with their own official inference API."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Metrics and terminal table
# ---------------------------------------------------------------------------

METRIC_NAMES = ('dice', 'iou', 'cldice', 'hd95')


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(values))


def binary_metrics(pred: np.ndarray, gt: np.ndarray,
                   empty_policy: str = 'penalty') -> dict[str, float]:
    # Use the same canonical metric functions as paper-metric collection
    # (medpy Dice/IoU conventions, no epsilon smoothing) so validation
    # checkpoint selection and reported test metrics share one definition.
    # ``empty_policy='penalty'`` keeps empty predictions in clDice/HD95
    # averages instead of silently dropping them through NaN.
    return canonical_binary_metrics(pred, gt, empty_policy=empty_policy)


@torch.no_grad()
def evaluate(adapter: BaselineAdapter, model: nn.Module,
             samples: Sequence[Sample], device: torch.device,
             empty_policy: str = 'penalty'
             ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    adapter.begin_validation()
    overall = {key: [] for key in (*METRIC_NAMES, 'empty')}
    per_dataset: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        prepared = adapter.prepare_eval_sample(sample, device)
        pred = adapter.predict_sample(model, prepared, sample)
        pred = np.asarray(pred)
        if pred.dtype != bool:
            pred = pred > 0.5
        gt = load_mask(sample.mask_path)
        if pred.shape != gt.shape:
            raise RuntimeError(
                f'{adapter.id}: prediction shape {pred.shape} != GT '
                f'{gt.shape} for {sample.image_path}')
        values = binary_metrics(pred, gt, empty_policy=empty_policy)
        bucket = per_dataset.setdefault(
            sample.dataset, {key: [] for key in (*METRIC_NAMES, 'empty')})
        for key, value in values.items():
            overall[key].append(value)
            bucket[key].append(value)

    metrics = {key: _mean(values) for key, values in overall.items()}
    metrics['empty_prediction_rate'] = metrics.pop('empty')
    datasets = {}
    for dataset, values in sorted(per_dataset.items()):
        datasets[dataset] = {
            key: _mean(items) for key, items in values.items()
            if key != 'empty'
        }
        datasets[dataset]['empty_prediction_rate'] = _mean(values['empty'])
    return metrics, datasets


LOG_SEP = ' │ '
LOG_COLS = ('ep', 'tr_loss', 'Dice↑', 'clDice↑', 'IoU↑', 'HD95↓',
            'lr', 'time')
LOG_WIDTHS = (5, 8, 7, 8, 7, 7, 8, 8)
LOG_RULE = '─┼─'.join('─' * width for width in LOG_WIDTHS)
LOG_FOOTER = '─' * (sum(LOG_WIDTHS) + 3 * (len(LOG_WIDTHS) - 1))


def _metric_text(value, digits=4):
    return '—' if value != value else f'{value:.{digits}f}'


def _hd95_text(value):
    if value != value:
        return '—'
    value = abs(value)
    if value >= 1000:
        return f'{value:.0f}'
    if value >= 100:
        return f'{value:.1f}'
    return f'{value:.2f}'


def print_epoch_header(adapter: BaselineAdapter, epochs, seed, device,
                       num_params, trainable=None):
    console.print()
    extra = ''
    if trainable is not None:
        extra = f'  trainable {trainable / 1e6:.1f}M'
    console.print(
        f' [bold cyan]{adapter.id}[/]  [dim]{num_params:,} params{extra}  '
        f'{epochs} epochs  seed {seed}  {device}[/]')
    console.print(
        LOG_SEP.join(f'{name:^{width}}'
                     for name, width in zip(LOG_COLS, LOG_WIDTHS)),
        style='bold cyan')
    console.print(LOG_RULE, style='dim')


def print_epoch_row(epoch, loss, metrics, lr, elapsed, improved):
    star = ' *' if improved else ''
    values = (
        f'E{epoch:03d}',
        f'{loss:.4f}',
        _metric_text(metrics['dice']),
        _metric_text(metrics['cldice']),
        _metric_text(metrics['iou']),
        _hd95_text(metrics['hd95']),
        f'{lr:.2e}',
        f'{elapsed:.0f}s{star}',
    )
    row = LOG_SEP.join(f'{value:^{width}}'
                       for value, width in zip(values, LOG_WIDTHS))
    console.print(row, style='bold green' if improved else None)


def print_epoch_footer():
    console.print(LOG_FOOTER, style='dim')


# ---------------------------------------------------------------------------
# Checkpoint/history helpers
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if value is None:
        return None
    value = float(value)
    return None if value != value else round(value, 6)


def _csv_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ''
    return '' if value != value else f'{value:.6f}'


def _save_checkpoint_atomic(payload, path: Path) -> None:
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _prune_periodic_checkpoints(directory: Path, keep: int) -> list[Path]:
    if not directory.is_dir():
        return []
    files = sorted(
        (path for path in directory.iterdir()
         if path.name.startswith('epoch_') and path.suffix == '.pth'),
        key=lambda path: (int(path.stem[6:])
                          if path.stem[6:].isdigit() else -1))
    for path in files[:-keep] if keep else files:
        try:
            path.unlink()
        except OSError:
            pass
    return files[-keep:] if keep else []


def _reject_temporary_checkpoint(path: Path) -> None:
    """Reject temporary resume snapshots and legacy last checkpoints."""
    name = Path(path).name.lower()
    if name == '.resume_state.pth' or 'last' in name:
        raise SystemExit(
            f'{path} is a temporary training snapshot and is never a valid '
            'evaluation checkpoint; use best_model.pth')


def _remove_temporary_checkpoints(workdir: Path, last_path: Path) -> None:
    """Remove crash-resume snapshots after a successful training run.

    ``.resume_state.pth`` is needed while training is in progress so a crashed
    run can resume.  It is never a result artifact: only ``best_model.pth``
    survives a completed run.  Periodic snapshots and legacy last snapshots
    are also removed for the same reason.
    """
    try:
        if last_path.is_file() or last_path.is_symlink():
            last_path.unlink()
    except OSError as exc:
        print(f'WARNING: could not remove temporary checkpoint '
              f'{last_path}: {exc}', flush=True)
    checkpoint_dir = workdir / 'checkpoints'
    if checkpoint_dir.is_dir():
        try:
            shutil.rmtree(checkpoint_dir)
        except OSError as exc:
            print(f'WARNING: could not remove periodic checkpoints '
                  f'{checkpoint_dir}: {exc}', flush=True)
    for backup in workdir.glob('best_model.before_*.pth'):
        try:
            backup.unlink()
        except OSError as exc:
            print(f'WARNING: could not remove checkpoint backup '
                  f'{backup}: {exc}', flush=True)
    for legacy in workdir.glob('*last*.pth'):
        try:
            legacy.unlink()
        except OSError as exc:
            print(f'WARNING: could not remove legacy checkpoint '
                  f'{legacy}: {exc}', flush=True)


def write_history(path: Path, config, history, best_epoch, best_score,
                  *, status: str = 'running',
                  last_checkpoint: Path | None = None,
                  artifacts: dict | None = None,
                  wandb: dict | None = None,
                  resumed_from: str | None = None,
                  error: str | None = None):
    payload = {
        'artifact_version': ARTIFACT_VERSION,
        'status': status,
        'config': config,
        'best_epoch': best_epoch,
        'best_score': None if best_score == -float('inf')
        else round(float(best_score), 6),
        'last_epoch': (history[-1].get('epoch') if history else None),
        'history': history,
        'last_checkpoint': (str(Path(last_checkpoint).resolve())
                            if last_checkpoint is not None else None),
        'resumed_from': resumed_from,
        'wandb': wandb or {},
        'error': error,
    }
    if artifacts:
        payload['artifacts'] = {
            key: artifacts.get(key) for key in (
                'train_config', 'data_manifest', 'provenance_dir',
                'data_manifest_sha256', 'base_checkpoint',
            )
        }
    try:
        write_json_atomic(path, payload)
    except (OSError, ValueError) as exc:
        print(f'WARNING: failed to write {path}: {exc}')


def _variant_from_workdir(workdir: Path) -> str:
    """Return the run variant encoded in ``results/runs/<method>/<variant>``."""
    try:
        from core.run_layout import RUNS_ROOT
        rel = Path(workdir).resolve().relative_to(RUNS_ROOT.resolve())
        if len(rel.parts) >= 2:
            return rel.parts[1]
    except (OSError, ValueError):
        pass
    return 'default'


def write_run_info(path: Path, *, method: str, seed: int, checkpoint: Path,
                   best_epoch: int, last_epoch: int, selection: str,
                   epochs: int, data_root: str | None = None,
                   reproducibility_info: dict | None = None,
                   status: str = 'done', last_checkpoint: Path | None = None,
                   wandb: dict | None = None,
                   artifacts: dict | None = None,
                   resumed_from: str | None = None,
                   variant: str = 'default',
                   periodic_checkpoints: list[str] | None = None,
                   error: str | None = None):
    """Write seed-aware checkpoint metadata next to metrics_history.json.

    This is additive: it does not change the checkpoint format or the
    baseline training/evaluation contract.  It gives every run one file
    from which the model and its seed can be found without guessing.
    """
    payload = {
        'artifact_version': ARTIFACT_VERSION,
        'family': 'baseline',
        'method': str(method),
        'variant': str(variant),
        'seed': int(seed),
        'status': str(status),
        'checkpoint': str(Path(checkpoint).resolve()),
        'best_checkpoint': str(Path(checkpoint).resolve()),
        'last_checkpoint': (str(Path(last_checkpoint).resolve())
                            if last_checkpoint is not None else None),
        'best_epoch': int(best_epoch),
        'last_epoch': int(last_epoch),
        'selection': str(selection),
        'epochs': int(epochs),
        'resumed_from': resumed_from,
        'wandb': wandb or {},
        'periodic_checkpoints': list(periodic_checkpoints or []),
        'error': error,
    }
    if artifacts:
        payload['artifacts'] = {
            key: artifacts.get(key) for key in (
                'train_config', 'data_manifest', 'provenance_dir',
                'data_manifest_sha256', 'base_checkpoint',
            )
        }
    if data_root is not None:
        payload['data_root'] = str(data_root)
    if reproducibility_info:
        payload['reproducibility_info'] = reproducibility_info
    try:
        write_json_atomic(path, payload)
    except (OSError, ValueError) as exc:
        print(f'WARNING: failed to write {path}: {exc}')


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_baseline_determinism(args: argparse.Namespace) -> dict:
    """Apply the same deterministic execution contract as the baseline runner.

    Baseline adapters remain responsible for their official recipes; this
    only fixes RNGs, deterministic kernels, TF32 and DataLoader workers.
    ``strict=False`` is the safe baseline default because a few third-party
    CUDA operators lack deterministic kernels; pass
    ``--deterministic-strict true`` when a model supports strict mode.
    """
    info = configure_determinism(
        int(getattr(args, 'seed', 42)),
        deterministic=bool(getattr(args, 'deterministic', True)),
        strict=bool(getattr(args, 'deterministic_strict', False)),
        num_threads=int(getattr(args, 'deterministic_threads', 1)),
        cuda_seed_all=True,
    )
    args.reproducibility_info = info
    print(f"{getattr(args, 'method', 'baseline')}: reproducibility={info}",
          flush=True)
    return info


def make_training_scheduler(adapter: BaselineAdapter,
                            optimizer: torch.optim.Optimizer,
                            args: argparse.Namespace):
    """Build either the adapter's official scheduler or a generic override.

    The shared runtime steps the scheduler once per epoch: inside
    ``train_epoch`` for epoch-based schedules, or via
    ``adapter.step_scheduler`` after validation for metric-based schedules
    (for example ReduceLROnPlateau).  The generic ``onecycle`` mode is
    therefore an epoch-granular OneCycle, not PyTorch's usual per-iteration
    schedule.
    """
    choice = getattr(args, 'scheduler', 'official') or 'official'
    if choice == 'official':
        # Keep the adapter's per-group LRs (e.g. TPSegformer head x10, DKA
        # head/adapter split).  Overwriting them here would erase those
        # official ratios for the first epoch.
        return adapter.make_scheduler(optimizer, args)

    # Generic overrides are expressed in the effective LR, so normalize the
    # base first.  They are intentionally not applied to official
    # multi-group schedules.
    for group in optimizer.param_groups:
        group['lr'] = float(args.lr)

    epochs = max(1, int(getattr(args, 'epochs', 1)))
    warmup_raw = getattr(args, 'warmup_epochs', None)
    warmup = 0 if warmup_raw is None else max(
        0, min(int(warmup_raw), epochs - 1))

    def warmup_factor(epoch: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / warmup)

    if choice == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: warmup_factor(epoch),
        )

    if choice == 'cosine':
        def cosine_lambda(epoch: int) -> float:
            if epoch < warmup:
                return warmup_factor(epoch)
            progress = (epoch - warmup + 1) / max(1, epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, cosine_lambda)

    if choice == 'poly':
        def poly_lambda(epoch: int) -> float:
            poly = max(0.0, 1.0 - (epoch + 1) / epochs) ** 0.9
            return warmup_factor(epoch) * poly

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, poly_lambda)

    if choice == 'onecycle':
        # Epoch-granular OneCycle implemented with LambdaLR.  PyTorch's
        # OneCycleLR assumes a larger number of steps and divides by zero for
        # very short schedules; this form has the same shape for the shared
        # once-per-epoch stepping.
        steps = max(2, epochs)
        if warmup_raw is None:
            pct_start = 0.3
        elif warmup <= 0:
            pct_start = 1.0 / steps
        else:
            pct_start = warmup / epochs
        pct_start = min(max(pct_start, 1.0 / steps), 0.9)
        div_factor = 25.0
        final_div_factor = 1e4

        def onecycle_lambda(epoch: int) -> float:
            pct = min(max(epoch / max(1, steps - 1), 0.0), 1.0)
            if pct <= pct_start:
                return (1.0 / div_factor
                        + (1.0 - 1.0 / div_factor)
                        * (pct / pct_start))
            progress = (pct - pct_start) / max(1e-8, 1.0 - pct_start)
            final = 1.0 / final_div_factor
            return final + (1.0 - final) * 0.5 * (
                1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, onecycle_lambda)

    raise ValueError(f'unknown scheduler override: {choice!r}')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_free(adapter: BaselineAdapter,
                      args: argparse.Namespace) -> Path:
    """Write a metadata-only checkpoint for a training-free adapter.

    The method's official weights are loaded by ``adapter.build_model`` at
    prediction time; the support bank is reconstructed from ``data_root``.
    The checkpoint carries the split hash and adapter config so the normal
    evaluation pipeline can validate and restore it.
    """
    configure_baseline_determinism(args)
    device = torch.device(args.device)
    _select_cuda_device(device)
    adapter.args = args
    adapter.device = device

    try:
        train_samples = load_split_samples(Path(args.data_root), 'train')
        val_samples = load_split_samples(Path(args.data_root), 'val')
    except SystemExit:
        if getattr(adapter, 'needs_split', True):
            raise
        # A zero-shot adapter (C2P) does not consume the protocol data at
        # all; a missing local split must not block its metadata checkpoint.
        train_samples, val_samples = [], []
    if args.limit_train:
        train_samples = train_samples[:args.limit_train]
    if args.limit_val:
        val_samples = val_samples[:args.limit_val]

    split_manifest = Path(args.data_root) / 'split_manifest.csv'
    split_sha = (file_sha256(split_manifest)
                 if split_manifest.is_file() else None)
    run_config = {
        'method': adapter.id,
        'display_name': adapter.display_name,
        'official_source': adapter.official_source,
        'official_entrypoint': adapter.official_entrypoint,
        'deviations': list(adapter.deviations),
        'data_root': str(args.data_root),
        'train_images': len(train_samples),
        'val_images': len(val_samples),
        'epochs': 0,
        'batch_size': args.batch_size,
        'input_size': adapter.input_size,
        'adapter_config': adapter.save_config(args),
        'selection': args.selection,
        'empty_metric_policy': getattr(
            args, 'empty_metric_policy', 'penalty'),
        'seed': args.seed,
        'training_free': True,
        'split_manifest_sha256': split_sha,
    }

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = workdir / 'best_model.pth'

    payload = {
        'adapter': adapter.id,
        'official_source': adapter.official_source,
        'deviations': list(adapter.deviations),
        'model_state_dict': {},
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'epoch': 0,
        'best_score': 0.0,
        'selection': args.selection,
        'val_dice': float('nan'),
        'val_metrics': {},
        'val_per_dataset': {},
        'input_size': adapter.input_size,
        'split_manifest_sha256': split_sha,
        'config': run_config,
    }
    tmp_path = ckpt_path.with_name(ckpt_path.name + '.tmp')
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    write_history(workdir / 'metrics_history.json', run_config, [], 0, 0.0)
    write_run_info(
        workdir / 'run_info.json',
        method=adapter.id,
        variant=_variant_from_workdir(workdir),
        seed=args.seed,
        checkpoint=ckpt_path,
        best_epoch=0,
        last_epoch=0,
        selection=args.selection,
        epochs=0,
        data_root=str(args.data_root),
        reproducibility_info=getattr(args, 'reproducibility_info', None),
    )

    console.print(
        f'[bold]{adapter.display_name}[/] is training-free; '
        f'metadata checkpoint -> {ckpt_path}')
    for deviation in adapter.deviations:
        console.print(f' [yellow]deviation:[/] {deviation}')
    return ckpt_path


def run_training(adapter: BaselineAdapter, args: argparse.Namespace) -> Path:
    if getattr(adapter, 'training_free', False):
        return run_training_free(adapter, args)
    configure_baseline_determinism(args)
    device = torch.device(args.device)
    _select_cuda_device(device)
    adapter.args = args
    adapter.device = device
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        artifact_info = prepare_training_artifacts(
            args, workdir,
            hash_files=bool(getattr(args, 'hash_data_files', True)),
            resume=bool(getattr(args, 'resume', False)),
        )
        print(f'Training artifacts -> {artifact_info["train_config"]}',
              flush=True)
    except OSError as exc:
        print(f'WARNING: provenance artifacts not written: {exc}',
              flush=True)
        artifact_info = {}

    train_samples = load_split_samples(Path(args.data_root), 'train')
    val_samples = load_split_samples(Path(args.data_root), 'val')
    if args.limit_train:
        train_samples = train_samples[:args.limit_train]
    if args.limit_val:
        val_samples = val_samples[:args.limit_val]

    sampler = adapter.make_train_sampler(train_samples, args)
    train_dataset = adapter.make_train_dataset(train_samples, args)
    if train_dataset is None:
        train_dataset = SampleListDataset(train_samples)
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
        drop_last=getattr(adapter, 'drop_last', False),
        collate_fn=lambda batch: batch)

    # Resume must build the model from the checkpoint's adapter config, not
    # from the current CLI defaults (geometry/norm/scale can differ).
    ckpt_path = workdir / 'best_model.pth'
    last_path = workdir / '.resume_state.pth'
    if args.resume:
        choice = getattr(args, 'resume_checkpoint', 'auto')
        if choice == 'last':
            preview_path = last_path
        elif choice == 'best':
            preview_path = ckpt_path
        else:
            preview_path = last_path if last_path.is_file() else ckpt_path
        if preview_path.is_file():
            preview = torch.load(preview_path, map_location='cpu',
                                 weights_only=False)
            stored_cfg = preview.get('config', {}) or {}
            stored_seed = stored_cfg.get('seed', preview.get('seed'))
            if stored_seed is not None and int(stored_seed) != int(args.seed):
                raise SystemExit(
                    f'resume checkpoint seed={stored_seed} but --seed '
                    f'is {args.seed}; refusing to mix seeds')
            adapter.restore_config(
                args, stored_cfg.get('adapter_config', {}))

    model = adapter.build_model(args).to(device)
    adapter.load_pretrained(model, args)
    adapter.configure_trainable(model)
    optimizer = adapter.make_optimizer(model, args)
    scheduler = make_training_scheduler(adapter, optimizer, args)

    # Record the actual optimization budget.  "800 epochs" is not a common
    # unit across models: a sampler can expand the dataset (LIOT repeat=30,
    # BCU-Net 18x augmentation), and accumulation changes the update count.
    samples_per_epoch = len(train_dataset)
    if sampler is not None and getattr(sampler, 'num_samples', None):
        samples_per_epoch = int(sampler.num_samples)
    batches_per_epoch = (
        samples_per_epoch // args.batch_size
        if getattr(adapter, 'drop_last', False)
        else math.ceil(samples_per_epoch / max(args.batch_size, 1))
    ) if samples_per_epoch else 0
    inner_steps = max(1, int(getattr(
        adapter, 'optimizer_steps_per_batch', 1)))
    updates_per_epoch = (
        math.ceil(batches_per_epoch / max(args.accum_steps, 1)) * inner_steps
        if batches_per_epoch else 0)

    split_manifest = Path(args.data_root) / 'split_manifest.csv'
    split_sha256 = (file_sha256(split_manifest)
                    if split_manifest.is_file() else None)
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_config = {
        'method': adapter.id,
        'display_name': adapter.display_name,
        'official_source': adapter.official_source,
        'official_entrypoint': adapter.official_entrypoint,
        'deviations': list(adapter.deviations),
        'data_root': str(args.data_root),
        'train_images': len(train_samples),
        'val_images': len(val_samples),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'accum_steps': args.accum_steps,
        'effective_batch_size': args.batch_size * args.accum_steps,
        'samples_per_epoch': samples_per_epoch,
        'batches_per_epoch': batches_per_epoch,
        'approx_optimizer_steps_per_epoch': updates_per_epoch,
        'approx_total_optimizer_steps': updates_per_epoch * args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': adapter.input_size,
        'adapter_config': adapter.save_config(args),
        'selection': args.selection,
        'empty_metric_policy': getattr(
            args, 'empty_metric_policy', 'penalty'),
        'seed': args.seed,
        'optimizer': type(optimizer).__name__,
        'scheduler': type(scheduler).__name__ if scheduler else None,
        'scheduler_override': getattr(args, 'scheduler', 'official'),
        'warmup_epochs': getattr(args, 'warmup_epochs', None),
        'split_manifest_sha256': split_sha256,
        'num_params': num_params,
        'trainable_params': trainable,
        'artifact_version': ARTIFACT_VERSION,
        'train_config': artifact_info.get('train_config'),
        'data_manifest': artifact_info.get('data_manifest'),
        'data_manifest_sha256': artifact_info.get(
            'data_manifest_sha256'),
        'base_checkpoint': artifact_info.get('base_checkpoint'),
    }
    run = setup_wandb(args, name=f'{adapter.id}-seed{args.seed}',
                      config=run_config)
    try:
        wandb_run_info = write_wandb_info(workdir, run)
    except OSError as exc:
        print(f'WARNING: wandb metadata save failed: {exc}', flush=True)
        wandb_run_info = {}

    ckpt_path = workdir / 'best_model.pth'
    last_path = workdir / '.resume_state.pth'

    history = []
    history_path = workdir / 'metrics_history.json'
    dataset_csv_path = workdir / 'val_metrics_by_dataset.csv'
    dataset_names = sorted({sample.dataset for sample in val_samples})
    csv_fields = (['epoch', 'train_loss', 'lr', 'is_best'] +
                  [f'{dataset}_{metric}'
                   for dataset in dataset_names for metric in METRIC_NAMES])
    if not args.resume:
        with dataset_csv_path.open('w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_score = -float('inf')
    best_metric = float('nan')
    best_epoch = 0
    saved_any = False
    start_epoch = 1
    resume_choice = getattr(args, 'resume_checkpoint', 'auto')
    last_checkpoint = (
        last_path
        if args.resume and resume_choice in ('auto', 'last')
        and last_path.is_file() else None)
    periodic_checkpoints: list[str] = []
    resume_path = None
    if args.resume:
        choice = getattr(args, 'resume_checkpoint', 'auto')
        if choice == 'last':
            resume_path = last_path
        elif choice == 'best':
            resume_path = ckpt_path
        else:
            resume_path = last_path if last_path.is_file() else ckpt_path
        if not resume_path.is_file():
            raise SystemExit(
                f'--resume requested but no checkpoint exists '
                f'(last={last_path}, best={ckpt_path})')
    if resume_path is not None:
        payload = torch.load(resume_path, map_location=device,
                             weights_only=False)
        stored_selection = payload.get('selection')
        if stored_selection and stored_selection != args.selection:
            raise SystemExit(
                f'checkpoint selection={stored_selection!r} but --selection '
                f'is {args.selection!r}; scores are not comparable')
        stored_split = payload.get('split_manifest_sha256')
        if stored_split is None:
            print('WARNING: resume checkpoint has no split_manifest_sha256; '
                  'cannot prove it belongs to the current --data-root')
        elif split_sha256 is None:
            print('WARNING: current --data-root has no split_manifest.csv; '
                  'cannot prove the resume split matches the checkpoint')
        elif stored_split != split_sha256:
            raise SystemExit(
                'resume checkpoint was trained on a different split '
                f'(checkpoint={stored_split}, current={split_sha256}); '
                'use a matching --data-root or start a fresh run')
        if payload.get('data_manifest_sha256') and artifact_info.get(
                'data_manifest_sha256') and (
                payload['data_manifest_sha256']
                != artifact_info['data_manifest_sha256']):
            raise SystemExit(
                'resume checkpoint was trained with a different data '
                'manifest; use the matching --data-root')
        adapter.load_state_dict(model, payload['model_state_dict'])
        if payload.get('optimizer_state_dict'):
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        else:
            print('WARNING: resume checkpoint has no optimizer state; '
                  'the optimizer restarts from scratch at this epoch',
                  flush=True)
        if scheduler is not None and payload.get('scheduler_state_dict'):
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        restore_rng_state(payload.get('rng_state'))
        restore_generator_state(
            train_generator, payload.get('train_generator_state'))
        completed = int(payload.get(
            'epoch_completed', payload.get('epoch', 0)))
        start_epoch = completed + 1
        best_score = float(payload.get('best_score', -float('inf')))
        best_epoch = int(payload.get('best_epoch', completed))
        best_metric = float(payload.get(
            'best_metric', payload.get('val_metrics', {}).get(
                args.selection, float('nan'))))
        # The existing checkpoint is a valid result even if no later epoch
        # improves on it.
        saved_any = True
        print(f'resume from {resume_path} at epoch {start_epoch}; '
              f'best_score={best_score:.4f}', flush=True)
        stored_history = payload.get('history')
        if not isinstance(stored_history, list) and history_path.is_file():
            stored_history = read_json(history_path).get('history', [])
        if isinstance(stored_history, list):
            history = [
                entry for entry in stored_history
                if int(entry.get('epoch', 0)) <= completed]
        # Rewrite the per-dataset CSV to the same truncated history;
        # appending would duplicate rows after a resume.
        with dataset_csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for entry in history:
                row = {
                    'epoch': entry.get('epoch', ''),
                    'train_loss': _csv_value(entry.get('train_loss')),
                    'lr': _csv_value(entry.get('lr')),
                    'is_best': int(bool(entry.get('is_best', False))),
                }
                datasets = entry.get('datasets', {})
                for dataset in dataset_names:
                    values = datasets.get(dataset, {})
                    for metric in METRIC_NAMES:
                        row[f'{dataset}_{metric}'] = _csv_value(
                            values.get(metric))
                writer.writerow(row)

    write_history(
        history_path, run_config, history, best_epoch, best_score,
        status='running', last_checkpoint=last_checkpoint,
        artifacts=artifact_info, wandb=wandb_run_info,
        resumed_from=(str(resume_path) if resume_path is not None else None))
    write_run_info(
        workdir / 'run_info.json',
        method=adapter.id,
        variant=_variant_from_workdir(workdir),
        seed=args.seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        last_epoch=(int(history[-1].get('epoch', 0))
                    if history else 0),
        selection=args.selection,
        epochs=args.epochs,
        data_root=str(args.data_root),
        reproducibility_info=getattr(args, 'reproducibility_info', None),
        status='running',
        last_checkpoint=last_checkpoint,
        wandb=wandb_run_info,
        artifacts=artifact_info,
        resumed_from=(str(resume_path) if resume_path is not None else None),
    )
    print(f'metrics history -> {history_path}')
    print(f'per-dataset metrics -> {dataset_csv_path}')
    for deviation in adapter.deviations:
        console.print(f' [yellow]deviation:[/] {deviation}')
    print_epoch_header(adapter, args.epochs, args.seed, device, num_params,
                       trainable)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        adapter._epoch = epoch
        train_loss = adapter.train_epoch(model, loader, optimizer, scheduler,
                                         args)
        val_metrics, val_per_dataset = evaluate(
            adapter, model, val_samples, device,
            empty_policy=getattr(args, 'empty_metric_policy', 'penalty'))
        if adapter.scheduler_after_validation:
            adapter.step_scheduler(scheduler, val_metrics)
        elapsed = time.time() - epoch_t0
        lr = adapter.display_lr(optimizer)
        score = adapter.selection_score(val_metrics, val_per_dataset)
        selection_metric = val_metrics.get(args.selection, float('nan'))
        wb = {'epoch': epoch, 'train/loss': train_loss, 'lr': lr,
              'val/selection': score}
        wb.update({f'val/{key}': value
                   for key, value in val_metrics.items()})
        for dataset, values in val_per_dataset.items():
            wb.update({f'val/{dataset}/{metric}': values[metric]
                       for metric in METRIC_NAMES})
        log_scalars(run, wb)

        improved = (adapter.checkpoint_eligible(epoch)
                    and score > best_score)
        print_epoch_row(epoch, train_loss, val_metrics, lr, elapsed, improved)
        if improved:
            best_score = score
            best_metric = selection_metric
            best_epoch = epoch
            payload = {
                'artifact_version': ARTIFACT_VERSION,
                'adapter': adapter.id,
                'official_source': adapter.official_source,
                'deviations': list(adapter.deviations),
                'model_state_dict': adapter.state_dict(model),
                'optimizer_state_dict': (optimizer.state_dict()
                                         if args.save_optimizer else None),
                'scheduler_state_dict': (scheduler.state_dict()
                                         if scheduler is not None else None),
                'epoch': epoch,
                'epoch_completed': epoch,
                'best_score': best_score,
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'selection': args.selection,
                'val_dice': val_metrics['dice'],
                'val_metrics': val_metrics,
                'val_per_dataset': val_per_dataset,
                'input_size': adapter.input_size,
                'split_manifest_sha256': split_sha256,
                'data_manifest_sha256': artifact_info.get(
                    'data_manifest_sha256'),
                'full_config': artifact_info.get('train_config_payload'),
                'history': list(history),
                'rng_state': capture_rng_state(),
                'train_generator_state': capture_generator_state(
                    train_generator),
                'wandb': wandb_run_info,
                'resumed_from': (
                    str(resume_path) if resume_path is not None else None),
                'config': run_config,
            }
            # Atomic replace: a kill during a multi-GB save must not destroy
            # the previous best checkpoint.
            tmp_path = ckpt_path.with_name(ckpt_path.name + '.tmp')
            torch.save(payload, tmp_path)
            os.replace(tmp_path, ckpt_path)
            saved_any = True

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'lr': lr,
            'is_best': improved,
            'selection': _json_value(selection_metric),
            'overall': {key: _json_value(value)
                        for key, value in val_metrics.items()},
            'datasets': {
                dataset: {key: _json_value(value)
                          for key, value in values.items()}
                for dataset, values in val_per_dataset.items()
            },
        })
        csv_row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'lr': f'{lr:.8g}',
            'is_best': int(improved),
        }
        for dataset in dataset_names:
            values = val_per_dataset.get(dataset, {})
            for metric in METRIC_NAMES:
                csv_row[f'{dataset}_{metric}'] = _csv_value(
                    values.get(metric))
        try:
            with dataset_csv_path.open('a', newline='') as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(csv_row)
        except OSError as exc:
            print(f'WARNING: failed to append {dataset_csv_path}: {exc}')
        write_history(
            history_path, run_config, history, best_epoch, best_score,
            status='running', last_checkpoint=last_checkpoint,
            artifacts=artifact_info, wandb=wandb_run_info,
            resumed_from=(
                str(resume_path) if resume_path is not None else None))
        last_payload = {
            'artifact_version': ARTIFACT_VERSION,
            'adapter': adapter.id,
            'official_source': adapter.official_source,
            'deviations': list(adapter.deviations),
            'model_state_dict': adapter.state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': (scheduler.state_dict()
                                     if scheduler is not None else None),
            'epoch': epoch,
            'epoch_completed': epoch,
            'best_score': best_score,
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'selection': args.selection,
            'val_dice': val_metrics['dice'],
            'val_metrics': val_metrics,
            'val_per_dataset': val_per_dataset,
            'input_size': adapter.input_size,
            'split_manifest_sha256': split_sha256,
            'data_manifest_sha256': artifact_info.get(
                'data_manifest_sha256'),
            'full_config': artifact_info.get('train_config_payload'),
            'history': list(history),
            'rng_state': capture_rng_state(),
            'train_generator_state': capture_generator_state(
                train_generator),
            'wandb': wandb_run_info,
            'resumed_from': (
                str(resume_path) if resume_path is not None else None),
            'config': run_config,
        }
        save_last_every = int(getattr(args, 'save_last_every', 1) or 0)
        if save_last_every > 0 and (
                epoch % save_last_every == 0 or epoch == args.epochs):
            try:
                _save_checkpoint_atomic(last_payload, last_path)
                last_checkpoint = last_path
            except (OSError, RuntimeError) as exc:
                print(f'WARNING: last checkpoint save failed: {exc}')
        save_every = int(getattr(args, 'save_every', 0) or 0)
        if save_every > 0 and epoch % save_every == 0:
            checkpoint_dir = workdir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            periodic_path = checkpoint_dir / f'epoch_{epoch:04d}.pth'
            try:
                _save_checkpoint_atomic(last_payload, periodic_path)
                periodic_checkpoints = [
                    str(path) for path in _prune_periodic_checkpoints(
                        checkpoint_dir,
                        int(getattr(args, 'keep_checkpoints', 2)))
                ]
            except (OSError, RuntimeError) as exc:
                print(f'WARNING: periodic checkpoint save failed: {exc}')
        write_history(
            history_path, run_config, history, best_epoch, best_score,
            status='running', last_checkpoint=last_checkpoint,
            artifacts=artifact_info, wandb=wandb_run_info,
            resumed_from=(
                str(resume_path) if resume_path is not None else None))
        write_run_info(
            workdir / 'run_info.json',
            method=adapter.id,
            variant=_variant_from_workdir(workdir),
            seed=args.seed,
            checkpoint=ckpt_path,
            best_epoch=best_epoch,
            last_epoch=epoch,
            selection=args.selection,
            epochs=args.epochs,
            data_root=str(args.data_root),
            reproducibility_info=getattr(args, 'reproducibility_info', None),
            status='running',
            last_checkpoint=last_checkpoint,
            wandb=wandb_run_info,
            artifacts=artifact_info,
            periodic_checkpoints=periodic_checkpoints,
            resumed_from=(
                str(resume_path) if resume_path is not None else None),
        )

    if not saved_any:
        error = (
            'training finished without writing a checkpoint: every '
            f'validation {args.selection} score was non-finite or never '
            'improved.')
        write_history(
            history_path, run_config, history, best_epoch, best_score,
            status='failed', last_checkpoint=last_checkpoint,
            artifacts=artifact_info, wandb=wandb_run_info,
            resumed_from=(
                str(resume_path) if resume_path is not None else None),
            error=error)
        write_run_info(
            workdir / 'run_info.json',
            method=adapter.id,
            variant=_variant_from_workdir(workdir),
            seed=args.seed, checkpoint=ckpt_path,
            best_epoch=best_epoch,
            last_epoch=(int(history[-1].get('epoch', 0))
                        if history else 0),
            selection=args.selection, epochs=args.epochs,
            data_root=str(args.data_root),
            reproducibility_info=getattr(
                args, 'reproducibility_info', None),
            status='failed', last_checkpoint=last_checkpoint,
            wandb=wandb_run_info, artifacts=artifact_info,
            periodic_checkpoints=periodic_checkpoints,
            resumed_from=(
                str(resume_path) if resume_path is not None else None),
            error=error)
        raise SystemExit(error)
    # Successful training exposes exactly one model artifact: best_model.pth.
    # The temporary resume snapshot and periodic snapshots are removed before
    # the terminal metadata is written, so no evaluation can select them.
    _remove_temporary_checkpoints(workdir, last_path)
    last_checkpoint = None
    periodic_checkpoints = []
    write_history(
        history_path, run_config, history, best_epoch, best_score,
        status='done', last_checkpoint=None,
        artifacts=artifact_info, wandb=wandb_run_info,
        resumed_from=(str(resume_path) if resume_path is not None else None))
    print_epoch_footer()
    console.print(
        f' [bold]Best[/]  E{best_epoch:03d}  '
        f'{args.selection}={best_metric:.4f}  -> {ckpt_path}')
    write_run_info(
        workdir / 'run_info.json',
        method=adapter.id,
        variant=_variant_from_workdir(workdir),
        seed=args.seed,
        checkpoint=ckpt_path,
        best_epoch=best_epoch,
        last_epoch=args.epochs,
        selection=args.selection,
        epochs=args.epochs,
        data_root=str(args.data_root),
        reproducibility_info=getattr(args, 'reproducibility_info', None),
        status='done',
        last_checkpoint=last_checkpoint,
        wandb=wandb_run_info,
        artifacts=artifact_info,
        periodic_checkpoints=periodic_checkpoints,
        resumed_from=(
            str(resume_path) if resume_path is not None else None),
    )
    if run is not None:
        if args.wandb_log_artifact:
            log_artifact(run, ckpt_path, name=f'{adapter.id}-seed{args.seed}',
                         artifact_type='model')
        else:
            print('W&B artifact upload skipped '
                  '(set WANDB_LOG_ARTIFACT=true to enable)')
    finish_run(run)
    return ckpt_path


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def run_prediction(adapter: BaselineAdapter, args: argparse.Namespace,
                   ckpt_path: Path) -> None:
    _reject_temporary_checkpoint(Path(ckpt_path))
    configure_baseline_determinism(args)
    device = torch.device(args.device)
    _select_cuda_device(device)
    adapter.args = args
    adapter.device = device
    # Distinguish final prediction from validation inside a training run.
    # Adapters that seed eval-time prompt sampling use this flag so they do
    # not reset the training RNG during per-epoch validation.  This also
    # clears the once-only seed latch when one adapter object is reused for
    # a second prediction run.
    adapter.begin_prediction()
    # A failed rerun must not leave metadata from the previous checkpoint.
    previous_meta = Path(args.pred_root) / '.prediction_meta.json'
    if previous_meta.is_file():
        previous_meta.unlink()
    # Checkpoints may carry large optimizer states (e.g. SAM-Med2D Adapters);
    # load them on CPU and move only the model to the GPU.
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    checkpoint_sha = file_sha256(ckpt_path)
    stored_adapter = payload.get('adapter')
    if stored_adapter and stored_adapter != adapter.id:
        raise SystemExit(
            f'checkpoint is for adapter {stored_adapter!r}, not '
            f'{adapter.id!r}')
    adapter.checkpoint_config = payload.get('config', {}) or {}
    adapter.restore_config(
        args, adapter.checkpoint_config.get('adapter_config', {}))
    # The checkpoint's construction size is authoritative.  Checking it
    # before restore_config rejected valid non-default configurations.
    saved_input = payload.get('input_size')
    if saved_input is not None and hasattr(args, 'input_size'):
        saved_input = int(saved_input)
        if int(args.input_size) != saved_input:
            print(
                f'[predict] checkpoint input_size={saved_input}; using it '
                f'instead of --input-size={args.input_size}', flush=True)
        args.input_size = saved_input
    model = adapter.build_model(args)
    state = payload.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        raise SystemExit(
            f'checkpoint {ckpt_path} has no model_state_dict; refusing to '
            'evaluate a randomly initialized model')
    adapter.load_state_dict(model, state)
    model.to(device).eval()

    datasets = []
    for spec in args.test_datasets:
        name, sep, root = spec.partition(':')
        if not sep or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        # External roots often use unprefixed stems (100_D.png, 405.png).
        # The explicit protocol name must win over the stem-prefix guess so
        # support-donor selection and per-dataset caches stay correct.
        test_set = RawPairDataset(Path(root), dataset_name=name)
        datasets.append((name, test_set.samples))
    total_images = sum(len(samples) for _, samples in datasets)
    status = RunStatus(method=adapter.id, stage='predict', heartbeat=10.0)
    status.update(datasets_total=len(datasets), overall_total=total_images)

    try:
        for dataset_index, (name, samples) in enumerate(datasets, 1):
            out_dir = Path(args.pred_root) / name
            out_dir.mkdir(parents=True, exist_ok=True)
            total = len(samples)
            status.start_dataset(name, index=dataset_index, total=total,
                                 overall_total=total_images)
            print(f'{name}: predicting {total} images on {device} -> {out_dir}',
                  flush=True)
            t0 = time.time()
            for done, sample in enumerate(samples, 1):
                with torch.no_grad():
                    prepared = adapter.prepare_eval_sample(sample, device)
                    pred = adapter.predict_sample(model, prepared, sample)
                pred = np.asarray(pred)
                if pred.dtype != bool:
                    pred = pred > 0.5
                from PIL import Image
                with Image.open(sample.mask_path) as mask_image:
                    expected_shape = (mask_image.size[1], mask_image.size[0])
                if pred.shape != expected_shape:
                    raise RuntimeError(
                        f'{adapter.id}: prediction shape {pred.shape} != '
                        f'GT shape {expected_shape} for {sample.image_path}')
                out = (pred.astype(np.uint8) * 255)
                Image.fromarray(out, mode='L').save(
                    out_dir / f'{sample.stem}.png')
                status.advance(1, write=(done % 10 == 0))
                if done % 50 == 0 and done != total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate > 0 else 0.0
                    print(f'  {name}: {done}/{total}  {rate:.1f} img/s  '
                          f'ETA {eta:.0f}s', flush=True)
            prediction_rows = [{
                'dataset': name,
                'image': sample.image_path.name,
                # Relative to out_dir: finalize_run moves prediction dirs.
                'pred_path': f'{sample.stem}.png',
                'gt_path': str(Path(sample.mask_path).resolve()),
                'image_path': str(Path(sample.image_path).resolve()),
            } for sample in samples]
            try:
                write_prediction_index(
                    out_dir, name, prediction_rows,
                    manifest_extra={
                        'method': adapter.id,
                        'checkpoint': str(ckpt_path),
                        'checkpoint_sha256': checkpoint_sha,
                        'prediction_suffix': '.png',
                        'input_size': int(adapter.input_size),
                        'data_root': str(Path(root)),
                    })
            except OSError as exc:
                print(f'WARNING: prediction index save failed for {name}: '
                      f'{exc}', flush=True)
            elapsed = time.time() - t0
            status.finish_dataset()
            print(f'{name}: done {total} predictions in {elapsed:.0f}s '
                  f'-> {out_dir}', flush=True)

        meta = {
            'method': adapter.id,
            'official_source': adapter.official_source,
            'deviations': list(adapter.deviations),
            'checkpoint': str(ckpt_path),
            'checkpoint_sha256': checkpoint_sha,
            'split_manifest_sha256': payload.get('split_manifest_sha256'),
            'datasets': [name for name, _ in datasets],
            'prediction_index': str(
                (Path(args.pred_root) / '{DATASET}' /
                 'prediction_index.csv').resolve()),
            'artifact_version': ARTIFACT_VERSION,
        }
        Path(args.pred_root).mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            Path(args.pred_root) / '.prediction_meta.json', meta)
        status.finish('predictions complete')
    except BaseException as exc:
        status.fail(exc)
        raise
    finally:
        adapter.prediction_mode = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configured_epochs(adapter: BaselineAdapter) -> int:
    """Read the training budget from the model config, the single source."""
    from .registry import load_models

    model = load_models().get(adapter.id)
    if model is None or 'epochs' not in model.defaults:
        raise SystemExit(
            f'{adapter.id}: missing defaults.epochs in configs/models')
    epochs = int(model.defaults['epochs'])
    if epochs < 0:
        raise SystemExit(f'{adapter.id}: negative epochs: {epochs}')
    return epochs


def build_parser(adapter: BaselineAdapter) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'{adapter.display_name} baseline '
                    f'({adapter.official_source})')
    parser.add_argument('--stage', choices=['train', 'predict', 'all'],
                        default='all')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='split root (default: '
                             'data/sacm_3shot_seed<seed> when present)')
    parser.add_argument('--workdir', type=Path, default=None,
                        help='canonical run root (default: '
                             'results/runs/<model>/default/seed_<seed>)')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--pred-root', type=Path, default=None,
                        help='prediction output root (default: '
                             '<workdir>/predictions/<model>)')
    parser.add_argument('--test-datasets', nargs='+', default=[])
    parser.add_argument('--epochs', type=int,
                        default=_configured_epochs(adapter))
    parser.add_argument('--batch-size', type=int,
                        default=adapter.train_batch_size)
    parser.add_argument('--accum-steps', type=int,
                        default=adapter.accumulation_steps)
    parser.add_argument('--lr', type=float, default=adapter.official_lr)
    parser.add_argument('--weight-decay', type=float,
                        default=adapter.official_weight_decay)
    parser.add_argument(
        '--grad-clip', type=float,
        default=getattr(adapter, 'default_grad_clip', 0.0))
    parser.add_argument(
        '--scheduler', default='official',
        choices=['official', 'constant', 'cosine', 'poly', 'onecycle'],
        help='scheduler override; official keeps the adapter recipe. '
             'Shared-runtime onecycle steps once per epoch.')
    parser.add_argument(
        '--warmup-epochs', type=int, default=None,
        help='warmup for generic schedulers; official schedulers keep their '
             'own warmup')
    parser.add_argument('--selection', default=adapter.selection,
                        choices=['dice', 'iou', 'cldice', 'hd95'],
                        help='validation metric for checkpoint selection')
    parser.add_argument(
        '--empty-metric-policy', default='penalty',
        choices=['penalty', 'nan'],
        help='empty prediction convention: penalty (default) keeps empty '
             'masks in clDice/HD95 with a finite worst-case value; nan keeps '
             'the raw official NaN semantics and silently drops them')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--deterministic', type=lambda value: str(value).lower()
        in ('1', 'true', 'yes', 'on'), default=True,
        help='enable deterministic kernels/TF32-off (default true)')
    parser.add_argument(
        '--deterministic-strict', type=lambda value: str(value).lower()
        in ('1', 'true', 'yes', 'on'), default=False,
        help='raise on ops without a deterministic CUDA kernel '
             '(default false for third-party baselines)')
    parser.add_argument(
        '--deterministic-threads', type=int, default=1,
        help='torch CPU threads when deterministic (default 1)')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--resume', action='store_true',
        help='resume at the next epoch boundary from the temporary '
             '.resume_state.pth when available, otherwise best_model.pth; the '
             'temporary file is removed after successful completion')
    parser.add_argument(
        '--resume-checkpoint', choices=['auto', 'last', 'best'],
        default='auto',
        help='resume source: auto=last if present else best (default)')
    parser.add_argument('--save-optimizer', action='store_true',
                        help='store optimizer state in the best checkpoint '
                             '(large for SAM-Med2D Adapters; default off)')
    parser.add_argument(
        '--save-last-every', type=int, default=1,
        help='write the temporary .resume_state.pth every N epochs for crash '
             'resume (default 1); it is deleted after successful training')
    parser.add_argument('--save-every', type=int, default=50,
                        help='write temporary checkpoints/epoch_XXXX.pth every '
                             'N epochs (0 disables); the directory is deleted '
                             'after successful training')
    parser.add_argument('--keep-checkpoints', type=int, default=2,
                        help='keep this many newest periodic checkpoints')
    parser.add_argument(
        '--hash-data-files',
        type=lambda value: str(value).lower() in ('1', 'true', 'yes', 'on'),
        default=True,
        help='hash train/val files and test masks into '
             'provenance/data_manifest.json (default true)')
    parser.add_argument('--limit-train', type=int, default=0,
                        help='debug only: truncate the train split')
    parser.add_argument('--limit-val', type=int, default=0,
                        help='debug only: truncate the val split')
    parser.add_argument(
        '--use_wandb', nargs='?', const=True, default=True,
        type=lambda value: str(value).lower() in ('1', 'true', 'yes', 'on'),
        help='log to W&B (default: true; pass --use_wandb false to disable)')
    parser.add_argument('--wandb_project', default='sacm')
    parser.add_argument('--wandb_entity',
                        default=os.environ.get('WANDB_ENTITY', 'buaazqk'))
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--wandb_group', default='baselines')
    parser.add_argument('--wandb_job_type', default='train')
    parser.add_argument('--wandb_tags',
                        default=f'{adapter.id},3shot')
    parser.add_argument('--wandb_mode',
                        choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE'))
    parser.add_argument('--wandb_host',
                        default=os.environ.get(
                            'WANDB_BASE_URL', 'http://172.16.1.7:8080'))
    parser.add_argument('--wandb_api_key',
                        default=os.environ.get('WANDB_API_KEY'))
    parser.add_argument('--wandb_dir', default=None)
    parser.add_argument(
        '--wandb_log_artifact', action='store_true',
        default=os.environ.get('WANDB_LOG_ARTIFACT', '').lower()
        in ('1', 'true', 'yes', 'on'))
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    adapter.add_args(parser)
    return parser


def main_for(adapter: BaselineAdapter, argv: list[str] | None = None) -> int:
    parser = build_parser(adapter)
    args = parser.parse_args(argv)
    if args.data_root is None:
        seed_root = (REPO_ROOT / 'data' /
                     f'sacm_3shot_seed{int(args.seed)}')
        if seed_root.is_dir():
            args.data_root = seed_root
        elif int(args.seed) == 42:
            args.data_root = REPO_ROOT / 'data' / 'sacm_3shot'
            print('[warn] seed 42: using legacy data/sacm_3shot split')
        else:
            raise SystemExit(
                f'missing seed-specific data root: {seed_root}; '
                'create the split or pass --data-root explicitly')
    if args.workdir is None:
        from core.run_layout import run_root
        args.workdir = run_root(adapter.id, 'default', int(args.seed))
    if args.pred_root is None:
        # Match the canonical predictions/<method>/<dataset> contract used
        # by collect_paper_metrics and run_baseline_test.
        args.pred_root = Path(args.workdir) / 'predictions' / adapter.id
    if args.stage == 'predict' and not args.test_datasets:
        raise SystemExit('--test-datasets is required for --stage predict')
    ckpt_path = args.checkpoint or (args.workdir / 'best_model.pth')
    if (args.stage in ('predict', 'all')
            and args.checkpoint is not None):
        _reject_temporary_checkpoint(Path(args.checkpoint))
    if args.stage in ('train', 'all') and args.checkpoint is not None:
        raise SystemExit('--checkpoint is a predict-time option; '
                         'use --workdir for training output')
    if (adapter.custom_pipeline and args.stage in ('train', 'all')
            and (args.scheduler != 'official'
                 or args.warmup_epochs is not None)):
        raise SystemExit(
            f'{adapter.id}: scheduler overrides are not wired into this '
            'custom official pipeline')
    if adapter.custom_pipeline:
        if args.stage in ('train', 'all'):
            ckpt_path = adapter.run_training(args)
        if args.stage == 'all' and not args.test_datasets:
            return 0
        if args.stage in ('predict', 'all'):
            adapter.run_prediction(args, ckpt_path)
        return 0
    if args.stage in ('train', 'all'):
        ckpt_path = run_training(adapter, args)
    if args.stage == 'all' and not args.test_datasets:
        return 0
    if args.stage in ('predict', 'all'):
        if not ckpt_path.is_file():
            raise SystemExit(f'missing checkpoint: {ckpt_path}')
        run_prediction(adapter, args, ckpt_path)
    return 0
