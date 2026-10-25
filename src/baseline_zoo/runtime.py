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
from core.metrics import (
    cldice_score,
    dice_score,
    hd95_score,
    iou_score,
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
    official_epochs: int
    official_lr: float
    official_weight_decay: float = 0.0
    train_batch_size: int = 1
    accumulation_steps: int = 1
    selection: str = 'dice'
    deviations: tuple[str, ...] = ()
    custom_pipeline: bool = False
    training_free: bool = False

    def __init__(self):
        self.args: argparse.Namespace | None = None
        self.device = torch.device('cpu')
        self.checkpoint_config: dict[str, Any] = {}

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
        if scheduler is not None:
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


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    # Use the same canonical metric functions as paper-metric collection
    # (medpy Dice/IoU conventions, no epsilon smoothing) so validation
    # checkpoint selection and reported test metrics share one definition.
    return {
        'dice': dice_score(pred, gt),
        'iou': iou_score(pred, gt),
        'cldice': cldice_score(pred, gt),
        'hd95': hd95_score(pred, gt),
        'empty': float(pred.sum() == 0),
    }


@torch.no_grad()
def evaluate(adapter: BaselineAdapter, model: nn.Module,
             samples: Sequence[Sample], device: torch.device
             ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
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
        values = binary_metrics(pred, gt)
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
LOG_COLS = ('ep', 'tr_loss', 'Dice↑', 'IoU↑', 'clDice↑', 'HD95↓',
            'lr', 'time')
LOG_WIDTHS = (5, 8, 7, 7, 8, 7, 8, 8)
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
        _metric_text(metrics['iou']),
        _metric_text(metrics['cldice']),
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


def write_history(path: Path, config, history, best_epoch, best_score):
    payload = {
        'config': config,
        'best_epoch': best_epoch,
        'best_score': None if best_score == -float('inf')
        else round(float(best_score), 6),
        'history': history,
    }
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        print(f'WARNING: failed to write {path}: {exc}')


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    seed_everything(args.seed)
    device = torch.device(args.device)
    adapter.args = args
    adapter.device = device

    train_samples = load_split_samples(Path(args.data_root), 'train')
    val_samples = load_split_samples(Path(args.data_root), 'val')
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
        'seed': args.seed,
        'training_free': True,
        'split_manifest_sha256': split_sha,
    }

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = workdir / 'best_model.pth'
    if ckpt_path.is_file() and not args.resume:
        backup = ckpt_path.with_name(
            f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        shutil.copy2(ckpt_path, backup)
        print(f'backup previous checkpoint -> {backup}')

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

    console.print(
        f'[bold]{adapter.display_name}[/] is training-free; '
        f'metadata checkpoint -> {ckpt_path}')
    for deviation in adapter.deviations:
        console.print(f' [yellow]deviation:[/] {deviation}')
    return ckpt_path


def run_training(adapter: BaselineAdapter, args: argparse.Namespace) -> Path:
    if getattr(adapter, 'training_free', False):
        return run_training_free(adapter, args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    adapter.args = args
    adapter.device = device

    train_samples = load_split_samples(Path(args.data_root), 'train')
    val_samples = load_split_samples(Path(args.data_root), 'val')
    if args.limit_train:
        train_samples = train_samples[:args.limit_train]
    if args.limit_val:
        val_samples = val_samples[:args.limit_val]

    loader = DataLoader(
        SampleListDataset(train_samples), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=getattr(adapter, 'drop_last', False),
        collate_fn=lambda batch: batch)

    model = adapter.build_model(args).to(device)
    adapter.load_pretrained(model, args)
    adapter.configure_trainable(model)
    optimizer = adapter.make_optimizer(model, args)
    scheduler = adapter.make_scheduler(optimizer, args)

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
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': adapter.input_size,
        'adapter_config': adapter.save_config(args),
        'selection': args.selection,
        'seed': args.seed,
        'optimizer': type(optimizer).__name__,
        'scheduler': type(scheduler).__name__ if scheduler else None,
        'split_manifest_sha256': split_sha256,
        'num_params': num_params,
        'trainable_params': trainable,
    }
    run = setup_wandb(args, name=f'{adapter.id}-seed{args.seed}',
                      config=run_config)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = workdir / 'best_model.pth'
    if ckpt_path.is_file() and not args.resume:
        backup = ckpt_path.with_name(
            f'best_model.before_{time.strftime("%Y%m%d_%H%M%S")}.pth')
        shutil.copy2(ckpt_path, backup)
        print(f'backup previous checkpoint -> {backup}')

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
    if args.resume and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device,
                             weights_only=False)
        stored_selection = payload.get('selection')
        if stored_selection and stored_selection != args.selection:
            raise SystemExit(
                f'checkpoint selection={stored_selection!r} but --selection '
                f'is {args.selection!r}; scores are not comparable')
        adapter.load_state_dict(model, payload['model_state_dict'])
        if payload.get('optimizer_state_dict'):
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler is not None and payload.get('scheduler_state_dict'):
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = int(payload.get('epoch', 0)) + 1
        best_score = float(payload.get('best_score', -float('inf')))
        best_epoch = int(payload.get('epoch', 0))
        best_metric = float(payload.get(
            'val_metrics', {}).get(args.selection, float('nan')))
        # The existing checkpoint is a valid result even if no later epoch
        # improves on it.
        saved_any = True
        print(f'resume from {ckpt_path} at epoch {start_epoch}; '
              f'best_score={best_score:.4f}')
        if history_path.is_file():
            history = [
                entry for entry in
                json.loads(history_path.read_text()).get('history', [])
                if int(entry.get('epoch', 0)) <= best_epoch]
            # Rewrite the per-dataset CSV to the same truncated history;
            # appending would duplicate rows after a weights-only resume.
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

    write_history(history_path, run_config, history, best_epoch, best_score)
    print_epoch_header(adapter, args.epochs, args.seed, device, num_params,
                       trainable)
    print(f'metrics history -> {history_path}')
    print(f'per-dataset metrics -> {dataset_csv_path}')
    for deviation in adapter.deviations:
        console.print(f' [yellow]deviation:[/] {deviation}')

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        adapter._epoch = epoch
        train_loss = adapter.train_epoch(model, loader, optimizer, scheduler,
                                         args)
        val_metrics, val_per_dataset = evaluate(
            adapter, model, val_samples, device)
        elapsed = time.time() - epoch_t0
        lr = optimizer.param_groups[0]['lr']
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

        improved = score > best_score
        print_epoch_row(epoch, train_loss, val_metrics, lr, elapsed, improved)
        if improved:
            best_score = score
            best_metric = selection_metric
            best_epoch = epoch
            payload = {
                'adapter': adapter.id,
                'official_source': adapter.official_source,
                'deviations': list(adapter.deviations),
                'model_state_dict': adapter.state_dict(model),
                'optimizer_state_dict': (optimizer.state_dict()
                                         if args.save_optimizer else None),
                'scheduler_state_dict': (scheduler.state_dict()
                                         if scheduler is not None else None),
                'epoch': epoch,
                'best_score': best_score,
                'selection': args.selection,
                'val_dice': val_metrics['dice'],
                'val_metrics': val_metrics,
                'val_per_dataset': val_per_dataset,
                'input_size': adapter.input_size,
                'split_manifest_sha256': split_sha256,
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
        write_history(history_path, run_config, history,
                      best_epoch, best_score)

    if not saved_any:
        raise SystemExit(
            'training finished without writing a checkpoint: every '
            f'validation {args.selection} score was non-finite or never '
            'improved. Check the validation masks/metric and rerun.')
    print_epoch_footer()
    console.print(
        f' [bold]Best[/]  E{best_epoch:03d}  '
        f'{args.selection}={best_metric:.4f}  -> {ckpt_path}')
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
    device = torch.device(args.device)
    adapter.args = args
    adapter.device = device
    # A failed rerun must not leave metadata from the previous checkpoint.
    previous_meta = Path(args.pred_root) / '.prediction_meta.json'
    if previous_meta.is_file():
        previous_meta.unlink()
    # Checkpoints may carry large optimizer states (e.g. SAM-Med2D Adapters);
    # load them on CPU and move only the model to the GPU.
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    stored_adapter = payload.get('adapter')
    if stored_adapter and stored_adapter != adapter.id:
        raise SystemExit(
            f'checkpoint is for adapter {stored_adapter!r}, not '
            f'{adapter.id!r}')
    saved_input = payload.get('input_size')
    if (saved_input is not None and hasattr(args, 'input_size')
            and int(args.input_size) != int(saved_input)):
        raise SystemExit(
            f'checkpoint was trained with input_size={saved_input}, but '
            f'current --input-size is {args.input_size}')
    adapter.checkpoint_config = payload.get('config', {}) or {}
    adapter.restore_config(
        args, adapter.checkpoint_config.get('adapter_config', {}))
    model = adapter.build_model(args)
    state = payload.get('model_state_dict') or {}
    if state:
        adapter.load_state_dict(model, state)
    model.to(device).eval()

    datasets = []
    for spec in args.test_datasets:
        name, sep, root = spec.partition(':')
        if not sep or not name or not root:
            raise SystemExit(
                f'--test-datasets entries must be NAME:DIR, got {spec!r}')
        test_set = RawPairDataset(Path(root))
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
            elapsed = time.time() - t0
            status.finish_dataset()
            print(f'{name}: done {total} predictions in {elapsed:.0f}s '
                  f'-> {out_dir}', flush=True)

        meta = {
            'method': adapter.id,
            'official_source': adapter.official_source,
            'deviations': list(adapter.deviations),
            'checkpoint': str(ckpt_path),
            'checkpoint_sha256': file_sha256(ckpt_path),
            'split_manifest_sha256': payload.get('split_manifest_sha256'),
            'datasets': [name for name, _ in datasets],
        }
        Path(args.pred_root).mkdir(parents=True, exist_ok=True)
        with open(Path(args.pred_root) / '.prediction_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
            f.write('\n')
        status.finish('predictions complete')
    except BaseException as exc:
        status.fail(exc)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser(adapter: BaselineAdapter) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'{adapter.display_name} baseline '
                    f'({adapter.official_source})')
    parser.add_argument('--stage', choices=['train', 'predict', 'all'],
                        default='all')
    parser.add_argument('--data-root', type=Path,
                        default=REPO_ROOT / 'data' / 'sacm_3shot')
    parser.add_argument('--workdir', type=Path,
                        default=REPO_ROOT / 'results' / 'workdirs' /
                        'baselines' / adapter.id)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--pred-root', type=Path,
                        default=REPO_ROOT / 'results' / 'predictions' /
                        adapter.id)
    parser.add_argument('--test-datasets', nargs='+', default=[])
    parser.add_argument('--epochs', type=int, default=adapter.official_epochs)
    parser.add_argument('--batch-size', type=int,
                        default=adapter.train_batch_size)
    parser.add_argument('--accum-steps', type=int,
                        default=adapter.accumulation_steps)
    parser.add_argument('--lr', type=float, default=adapter.official_lr)
    parser.add_argument('--weight-decay', type=float,
                        default=adapter.official_weight_decay)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--selection', default=adapter.selection,
                        choices=['dice', 'iou', 'cldice', 'hd95'],
                        help='validation metric for checkpoint selection')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--resume', action='store_true',
        help='resume from the best checkpoint (weights-only unless '
             '--save-optimizer was used; history/CSV are truncated to the '
             'checkpoint epoch)')
    parser.add_argument('--save-optimizer', action='store_true',
                        help='store optimizer state in the best checkpoint '
                             '(large for SAM-Med2D Adapters; default off)')
    parser.add_argument('--limit-train', type=int, default=0,
                        help='debug only: truncate the train split')
    parser.add_argument('--limit-val', type=int, default=0,
                        help='debug only: truncate the val split')
    parser.add_argument('--use_wandb', action='store_true')
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
                        default=None)
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
    if args.stage == 'predict' and not args.test_datasets:
        raise SystemExit('--test-datasets is required for --stage predict')
    ckpt_path = args.checkpoint or (args.workdir / 'best_model.pth')
    if args.stage in ('train', 'all') and args.checkpoint is not None:
        raise SystemExit('--checkpoint is a predict-time option; '
                         'use --workdir for training output')
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
