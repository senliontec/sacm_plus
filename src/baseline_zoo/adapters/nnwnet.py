"""nnWNet baseline running the official nnU-Net trainer pipeline.

The trainer plugin is copied into ``baseline_zoo.models.nnwnet`` and needs
``nnunetv2==2.8.1``.  Data are staged into the official nnU-Net raw layout
(24 protocol cases: 18 train + 6 val), a fixed one-fold split is written,
and training/prediction use nnU-Net's own planner, trainer and predictor.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline_zoo.data import RawPairDataset, load_mask, load_rgb, load_split_samples
from baseline_zoo.runtime import BaselineAdapter, binary_metrics, file_sha256
from core.wandb_utils import (
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

DATASET_ID = 501
DATASET_NAME = f'Dataset{DATASET_ID}_Curvilinear3Shot'


def _resolve_device(spec: str) -> torch.device:
    """Resolve the requested device and make CUDA devices current.

    ``nnUNetTrainer.__init__`` rewrites every CUDA device to ``cuda:0``
    because the official CLI selects cards with ``CUDA_VISIBLE_DEVICES``.
    We construct the trainer directly, so the current device must be set
    here as well; otherwise the model silently trains on physical GPU 0.
    """
    device = torch.device(spec)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def _restore_trainer_device(trainer, device: torch.device) -> None:
    """Undo nnUNetTrainer's ``cuda:0`` rewrite before ``initialize()``."""
    trainer.device = device
    init_kwargs = getattr(trainer, 'my_init_kwargs', None)
    if isinstance(init_kwargs, dict):
        # Keep checkpoint provenance consistent with the actual device.
        init_kwargs['device'] = device


class NNWNetAdapter(BaselineAdapter):
    id = 'nnwnet'
    display_name = 'nnWNet'
    official_source = 'nnWNet official nnU-Net trainer plugin'
    official_entrypoint = 'nnUNetv2_train ... -tr nnUNetTrainer_WNet2D'
    input_size = 512
    official_lr = 1e-2
    official_weight_decay = 3e-5
    train_batch_size = 1
    selection = 'dice'
    custom_pipeline = True
    deviations = (
        'One pooled 18-image train / 6-image val model replaces the official '
        'per-dataset 5-fold CV; training uses fold 0 only via a fixed '
        'splits_final.json.',
        'nnU-Net planning and normalization are fitted on the 18 train cases '
        'only; the 6 val cases are staged after planning and preprocessed '
        'with the frozen plans. Val images therefore do not affect planner '
        'choices or normalization statistics.',
        'The official 2D trainer recipe (SGD 1e-2, AMP, deep supervision, '
        'sliding-window/TTA inference) is kept; the official 200-epoch '
        'default is replaced by the shared 800-epoch budget. Every '
        '--nnwnet-val-interval epochs the adapter runs full-image nnU-Net '
        'validation and shared Dice/IoU/clDice/HD95; checkpoint selection '
        'uses the shared validation Dice. nnU-Net patch-based EMA pseudo '
        'Dice is logged only under nnunet/*.',
        'The training W&B run uses the shared run_config fields and the '
        'standard val/* scalar/per-dataset schema at full-validation '
        'epochs. Off-interval epochs log train/loss, lr and nnunet/* only; '
        'the final epoch is always fully validated.',
        'The adapter seeds Python/NumPy/Torch before training, but nnU-Net '
        'may use additional internal RNGs; --seed is not a bitwise guarantee.',
        '--resume is rejected because nnU-Net continue-training is not wired '
        'into this custom pipeline; reruns re-stage and re-plan from scratch.',
        'Official predictions are 0/1 label maps; this adapter writes 0/255 '
        'PNGs at the original GT resolution for the shared evaluator.',
        'A single nnU-Net dataset needs one channel count. The default rgb '
        'policy harmonizes grayscale cases to three channels; '
        '--nnwnet-channels grayscale converts every case to one channel for '
        'all-grayscale protocols. The policy is recorded and reused at '
        'prediction; mixed 1/3-channel datasets are not supported.',
    )

    def build_model(self, args):
        raise RuntimeError('nnWNet uses the nnU-Net custom pipeline')

    def prepare_train_batch(self, samples, rng):
        raise RuntimeError('nnWNet uses the nnU-Net custom pipeline')

    def train_step(self, model, batch):
        raise RuntimeError('nnWNet uses the nnU-Net custom pipeline')

    def prepare_eval_sample(self, sample, device):
        raise RuntimeError('nnWNet uses the nnU-Net custom pipeline')

    def predict_sample(self, model, prepared, sample):
        raise RuntimeError('nnWNet uses the nnU-Net custom pipeline')

    def add_args(self, parser):
        parser.add_argument(
            '--nnwnet-channels', choices=['rgb', 'grayscale'],
            default='rgb',
            help='channel policy for staged nnU-Net images: rgb keeps color '
                 'and replicates grayscale sources to 3 channels; grayscale '
                 'converts every case to one channel for all-grayscale '
                 'protocols. A single nnU-Net dataset cannot mix 1/3 '
                 'channels.')
        parser.add_argument(
            '--nnwnet-val-interval', type=int, default=1,
            help='run full-image shared validation and val-Dice checkpoint '
                 'selection every N epochs (default 1 for fully unified '
                 'curves; increase to reduce validation cost); the final '
                 'epoch is always validated')

    # ---- staging ------------------------------------------------------
    @staticmethod
    def _stage_image(path: Path, destination: Path, channels: str) -> None:
        if channels == 'grayscale':
            with Image.open(path) as image:
                image.convert('L').save(destination)
        else:
            load_rgb(path).save(destination)

    @staticmethod
    def _case_id(stem: str, used: set[str]) -> str:
        base = re.sub(r'[^A-Za-z0-9_]', '_', stem)
        case = base
        index = 1
        while case in used:
            case = f'{base}_{index}'
            index += 1
        return case

    def _stage_raw(self, raw: Path, data_root: Path,
                   include_val: bool = True, channels: str = 'rgb'):
        if channels not in ('rgb', 'grayscale'):
            raise ValueError(f'unsupported channel policy: {channels!r}')
        dataset_dir = raw / DATASET_NAME
        images_tr = dataset_dir / 'imagesTr'
        labels_tr = dataset_dir / 'labelsTr'
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)

        train = load_split_samples(data_root, 'train')
        val = load_split_samples(data_root, 'val') if include_val else []
        used: set[str] = set()
        train_cases, val_cases = [], []
        for samples, cases in ((train, train_cases), (val, val_cases)):
            for sample in samples:
                case = self._case_id(sample.stem, used)
                used.add(case)
                cases.append(case)
                self._stage_image(
                    sample.image_path, images_tr / f'{case}_0000.png',
                    channels)
                mask = load_mask(sample.mask_path).astype(np.uint8)
                Image.fromarray(mask).save(labels_tr / f'{case}.png')

        channel_names = (
            {'0': 'grayscale'} if channels == 'grayscale'
            else {'0': 'R', '1': 'G', '2': 'B'})
        dataset_json = {
            'channel_names': channel_names,
            'labels': {'background': 0, 'foreground': 1},
            'numTraining': len(used),
            'file_ending': '.png',
            'overwrite_image_reader_writer': 'NaturalImage2DIO',
        }
        (dataset_dir / 'dataset.json').write_text(
            json.dumps(dataset_json, indent=2) + '\n')
        return train_cases, val_cases

    @staticmethod
    def _plan_command():
        return [
            sys.executable, '-m',
            'nnunetv2.experiment_planning.'
            'plan_and_preprocess_entrypoints',
            '-d', str(DATASET_ID), '-c', '2d',
            '--verify_dataset_integrity', '--no_pp',
        ]

    @staticmethod
    def _sync_preprocessed_dataset_json(nn_root: Path,
                                        preprocessed: Path) -> None:
        preprocessed.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            nn_root / 'raw' / DATASET_NAME / 'dataset.json',
            preprocessed / 'dataset.json')

    @staticmethod
    def _write_splits(preprocessed: Path, train_cases, val_cases):
        preprocessed.mkdir(parents=True, exist_ok=True)
        (preprocessed / 'splits_final.json').write_text(
            json.dumps([{'train': train_cases, 'val': val_cases}],
                       indent=2) + '\n')

    @staticmethod
    def _nnunet_env(nn_root: Path):
        env = os.environ.copy()
        # nnU-Net subprocesses run with cwd=workdir; relative nnUNet_* paths
        # would then be resolved twice and point at a nonexistent Dataset501.
        nn_root = Path(nn_root).resolve()
        env['nnUNet_raw'] = str(nn_root / 'raw')
        env['nnUNet_preprocessed'] = str(nn_root / 'preprocessed')
        env['nnUNet_results'] = str(nn_root / 'results')
        # The custom WNet architecture is not beneficial to compile.
        env['nnUNet_compile'] = '0'
        env['nnUNet_extTrainer'] = str(
            Path(__file__).resolve().parents[1] / 'models' / 'nnwnet')
        return env

    def _init_env(self, workdir: Path):
        nn_root = workdir / 'nnunet'
        for sub in ('raw', 'preprocessed', 'results'):
            (nn_root / sub).mkdir(parents=True, exist_ok=True)
        env = self._nnunet_env(nn_root)
        os.environ.update({
            'nnUNet_raw': env['nnUNet_raw'],
            'nnUNet_preprocessed': env['nnUNet_preprocessed'],
            'nnUNet_results': env['nnUNet_results'],
            'nnUNet_compile': env['nnUNet_compile'],
            'nnUNet_extTrainer': env['nnUNet_extTrainer'],
        })
        return nn_root, env

    # ---- official training -------------------------------------------
    @staticmethod
    def _find_checkpoint(model_dir: Path, kind: str) -> Path | None:
        """Find checkpoint_best/final with either .pth or .safetensors."""
        fold = model_dir / 'fold_0'
        for suffix in ('.pth', '.safetensors'):
            candidate = fold / f'checkpoint_{kind}{suffix}'
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _clear_stale_results(nn_root: Path) -> None:
        """Remove this dataset's previous nnU-Net results before retraining.

        nnU-Net writes ``checkpoint_best`` only when validation improves.
        Reusing a workdir without clearing results can therefore leave an old
        best checkpoint that prediction would silently pick if the new run
        fails before writing its own checkpoint.  Other dataset IDs in the
        same nnU-Net root are untouched.
        """
        shutil.rmtree(nn_root / 'results' / DATASET_NAME,
                      ignore_errors=True)

    @staticmethod
    def _clear_prediction_dirs(workdir: Path, pred_root: Path,
                               name: str) -> None:
        """Remove stale staging/output files for one prediction dataset."""
        for path in (workdir / 'predict_input' / name,
                     workdir / 'predict_output' / name,
                     pred_root / name):
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _mean(values):
        finite = [float(value) for value in values
                  if value is not None and np.isfinite(value)]
        return float(np.mean(finite)) if finite else float('nan')

    @classmethod
    def _aggregate_shared_metrics(cls, rows):
        """Aggregate per-case shared-evaluator metrics.

        ``rows`` is a sequence of ``(dataset, binary_metrics_dict)``.
        """
        metric_names = ('dice', 'iou', 'cldice', 'hd95')
        overall = {name: [] for name in metric_names}
        per_dataset = {}
        empty = []
        for dataset, values in rows:
            bucket = per_dataset.setdefault(
                dataset, {name: [] for name in metric_names})
            for name in metric_names:
                value = values.get(name)
                if value is not None and np.isfinite(value):
                    overall[name].append(float(value))
                    bucket[name].append(float(value))
            empty.append(bool(values.get('empty', False)))
        metrics = {
            name: cls._mean(overall[name]) for name in metric_names}
        metrics['empty_prediction_rate'] = (
            float(np.mean(empty)) if empty else float('nan'))
        datasets = {}
        for dataset, values in per_dataset.items():
            datasets[dataset] = {
                name: cls._mean(values[name]) for name in metric_names}
        return metrics, datasets

    def _shared_full_validation(self, trainer, val_case_samples):
        """Run nnU-Net full-image validation and the shared evaluator."""
        validation_dir = Path(trainer.output_folder) / 'validation'
        shutil.rmtree(validation_dir, ignore_errors=True)
        trainer.perform_actual_validation(save_probabilities=False)
        trainer.network.train()
        file_ending = str(
            trainer.dataset_json.get('file_ending', '.png'))
        _, val_keys = trainer.do_split()
        rows = []
        for case in val_keys:
            sample = val_case_samples.get(case)
            if sample is None:
                raise SystemExit(
                    f'nnWNet validation case {case!r} has no source sample')
            prediction_path = validation_dir / f'{case}{file_ending}'
            if not prediction_path.is_file():
                raise SystemExit(
                    f'nnWNet full validation did not write '
                    f'{prediction_path}')
            prediction = np.asarray(
                Image.open(prediction_path).convert('L')) > 0
            ground_truth = load_mask(sample.mask_path)
            if prediction.shape != ground_truth.shape:
                prediction = np.asarray(Image.fromarray(
                    prediction.astype(np.uint8) * 255).resize(
                        (ground_truth.shape[1], ground_truth.shape[0]),
                        Image.NEAREST)) > 127
            rows.append((sample.dataset,
                         binary_metrics(prediction, ground_truth)))
        return self._aggregate_shared_metrics(rows)

    def run_training(self, args):
        if args.resume:
            raise SystemExit(
                'nnWNet does not support intra-run --resume; nnU-Net '
                'continue-training is not wired into this custom pipeline')
        unsupported = []
        if int(args.accum_steps) != 1:
            unsupported.append('--accum-steps')
        if float(args.grad_clip) != 0:
            unsupported.append('--grad-clip')
        if int(args.batch_size) != 1:
            unsupported.append('--batch-size (planner-controlled)')
        if args.selection != self.selection:
            unsupported.append('--selection (EMA pseudo-Dice is fixed)')
        if int(getattr(args, 'num_workers', 0)) != 0:
            unsupported.append('--num-workers (nnU-Net loader-controlled)')
        if unsupported:
            raise SystemExit(
                'nnWNet custom pipeline does not implement: '
                + ', '.join(unsupported))
        from baseline_zoo.runtime import seed_everything
        seed_everything(args.seed)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        channels = str(getattr(args, 'nnwnet_channels', 'rgb'))
        if channels not in ('rgb', 'grayscale'):
            raise SystemExit(
                f'unsupported --nnwnet-channels: {channels!r}')
        val_interval = int(getattr(args, 'nnwnet_val_interval', 1))
        if val_interval < 1:
            raise SystemExit(
                '--nnwnet-val-interval must be >= 1')
        (workdir / 'nnunet_channels.txt').write_text(channels + '\n')
        nn_root, env = self._init_env(workdir)
        # A reused workdir must not mix old results/checkpoints, raw PNGs or
        # preprocessed arrays from a different protocol.
        self._clear_stale_results(nn_root)
        stale_pointer = workdir / 'nnunet_model_dir.txt'
        if stale_pointer.exists():
            stale_pointer.unlink()
        shutil.rmtree(nn_root / 'raw' / DATASET_NAME, ignore_errors=True)
        shutil.rmtree(nn_root / 'preprocessed' / DATASET_NAME,
                      ignore_errors=True)
        # Fit the planner/preprocessor on train only.  Staging val before
        # planning leaked the 6 validation images into normalization and
        # planner choices.
        train_cases, _ = self._stage_raw(
            nn_root / 'raw', Path(args.data_root), include_val=False,
            channels=channels)

        preprocessed = nn_root / 'preprocessed' / DATASET_NAME
        plans_path = preprocessed / 'nnUNetPlans.json'
        # Plan on the 18 train cases only; --no_pp is essential here.
        # Without it nnU-Net also preprocesses those 18 cases with its
        # default 8 workers, and the second preprocess below would discard
        # that work and rebuild all 24 cases.
        command = self._plan_command()
        print('+ ' + ' '.join(command), flush=True)
        subprocess.run(command, env=env, check=True, cwd=workdir)

        # Add val only after the plans are frozen, then preprocess it with
        # the existing plans.  nnUNetv2_preprocess does not re-plan.
        train_cases, val_cases = self._stage_raw(
            nn_root / 'raw', Path(args.data_root), include_val=True,
            channels=channels)
        val_samples = load_split_samples(Path(args.data_root), 'val')
        val_case_samples = dict(zip(val_cases, val_samples))
        # DefaultPreprocessor reads preprocessed/dataset.json, not the raw
        # copy.  The planning pass wrote the 18-case file; refresh it to the
        # 24-case raw metadata before the second preprocessing pass.
        self._sync_preprocessed_dataset_json(nn_root, preprocessed)

        preprocess_bin = Path(sys.executable).with_name('nnUNetv2_preprocess')
        if preprocess_bin.is_file():
            command = [str(preprocess_bin)]
        else:
            command = [
                sys.executable, '-c',
                'from nnunetv2.experiment_planning.'
                'plan_and_preprocess_entrypoints import preprocess_entry; '
                'preprocess_entry()',
            ]
        command += ['-d', str(DATASET_ID), '-c', '2d', '-np', '2']
        print('+ ' + ' '.join(command), flush=True)
        subprocess.run(command, env=env, check=True, cwd=workdir)
        self._write_splits(preprocessed, train_cases, val_cases)

        from batchgenerators.utilities.file_and_folder_operations import (
            load_json,
        )

        from baseline_zoo.models.nnwnet import get_trainer_class

        plans = load_json(str(plans_path))
        dataset_json = load_json(
            str(nn_root / 'raw' / DATASET_NAME / 'dataset.json'))
        plan_config = plans.get('configurations', {}).get('2d', {})
        actual_batch = int(plan_config.get('batch_size', args.batch_size))

        # nnU-Net's own CLI injects these before constructing the trainer.
        plans.setdefault('plans_name', 'nnUNetPlans')
        plans['continue_training'] = False
        trainer_class = get_trainer_class()
        device = _resolve_device(args.device)
        trainer = trainer_class(
            plans, '2d', 0, dataset_json, device=device)
        # nnUNetTrainer changes a non-DDP CUDA device to cuda:0; restore the
        # adapter-requested card before initialize() moves the network.
        _restore_trainer_device(trainer, device)
        # The shared CLI/config is the source of truth for optimizer
        # hyperparameters; configure_optimizers() runs inside initialize().
        trainer.initial_lr = float(args.lr)
        trainer.weight_decay = float(args.weight_decay)
        trainer.num_epochs = int(args.epochs)
        trainer.initialize()

        # nnU-Net has no shared-runtime epoch table.  Keep its official
        # training loop, but render the same table and W&B keys as every
        # other baseline.  nnU-Net only exposes patch-based pseudo Dice
        # during training; the shared evaluator supplies IoU/clDice/HD95
        # on the final test predictions.
        from baseline_zoo.runtime import (
            print_epoch_footer,
            print_epoch_header,
            print_epoch_row,
        )

        num_params = sum(p.numel() for p in trainer.network.parameters())
        trainable = sum(p.numel() for p in trainer.network.parameters()
                        if p.requires_grad)
        split_manifest = Path(args.data_root) / 'split_manifest.csv'
        split_sha256 = (file_sha256(split_manifest)
                        if split_manifest.is_file() else None)
        run_config = {
            'method': self.id,
            'display_name': self.display_name,
            'official_source': self.official_source,
            'official_entrypoint': self.official_entrypoint,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'train_images': len(train_cases),
            'val_images': len(val_cases),
            'epochs': args.epochs,
            'batch_size': actual_batch,
            'accum_steps': 1,
            'effective_batch_size': actual_batch,
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'input_size': list(plan_config.get('patch_size', [])),
            'adapter_config': {
                'channels': channels,
                'val_interval': val_interval,
            },
            'selection': args.selection,
            'selection_source': 'shared validation Dice (full image)',
            'seed': args.seed,
            'optimizer': type(trainer.optimizer).__name__,
            'scheduler': (type(trainer.lr_scheduler).__name__
                          if trainer.lr_scheduler is not None else None),
            'scheduler_override': 'official',
            'warmup_epochs': None,
            'split_manifest_sha256': split_sha256,
            'num_params': num_params,
            'trainable_params': trainable,
            'dataset': DATASET_NAME,
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'patch_size': list(plan_config.get('patch_size', [])),
            'channel_mode': channels,
            'val_interval': val_interval,
            'val_dice_source': (
                'shared evaluator: full-image nnU-Net validation every '
                f'{val_interval} epochs'),
        }
        run = setup_wandb(args, name=f'nnwnet-seed{args.seed}',
                          config=run_config)
        from core.console import console

        for deviation in self.deviations:
            console.print(f' [yellow]deviation:[/] {deviation}')
        print_epoch_header(self, args.epochs, args.seed, device,
                           num_params, trainable)
        print(f'note: full-image shared validation every {val_interval} '
              'epochs; off-interval rows show only train/loss. nnU-Net '
              'patch pseudo Dice is logged under nnunet/*.', flush=True)
        benchmark_ckpt = (
            Path(trainer.output_folder) / 'checkpoint_benchmark_best.pth')

        from types import MethodType

        original_on_epoch_end = trainer.on_epoch_end
        best_shared = -float('inf')
        best_shared_epoch = 0

        def _last_scalar(key):
            try:
                value = trainer.logger.get_value(key, step=-1)
            except Exception:
                return None
            if value is None:
                return None
            array = np.asarray(value)
            if array.size == 0:
                return None
            return float(array.reshape(-1)[-1])

        def _shared_scalars(shared_metrics, shared_per_dataset):
            scalars = {
                'val/selection': shared_metrics['dice'],
                'val/dice': shared_metrics['dice'],
                'val/iou': shared_metrics['iou'],
                'val/cldice': shared_metrics['cldice'],
                'val/hd95': shared_metrics['hd95'],
                'val/empty_prediction_rate': shared_metrics[
                    'empty_prediction_rate'],
            }
            for dataset, values in shared_per_dataset.items():
                for metric in ('dice', 'iou', 'cldice', 'hd95'):
                    scalars[f'val/{dataset}/{metric}'] = values[metric]
            return scalars

        def _run_shared_validation(epoch):
            nonlocal best_shared, best_shared_epoch
            shared_metrics, shared_per_dataset = (
                self._shared_full_validation(trainer, val_case_samples))
            dice = shared_metrics['dice']
            if np.isfinite(dice) and dice > best_shared:
                best_shared = dice
                best_shared_epoch = epoch
                trainer.save_checkpoint(str(benchmark_ckpt))
                print(f'  shared-val best: epoch {epoch} dice={dice:.4f} '
                      f'-> {benchmark_ckpt.name}', flush=True)
            return shared_metrics, shared_per_dataset

        def _on_epoch_end(trainer_self):
            epoch = int(trainer_self.current_epoch) + 1
            train_loss = _last_scalar('train_losses')
            pseudo_loss = _last_scalar('val_losses')
            pseudo_dice = _last_scalar('mean_fg_dice')
            ema_dice = _last_scalar('ema_fg_dice')
            lr = float(trainer_self.optimizer.param_groups[0]['lr'])
            start = _last_scalar('epoch_start_timestamps')
            elapsed = (time.time() - start) if start is not None else 0.0

            do_full_val = (epoch % val_interval == 0
                           or epoch == int(args.epochs))
            shared_metrics = None
            shared_per_dataset = None
            if do_full_val:
                shared_metrics, shared_per_dataset = (
                    _run_shared_validation(epoch))
                table_metrics = shared_metrics
                improved = (np.isfinite(shared_metrics['dice'])
                            and shared_metrics['dice'] == best_shared)
            else:
                table_metrics = {
                    'dice': float('nan'),
                    'iou': float('nan'),
                    'cldice': float('nan'),
                    'hd95': float('nan'),
                }
                improved = False

            print_epoch_row(
                epoch,
                float('nan') if train_loss is None else train_loss,
                table_metrics, lr, elapsed, improved)

            scalars = {
                'epoch': epoch,
                'train/loss': train_loss,
                'lr': lr,
                'nnunet/patch_pseudo_dice': pseudo_dice,
                'nnunet/ema_patch_pseudo_dice': ema_dice,
                'nnunet/val_pseudo_loss': pseudo_loss,
            }
            scalars = {key: value for key, value in scalars.items()
                       if value is not None}
            if shared_metrics is not None:
                scalars.update(_shared_scalars(
                    shared_metrics, shared_per_dataset))
            for key in ('train_losses', 'val_losses', 'ema_fg_dice'):
                value = _last_scalar(key)
                if value is not None:
                    scalars[f'nnunet/{key}'] = value
            try:
                dice_values = trainer_self.logger.get_value(
                    'dice_per_class_or_region', step=-1)
                if dice_values is not None:
                    values = np.asarray(dice_values).reshape(-1)
                    if values.size:
                        scalars['nnunet/mean_fg_dice'] = float(
                            np.mean(values))
            except Exception:
                pass
            if run is not None:
                log_scalars(run, scalars)
            original_on_epoch_end()

        trainer.on_epoch_end = MethodType(_on_epoch_end, trainer)

        try:
            trainer.run_training()
        except BaseException:
            finish_run(run)
            raise
        print_epoch_footer()

        model_dir = Path(trainer.output_folder_base)
        logs = getattr(trainer.logger, 'my_fantastic_logging', {})
        if run is not None:
            scalars = {}
            for key in ('train_losses', 'val_losses', 'ema_fg_dice'):
                values = logs.get(key)
                if values:
                    scalars[f'nnunet/{key}'] = float(values[-1])
            if best_shared_epoch:
                scalars['val/final_best_epoch'] = best_shared_epoch
                scalars['val/final_best_dice'] = best_shared
            log_scalars(run, scalars)
            if getattr(args, 'wandb_log_artifact', False):
                best = self._find_checkpoint(
                    model_dir, 'benchmark_best')
                if best is None:
                    best = self._find_checkpoint(model_dir, 'best')
                if best is not None:
                    log_artifact(
                        run, best, name=f'nnwnet-seed{args.seed}',
                        artifact_type='model')
        finish_run(run)

        (workdir / 'nnunet_model_dir.txt').write_text(str(model_dir) + '\n')
        (model_dir / 'benchmark_channels.txt').write_text(channels + '\n')
        (model_dir / 'benchmark_run_config.json').write_text(
            json.dumps(run_config, indent=2) + '\n')
        return model_dir

    # ---- official inference ------------------------------------------
    def run_prediction(self, args, checkpoint: Path):
        workdir = Path(args.workdir)
        nn_root, env = self._init_env(workdir)
        os.environ.update({key: env[key] for key in (
            'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')})

        if args.checkpoint is not None:
            candidate = Path(args.checkpoint)
            if candidate.is_file() and candidate.suffix == '.txt':
                candidate = Path(candidate.read_text().strip())
            model_dir = candidate
        elif (workdir / 'nnunet_model_dir.txt').is_file():
            model_dir = Path(
                (workdir / 'nnunet_model_dir.txt').read_text().strip())
        else:
            model_dir = Path(checkpoint)
        if not model_dir.is_dir():
            raise SystemExit(f'missing nnU-Net model folder: {model_dir}')

        channels = str(getattr(args, 'nnwnet_channels', 'rgb'))
        for sidecar in (workdir / 'nnunet_channels.txt',
                        model_dir / 'benchmark_channels.txt'):
            if sidecar.is_file():
                channels = sidecar.read_text().strip() or channels
                break
        if channels not in ('rgb', 'grayscale'):
            raise SystemExit(
                f'unsupported nnU-Net channel policy: {channels!r}')

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = _resolve_device(args.device)
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=False,
            device=device, verbose=False,
            allow_tqdm=True)
        resolved = self._find_checkpoint(model_dir, 'benchmark_best')
        if resolved is None:
            resolved = self._find_checkpoint(model_dir, 'best')
        if resolved is None:
            resolved = self._find_checkpoint(model_dir, 'final')
        if resolved is None:
            raise SystemExit(
                f'no checkpoint_best/final (.pth or .safetensors) under '
                f'{model_dir / "fold_0"}')
        checkpoint_name = resolved.name
        if resolved.suffix != '.pth':
            raise SystemExit(
                f'nnU-Net runtime found {resolved.name}, but the installed '
                'nnUNetPredictor expects .pth checkpoints; install the '
                'declared nnunetv2 2.8.1 runtime instead of the vendored '
                'safetensors fork')
        predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(0,), checkpoint_name=checkpoint_name)

        pred_root = Path(args.pred_root)
        pred_root.mkdir(parents=True, exist_ok=True)
        stale_meta = pred_root / '.prediction_meta.json'
        if stale_meta.is_file():
            stale_meta.unlink()
        for spec in args.test_datasets:
            name, sep, root = spec.partition(':')
            if not sep:
                raise SystemExit(f'bad test spec: {spec!r}')
            samples = RawPairDataset(Path(root)).samples
            self._clear_prediction_dirs(workdir, pred_root, name)
            input_dir = workdir / 'predict_input' / name
            output_dir = workdir / 'predict_output' / name
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            inputs, outputs = [], []
            for sample in samples:
                staged = input_dir / f'{sample.stem}_0000.png'
                self._stage_image(sample.image_path, staged, channels)
                inputs.append([str(staged)])
                # nnU-Net appends the dataset file ending to this prefix.
                outputs.append(str(output_dir / sample.stem))
            print(f'{name}: predicting {len(samples)} images -> '
                  f'{pred_root / name}', flush=True)
            predictor.predict_from_files(
                inputs, outputs, save_probabilities=False, overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1)
            out_dir = pred_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for sample, output in zip(samples, outputs):
                label = np.asarray(Image.open(output + '.png').convert('L'))
                pred = label > 0
                gt = load_mask(sample.mask_path)
                if pred.shape != gt.shape:
                    pred = np.asarray(Image.fromarray(
                        pred.astype(np.uint8) * 255).resize(
                            (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
                Image.fromarray((pred.astype(np.uint8) * 255)).save(
                    out_dir / f'{sample.stem}.png')

        split_manifest = Path(args.data_root) / 'split_manifest.csv'
        meta = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'checkpoint': str(model_dir),
            'checkpoint_name': checkpoint_name,
            'channel_mode': channels,
            'split_manifest_sha256': (file_sha256(split_manifest)
                                      if split_manifest.is_file() else None),
            'datasets': [spec.split(':', 1)[0]
                         for spec in args.test_datasets],
        }
        (pred_root / '.prediction_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n')
