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
from baseline_zoo.runtime import BaselineAdapter, file_sha256
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
    official_epochs = 200
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
        'The official 2D trainer defaults are kept (200 epochs, SGD 1e-2, '
        'AMP, deep supervision, sliding-window/TTA inference); checkpoint '
        'selection is nnU-Net EMA pseudo-Dice on the val split.',
        'The adapter seeds Python/NumPy/Torch before training, but nnU-Net '
        'may use additional internal RNGs; --seed is not a bitwise guarantee.',
        '--resume is rejected because nnU-Net continue-training is not wired '
        'into this custom pipeline; reruns re-stage and re-plan from scratch.',
        'Official predictions are 0/1 label maps; this adapter writes 0/255 '
        'PNGs at the original GT resolution for the shared evaluator.',
        'All 24 cases are converted to RGB PNG for a single nnU-Net file '
        'ending; the official natural-image reader supplies 0-255 RGB and '
        'nnU-Net applies its own preprocessing/normalization.',
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

    # ---- staging ------------------------------------------------------
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
                   include_val: bool = True):
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
                load_rgb(sample.image_path).save(
                    images_tr / f'{case}_0000.png')
                mask = load_mask(sample.mask_path).astype(np.uint8)
                Image.fromarray(mask).save(labels_tr / f'{case}.png')

        dataset_json = {
            'channel_names': {'0': 'R', '1': 'G', '2': 'B'},
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

    def run_training(self, args):
        if args.resume:
            raise SystemExit(
                'nnWNet does not support intra-run --resume; nnU-Net '
                'continue-training is not wired into this custom pipeline')
        from baseline_zoo.runtime import seed_everything
        seed_everything(args.seed)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        nn_root, env = self._init_env(workdir)
        # A reused workdir must not mix raw PNGs/preprocessed arrays from a
        # different protocol; re-stage and re-plan deterministically.
        shutil.rmtree(nn_root / 'raw' / DATASET_NAME, ignore_errors=True)
        shutil.rmtree(nn_root / 'preprocessed' / DATASET_NAME,
                      ignore_errors=True)
        # Fit the planner/preprocessor on train only.  Staging val before
        # planning leaked the 6 validation images into normalization and
        # planner choices.
        train_cases, _ = self._stage_raw(
            nn_root / 'raw', Path(args.data_root), include_val=False)

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
            nn_root / 'raw', Path(args.data_root), include_val=True)
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
        run_config = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'dataset': DATASET_NAME,
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'epochs': args.epochs,
            'batch_size': actual_batch,
            'patch_size': list(plan_config.get('patch_size', [])),
            'effective_batch_size': actual_batch,
            'seed': args.seed,
            'selection': 'nnU-Net EMA pseudo-Dice (official)',
            'val_dice_source': 'nnU-Net mean_fg_dice (patch-based pseudo Dice); '
                               'IoU/clDice/HD95 are computed by the final '
                               'evaluator',
        }
        run = setup_wandb(args, name=f'nnwnet-seed{args.seed}',
                          config=run_config)

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
        trainer.num_epochs = args.epochs
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
        from core.console import console

        for deviation in self.deviations:
            console.print(f' [yellow]deviation:[/] {deviation}')
        print_epoch_header(self, args.epochs, args.seed, device,
                           num_params, trainable)
        print('note: Dice is nnU-Net pseudo Dice (patch-based); '
              'IoU/clDice/HD95 are computed by the final evaluator.',
              flush=True)

        from types import MethodType

        original_on_epoch_end = trainer.on_epoch_end

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

        def _on_epoch_end(self):
            epoch = int(self.current_epoch) + 1
            train_loss = _last_scalar('train_losses')
            val_loss = _last_scalar('val_losses')
            dice = _last_scalar('mean_fg_dice')
            ema = _last_scalar('ema_fg_dice')
            lr = float(self.optimizer.param_groups[0]['lr'])
            start = _last_scalar('epoch_start_timestamps')
            elapsed = (time.time() - start) if start is not None else 0.0
            previous_best = getattr(self, '_best_ema', None)
            improved = (ema is not None
                        and (previous_best is None or ema > previous_best))
            table_metrics = {
                'dice': float('nan') if dice is None else dice,
                'iou': float('nan'),
                'cldice': float('nan'),
                'hd95': float('nan'),
            }
            print_epoch_row(
                epoch,
                float('nan') if train_loss is None else train_loss,
                table_metrics, lr, elapsed, improved)

            scalars = {
                'epoch': epoch,
                'train/loss': train_loss,
                'lr': lr,
                'val/selection': ema if ema is not None else dice,
                'val/dice': dice,
                'val/loss': val_loss,
                'val/pseudo_dice': dice,
                'val/ema_pseudo_dice': ema,
            }
            scalars = {key: value for key, value in scalars.items()
                       if value is not None}
            for key in ('train_losses', 'val_losses', 'ema_fg_dice'):
                value = _last_scalar(key)
                if value is not None:
                    scalars[f'nnunet/{key}'] = value
            try:
                dice_values = self.logger.get_value(
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

        trainer.run_training()
        print_epoch_footer()

        logs = getattr(trainer.logger, 'my_fantastic_logging', {})
        if run is not None:
            scalars = {}
            for key in ('train_losses', 'val_losses', 'ema_fg_dice'):
                values = logs.get(key)
                if values:
                    scalars[f'nnunet/{key}'] = float(values[-1])
            log_scalars(run, scalars)
            best = self._find_checkpoint(
                Path(trainer.output_folder_base), 'best')
            if best is not None:
                log_artifact(run, best, name=f'nnwnet-seed{args.seed}',
                             artifact_type='model')
        finish_run(run)

        model_dir = Path(trainer.output_folder_base)
        (workdir / 'nnunet_model_dir.txt').write_text(str(model_dir) + '\n')
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

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = _resolve_device(args.device)
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=False,
            device=device, verbose=False,
            allow_tqdm=True)
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
            input_dir = workdir / 'predict_input' / name
            output_dir = workdir / 'predict_output' / name
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            inputs, outputs = [], []
            for sample in samples:
                staged = input_dir / f'{sample.stem}_0000.png'
                load_rgb(sample.image_path).save(staged)
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
            'split_manifest_sha256': (file_sha256(split_manifest)
                                      if split_manifest.is_file() else None),
            'datasets': [spec.split(':', 1)[0]
                         for spec in args.test_datasets],
        }
        (pred_root / '.prediction_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n')
