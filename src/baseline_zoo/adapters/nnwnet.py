"""nnWNet baseline running the official nnU-Net trainer pipeline.

The trainer plugin is copied into ``baseline_zoo.models.nnwnet`` and needs
``nnunetv2==2.5.1``.  Data are staged into the official nnU-Net raw layout
(24 protocol cases: 18 train + 6 val), a fixed one-fold split is written,
and training/prediction use nnU-Net's own planner, trainer and predictor.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline_zoo.data import RawPairDataset, load_mask, load_rgb
from baseline_zoo.runtime import BaselineAdapter
from core.wandb_utils import (
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

DATASET_ID = 501
DATASET_NAME = f'Dataset{DATASET_ID}_Curvilinear3Shot'


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
        'nnU-Net planning sees all 24 staged cases (train+val), so '
        'normalization statistics and planner choices include the 6 val '
        'cases. This is a soft protocol deviation of the official pipeline.',
        'The official 2D trainer defaults are kept (200 epochs, SGD 1e-2, '
        'AMP, deep supervision, sliding-window/TTA inference); checkpoint '
        'selection is nnU-Net EMA pseudo-Dice on the val split.',
        'Official predictions are 0/1 label maps; this adapter writes 0/255 '
        'PNGs at the original GT resolution for the shared evaluator.',
        'All 24 cases are converted to RGB PNG for a single nnU-Net file '
        'ending; the official natural-image reader applies RGB 0-1 '
        'normalization.',
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

    def _stage_raw(self, raw: Path, data_root: Path):
        dataset_dir = raw / DATASET_NAME
        images_tr = dataset_dir / 'imagesTr'
        labels_tr = dataset_dir / 'labelsTr'
        images_tr.mkdir(parents=True, exist_ok=True)
        labels_tr.mkdir(parents=True, exist_ok=True)

        train = RawPairDataset(data_root / 'train').samples
        val = RawPairDataset(data_root / 'val').samples
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
    def _write_splits(preprocessed: Path, train_cases, val_cases):
        preprocessed.mkdir(parents=True, exist_ok=True)
        (preprocessed / 'splits_final.json').write_text(
            json.dumps([{'train': train_cases, 'val': val_cases}],
                       indent=2) + '\n')

    @staticmethod
    def _nnunet_env(nn_root: Path):
        env = os.environ.copy()
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
    def run_training(self, args):
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        nn_root, env = self._init_env(workdir)
        train_cases, val_cases = self._stage_raw(
            nn_root / 'raw', Path(args.data_root))

        preprocessed = nn_root / 'preprocessed' / DATASET_NAME
        plans_path = preprocessed / 'nnUNetPlans.json'
        if not plans_path.is_file():
            command = [
                sys.executable, '-m',
                'nnunetv2.experiment_planning.'
                'plan_and_preprocess_entrypoints',
                '-d', str(DATASET_ID), '-c', '2d',
                '--verify_dataset_integrity',
            ]
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
        run_config = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'dataset': DATASET_NAME,
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'epochs': args.epochs,
            'seed': args.seed,
            'selection': 'nnU-Net EMA pseudo-Dice (official)',
        }
        run = setup_wandb(args, name=f'nnwnet-seed{args.seed}',
                          config=run_config)

        # nnU-Net's own CLI injects these before constructing the trainer.
        plans.setdefault('plans_name', 'nnUNetPlans')
        plans['continue_training'] = False
        trainer_class = get_trainer_class()
        trainer = trainer_class(
            plans, '2d', 0, dataset_json, device=torch.device(args.device))
        trainer.num_epochs = args.epochs
        trainer.initialize()
        trainer.run_training()

        logs = getattr(trainer.logger, 'my_fantastic_logging', {})
        if run is not None:
            scalars = {}
            for key in ('train_losses', 'val_losses', 'ema_fg_dice'):
                values = logs.get(key)
                if values:
                    scalars[f'nnunet/{key}'] = float(values[-1])
            log_scalars(run, scalars)
            best = Path(trainer.output_folder) / 'checkpoint_best.pth'
            if best.is_file():
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

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=False,
            device=torch.device(args.device), verbose=False,
            allow_tqdm=True)
        checkpoint_name = 'checkpoint_best.pth'
        if not (model_dir / 'fold_0' / checkpoint_name).is_file():
            checkpoint_name = 'checkpoint_final.pth'
        predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(0,), checkpoint_name=checkpoint_name)

        pred_root = Path(args.pred_root)
        pred_root.mkdir(parents=True, exist_ok=True)
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

        meta = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'checkpoint': str(model_dir),
            'checkpoint_name': checkpoint_name,
            'datasets': [spec.split(':', 1)[0]
                         for spec in args.test_datasets],
        }
        (pred_root / '.prediction_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n')
