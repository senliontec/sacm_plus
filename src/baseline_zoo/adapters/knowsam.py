"""KnowSAM baseline with the official semi-supervised training loop.

Official reference: vendored ``baselines/knowsam``.  The SAM adapter, UNet,
VNet, fusion/discriminator, losses, transforms, two-stream sampler and
trainer are copied into ``baseline_zoo.models.knowsam``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from baseline_zoo.data import RawPairDataset, load_mask, load_rgb
from baseline_zoo.models.knowsam import KnowSAM, Trainer
from baseline_zoo.models.knowsam.dataset import build_Dataset
from baseline_zoo.models.knowsam.sampler import TwoStreamBatchSampler
from baseline_zoo.models.knowsam.transforms import (
    build_transforms,
    build_weak_strong_transforms,
)
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter
from core.wandb_utils import (
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

SOURCE_DOMAINS = ('DRIVE', 'CHASEDB1', 'DCA1', 'CREMI', 'CrackTree200', 'CORN')


class KnowSAMAdapter(BaselineAdapter):
    id = 'knowsam'
    display_name = 'KnowSAM'
    official_source = 'KnowSAM (self-contained copy)'
    official_entrypoint = 'upstream train_semi_SAM.py'
    input_size = 256
    official_epochs = 50000
    official_lr = 1e-4
    official_weight_decay = 1e-4
    train_batch_size = 6
    selection = 'dice'
    custom_pipeline = True
    deviations = (
        'The official semi-supervised sampler uses 12 labeled + 12 unlabeled '
        'images; the 3-shot protocol supplies 18 labeled images, so the '
        'adapter uses a configurable 3+3 batch (default total 6).',
        'Unlabeled data are drawn from the six seen-domain full training '
        'pools, excluding the 18 labeled and 6 val images (capped by '
        '--unlabeled-limit). Upstream uses per-dataset unlabeled pools with '
        'patients_to_slices.',
        'The official 50,000-iteration budget, consistency/MixUp schedule, '
        'losses, optimizers and SGDL validation selection are preserved; '
        'the seed is the protocol seed.',
        'ViT-B is used as in the official KnowSAM code; the paper states the '
        'shared ViT-L backbone, but KnowSAM ships a ViT-B SAM adapter.',
        'Predictions are restored to the original GT resolution; upstream '
        'reports metrics at 256x256. The shipped prediction.py '
        '(bilinear=True) is incompatible with the trainer checkpoint '
        '(bilinear=False), so this port uses the trainer architecture.',
    )

    def build_model(self, args):
        raise RuntimeError('KnowSAM uses the custom semi-supervised pipeline')

    def prepare_train_batch(self, samples, rng):
        raise RuntimeError('KnowSAM uses the custom semi-supervised pipeline')

    def train_step(self, model, batch):
        raise RuntimeError('KnowSAM uses the custom semi-supervised pipeline')

    def prepare_eval_sample(self, sample, device):
        raise RuntimeError('KnowSAM uses the custom semi-supervised pipeline')

    def predict_sample(self, model, prepared, sample):
        raise RuntimeError('KnowSAM uses the custom semi-supervised pipeline')

    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth')
        parser.add_argument('--max-iterations', type=int, default=50000)
        parser.add_argument('--val-interval', type=int, default=200)
        parser.add_argument('--unlabeled-limit', type=int, default=384)
        parser.add_argument('--labeled-batch', type=int, default=3)

    # ---- staging ------------------------------------------------------
    @staticmethod
    def _save_pair(folder: Path, sample, stem: str):
        image_dir = folder / 'image'
        mask_dir = folder / 'mask'
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        load_rgb(sample.image_path).save(image_dir / f'{stem}_image.png')
        mask = load_mask(sample.mask_path)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            mask_dir / f'{stem}_mask.png')

    def _stage(self, root: Path, data_root: Path, unlabeled_limit: int):
        labeled = RawPairDataset(data_root / 'train').samples
        val = RawPairDataset(data_root / 'val').samples
        excluded = {sample.stem for sample in (*labeled, *val)}
        for sample in labeled:
            self._save_pair(root / 'labeled', sample, sample.stem)
        for sample in val:
            self._save_pair(root / 'val', sample, sample.stem)

        count = 0
        for domain in SOURCE_DOMAINS:
            domain_root = REPO_ROOT / 'datasets' / 'seen' / domain
            if not domain_root.is_dir():
                continue
            for sample in RawPairDataset(domain_root).samples:
                if sample.stem in excluded:
                    continue
                self._save_pair(root / 'unlabeled', sample,
                                f'{domain}_{sample.stem}')
                count += 1
                if count >= unlabeled_limit:
                    break
            if count >= unlabeled_limit:
                break
        if count == 0:
            raise SystemExit('no unlabeled images found for KnowSAM')
        return len(labeled), count, len(val)

    @staticmethod
    def _args(args, root: Path, labeled_batch: int):
        return SimpleNamespace(
            data_path=str(root), dataset='', num_classes=2, in_channels=3,
            lr=args.lr, UNet_lr=0.01, VNet_lr=0.01,
            image_size=256, point_nums=5, box_nums=1, mod='sam_adpt',
            model_type='vit_b', thd=False,
            batch_size=args.batch_size, labeled_bs=labeled_batch,
            seed=args.seed, mixed_iterations=12000,
            max_iterations=args.max_iterations, n_fold=1,
            consistency=0.1, consistency_rampup=200.0, device='cuda',
            multimask=False, encoder_adapter=True,
            sam_checkpoint=str(args.sam_checkpoint),
        )

    # ---- training -----------------------------------------------------
    def run_training(self, args):
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        root = workdir / 'knowsam_data'
        n_labeled, n_unlabeled, n_val = self._stage(
            root, Path(args.data_root), args.unlabeled_limit)
        if args.labeled_batch * 2 > args.batch_size:
            raise SystemExit('--labeled-batch must be <= batch-size / 2')

        torch.cuda.set_device(torch.device(args.device))
        kargs = self._args(args, root, args.labeled_batch)
        run_config = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'labeled': n_labeled, 'unlabeled': n_unlabeled, 'val': n_val,
            'batch_size': args.batch_size,
            'labeled_batch': args.labeled_batch,
            'max_iterations': args.max_iterations,
            'seed': args.seed,
            'selection': 'SGDL validation Dice (official)',
        }
        run = setup_wandb(args, name=f'knowsam-seed{args.seed}',
                          config=run_config)

        trainer = Trainer(kargs)
        transforms = build_weak_strong_transforms(kargs)
        train_set = build_Dataset(kargs, str(root), 'train_semi', transforms)
        val_set = build_Dataset(kargs, str(root), 'val',
                                transforms['valid_test'])
        labeled_count = len(train_set.sample_list_labeled)
        sampler = TwoStreamBatchSampler(
            list(range(labeled_count)),
            list(range(labeled_count, len(train_set))),
            args.batch_size, args.batch_size - args.labeled_batch)
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0,
                            pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=0)

        iteration = 0
        while iteration < args.max_iterations:
            for batch in loader:
                images = batch['image'].to(args.device)
                labels = batch['label'].to(args.device)
                trainer.train(images, labels, iteration)
                iteration += 1
                if iteration % args.val_interval == 0:
                    trainer.val(val_loader, str(workdir), iteration)
                    if run is not None:
                        log_scalars(run, {
                            'iteration': iteration,
                            'val/SGDL_best_dice':
                                trainer.best_performance_SGDL,
                            'val/SAM_best_dice':
                                trainer.best_performance_sam,
                        })
                if iteration >= args.max_iterations:
                    break

        best = workdir / 'SGDL_best_model.pth'
        if not best.is_file():
            torch.save(trainer.SGDL.state_dict(), best)
        if run is not None:
            log_scalars(run, {
                'final/SGDL_best_dice': trainer.best_performance_SGDL,
                'final/SAM_best_dice': trainer.best_performance_sam,
            })
            log_artifact(run, best, name=f'knowsam-seed{args.seed}',
                         artifact_type='model')
        finish_run(run)
        (workdir / 'knowsam_model.txt').write_text(str(best) + '\n')
        return best

    # ---- inference ----------------------------------------------------
    def run_prediction(self, args, checkpoint: Path):
        workdir = Path(args.workdir)
        root = workdir / 'knowsam_data'
        if not root.is_dir():
            root = workdir
        torch.cuda.set_device(torch.device(args.device))
        kargs = self._args(args, root, args.labeled_batch)
        model = KnowSAM(kargs, bilinear=False).to(args.device).eval()
        if args.checkpoint is None and (workdir / 'knowsam_model.txt').is_file():
            checkpoint = Path(
                (workdir / 'knowsam_model.txt').read_text().strip())
        payload = torch.load(checkpoint, map_location='cpu',
                             weights_only=False)
        state = payload.get('model_state_dict', payload)
        model.load_state_dict(state)
        transform = build_transforms(kargs)['valid_test']

        pred_root = Path(args.pred_root)
        pred_root.mkdir(parents=True, exist_ok=True)
        for spec in args.test_datasets:
            name, sep, dataset_root = spec.partition(':')
            samples = RawPairDataset(Path(dataset_root)).samples
            out_dir = pred_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f'{name}: predicting {len(samples)} images -> {out_dir}',
                  flush=True)
            for sample in samples:
                image = np.asarray(load_rgb(sample.image_path),
                                   dtype=np.float32) / 255.0
                dummy = np.zeros(image.shape[:2], dtype=np.uint8)
                augmented = transform(image=image, mask=dummy)
                tensor = torch.from_numpy(
                    augmented['image'].transpose(2, 0, 1).copy()
                ).float().unsqueeze(0).to(args.device)
                with torch.no_grad():
                    outputs = model(tensor)
                    fusion = outputs[-1]
                    prob = torch.softmax(fusion, dim=1)[0, 1]
                gt = load_mask(sample.mask_path)
                prob = F.interpolate(
                    prob[None, None], size=(gt.shape[0], gt.shape[1]),
                    mode='bilinear', align_corners=False)[0, 0]
                pred = prob.cpu().numpy() > 0.5
                Image.fromarray((pred.astype(np.uint8) * 255)).save(
                    out_dir / f'{sample.stem}.png')

        meta = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'checkpoint': str(checkpoint),
            'datasets': [spec.split(':', 1)[0]
                         for spec in args.test_datasets],
        }
        (pred_root / '.prediction_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n')
