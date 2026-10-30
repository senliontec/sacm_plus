"""KnowSAM baseline with the official semi-supervised training loop.

Official reference: vendored ``baselines/knowsam``.  The SAM adapter, UNet,
VNet, fusion/discriminator, losses, transforms, two-stream sampler and
trainer are copied into ``baseline_zoo.models.knowsam``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from baseline_zoo.data import RawPairDataset, load_mask, load_rgb, load_split_samples
from baseline_zoo.models.knowsam import KnowSAM, Trainer
from baseline_zoo.models.knowsam.dataset import build_Dataset
from baseline_zoo.models.knowsam.sampler import TwoStreamBatchSampler
from baseline_zoo.models.knowsam.transforms import (
    build_transforms,
    build_weak_strong_transforms,
)
from baseline_zoo.runtime import (
    REPO_ROOT,
    BaselineAdapter,
    binary_metrics,
    file_sha256,
)
from core.wandb_utils import (
    finish_run,
    log_artifact,
    log_scalars,
    setup_wandb,
)

SOURCE_DOMAINS = ('DRIVE', 'CHASEDB1', 'DCA1', 'CREMI', 'CrackTree200', 'CORN')


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Official ``getLargestCC``: keep only the largest foreground component."""
    from skimage.measure import label

    labels = label(mask)
    if labels.max() == 0:
        return mask
    largest = int(np.argmax(np.bincount(labels.flat)[1:])) + 1
    return labels == largest


def _postprocess_prediction(pred: np.ndarray, largest_cc: bool) -> np.ndarray:
    """Apply the optional ACDC largest-component rule.

    The generic official prediction path leaves ``Largest = False``; only
    the ACDC-specific script keeps the largest component.  Curvilinear GT
    masks are often multi-component, so making this rule unconditional caps
    their Dice score.
    """
    if largest_cc:
        return _largest_cc(pred)
    return pred


_SHARED_METRICS = ('dice', 'iou', 'cldice', 'hd95')


def _mean_metric(values):
    """NaN-aware mean matching ``runtime._mean`` for shared metrics."""
    finite = [float(value) for value in values if value == value]
    return float(np.mean(finite)) if finite else float('nan')


class KnowSAMAdapter(BaselineAdapter):
    id = 'knowsam'
    display_name = 'KnowSAM'
    official_source = 'KnowSAM (self-contained copy)'
    official_entrypoint = 'upstream train_semi_SAM.py'
    input_size = 256
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
        'pools. Every source stem present in split_manifest.csv (train, '
        'validation and test) is excluded, so the semi-supervised pool '
        'cannot see any protocol test image; capped by --unlabeled-limit. '
        'Upstream uses per-dataset unlabeled pools with patients_to_slices. '
        'After excluding all protocol images, only CHASEDB1 (4), CREMI (246) '
        'and CORN (134) contribute non-test unlabeled images; DRIVE, DCA1 and '
        'CrackTree200 have none because their full pools are in the test '
        'manifest. The 384-image cap is therefore dominated by CREMI/CORN, '
        'not six-domain coverage; this limitation must be reported.',
        'Official DataLoader workers are 2 (labeled) and 1 (unlabeled); this '
        'port uses workers=0 and the shared seeded per-epoch RNG, so '
        'augmentation/data order is not bitwise identical.',
        'The custom loop uses --epochs as the iteration budget unless '
        '--max-iterations is passed, seeds Python/NumPy/Torch with the '
        'protocol seed, and requires labeled_batch == batch_size/2. It does '
        'not implement nnU-Net-style intra-run resume.',
        'The official 50,000-iteration budget, consistency/MixUp schedule, '
        'losses, optimizers and SGDL validation selection are preserved; '
        'the seed is the protocol seed.',
        'ViT-B is used as in the official KnowSAM code; the paper states the '
        'shared ViT-L backbone, but KnowSAM ships a ViT-B SAM adapter.',
        'Upstream postprocessing takes argmax over the 256x256 logits and '
        'zooms with order=0. The generic official prediction.py leaves '
        'Largest=False; only the ACDC-specific prediction_ACDC.py keeps the '
        'largest connected component. This port matches the trainer '
        'architecture (the shipped prediction.py bilinear=True is '
        'incompatible with its own checkpoint), zooms the argmax mask to GT '
        'size with scipy order=0 as prediction_ACDC.py does, and keeps the '
        'largest component only when --knowsam-largest-cc is passed. '
        'Unconditional largest-component filtering is invalid for '
        'multi-component curvilinear targets. Upstream reports metrics at '
        '256x256; the shared evaluator uses GT resolution.',
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
        parser.add_argument(
            '--max-iterations', type=int, default=None,
            help='defaults to --epochs (official: 50000 iterations)')
        parser.add_argument('--val-interval', type=int, default=200)
        parser.add_argument(
            '--log-interval', type=int, default=50,
            help='iterations between shared train/loss and lr W&B points')
        parser.add_argument(
            '--knowsam-shared-val', type=int, default=1,
            help='1 = also log the shared Dice/IoU/clDice/HD95 validation '
                 'metrics with the same W&B keys as other baselines; '
                 '0 = official branch Dice only')
        parser.add_argument('--unlabeled-limit', type=int, default=384)
        parser.add_argument('--labeled-batch', type=int, default=3)
        parser.add_argument(
            '--input-size', type=int, default=256,
            help='KnowSAM network/transform resolution (official: 256)')
        parser.add_argument(
            '--knowsam-largest-cc', action='store_true',
            help='keep only the largest connected component at prediction '
                 '(ACDC-specific official behaviour; off by default because '
                 'curvilinear targets are often multi-component)')
        parser.add_argument(
            '--unet-lr', type=float, default=0.01,
            help='SGD LR for the SGDL/UNet-VNet branch (official: 0.01)')

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
        labeled = load_split_samples(data_root, 'train')
        val = load_split_samples(data_root, 'val')
        for sample in labeled:
            self._save_pair(root / 'labeled', sample, sample.stem)
        for sample in val:
            self._save_pair(root / 'val', sample, sample.stem)

        # The protocol stem is <DOMAIN>_<source stem>, while the source pool
        # uses <source stem>.  Read split_manifest.csv and exclude every
        # train/val/test source image; comparing protocol stems directly
        # silently leaked validation/test images into the pool.
        excluded = {}
        manifest = data_root / 'split_manifest.csv'
        if not manifest.is_file():
            raise SystemExit(
                'KnowSAM requires split_manifest.csv to exclude test images '
                f'from the unlabeled pool: {manifest}')
        # Exclude every protocol image (train, val and seen_test).  Test
        # images are not unlabeled training data and must never enter the
        # semi-supervised pool.
        with manifest.open(newline='') as handle:
            for row in csv.DictReader(handle):
                excluded.setdefault(row['dataset'], set()).add(
                    Path(row['source_image']).stem)

        count = 0
        for domain in SOURCE_DOMAINS:
            domain_root = REPO_ROOT / 'datasets' / 'seen' / domain
            if domain == 'CORN':
                domain_root = domain_root / 'train'
            if not (domain_root / 'images').is_dir():
                continue
            for sample in RawPairDataset(domain_root).samples:
                if sample.stem in excluded.get(domain, set()):
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
            lr=args.lr, UNet_lr=args.unet_lr, VNet_lr=0.01,
            weight_decay=args.weight_decay,
            image_size=int(args.input_size), point_nums=5, box_nums=1, mod='sam_adpt',
            model_type='vit_b', thd=False,
            batch_size=args.batch_size, labeled_bs=labeled_batch,
            seed=args.seed, mixed_iterations=12000,
            max_iterations=args.max_iterations, n_fold=1,
            consistency=0.1, consistency_rampup=200.0,
            device=args.device,
            multimask=False, encoder_adapter=True,
            sam_checkpoint=str(args.sam_checkpoint),
        )

    @torch.no_grad()
    def _shared_val_metrics(self, trainer, kargs, samples, device, args):
        """Evaluate the current SGDL prediction with the shared metric suite.

        Mirrors ``run_prediction``'s GT-resolution prediction path (argmax at
        network resolution, nearest zoom to GT) and the shared
        ``runtime.binary_metrics`` definitions.  Used only for W&B monitoring;
        the official SGDL-Dice checkpoint selection is unchanged.
        """
        was_training = trainer.SGDL.training
        trainer.SGDL.eval()
        transform = build_transforms(kargs)['valid_test']
        overall = {key: [] for key in (*_SHARED_METRICS, 'empty')}
        per_dataset = {}
        try:
            for sample in samples:
                image = np.asarray(
                    load_rgb(sample.image_path), dtype=np.float32) / 255.0
                dummy = np.zeros(image.shape[:2], dtype=np.uint8)
                augmented = transform(image=image, mask=dummy)
                tensor = torch.from_numpy(
                    augmented['image'].transpose(2, 0, 1).copy()
                ).float().unsqueeze(0).to(device)
                outputs = trainer.SGDL(tensor)
                fusion = outputs[-1]
                pred_low = torch.argmax(
                    torch.softmax(fusion, dim=1), dim=1)[0]
                gt = load_mask(sample.mask_path)
                pred = zoom(
                    pred_low.cpu().numpy().astype(np.uint8),
                    (gt.shape[0] / pred_low.shape[0],
                     gt.shape[1] / pred_low.shape[1]),
                    order=0) > 0
                pred = _postprocess_prediction(
                    pred, bool(getattr(args, 'knowsam_largest_cc', False)))
                values = binary_metrics(pred, gt)
                bucket = per_dataset.setdefault(
                    sample.dataset,
                    {key: [] for key in (*_SHARED_METRICS, 'empty')})
                for key, value in values.items():
                    overall[key].append(value)
                    bucket[key].append(value)
        finally:
            if was_training:
                trainer.SGDL.train()

        metrics = {key: _mean_metric(values)
                   for key, values in overall.items()}
        metrics['empty_prediction_rate'] = metrics.pop('empty')
        datasets = {}
        for dataset, values in sorted(per_dataset.items()):
            datasets[dataset] = {
                key: _mean_metric(items) for key, items in values.items()
                if key != 'empty'}
            datasets[dataset]['empty_prediction_rate'] = _mean_metric(
                values['empty'])
        return metrics, datasets

    # ---- training -----------------------------------------------------
    def run_training(self, args):
        if args.resume:
            raise SystemExit(
                'KnowSAM does not support intra-run --resume; the custom '
                'semi-supervised loop only writes the best SGDL checkpoint')
        unsupported = []
        if int(args.accum_steps) != 1:
            unsupported.append('--accum-steps')
        if float(args.grad_clip) != 0:
            unsupported.append('--grad-clip')
        if args.selection != self.selection:
            unsupported.append('--selection (SGDL Dice is official)')
        if int(getattr(args, 'num_workers', 0)) != 0:
            unsupported.append('--num-workers (custom loader uses 0)')
        if int(args.input_size) % 16:
            unsupported.append('--input-size must be divisible by 16')
        if unsupported:
            raise SystemExit(
                'KnowSAM custom pipeline does not implement: '
                + ', '.join(unsupported))
        from baseline_zoo.runtime import seed_everything
        seed_everything(args.seed)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.args = args
        val_samples = load_split_samples(Path(args.data_root), 'val')
        if args.limit_val:
            val_samples = val_samples[:args.limit_val]
        # A rerun must not silently keep a checkpoint from an earlier run.
        for stale_name in (
                'SGDL_best_model.pth', 'sam_best_model.pth',
                'knowsam_model.txt', 'knowsam_config.json'):
            (workdir / stale_name).unlink(missing_ok=True)
        root = workdir / 'knowsam_data'
        # Staging is enumerated with os.listdir; stale files from a previous
        # protocol/limit would silently change the training and val sets.
        shutil.rmtree(root, ignore_errors=True)
        n_labeled, n_unlabeled, n_val = self._stage(
            root, Path(args.data_root), args.unlabeled_limit)
        if args.labeled_batch * 2 != args.batch_size:
            raise SystemExit(
                'KnowSAM requires --labeled-batch == batch_size / 2 '
                '(two equal streams); got '
                f'{args.labeled_batch}+{args.batch_size - args.labeled_batch}')
        iterations = (args.max_iterations if args.max_iterations is not None
                      else args.epochs)
        if iterations <= 0:
            raise SystemExit('--max-iterations/--epochs must be positive')
        if args.val_interval <= 0 or args.log_interval <= 0:
            raise SystemExit('--val-interval and --log-interval must be positive')

        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_device(torch.device(args.device))
        kargs = self._args(args, root, args.labeled_batch)
        kargs.max_iterations = iterations
        sam_checkpoint = Path(args.sam_checkpoint)
        construction = {
            'input_size': int(args.input_size),
            'sam_checkpoint': str(sam_checkpoint),
            'sam_checkpoint_sha256': (
                file_sha256(sam_checkpoint)
                if sam_checkpoint.is_file() else None),
            'labeled_batch': int(args.labeled_batch),
            'batch_size': int(args.batch_size),
            'max_iterations': int(iterations),
            'unet_lr': float(kargs.UNet_lr),
        }
        (workdir / 'knowsam_config.json').write_text(
            json.dumps(construction, indent=2) + '\n')
        run_config = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'data_root': str(args.data_root),
            'labeled': n_labeled, 'unlabeled': n_unlabeled, 'val': n_val,
            'batch_size': args.batch_size,
            'labeled_batch': args.labeled_batch,
            'max_iterations': iterations,
            'log_interval': args.log_interval,
            'val_interval': args.val_interval,
            'seed': args.seed,
            'selection': 'SGDL validation Dice (official)',
            'val_dice_source': 'current SGDL validation Dice; best_* remain '
                               'the official best-so-far checkpoint values',
            'wandb_val_metrics': (
                'shared Dice/IoU/clDice/HD95 on the val split; official '
                'SGDL Dice remains the checkpoint-selection metric'),
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
        train_window = []
        while iteration < iterations:
            for batch in loader:
                images = batch['image'].to(args.device)
                labels = batch['label'].to(args.device)
                trainer.train(images, labels, iteration)
                loss_value = getattr(trainer, 'last_losses', {}).get(
                    'SGDL_loss')
                if loss_value is not None:
                    train_window.append(float(loss_value))
                iteration += 1
                if run is not None and iteration % args.log_interval == 0:
                    losses = getattr(trainer, 'last_losses', {})
                    lrs = getattr(trainer, 'last_lrs', {})
                    scalars = {
                        'iteration': iteration,
                        'train/loss': losses.get('SGDL_loss'),
                        'train/sgdl_loss': losses.get('SGDL_loss'),
                        'train/sam_loss': losses.get('sam_loss'),
                        'train/unet_vnet_loss': losses.get('unet_vnet_loss'),
                        'train/fusion_loss': losses.get('fusion_loss'),
                        'lr': lrs.get('sgdl'),
                        'lr/sam': lrs.get('sam'),
                        'lr/sgdl': lrs.get('sgdl'),
                    }
                    log_scalars(run, {
                        key: value for key, value in scalars.items()
                        if value is not None})
                if iteration % args.val_interval == 0:
                    trainer.val(val_loader, str(workdir), iteration)
                    window_loss = (_mean_metric(train_window)
                                   if train_window else None)
                    train_window = []
                    if run is not None:
                        current = getattr(trainer, 'last_val_dice', {})
                        scalars = {
                            'epoch': iteration // args.val_interval,
                            'iteration': iteration,
                            'train/loss': window_loss,
                            'lr': getattr(trainer, 'last_lrs', {}).get(
                                'sgdl'),
                            # Official KnowSAM selection / checkpoint metric.
                            'val/knowsam_selection': current.get('SGDL'),
                            'val/knowsam/sgdl_dice': current.get('SGDL'),
                            'val/knowsam/sam_dice': current.get('sam'),
                            'val/knowsam/unet_dice': current.get('unet'),
                            'val/knowsam/vnet_dice': current.get('vnet'),
                            'val/knowsam/SGDL_best_dice':
                                trainer.best_performance_SGDL,
                            'val/knowsam/SAM_best_dice':
                                trainer.best_performance_sam,
                        }
                        if int(getattr(args, 'knowsam_shared_val', 1)):
                            try:
                                shared_metrics, shared_datasets = (
                                    self._shared_val_metrics(
                                        trainer, kargs, val_samples,
                                        args.device, args))
                            except Exception as exc:
                                print('[knowsam] shared validation failed: '
                                      f'{exc}', flush=True)
                            else:
                                scalars.update({
                                    'val/selection': shared_metrics['dice'],
                                    'val/dice': shared_metrics['dice'],
                                    'val/iou': shared_metrics['iou'],
                                    'val/cldice': shared_metrics['cldice'],
                                    'val/hd95': shared_metrics['hd95'],
                                    'val/empty_prediction_rate':
                                        shared_metrics[
                                            'empty_prediction_rate'],
                                })
                                for dataset, values in (
                                        shared_datasets.items()):
                                    for metric in ('dice', 'iou', 'cldice',
                                                   'hd95'):
                                        scalars[
                                            f'val/{dataset}/{metric}'] = (
                                                values[metric])
                        else:
                            # Backward-compatible monitor when the shared
                            # evaluation is explicitly disabled.
                            scalars['val/selection'] = current.get('SGDL')
                            scalars['val/dice'] = current.get('SGDL')
                        log_scalars(run, {
                            key: value for key, value in scalars.items()
                            if value is not None})
                if iteration >= iterations:
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
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_device(torch.device(args.device))
        sidecar = workdir / 'knowsam_config.json'
        if sidecar.is_file():
            construction = json.loads(sidecar.read_text())
            args.input_size = int(construction.get(
                'input_size', args.input_size))
            args.labeled_batch = int(construction.get(
                'labeled_batch', args.labeled_batch))
            # KnowSAM inference uses the SGDL checkpoint only; SAM is a
            # training-time pseudo-label teacher and is not rebuilt here.
        else:
            print('[knowsam] warning: missing knowsam_config.json; '
                  'using eval-time defaults', flush=True)
        kargs = self._args(args, root, args.labeled_batch)
        kargs.max_iterations = (args.max_iterations
                                if args.max_iterations is not None
                                else args.epochs)
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
        stale_meta = pred_root / '.prediction_meta.json'
        if stale_meta.is_file():
            stale_meta.unlink()
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
                    # Argmax at the 256x256 network resolution.
                    pred_low = torch.argmax(
                        torch.softmax(fusion, dim=1), dim=1)[0]
                gt = load_mask(sample.mask_path)
                # Official ACDC path: scipy.ndimage.zoom(..., order=0) from
                # the 256x256 argmax to the native slice size.  PyTorch
                # ``mode='nearest'`` is not equivalent for non-integer
                # scales, so use the same scipy mapping.
                pred = zoom(
                    pred_low.cpu().numpy().astype(np.uint8),
                    (gt.shape[0] / pred_low.shape[0],
                     gt.shape[1] / pred_low.shape[1]),
                    order=0) > 0
                pred = _postprocess_prediction(
                    pred, bool(getattr(args, 'knowsam_largest_cc', False)))
                Image.fromarray((pred.astype(np.uint8) * 255)).save(
                    out_dir / f'{sample.stem}.png')

        split_manifest = Path(args.data_root) / 'split_manifest.csv'
        meta = {
            'method': self.id,
            'official_source': self.official_source,
            'deviations': list(self.deviations),
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': (file_sha256(Path(checkpoint))
                                  if Path(checkpoint).is_file() else None),
            'split_manifest_sha256': (file_sha256(split_manifest)
                                      if split_manifest.is_file() else None),
            'datasets': [spec.split(':', 1)[0]
                         for spec in args.test_datasets],
            'largest_cc': bool(getattr(
                args, 'knowsam_largest_cc', False)),
        }
        (pred_root / '.prediction_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n')
