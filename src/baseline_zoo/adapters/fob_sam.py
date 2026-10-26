"""FoB-SAM baseline (CVPR 2026) with the official episode logic.

Official reference: vendored ``baselines/FoB_SAM`` (Apache-2.0).  FoB is
a few-shot prompt localizer: a ResNet-101 episode encoder produces
positive/negative point prompts from support images, and a frozen SAM
turns those prompts into the query mask.

The official medical protocol trains and evaluates on consecutive slices
of 3D volumes with supervoxel pseudo-classes.  The 20-dataset protocol is
2D image pairs, so this adapter keeps the official model/loss/prompt logic
and adapts only the episode construction (documented below).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose

from baseline_zoo.data import Sample, list_pairs, load_mask, load_rgb
from baseline_zoo.models.fob_sam import SAM, FewShotSeg
from baseline_zoo.models.fob_sam.transforms import ElasticTransform, RandomAffine
from baseline_zoo.runtime import REPO_ROOT, BaselineAdapter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 256

# Auto donor map for datasets that have no labeled support in the 3-shot
# training split.  The paper's cross-domain experiments use source-domain
# support, which is the same idea.
CRACK_LIKE = {
    'DeepCrack', 'CHCrack5k', 'TopoMortar', 'TYRE', 'WIRE'}
VESSEL_LIKE = {
    'FIVES', 'STARE', 'XCAD', 'HRF', 'LES-AV', 'OCTA-500', 'ROAD',
    'LEAF', 'DeepGlobe'}
DONOR_PREFERENCE = ('DRIVE', 'CHASEDB1', 'DCA1', 'CREMI', 'CrackTree200',
                    'CORN')


class FoBSAMAdapter(BaselineAdapter):
    id = 'fob_sam'
    display_name = 'FoB-SAM'
    official_source = 'FoB (CVPR 2026)'
    official_entrypoint = 'upstream train.py/test.py (FewShotSeg + frozen SAM)'
    input_size = SIZE
    # 39001 optimizer iterations with max_iters_per_load=3000 in the
    # official scripts; 13 epochs x 3000 iterations = 39000.
    official_epochs = 13
    official_lr = 1e-4
    official_weight_decay = 0.0
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'Official training uses consecutive slices of 3D medical volumes '
        'and supervoxel pseudo-classes; this 2D 20-dataset port uses support/'
        'query image pairs with the provided GT masks (the official '
        'use_gt=True variant), so no supervoxels are generated.',
        'Official episodes are 1-shot/1-query and same-volume; training here '
        'samples 1 support + 1 query from the same seen domain (18 images, '
        '6 domains x 3). Evaluation support is the same-domain train shot '
        'when available; unseen/novel datasets use one donor domain selected '
        'by structure (vessel-like -> DRIVE, crack-like -> CrackTree200).',
        'The official checkpoint is selected manually; this port selects by '
        'the shared 6-image validation Dice so every baseline uses one rule.',
        'SAM ViT-H is the paper backbone and the default. --sam-model-type '
        'vit_b/vit_l exist only for smoke tests; reported results must use '
        'vit_h.',
        'ResNet-101 is initialized from the official DeepLabV3 COCO '
        'checkpoint instead of the hard-coded absolute path.',
        'The official test passes the query GT into forward only to guard '
        'the empty-foreground branch; this port passes an all-ones dummy at '
        'evaluation, so no test GT influences prompts or masks.',
        'SAM masks are predicted at 256x256 and resized to the original GT '
        'resolution before the shared metrics; the official protocol keeps '
        'volumes at 256.',
        'Official test support is a held-out volume inside the test fold and '
        'only labeled query slices are evaluated; this port uses training/'
        'donor support and evaluates all query images. Metrics are therefore '
        'not directly comparable.',
        'The unused InfoNCE attribute is kept as a dependency-free '
        'placeholder; it is never called by the official forward.',
        '2D protocol images use per-image grayscale standardization or '
        'ImageNet RGB normalization; upstream standardizes per volume. '
        'Gamma and geometry augmentation choose their branch independently, '
        'matching the released dataset code.',
        'Official training uses seed 2021 and workers=16; this port uses the '
        'protocol seed and workers=0, so the data/RNG stream differs.',
    )

    def __init__(self):
        super().__init__()
        self._train_pairs: list[Sample] | None = None
        self._train_by_dataset: dict[str, list[Sample]] | None = None
        self._pair_cache: dict[Sample, tuple[torch.Tensor, torch.Tensor]] = {}
        self._support_cache: dict[str, tuple[list[torch.Tensor],
                                             list[torch.Tensor]]] = {}
        self._sam = None

    # ---- CLI ----------------------------------------------------------
    def add_args(self, parser):
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth',
            help='Frozen SAM checkpoint (paper: ViT-H sam_vit_h_4b8939.pth)')
        parser.add_argument(
            '--sam-model-type', default='vit_h',
            choices=['vit_h', 'vit_l', 'vit_b'],
            help='SAM backbone; keep vit_h for reported results')
        parser.add_argument(
            '--resnet-checkpoint', type=Path,
            default=REPO_ROOT / 'checkpoints' /
            'deeplabv3_resnet101_coco-586e9e4e.pth',
            help='Official DeepLabV3 COCO ResNet-101 checkpoint')
        parser.add_argument(
            '--fob-shots', type=int, default=1,
            help='Evaluation support images per query (official: 1); '
                 'training episodes stay 1-shot as in the official code')
        parser.add_argument(
            '--fob-iters-per-epoch', type=int, default=3000,
            help='Official max_iters_per_load; total iterations = '
                 'epochs * this value')
        parser.add_argument(
            '--fob-support-dataset', default='auto',
            help="Support source for a test dataset: 'auto', or a training "
                 "dataset name (e.g. DRIVE / CrackTree200)")
        parser.add_argument(
            '--fob-random-init', action='store_true',
            help='Smoke only: do not load ResNet-101 pretrained weights')
        parser.add_argument(
            '--fob-best-pred-idx', type=int, default=0,
            help='SAM multimask index (official: 0 for medical, 1 for ISIC)')

    # ---- model --------------------------------------------------------
    def build_model(self, args):
        weights = None
        if not args.fob_random_init:
            if not args.resnet_checkpoint.is_file():
                raise SystemExit(
                    f'missing ResNet-101 COCO checkpoint: '
                    f'{args.resnet_checkpoint}\n'
                    'download: curl -L -o checkpoints/'
                    'deeplabv3_resnet101_coco-586e9e4e.pth '
                    'https://download.pytorch.org/models/'
                    'deeplabv3_resnet101_coco-586e9e4e.pth')
            weights = args.resnet_checkpoint
        model = FewShotSeg({}, pretrained_weights=weights)
        return model

    def configure_trainable(self, model):
        # Official training optimizes every FoB parameter; SAM is frozen
        # and is only instantiated lazily for validation/prediction.
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def make_optimizer(self, model, args):
        # Upstream: Adam(model.parameters(), lr=...), no weight decay.
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.0)

    def make_scheduler(self, optimizer, args):
        # Official MultiStepLR milestones are every max_iters_per_load
        # (3000 iterations); the shared loop steps once per epoch, which
        # is exactly one 3000-iteration block.
        milestones = list(range(1, max(int(args.epochs), 1)))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.98)

    def state_dict(self, model):
        return model.state_dict()

    def load_state_dict(self, model, state):
        result = model.load_state_dict(state, strict=False)
        # Upstream checkpoints do not contain the sin_pos buffer added by
        # this device-agnostic port; everything else must match.
        ignorable = {'sin_pos', 'masked_attention.sin_pos'}
        missing = [key for key in result.missing_keys
                   if key not in ignorable]
        if missing:
            raise RuntimeError(
                f'fob_sam checkpoint is missing model tensors: {missing[:5]}')

    # ---- preprocessing ------------------------------------------------
    @staticmethod
    def _normalize(image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        gray = (np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2]))
        if gray:
            # Official medical path: per-image standardization, grayscale
            # replicated to 3 channels.
            gray_array = array[..., 0]
            gray_array = ((gray_array - gray_array.mean())
                          / (gray_array.std() + 1e-6))
            array = np.stack([gray_array] * 3, axis=-1)
        else:
            # Official ISIC path: ImageNet normalization.
            array = array / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float()

    @staticmethod
    def _resize_mask(path: Path) -> torch.Tensor:
        mask = load_mask(path).astype(np.uint8) * 255
        image = Image.fromarray(mask, mode='L').resize(
            (SIZE, SIZE), Image.NEAREST)
        return (torch.from_numpy(np.asarray(image) > 127)
                .float().unsqueeze(0))

    def _load_pair(self, sample: Sample, cache: bool = True
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._pair_cache.get(sample) if cache else None
        if cached is None:
            image = self._normalize(load_rgb(sample.image_path))
            mask = self._resize_mask(sample.mask_path)
            cached = (image, mask)
            if cache:
                self._pair_cache[sample] = cached
        return cached

    @staticmethod
    def _gamma(image: np.ndarray) -> np.ndarray:
        gamma = random.uniform(0.5, 1.5)
        cmin = image.min()
        irange = image.max() - cmin + 1e-5
        image = image - cmin + 1e-5
        return irange * np.power(image / irange, gamma) + cmin

    @staticmethod
    def _geom(image: np.ndarray, mask: np.ndarray):
        transform = Compose([
            RandomAffine(5, (5, 5), 5, (0.9, 1.2), True, order=3),
            ElasticTransform(10, 5),
        ])
        stacked = np.concatenate([image, mask[..., None]], axis=2)
        stacked = transform(stacked)
        return stacked[..., :3], np.rint(stacked[..., 3])

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy().copy()
        mask_np = mask[0].numpy().copy()
        if random.random() < 0.5:
            image_np = self._gamma(image_np)
        if random.random() < 0.5:
            image_np, mask_np = self._geom(image_np, mask_np)
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask = torch.from_numpy((mask_np > 0.5).astype(np.float32)).unsqueeze(0)
        return image, mask

    # ---- support/episode construction ---------------------------------
    def _ensure_train_pairs(self):
        if self._train_pairs is not None:
            return
        roots = [Path(self.args.data_root) / 'train']
        trained_on = getattr(self, 'checkpoint_config', {}).get('data_root')
        if trained_on:
            roots.append(Path(trained_on) / 'train')
        roots.append(REPO_ROOT / 'data' / 'sacm_3shot' / 'train')
        root = next((path for path in roots if path.is_dir()), None)
        if root is None:
            raise SystemExit(
                'fob_sam needs the training split for support images; '
                f'not found under {roots[0]}')
        if root != roots[0]:
            print(f'WARNING: fob_sam support split fell back to {root}; '
                  f'expected {roots[0]}')
        pairs = []
        for image, mask in list_pairs(root):
            # The 3-shot split is flat: train/images/<DATASET>_<case>.png.
            dataset = image.stem.split('_', 1)[0]
            pairs.append(Sample(image, mask, dataset, image.stem))
        self._train_pairs = pairs
        self._train_by_dataset = defaultdict(list)
        for sample in pairs:
            self._train_by_dataset[sample.dataset].append(sample)
        for values in self._train_by_dataset.values():
            values.sort(key=lambda item: item.stem)

    @staticmethod
    def _canonical_dataset(sample: Sample) -> str:
        """Resolve an unprefixed test stem back to its dataset root.

        ``RawPairDataset`` derives ``sample.dataset`` from the first
        underscore-separated stem token, which is only correct for the flat
        training split.  Test roots use names such as ``DeepCrack/111212-1``,
        so prefer the dataset root directory when it is meaningful.
        """
        root_name = sample.image_path.parent.parent.name
        if root_name and root_name not in {'train', 'val', 'test'}:
            return root_name
        return sample.dataset

    def _support_dataset(self, name: str) -> str:
        self._ensure_train_pairs()
        assert self._train_by_dataset is not None
        available = set(self._train_by_dataset)
        override = getattr(self.args, 'fob_support_dataset', 'auto')
        if override and override != 'auto':
            if override not in available:
                raise SystemExit(
                    f'--fob-support-dataset={override!r} is not in the '
                    f'training split; available: {sorted(available)}')
            return override
        if name in available:
            return name
        preferred = []
        if name in CRACK_LIKE:
            preferred.append('CrackTree200')
        elif name in VESSEL_LIKE:
            preferred.append('DRIVE')
        preferred.extend(DONOR_PREFERENCE)
        for candidate in preferred:
            if candidate in available:
                return candidate
        return sorted(available)[0]

    def _support_pairs(self, name: str) -> list[Sample]:
        self._ensure_train_pairs()
        assert self._train_by_dataset is not None
        donor = self._support_dataset(name)
        pairs = [sample for sample in self._train_by_dataset[donor]
                 if self._resize_mask(sample.mask_path).sum() > 0]
        if not pairs:
            raise RuntimeError(f'no non-empty support mask in {donor}')
        shots = max(int(getattr(self.args, 'fob_shots', 1)), 1)
        return pairs[:shots]

    def _query_support(self, name: str):
        if name not in self._support_cache:
            pairs = self._support_pairs(name)
            images, masks = [], []
            for sample in pairs:
                image, mask = self._load_pair(sample)
                images.append(image)
                masks.append(mask)
            self._support_cache[name] = (images, masks)
        return self._support_cache[name]

    # ---- training -----------------------------------------------------
    def prepare_train_batch(self, samples, rng: random.Random):
        self._ensure_train_pairs()
        assert self._train_by_dataset is not None
        query_sample = samples[0]
        candidates = [sample for sample in self._train_by_dataset[
            query_sample.dataset] if sample != query_sample]
        if not candidates:
            candidates = [sample for values in self._train_by_dataset.values()
                          for sample in values if sample != query_sample]
        support_sample = rng.choice(candidates)
        support_image, support_mask = self._load_pair(support_sample)
        query_image, query_mask = self._load_pair(query_sample)
        if rng.random() < 0.5:
            query_image, query_mask = self._augment(query_image, query_mask)
        else:
            support_image, support_mask = self._augment(
                support_image, support_mask)
        device = self.device
        return {
            'support_images': [[support_image.unsqueeze(0).to(device)]],
            'support_masks': [[support_mask.unsqueeze(0).to(device)]],
            'query_image': query_image.unsqueeze(0).to(device),
            'query_label': query_mask.long().to(device),
        }

    @staticmethod
    def _episode_loss(model, batch):
        prompt_loss, rac_loss, foreground_loss = model(
            batch['support_images'], batch['support_masks'],
            [batch['query_image']], batch['query_label'], True)
        return prompt_loss + rac_loss + foreground_loss

    def train_step(self, model, batch):
        return self._episode_loss(model, batch)

    def train_epoch(self, model, loader, optimizer, scheduler, args):
        # Official training is iteration-based (39001 iterations with
        # 3000-iteration "epochs"), not one pass over the 18 images.
        model.train()
        samples = list(loader.dataset.samples)
        valid = [sample for sample in samples
                 if self._load_pair(sample)[1].sum() > 0]
        by_dataset = defaultdict(list)
        for sample in valid:
            by_dataset[sample.dataset].append(sample)
        eligible = [values for values in by_dataset.values()
                    if len(values) >= 2]
        if not eligible:
            raise RuntimeError('fob_sam needs at least two images per domain')

        iterations = max(int(getattr(args, 'fob_iters_per_epoch', 3000)), 1)
        running = 0.0
        for _ in range(iterations):
            domain = random.choice(eligible)
            support_sample, query_sample = random.sample(domain, 2)
            support_image, support_mask = self._load_pair(support_sample)
            query_image, query_mask = self._load_pair(query_sample)
            if random.random() < 0.5:
                query_image, query_mask = self._augment(
                    query_image, query_mask)
            else:
                support_image, support_mask = self._augment(
                    support_image, support_mask)

            device = self.device
            batch = {
                'support_images': [[support_image.unsqueeze(0).to(device)]],
                'support_masks': [[support_mask.unsqueeze(0).to(device)]],
                'query_image': query_image.unsqueeze(0).to(device),
                'query_label': query_mask.long().to(device),
            }
            loss = self._episode_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
            optimizer.step()
            running += float(loss.detach())
        # This adapter overrides the shared train_epoch, so it must step the
        # scheduler itself; one step per 3000-iteration epoch matches the
        # official milestone spacing.
        if scheduler is not None:
            scheduler.step()
        return running / iterations

    # ---- inference ----------------------------------------------------
    def prepare_eval_sample(self, sample: Sample, device):
        dataset = self._canonical_dataset(sample)
        support_images, support_masks = self._query_support(dataset)
        # Test queries are streamed; caching all 3,580 pairs would retain
        # ~3.5 GiB of CPU RAM.  Support pairs remain cached (1-3 per domain).
        query_image, _ = self._load_pair(sample, cache=False)
        height, width = load_mask(sample.mask_path).shape
        return {
            'support_images': [
                [image.unsqueeze(0).to(device) for image in support_images]],
            'support_masks': [
                [mask.unsqueeze(0).to(device) for mask in support_masks]],
            'query_image': query_image.unsqueeze(0).to(device),
            # Upstream uses qry_labels only in the
            # `supp_mask.max() > 0 and qry_labels.max() > 0` guard.
            'query_label': torch.ones(1, SIZE, SIZE, device=device),
            'original_size': (height, width),
        }

    def save_config(self, args):
        return {
            'input_size': int(self.input_size),
            'data_root': str(getattr(args, 'data_root', '')),
            'sam_checkpoint': str(getattr(args, 'sam_checkpoint', '')),
            'resnet_checkpoint': str(
                getattr(args, 'resnet_checkpoint', '')),
            'sam_model_type': str(getattr(args, 'sam_model_type', 'vit_h')),
            'fob_shots': int(getattr(args, 'fob_shots', 1)),
            'fob_support_dataset': str(
                getattr(args, 'fob_support_dataset', 'auto')),
            'fob_best_pred_idx': int(
                getattr(args, 'fob_best_pred_idx', 0)),
            'fob_iters_per_epoch': int(
                getattr(args, 'fob_iters_per_epoch', 3000)),
        }

    def restore_config(self, args, config):
        for key in ('fob_shots', 'fob_best_pred_idx',
                    'fob_iters_per_epoch'):
            if key in config:
                setattr(args, key, int(config[key]))
        for key in ('sam_model_type', 'fob_support_dataset'):
            if key in config:
                setattr(args, key, str(config[key]))
        for key in ('data_root', 'sam_checkpoint', 'resnet_checkpoint'):
            if config.get(key):
                setattr(args, key, Path(config[key]))

    def _ensure_sam(self):
        if self._sam is None:
            checkpoint = Path(self.args.sam_checkpoint)
            if not checkpoint.is_file():
                raise SystemExit(
                    f'missing SAM checkpoint: {checkpoint}\n'
                    'download: curl -L -o checkpoints/'
                    'sam_vit_h_4b8939.pth '
                    'https://dl.fbaipublicfiles.com/segment_anything/'
                    'sam_vit_h_4b8939.pth')
            self._sam = SAM(
                checkpoint, model_type=self.args.sam_model_type,
                device=self.device,
                best_pred_idx=self.args.fob_best_pred_idx)
        return self._sam

    def predict_sample(self, model, prepared, sample):
        with torch.no_grad():
            neg_point, pos_point = model(
                prepared['support_images'], prepared['support_masks'],
                [prepared['query_image']], prepared['query_label'], False)
        if neg_point is None or pos_point is None:
            return np.zeros(prepared['original_size'], dtype=bool)
        sam = self._ensure_sam()
        mask = sam(prepared['query_image'][0], pos_point, neg_point)
        height, width = prepared['original_size']
        if mask.shape != (height, width):
            tensor = torch.from_numpy(
                mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor, size=(height, width), mode='bilinear',
                align_corners=False)
            mask = tensor[0, 0].numpy() > 0.5
        return mask.astype(bool)
