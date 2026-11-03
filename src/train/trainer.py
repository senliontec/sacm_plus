"""Training CLI entry point (algorithm-agnostic).

Thin wrapper: parses arguments, builds everything (model via the model
registry, losses, optimizer, scheduler, data), then delegates the whole
training schedule to core.trainer.Trainer.

Usage:
    python train.py --algorithm sacm --model sam_l --preset stage2 \
        --data_root <root> --checkpoint <sam.pth>
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# src-layout bootstrap: make the packages importable without an editable
# install (harmless when `pip install -e .` is also in effect).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

import models  # noqa: F401  (registers every model family's ModelSpecs)
from core.augmentation import JointAugment
from core.data import SegmentationDataset, TestDataset
from core.losses import (
    TOPOLOGY_LOSS_REGISTRY,
    BoundaryLoss,
    DiceBCELoss,
    SCNPCriterion,
    SoftclDiceLoss,
    build_topology_loss,
)
from core.registry import MODEL_REGISTRY, get_model_spec
from core.reproducibility import configure_determinism, seed_worker
from core.run_artifacts import (
    prepare_training_artifacts,
    read_json,
    restore_generator_state,
    restore_rng_state,
)
from core.trainer import ForwardWrapper, Trainer
from core.wandb_utils import WANDB_API_KEY_DEFAULT, redact_secrets
from models.sacm_x.configs import PRESETS, apply_preset, str2bool


class DistributedEvalSampler(Sampler):
    """Unique, unpadded validation shards for DDP.

    ``DistributedSampler`` with ``drop_last=False`` pads the dataset with
    duplicate indices, which double-counts tiny 3-shot validation sets.
    This sampler gives every rank a disjoint subset and lets trailing ranks
    be empty; the validation aggregator already handles empty shards.
    """

    def __init__(self, dataset):
        import torch.distributed as dist
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


def setup_logging(args):
    # exist_ok: DDP 下 8 个 rank 同时建目录,避免 FileExistsError 竞态
    os.makedirs('results/logs', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'results/logs/training_{timestamp}.log'

    stream = logging.StreamHandler()
    stream.setLevel(logging.WARNING)  # 终端只显示警告/错误;INFO 全量进日志文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            stream,
        ]
    )

    logging.info("Training Configuration:")
    for arg, value in redact_secrets(args).items():
        logging.info(f"{arg}: {value}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _resolve_resume_path(args, save_dir):
    raw = getattr(args, 'resume', None)
    if not raw:
        return None
    if str(raw).lower() in ('last', 'best'):
        path = Path(save_dir) / f'{str(raw).lower()}_model.pth'
    else:
        path = Path(raw)
    if not path.is_file():
        raise SystemExit(f'resume checkpoint not found: {path}')
    return path


def _completed_epoch(payload):
    """Return the number of completed epochs (1-based)."""
    if payload.get('epoch_completed') is not None:
        return int(payload['epoch_completed'])
    return int(payload.get('epoch', -1)) + 1


def _history_from_checkpoint(payload, history_path, completed_epoch):
    history = payload.get('history')
    if not isinstance(history, list):
        history = read_json(history_path).get('history', [])
    entries = []
    for entry in history if isinstance(history, list) else []:
        try:
            epoch = int(entry.get('epoch', 0))
        except (TypeError, ValueError):
            continue
        if epoch <= completed_epoch:
            entries.append(entry)
    return entries


def _validate_resume(payload, args, artifact_info):
    """Reject silent architecture/seed/data mismatches on resume."""
    config = payload.get('config') or {}
    for key in (
        'model', 'preset', 'use_geo_i', 'use_geo_e', 'geo_e_layers',
        'use_coarse_to_fine', 'use_fusion_v2', 'use_multi_depth',
        'use_typed_heads', 'adapter_dim_ratio',
    ):
        if key in config:
            current = getattr(args, key, None)
            if str(config[key]) != str(current):
                raise SystemExit(
                    f'resume checkpoint {key}={config[key]!r} does not match '
                    f'current --{key}={current!r}')
    saved_seed = config.get('seed', payload.get('seed'))
    if saved_seed is not None and int(saved_seed) != int(args.seed):
        raise SystemExit(
            f'resume checkpoint seed={saved_seed} != current seed={args.seed}')
    saved_data = payload.get('data_manifest_sha256')
    current_data = (artifact_info or {}).get('data_manifest_sha256')
    if saved_data and current_data and saved_data != current_data:
        raise SystemExit(
            'resume checkpoint was trained with a different data manifest; '
            'use the matching --data-root or start a fresh run')


def _apply_resume_state(
    payload, args, optimizer, scheduler, train_generator, val_generator,
    save_dir, artifact_info,
):
    _validate_resume(payload, args, artifact_info)
    if payload.get('optimizer_state_dict'):
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    if scheduler is not None and payload.get('scheduler_state_dict'):
        scheduler.load_state_dict(payload['scheduler_state_dict'])
    restore_rng_state(payload.get('rng_state'))
    restore_generator_state(
        train_generator, payload.get('train_generator_state'))
    restore_generator_state(
        val_generator, payload.get('val_generator_state'))

    completed = _completed_epoch(payload)
    history = _history_from_checkpoint(
        payload, Path(save_dir) / 'metrics_history.json', completed)
    best_score = payload.get(
        'best_selection_score', payload.get('best_score', -float('inf')))
    best_score = (float(best_score)
                  if best_score is not None else -float('inf'))
    best_f1 = payload.get(
        'best_f1_score', payload.get('f1_score', 0.0))
    best_epoch_raw = payload.get('best_epoch', completed)
    best_epoch = (int(best_epoch_raw)
                  if best_epoch_raw is not None else int(completed))
    return {
        'start_epoch': completed,
        'history': history,
        'best_selection_score': best_score,
        'best_f1_score': float(best_f1 or 0.0),
        'best_epoch': best_epoch,
        'resumed_from': str(payload.get('resumed_from', '')),
    }


def train(args):
    setup_logging(args)

    # The save path may point at a not-yet-existing job dir (run_paper.sh /
    # run_experiments use results/<preset>/<dataset>/); create it up front
    # so checkpoint + metrics_history.json saves never fail.
    save_dir = Path(args.save_path).parent
    os.makedirs(save_dir or '.', exist_ok=True)
    resume_path = _resolve_resume_path(args, save_dir)

    # DDP:由 torchrun 启动时注入 LOCAL_RANK;否则退化为单卡。
    # 关键顺序(此前 manual_seed_all 在 init 之前执行,会在每个 rank 进程里
    # 初始化全部 8 张卡的 CUDA 上下文,导致与 NCCL 冲突无声崩溃):
    # 先建进程组 → 绑定本地卡 → 只播本地卡的种子。
    is_ddp = 'LOCAL_RANK' in os.environ
    # Seed CPU RNGs and deterministic kernels before DDP.  Do not
    # manual_seed_all here: that can create CUDA contexts on all visible
    # GPUs inside every rank.  The selected device is seeded below.
    reproducibility = configure_determinism(
        args.seed,
        deterministic=getattr(args, 'deterministic', True),
        strict=getattr(args, 'deterministic_strict', True),
        num_threads=getattr(args, 'deterministic_threads', 1),
        cuda_seed_all=False,
    )
    args.reproducibility_info = reproducibility
    logging.info("Reproducibility: %s", reproducibility)

    if is_ddp:
        import torch.distributed as dist
        local_rank = int(os.environ['LOCAL_RANK'])
        # fencing-algs 模式:显式传入 device_id,barrier 等集合通信不再报
        # "using the device under current context" 警告
        dist.init_process_group(backend='nccl',
                                device_id=torch.device(f'cuda:{local_rank}'))
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    logging.info(f"Using device: {device} (ddp={is_ddp})")

    # Provenance hashing runs after deterministic CUDA settings are applied,
    # so querying GPU names cannot initialize a context under other flags.
    # Only global rank 0 writes provenance; other DDP ranks must not race on
    # the same train_config/provenance files.
    artifact_rank0 = True
    if is_ddp:
        import torch.distributed as dist
        artifact_rank0 = dist.get_rank() == 0
    if artifact_rank0:
        artifact_info = prepare_training_artifacts(
            args, save_dir or Path('.'),
            hash_files=bool(getattr(args, 'hash_data_files', True)),
            resume=resume_path is not None,
        )
        logging.info(
            'Training artifacts: config=%s data_manifest=%s',
            artifact_info['train_config'], artifact_info['data_manifest'])
    else:
        artifact_info = {}

    # Build the model through the model registry (extensible to other
    # architectures without touching this script).
    spec = get_model_spec(args.model)
    logging.info(f"Model: {args.model} — {spec.description}")
    sam = spec.build(
        checkpoint=args.checkpoint,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        geo_e_layers=args.geo_e_layers,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
        use_dynamic_strip=getattr(args, 'use_dynamic_strip', False),
        use_typed_heads=getattr(args, 'use_typed_heads', False),
        init_stage2_from_stage1=getattr(args, 'init_stage2_from_stage1', False),
        use_tcsu=getattr(args, 'use_tcsu', False),
        tcsu_scale=getattr(args, 'tcsu_scale', 4),
        tcsu_mid_channels=getattr(args, 'tcsu_mid_channels', 32),
        tcsu_max_length=getattr(args, 'tcsu_max_length', 5),
        use_tadt=getattr(args, 'use_tadt', False),
        tadt_hidden=getattr(args, 'tadt_hidden', 128),
        tadt_t_min=getattr(args, 'tadt_t_min', -4.0),
        tadt_t_max=getattr(args, 'tadt_t_max', 4.0),
        use_ranker=getattr(args, 'use_ranker', False),
        ranker_hidden=getattr(args, 'ranker_hidden', 128),
        ranker_pair_hidden=getattr(args, 'ranker_pair_hidden', 128),
        use_detail_stem=getattr(args, 'use_detail_stem', False),
        detail_channels=getattr(args, 'detail_channels', 32),
        detail_out_channels=getattr(args, 'detail_out_channels', 32),
        high_res_output=getattr(args, 'high_res_output', False),
        high_res_size=getattr(args, 'high_res_size', 512),
        high_res_refine=getattr(args, 'high_res_refine', False),
        high_res_refine_size=getattr(args, 'high_res_refine_size', 1024),
        single_head=getattr(args, 'single_head', False),
        use_v3=getattr(args, 'use_v3', False),
        v3_feature_dim=getattr(args, 'v3_feature_dim', 128),
        v3_mid_dim=getattr(args, 'v3_mid_dim', 96),
        v3_fine_dim=getattr(args, 'v3_fine_dim', 64),
        v3_refine_channels=getattr(args, 'v3_refine_channels', 16),
        v3_aux_centerline=getattr(args, 'v3_aux_centerline', True),
        v3_aux_boundary=getattr(args, 'v3_aux_boundary', True),
        v3_aux_width=getattr(args, 'v3_aux_width', True),
        use_internal_adapter=getattr(args, 'use_internal_adapter', True),
        collect_adapter_features=getattr(
            args, 'collect_adapter_features', True),
    )
    sam.to(device)
    if hasattr(sam.image_encoder, 'grad_checkpointing'):
        sam.image_encoder.grad_checkpointing = bool(
            getattr(args, 'grad_checkpointing', True))
        logging.info(
            "Encoder gradient checkpointing: %s",
            sam.image_encoder.grad_checkpointing)

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Model-specific freezing protocol (SAM: encoder adapters + mask
    # decoder only; other models define their own in their ModelSpec)
    spec.freeze(sam)
    base_model = sam

    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(
            resume_path, map_location='cpu', weights_only=False)
        state = resume_payload.get('model_state_dict')
        if not state:
            raise SystemExit(
                f'resume checkpoint has no model_state_dict: {resume_path}')
        _validate_resume(resume_payload, args, artifact_info)
        try:
            base_model.load_state_dict(state)
        except RuntimeError as exc:
            raise SystemExit(
                f'resume model does not match the current architecture: '
                f'{exc}') from exc
        logging.info(
            'Resumed model weights from %s (completed epochs=%d)',
            resume_path, _completed_epoch(resume_payload))

    if is_ddp:
        # Wrap the prompt-free forward, not just the bare Sam, so DDP's
        # reducer actually sees the training forward.  The optimizer must
        # still be built from the unwrapped model's parameter names.
        sam = torch.nn.parallel.DistributedDataParallel(
            ForwardWrapper(base_model, spec), device_ids=[local_rank],
            find_unused_parameters=True,  # 层子集等条件分支会产生未用参数
        )

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"After freezing - Total parameters: {total_params:,}")
    logging.info(f"After freezing - Trainable parameters: {trainable_params:,}")
    logging.info(f"After freezing - Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")

    # Legacy ``resize`` stretches to a square.  ``aspect_pad`` preserves the
    # source aspect ratio, and ``native_crop`` keeps native pixel scale for
    # large images (e.g. 3504x2336 HRF) instead of shrinking every 1-2px
    # structure to sub-pixel width.
    if args.train_input_mode == 'resize':
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()

    # Joint augmentation for the training split only (validation stays
    # clean). Defaults are the curvature-safe set.
    augment = None
    if not args.no_augment:
        augment = JointAugment(
            p_flip=args.aug_flip_p,
            p_rot90=args.aug_rot90_p,
            p_elastic=args.aug_elastic_p,
            elastic_alpha=args.aug_elastic_alpha,
            elastic_sigma=args.aug_elastic_sigma,
            p_color=args.aug_color_p,
        )
        logging.info(
            f"Joint augmentation enabled: flip_p={args.aug_flip_p}, rot90_p={args.aug_rot90_p}, "
            f"elastic_p={args.aug_elastic_p} (alpha={args.aug_elastic_alpha}, sigma={args.aug_elastic_sigma}), "
            f"color_p={args.aug_color_p}"
        )
    else:
        logging.info("Joint augmentation disabled")

    train_dataset = SegmentationDataset(
        args.data_root, 'train', transform, augment=augment,
        input_mode=args.train_input_mode,
        image_size=1024,
        foreground_crop_prob=args.foreground_crop_prob)
    # Validation keeps GT at its ORIGINAL resolution so val HD95/clDice/NSD
    # are directly comparable to test metrics (not 1024-space metrics).
    val_dataset = TestDataset(
        args.data_root, split='val', input_mode=args.eval_input_mode,
        image_size=1024)
    logging.info(
        f"Input modes: train={args.train_input_mode} "
        f"(fg_crop_prob={args.foreground_crop_prob}), "
        f"eval={args.eval_input_mode}")

    train_sampler = None
    val_sampler = None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, seed=args.seed, shuffle=True)
        # Unique validation shards: no padding/duplicate samples on small
        # 3-shot val sets.  Empty trailing ranks are handled by _validate.
        val_sampler = DistributedEvalSampler(val_dataset)
    # A local generator fixes RandomSampler/worker base seeds; worker_init_fn
    # fixes Python/NumPy RNGs inside each worker.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    num_workers = int(getattr(args, 'num_workers', 4))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=train_generator)
    # Variable original GT sizes: batch_size must be 1 for validation.
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=val_generator)

    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # Losses
    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    if args.scnp:
        criterion = SCNPCriterion(
            criterion,
            neighborhood_size=args.scnp_window,
            original_weight=args.scnp_original_weight,
        )
        logging.info(
            f"SCNP enabled: window={args.scnp_window}, "
            f"original_weight={args.scnp_original_weight} "
            f"(main BCE/Dice and Stage-1 deep supervision use SCNP logits)")
    soft_cl = SoftclDiceLoss(num_iter=args.cl_dice_iters)
    boundary_loss = None
    if args.boundary_loss_weight > 0:
        boundary_loss = BoundaryLoss(
            resolution=args.boundary_loss_resolution)
        logging.info(
            "Boundary loss: weight=%s resolution=%d warmup=%d ramp=%d",
            args.boundary_loss_weight, args.boundary_loss_resolution,
            args.boundary_loss_warmup, args.boundary_loss_ramp)
    try:
        topo_loss = build_topology_loss(args.topology_loss,
                                        resolution=args.topology_resolution)
    except TypeError:
        # 纯 torch 损失(decl 等)不收 resolution 参数
        topo_loss = build_topology_loss(args.topology_loss)
    if topo_loss is not None:
        logging.info(f"Extra topology loss: {args.topology_loss} x {args.topology_loss_weight}")
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    logging.info(
        f"Deep supervision weight: {args.deep_sup_weight}, clDice weight: {args.cl_dice_weight} "
        f"(warmup {args.cl_dice_warmup} ep, ramp {args.cl_dice_ramp} ep), IoU loss weight: {args.iou_loss_weight}"
    )

    # Parameter groups with separate learning rates (model-specific)
    optimizer = optim.AdamW(spec.param_groups(base_model, args))

    # Set up learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.patience, verbose=True
        )
    else:
        scheduler = None

    logging.info(f"Adapter learning rate: {args.adapter_lr}")
    logging.info(f"New decoder module learning rate: {args.new_module_lr}")
    logging.info(f"Original decoder learning rate: {args.decoder_lr}")
    logging.info(f"Scheduler: {args.scheduler}")

    resume_state = {}
    if resume_payload is not None:
        resume_state = _apply_resume_state(
            resume_payload, args, optimizer, scheduler,
            train_generator, val_generator, save_dir, artifact_info)
        resume_state['resumed_from'] = str(resume_path)
        if Path(resume_path).name == 'last_model.pth':
            resume_state['last_checkpoint'] = str(resume_path)
        logging.info(
            'Resume state: start_epoch=%d best_epoch=%s best_score=%s',
            resume_state['start_epoch'], resume_state['best_epoch'],
            resume_state['best_selection_score'])

    trainer = Trainer(
        model=sam, spec=spec, device=device,
        criterion=criterion, soft_cl=soft_cl, topo_loss=topo_loss,
        optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        args=args, is_ddp=is_ddp, train_sampler=train_sampler,
        boundary_loss=boundary_loss,
        artifact_info=artifact_info,
        train_generator=train_generator,
        val_generator=val_generator,
        resume_state=resume_state,
    )
    try:
        trainer.fit()
    finally:
        # fencing-algs 模式:进程组显式销毁(rank 异常退出时由 torchrun 兜底)
        if is_ddp:
            import torch.distributed as dist
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam_vit_l_0b3195.pth', help='Path to SAM checkpoint')
    parser.add_argument('--model', type=str, default='sam_l', choices=sorted(MODEL_REGISTRY),
                        help='Model architecture from the model registry (see docs/MODEL_MANAGEMENT.md)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1, help='Ratio of adapter dimension to model dimension')
    parser.add_argument('--adapter_lr', type=float, default=3e-4, help='Learning rate for adapter modules')
    parser.add_argument('--new_module_lr', type=float, default=1e-4, help='Learning rate for newly added decoder modules')
    parser.add_argument('--decoder_lr', type=float, default=1e-5, help='Learning rate for original mask decoder parameters (lowered to curb few-shot overfitting)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs (SACM-X default budget; explicit value overrides)')
    parser.add_argument('--val_interval', type=int, default=None,
                        help='Validation interval (epochs); default 1 for '
                             'log_style=baseline, 10 otherwise')
    parser.add_argument('--grad_checkpointing', type=str2bool, default=True,
                        help='Block-wise gradient checkpointing for the ViT '
                             'encoder (reduces training memory; default ON)')
    parser.add_argument('--train_input_mode', type=str, default='aspect_pad',
                        choices=['resize', 'aspect_pad', 'native_crop'],
                        help='Training geometry: resize (legacy square stretch), '
                             'aspect_pad (preserve aspect ratio + pad), or '
                             'native_crop (foreground-biased native-resolution '
                             'crop for large images; small images use aspect_pad)')
    parser.add_argument('--eval_input_mode', type=str, default='aspect_pad',
                        choices=['resize', 'aspect_pad'],
                        help='Validation/test geometry; aspect_pad preserves '
                             'the source aspect ratio. Use resize only for '
                             'legacy checkpoint comparison')
    parser.add_argument('--foreground_crop_prob', type=float, default=0.5,
                        help='native_crop: probability of centring the crop on '
                             'a random foreground pixel instead of a random '
                             'location')
    parser.add_argument('--fast_val', type=str2bool, default=False,
                        help='Validation computes only loss + F1; skips the full metric suite and '
                             'topology monitor (training-time speed; full metrics stay in evaluate.py). '
                             'log_style=baseline requires fast_val=false.')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', type=str2bool, default=True,
                        help='Enable deterministic RNG/kernel settings; '
                             'keep on for reproducible comparisons')
    parser.add_argument('--deterministic_strict', type=str2bool,
                        default=True,
                        help='Fail if an operator has no deterministic CUDA '
                             'implementation. Set false only for a '
                             'best-effort (not bitwise) run.')
    parser.add_argument('--deterministic_threads', type=int, default=1,
                        help='torch CPU threads in deterministic mode')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers; keep fixed across runs')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Where to save the best checkpoint (per-job paths for parallel runs)')
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from a checkpoint path, or 'last'/'best' "
                             'for <save_path dir>/last_model.pth or '
                             'best_model.pth; resumes at the next epoch boundary')
    parser.add_argument('--save_last_every', type=int, default=1,
                        help='Write last_model.pth every N epochs (default: 1)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Write checkpoints/epoch_XXXX.pth every N epochs '
                             '(0 disables periodic snapshots)')
    parser.add_argument('--keep_checkpoints', type=int, default=2,
                        help='Keep this many newest periodic checkpoints')
    parser.add_argument('--hash_data_files', type=str2bool, default=True,
                        help='Hash train/val files and test masks in '
                             'provenance/data_manifest.json (slower startup)')

    # Architecture preset (applied after parsing; use "none" for manual control)
    parser.add_argument('--preset', type=str, default='full', choices=sorted(PRESETS), help='Architecture/training preset for ablation experiments')
    parser.add_argument('--use_geo_i', type=str2bool, default=True, help='Geometric internal adapters (strip branch)')
    parser.add_argument('--use_geo_e', type=str2bool, default=True, help='Geometric external adapters (SE gate + strip, window layers only)')
    parser.add_argument('--geo_e_layers', type=str, default='all', choices=['all', 'shallow', 'deep'],
                        help="A2 layer placement: 'all' window layers / 'shallow' first half / 'deep' second half "
                             "(for ViT-L: {0-4,6-10} / {12-16,18-22})")
    parser.add_argument('--use_coarse_to_fine', type=str2bool, default=True, help='Closed-loop coarse-to-fine refinement')
    parser.add_argument('--use_fusion_v2', type=str2bool, default=True, help='Fusion v2 (concat+FFN weights + spatial gate); off = original SACM fusion')
    parser.add_argument('--use_multi_depth', type=str2bool, default=True, help='Multi-depth semantic path from encoder intermediates')
    parser.add_argument('--use_dynamic_strip', type=str2bool, default=False,
                        help='Use deformable (dynamic) strip branches in the '
                             'geometric adapters; requires torchvision deform_conv2d')
    parser.add_argument('--use_typed_heads', type=str2bool, default=False,
                        help='Typed structural Stage-2 heads (thin/blob/boundary/recall); '
                             'opt-in via preset full_typed or this flag')
    parser.add_argument('--use_tcsu', type=str2bool, default=False,
                        help='PEPA target-conditioned snake upsampling residual '
                             '(opt-in through preset full_pepa)')
    parser.add_argument('--tcsu_scale', type=int, default=4,
                        help='TCSU upsampling factor (4 maps 64x64 to 256x256)')
    parser.add_argument('--tcsu_mid_channels', type=int, default=32,
                        help='TCSU snake branch width')
    parser.add_argument('--tcsu_max_length', type=int, default=5,
                        help='TCSU maximum odd snake chain length')
    parser.add_argument('--use_tadt', type=str2bool, default=False,
                        help='PEPA per-candidate adaptive logit threshold '
                             '(opt-in through preset full_pepa)')
    parser.add_argument('--tadt_consistency_weight', type=float, default=0.0,
                        help='Weight of the threshold-perturbation consistency '
                             'loss for TADT')
    parser.add_argument('--tadt_delta', type=float, default=0.5,
                        help='Logit perturbation margin for TADT consistency')
    parser.add_argument('--tadt_hidden', type=int, default=128,
                        help='TADT MLP hidden width')
    parser.add_argument('--tadt_t_min', type=float, default=-4.0,
                        help='TADT minimum logit threshold')
    parser.add_argument('--tadt_t_max', type=float, default=4.0,
                        help='TADT maximum logit threshold')
    parser.add_argument('--use_ranker', type=str2bool, default=False,
                        help='PromptMoE-style candidate ranker replacing the '
                             'IoU head for selection (opt-in via full_pepa_rank)')
    parser.add_argument('--ranker_hidden', type=int, default=128,
                        help='Candidate ranker utility MLP width')
    parser.add_argument('--ranker_pair_hidden', type=int, default=128,
                        help='Candidate ranker pairwise MLP width')
    parser.add_argument('--rank_loss_weight', type=float, default=0.0,
                        help='Listwise candidate-ranking loss weight')
    parser.add_argument('--use_detail_stem', type=str2bool, default=False,
                        help='Parallel high-resolution detail stem '
                             '(opt-in via full_pepa_hr)')
    parser.add_argument('--detail_channels', type=int, default=32,
                        help='Detail stem 1/2-resolution width')
    parser.add_argument('--detail_out_channels', type=int, default=32,
                        help='Detail stem output channels')
    parser.add_argument('--high_res_output', type=str2bool, default=False,
                        help='Produce Stage-2 masks at high_res_size')
    parser.add_argument('--high_res_size', type=int, default=512,
                        help='High-resolution Stage-2 output size')
    parser.add_argument('--high_res_refine', type=str2bool, default=False,
                        help='Image-guided full-resolution logit refinement')
    parser.add_argument('--high_res_refine_size', type=int, default=1024,
                        help='Output size of the image-guided refinement')
    parser.add_argument('--single_head', type=str2bool, default=False,
                        help='Use one Stage-2 mask head (no fixed2/selector); '
                             'opt-in via full_v2')
    # ---- V3 dense structural decoder ----
    parser.add_argument('--use_v3', type=str2bool, default=False,
                        help='Use the v3 dense structural decoder instead of '
                             'the SAM prompt decoder (opt-in via full_v3)')
    parser.add_argument('--v3_feature_dim', type=int, default=128,
                        help='V3 FPN feature width')
    parser.add_argument('--v3_mid_dim', type=int, default=96,
                        help='V3 1/4-resolution feature width')
    parser.add_argument('--v3_fine_dim', type=int, default=64,
                        help='V3 1/2-resolution feature width')
    parser.add_argument('--v3_refine_channels', type=int, default=16,
                        help='V3 1024-refinement hidden width')
    parser.add_argument('--v3_aux_centerline', type=str2bool, default=True,
                        help='V3: predict a training-only centerline head')
    parser.add_argument('--v3_aux_boundary', type=str2bool, default=True,
                        help='V3: predict a training-only boundary head')
    parser.add_argument('--v3_aux_width', type=str2bool, default=True,
                        help='V3: predict a training-only width head')
    parser.add_argument('--v3_centerline_weight', type=float, default=0.0,
                        help='V3 centerline BCE+Dice loss weight')
    parser.add_argument('--v3_boundary_weight', type=float, default=0.0,
                        help='V3 boundary BCE+Dice loss weight')
    parser.add_argument('--v3_width_weight', type=float, default=0.0,
                        help='V3 width L1 loss weight')
    parser.add_argument('--v3_area_weight', type=float, default=0.0,
                        help='V3 foreground-area consistency weight')
    parser.add_argument('--use_internal_adapter', type=str2bool, default=True,
                        help='Keep the per-block internal adapters. V3 sets '
                             'this to False to remove the duplicate adapter')
    parser.add_argument('--collect_adapter_features', type=str2bool,
                        default=True,
                        help='Collect all external-adapter outputs for the '
                             'decoder fusion. V3 disables it to save memory')
    parser.add_argument('--init_stage2_from_stage1', type=str2bool, default=True,
                        help='Warm-start the new Stage-2 mask heads from the '
                             'loaded pretrained Stage-1 heads (applied after '
                             'the base SAM checkpoint is loaded); default ON '
                             'for SACM-X')

    # Auxiliary losses
    parser.add_argument('--deep_sup_weight', type=float, default=0.3, help='Weight of Stage-1 deep supervision (all heads)')
    parser.add_argument('--stage2_supervision', type=str, default='soft',
                        choices=['wta', 'allhead', 'soft', 'typed'],
                        help='Stage-2 candidate supervision: soft (default D8 fix; '
                             'uniform + IoU-softmax mixture, all candidates get gradient), '
                             'allhead (mean over all candidates; may collapse), '
                             'typed (per-head structural losses; requires --use_typed_heads), '
                             'wta (original winner-take-all; baseline/ablation only)')
    parser.add_argument('--soft_select_mix', type=float, default=0.5,
                        help='soft mode: uniform/(IoU-softmax) mixture; 0=uniform all-head, 1=trust IoU head')
    parser.add_argument('--soft_select_tau', type=float, default=1.0,
                        help='soft mode: temperature for softmax(iou_pred / tau)')
    parser.add_argument('--val_selection', type=str, default='auto',
                        choices=['auto', 'iou', 'gating', 'quality', 'fixed'],
                        help='Mask used for validation/checkpoint selection: '
                             'auto -> iou when iou_loss_weight>0, else gating; '
                             'quality = IoU score minus fragmentation penalties; '
                             'fixed = one fixed Stage-2 head (--val_fixed_head)')
    parser.add_argument('--val_fixed_head', type=int, default=2,
                        choices=[0, 1, 2, 3],
                        help='Stage-2 head used by --val_selection fixed '
                             '(default: 2)')
    parser.add_argument('--val_empty_guard', type=str2bool, default=False,
                        help='Apply the largest-area fallback during validation '
                             'when the fixed/quality selection is empty')
    parser.add_argument('--quality_small_component_weight', type=float,
                        default=0.5,
                        help='quality selection: penalty for small-component rate')
    parser.add_argument('--quality_component_weight', type=float, default=0.05,
                        help='quality selection: penalty for log(1 + n_components)')
    parser.add_argument('--selection_metric', type=str, default='dice',
                        choices=['dice', 'f1', 'cldice'],
                        help='Validation metric used to select the best checkpoint: '
                             'dice (default; macro per-image Dice under full '
                             'validation, matching the baseline runner; micro F1 '
                             'fallback under fast_val), f1 (global micro F1), or '
                             'cldice (requires full validation).')
    parser.add_argument('--pred_threshold', type=float, default=0.0,
                        help='Logit threshold for validation F1/metrics; keep equal '
                             'to evaluation --pred_threshold')
    parser.add_argument('--log_style', type=str, default='table',
                        choices=['table', 'official', 'baseline'],
                        help="Terminal/W&B monitoring style: 'table' = rich "
                             "full-metric table; 'official' = SACM-style train "
                             "loss / val loss / F1 / LR only; 'baseline' = the "
                             "baseline runner's macro Dice/IoU/clDice/HD95 + "
                             "per-dataset W&B keys")
    parser.add_argument('--no-fov', dest='no_fov', action='store_true',
                        help='Do not mask predictions/GT by the FOV; matches the '
                             'official SACM adapter and paper-metric predictions')
    parser.add_argument('--cl_dice_weight', type=float, default=0.5, help='Max weight of the differentiable clDice loss')
    parser.add_argument('--cl_dice_warmup', type=int, default=10, help='Epochs before the clDice loss activates')
    parser.add_argument('--cl_dice_ramp', type=int, default=5, help='Epochs for the clDice weight to ramp to its max')
    parser.add_argument('--cl_dice_iters', type=int, default=10, help='Soft-skeleton iterations of the clDice loss (official clDice repo default: 10)')
    parser.add_argument('--iou_loss_weight', type=float, default=1.0, help='Weight of the IoU head MSE supervision')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.0,
                        help='Max weight of the distance-transform boundary '
                             'loss; 0 disables it (legacy behaviour).')
    parser.add_argument('--boundary_loss_resolution', type=int, default=512,
                        help='Resolution used for the boundary distance maps')
    parser.add_argument('--boundary_loss_warmup', type=int, default=10,
                        help='Epochs before the boundary loss activates')
    parser.add_argument('--boundary_loss_ramp', type=int, default=5,
                        help='Epochs for the boundary weight to ramp up')
    parser.add_argument('--tversky_weight', type=float, default=0.0,
                        help='Typed-head recall loss weight (Tversky, beta>alpha); '
                             '0 disables it outside the full_typed preset')
    parser.add_argument('--tversky_alpha', type=float, default=0.3,
                        help='Tversky FP weight (lower -> recall-oriented when beta>alpha)')
    parser.add_argument('--tversky_beta', type=float, default=0.7,
                        help='Tversky FN weight (higher -> recall-oriented)')

    # Full val metric suite: 16 常规指标 + 4 可选指标全部默认开启
    # (few-shot val 集仅 3 图,引擎级指标秒级可承受;传 --val_auc false 等可关)
    parser.add_argument('--val_auc', type=str2bool, default=True,
                        help='Compute dice_auc/cldice_auc on the val set (~9x skeletonization cost)')
    parser.add_argument('--val_betti_matching', type=str2bool, default=True,
                        help='Compute betti_matching on the val set (persistence engine, seconds per image)')
    parser.add_argument('--val_topograph', type=str2bool, default=True,
                        help='Compute topograph_error on the val set (component graph, seconds per image)')

    # Extra topology loss (strict ports of official implementations)
    parser.add_argument('--topology_loss', type=str, default='none',
                        choices=['none'] + sorted(TOPOLOGY_LOSS_REGISTRY),
                        help='Extra topology loss added to the training objective. Official ports: '
                             'decl (endpoint connectivity), betti (Betti matching), dice_betti '
                             '(Dice+Betti), wasserstein / composed_wasserstein (Wasserstein '
                             'matching family; composed needs gudhi), centerline_ce / dice_cldice / '
                             'ce_cldice / ce_clce (centerline-CE family, MICCAI 2024), satloss '
                             '(ICCVW 2025; needs gudhi+torch-topological+POT), topograph (component '
                             'graph; needs networkx+scipy+Python>=3.10). Losses that internally '
                             'combine their own Dice should use a small --topology_loss_weight. '
                             'betti/dice_betti/wasserstein/composed_wasserstein/satloss/topograph '
                             'evaluate at 256 resolution (engine performance).')
    parser.add_argument('--topology_loss_weight', type=float, default=0.1,
                        help='Weight of the extra topology loss term')
    parser.add_argument('--topology_resolution', type=int, default=256,
                        help='Resolution for engine-based topology losses (betti/topograph 等); '
                             '训练太慢可降到 128')

    # wandb monitoring (self-hosted server at 172.16.1.7; failures are non-fatal)
    parser.add_argument('--use_wandb', type=str2bool, default=True,
                        help='Log training/validation metrics to wandb (project sacm on the 172.16.1.7 server)')
    parser.add_argument('--wandb_project', type=str, default='sacm', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='buaazqk', help='wandb entity')
    parser.add_argument('--wandb_host', type=str, default='http://172.16.1.7:8080',
                        help='Self-hosted wandb base URL')
    parser.add_argument('--wandb_api_key', type=str, default=WANDB_API_KEY_DEFAULT,
                        help='wandb API key (default: WANDB_API_KEY env; empty '
                             'falls back to the wandb login/config store)')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name (default: model_preset)')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated wandb tags')

    # Data augmentation
    parser.add_argument('--no_augment', action='store_true', help='Disable joint data augmentation')
    parser.add_argument('--aug_flip_p', type=float, default=0.5, help='Probability of horizontal/vertical flips')
    parser.add_argument('--aug_rot90_p', type=float, default=0.5, help='Probability of 90-degree rotations')
    parser.add_argument('--aug_elastic_p', type=float, default=0.3, help='Probability of elastic deformation')
    parser.add_argument('--aug_elastic_alpha', type=float, default=8.0, help='Elastic deformation magnitude (pixels)')
    parser.add_argument('--aug_elastic_sigma', type=float, default=3.0, help='Elastic deformation smoothness')
    parser.add_argument('--aug_color_p', type=float, default=0.2, help='Color augmentation probability (brightness/contrast); automatically skipped for grayscale images (R==G==B), so RGB and grayscale modalities can share one training split')

    # Loss function weights
    parser.add_argument('--bce_weight', type=float, default=0.7, help='Weight for BCE loss in combined loss')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Weight for Dice loss in combined loss')
    parser.add_argument('--scnp', type=str2bool, default=False,
                        help='Enable Same-Class Neighbor Penalization (SCNP, CVPR 2026): '
                             'replace each logit with the hardest same-class neighbor '
                             'logit before BCE/Dice. Training-only, no inference cost; '
                             'orthogonal to clDice/topology terms.')
    parser.add_argument('--scnp_window', type=int, default=3,
                        help='SCNP pooling window (positive odd integer; official default: 3)')
    parser.add_argument('--scnp_original_weight', type=float, default=0.0,
                        help='SCNP: extra weight of the raw-logit BCE/Dice term; '
                             '0 = official SCNP(CEDice) objective (only transformed logits)')

    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'], help='LR scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR and ReduceLROnPlateau schedulers')
    parser.add_argument('--patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler')

    # YAML config overrides (applied after the preset)
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML file (see configs/presets.yaml) applied on top of the preset')

    args = parser.parse_args()
    apply_preset(args, args.preset)
    if args.config is not None:
        from models.sacm_x.configs import apply_yaml_presets
        apply_yaml_presets(args, args.config)
    if args.val_interval is None:
        args.val_interval = 1 if args.log_style == 'baseline' else 10
    if args.log_style == 'baseline' and args.fast_val:
        parser.error(
            "--log_style baseline requires full validation; "
            "use --fast_val false (or MONITOR=official for F1-only).")
    train(args)
