"""Training CLI entry point (algorithm-agnostic).

Thin wrapper: parses arguments, builds everything (model via the model
registry, losses, optimizer, scheduler, data), then delegates the whole
training schedule to core.trainer.Trainer.

Usage:
    python train.py --algorithm sacm --model sam_l --preset stage2 \
        --data_root <root> --checkpoint <sam.pth>
"""

import os
import random
import argparse
import logging
import sys
from datetime import datetime

# src-layout bootstrap: make the packages importable without an editable
# install (harmless when `pip install -e .` is also in effect).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import models  # noqa: F401  (registers every model family's ModelSpecs)
from core.wandb_utils import WANDB_API_KEY_DEFAULT
from models.sacm.configs import PRESETS, apply_preset, str2bool
from core.augmentation import JointAugment
from core.data import SegmentationDataset
from core.losses import DiceBCELoss, SoftclDiceLoss, TOPOLOGY_LOSS_REGISTRY, build_topology_loss
from core.registry import MODEL_REGISTRY, get_model_spec
from core.trainer import Trainer


def setup_logging(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train(args):
    setup_logging(args)

    # The save path may point at a not-yet-existing job dir (run_all.sh /
    # run_experiments use results/<preset>/<dataset>/); create it up front
    # so checkpoint + metrics_history.json saves never fail.
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

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
    )
    sam.to(device)

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Model-specific freezing protocol (SAM: encoder adapters + mask
    # decoder only; other models define their own in their ModelSpec)
    spec.freeze(sam)

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"After freezing - Total parameters: {total_params:,}")
    logging.info(f"After freezing - Trainable parameters: {trainable_params:,}")
    logging.info(f"After freezing - Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

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

    train_dataset = SegmentationDataset(args.data_root, 'train', transform, augment=augment)
    val_dataset = SegmentationDataset(args.data_root, 'val', transform, augment=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # Losses
    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    soft_cl = SoftclDiceLoss(num_iter=args.cl_dice_iters)
    topo_loss = build_topology_loss(args.topology_loss)
    if topo_loss is not None:
        logging.info(f"Extra topology loss: {args.topology_loss} x {args.topology_loss_weight}")
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    logging.info(
        f"Deep supervision weight: {args.deep_sup_weight}, clDice weight: {args.cl_dice_weight} "
        f"(warmup {args.cl_dice_warmup} ep, ramp {args.cl_dice_ramp} ep), IoU loss weight: {args.iou_loss_weight}"
    )

    # Parameter groups with separate learning rates (model-specific)
    optimizer = optim.AdamW(spec.param_groups(sam, args))

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

    trainer = Trainer(
        model=sam, spec=spec, device=device,
        criterion=criterion, soft_cl=soft_cl, topo_loss=topo_loss,
        optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        args=args,
    )
    trainer.fit()


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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (aligned with the paper protocol)')
    parser.add_argument('--val_interval', type=int, default=10, help='Validation interval (epochs)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Where to save the best checkpoint (per-job paths for parallel runs)')

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

    # Auxiliary losses
    parser.add_argument('--deep_sup_weight', type=float, default=0.3, help='Weight of Stage-1 deep supervision (all heads)')
    parser.add_argument('--cl_dice_weight', type=float, default=0.5, help='Max weight of the differentiable clDice loss')
    parser.add_argument('--cl_dice_warmup', type=int, default=10, help='Epochs before the clDice loss activates')
    parser.add_argument('--cl_dice_ramp', type=int, default=5, help='Epochs for the clDice weight to ramp to its max')
    parser.add_argument('--cl_dice_iters', type=int, default=10, help='Soft-skeleton iterations of the clDice loss (official clDice repo default: 10)')
    parser.add_argument('--iou_loss_weight', type=float, default=1.0, help='Weight of the IoU head MSE supervision')

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

    # wandb monitoring (self-hosted server at 172.16.1.7; failures are non-fatal)
    parser.add_argument('--use_wandb', type=str2bool, default=True,
                        help='Log training/validation metrics to wandb (project sacm on the 172.16.1.7 server)')
    parser.add_argument('--wandb_project', type=str, default='sacm', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='buaazqk', help='wandb entity')
    parser.add_argument('--wandb_host', type=str, default='http://172.16.1.7:8080',
                        help='Self-hosted wandb base URL')
    parser.add_argument('--wandb_api_key', type=str, default=WANDB_API_KEY_DEFAULT,
                        help='wandb API key (self-hosted 172.16.1.7 local key, hardcoded per user request)')
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
        from models.sacm.configs import apply_yaml_presets
        apply_yaml_presets(args, args.config)
    train(args)
