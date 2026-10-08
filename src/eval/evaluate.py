"""Evaluation CLI entry point (algorithm-agnostic).

Thin wrapper: parses arguments, builds the model via the registry, loads
weights, then delegates the whole evaluation to core.evaluator.Evaluator.

Usage:
    python evaluate.py --algorithm sacm --model sam_l \
        --data_root <root> --trained_weights best_model.pth \
        --output_dir results/name --selection iou --tta
"""

import os
import argparse
import logging
import sys
from datetime import datetime

# src-layout bootstrap: make the packages importable without an editable
# install (harmless when `pip install -e .` is also in effect).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import models  # noqa: F401  (registers every model family's ModelSpecs)
from models.sacm.configs import PRESETS, apply_preset, str2bool
from core.data import TestDataset
from core.evaluator import (
    AUC_KEYS,
    BETTI_MATCHING_KEYS,
    METRIC_KEYS,
    TOPOGRAPH_KEYS,
    Evaluator,
)
from core.losses import DiceBCELoss
from core.wandb_utils import WANDB_API_KEY_DEFAULT
from core.registry import MODEL_REGISTRY, get_model_spec


def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_{timestamp}.log'

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


def evaluate(args):
    setup_logging()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Test Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Load model through the registry (architecture switches must match
    # the training configuration)
    spec = get_model_spec(args.model)
    logging.info(f"Model: {args.model} — {spec.description}")
    sam = spec.build(
        checkpoint=None,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        geo_e_layers=args.geo_e_layers,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
    )
    sam.to(device)

    # Log adapter information
    external_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'external_adapters' in name)
    internal_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'blocks' in name and 'adapter' in name)
    logging.info(f"Model has {internal_adapter_count} internal block adapters and {external_adapter_count} external adapters")

    # Load trained weights
    checkpoint = torch.load(args.trained_weights, map_location=device)
    try:
        sam.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"权重与当前架构不匹配(常见原因:训练/评测的 --preset 或 --geo_e_layers "
            f"不一致;checkpoint 记录的 config = {checkpoint.get('config')}): {e}"
        ) from e
    logging.info(f"Loaded trained weights from {args.trained_weights}")
    if 'f1_score' in checkpoint:
        logging.info(f"Checkpoint validation F1 score: {checkpoint['f1_score']:.4f}")
    if 'config' in checkpoint:
        logging.info(f"Checkpoint architecture config: {checkpoint['config']}")

    dataset = TestDataset(args.data_root)
    logging.info(f"Validation dataset size: {len(dataset)}")

    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    logging.info(f"Mask selection: {args.selection}; TTA: {args.tta}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Expensive metrics are opt-in: the AUC sweep costs ~9x skeletonization;
    # Betti matching / Topograph engines cost seconds per image
    metric_keys = METRIC_KEYS + (AUC_KEYS if args.auc else []) + (
        BETTI_MATCHING_KEYS if args.betti_matching else []) + (
        TOPOGRAPH_KEYS if args.topograph_metric else [])

    evaluator = Evaluator(sam, spec, device, criterion, args, metric_keys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    summary, csv_rows, n_hd95_nan, avg_loss = evaluator.run(loader, args.output_dir)

    print("\nTest Results:")
    for k, (mean, std) in summary.items():
        print(f"Average {k.capitalize()}: {mean:.4f} ± {std:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    Evaluator.save_outputs(args.output_dir, summary, csv_rows, n_hd95_nan, avg_loss, args)
    logging.info(f"Per-image metrics saved to {os.path.join(args.output_dir, 'metrics.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--trained_weights', type=str, default='best_model.pth', help='Path to trained weights')
    parser.add_argument('--output_dir', type=str, default='outputs/tyre', help='Directory to save test results')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1, help='Ratio of adapter dimension to model dimension')
    parser.add_argument('--model', type=str, default='sam_l', choices=sorted(MODEL_REGISTRY),
                        help='Model architecture from the model registry (must match training)')
    parser.add_argument('--bce_weight', type=float, default=0.7, help='Weight for BCE loss in combined loss')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Weight for Dice loss in combined loss')

    # Architecture switches (must match the trained checkpoint)
    parser.add_argument('--preset', type=str, default='full', choices=sorted(PRESETS), help='Architecture preset (must match training)')
    parser.add_argument('--use_geo_i', type=str2bool, default=True, help='Geometric internal adapters (strip branch)')
    parser.add_argument('--use_geo_e', type=str2bool, default=True, help='Geometric external adapters (SE gate + strip, window layers only)')
    parser.add_argument('--geo_e_layers', type=str, default='all', choices=['all', 'shallow', 'deep'],
                        help="A2 layer placement: 'all' window layers / 'shallow' first half / 'deep' second half "
                             "(for ViT-L: {0-4,6-10} / {12-16,18-22})")
    parser.add_argument('--use_coarse_to_fine', type=str2bool, default=True, help='Closed-loop coarse-to-fine refinement')
    parser.add_argument('--use_fusion_v2', type=str2bool, default=True, help='Fusion v2 (concat+FFN weights + spatial gate)')
    parser.add_argument('--use_multi_depth', type=str2bool, default=True, help='Multi-depth semantic path from encoder intermediates')

    # Inference options
    parser.add_argument('--selection', type=str, default='iou', choices=['iou', 'gating'],
                        help="Final mask selection: 'iou' = trained IoU head; 'gating' = gating-ranked head 0")
    parser.add_argument('--pred_threshold', type=float, default=0.0,
                        help='Binarization threshold on logits; 0.0 = probability 0.5 (standard). '
                             'Set 0.5 to replicate the original SACM evaluation convention '
                             '(used by the fidelity gate of the stage-0 protocol).')
    parser.add_argument('--nsd_tolerance', type=float, default=1.0,
                        help='Tolerance (pixels) for the surface Dice / NSD metric')
    parser.add_argument('--tta', action='store_true', help='Enable four-way flip test-time augmentation')
    parser.add_argument('--auc', action='store_true',
                        help='Additionally compute dice_auc / cldice_auc (mean over a '
                             'probability-threshold sweep in [0.1, 0.9]); ~9x slower '
                             'evaluation due to repeated skeletonization')
    parser.add_argument('--betti_matching', action='store_true',
                        help='Additionally compute betti_matching (persistence-based Betti '
                             'matching error, official engine, evaluated at 256 resolution; '
                             'seconds per image — use on small subsets)')
    parser.add_argument('--topograph_metric', action='store_true',
                        help='Additionally compute topograph_error (official Topograph '
                             'critical-neighbor error count, evaluated at 256 resolution; '
                             'requires networkx + scipy)')

    # wandb monitoring (self-hosted server at 172.16.1.7; failures are non-fatal)
    parser.add_argument('--use_wandb', type=str2bool, default=True,
                        help='Log the evaluation metrics to wandb (project sacm on the 172.16.1.7 server)')
    parser.add_argument('--wandb_project', type=str, default='sacm', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='buaazqk', help='wandb entity')
    parser.add_argument('--wandb_host', type=str, default='http://172.16.1.7:8080',
                        help='Self-hosted wandb base URL')
    parser.add_argument('--wandb_api_key', type=str, default=WANDB_API_KEY_DEFAULT,
                        help='wandb API key (self-hosted 172.16.1.7 local key, hardcoded per user request)')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated wandb tags')

    # YAML config overrides (applied after the preset)
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML file (see configs/presets.yaml) applied on top of the preset')

    args = parser.parse_args()
    apply_preset(args, args.preset)
    if args.config is not None:
        from models.sacm.configs import apply_yaml_presets
        apply_yaml_presets(args, args.config)
    evaluate(args)
