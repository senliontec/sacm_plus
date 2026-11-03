"""Evaluation CLI entry point (algorithm-agnostic).

Thin wrapper: parses arguments, builds the model via the registry, loads
weights, then delegates the whole evaluation to core.evaluator.Evaluator.

Usage:
    python evaluate.py --algorithm sacm --model sam_l \
        --data_root <root> --trained_weights best_model.pth \
        --output_dir results/name --selection iou --tta
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# src-layout bootstrap: make the packages importable without an editable
# install (harmless when `pip install -e .` is also in effect).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import models  # noqa: F401  (registers every model family's ModelSpecs)
from core.data import TestDataset
from core.evaluator import (
    AUC_KEYS,
    BETTI_MATCHING_KEYS,
    METRIC_KEYS,
    TOPOGRAPH_KEYS,
    Evaluator,
    resolve_selection,
)
from core.losses import DiceBCELoss
from core.registry import MODEL_REGISTRY, get_model_spec
from core.reproducibility import configure_determinism, seed_worker
from core.wandb_utils import WANDB_API_KEY_DEFAULT, redact_secrets
from models.sacm_x.configs import PRESETS, apply_preset, str2bool


def setup_logging():
    os.makedirs('results/logs', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'results/logs/test_{timestamp}.log'

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

    reproducibility = configure_determinism(
        getattr(args, 'seed', 0),
        deterministic=getattr(args, 'deterministic', True),
        strict=getattr(args, 'deterministic_strict', True),
        num_threads=getattr(args, 'deterministic_threads', 1),
        cuda_seed_all=True,
    )
    args.reproducibility_info = reproducibility
    logging.info("Reproducibility: %s", reproducibility)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Test Configuration:")
    for arg, value in redact_secrets(args).items():
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
        use_typed_heads=getattr(args, 'use_typed_heads', False),
    )
    sam.to(device)

    # Log adapter information
    external_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'external_adapters' in name)
    internal_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'blocks' in name and 'adapter' in name)
    logging.info(f"Model has {internal_adapter_count} internal block adapters and {external_adapter_count} external adapters")

    # Load trained weights.  PyTorch >=2.6 defaults weights_only=True, but
    # this is a trusted project checkpoint containing config metadata and a
    # TorchVersion object; the baseline runtime already loads it with
    # weights_only=False.
    checkpoint = torch.load(
        args.trained_weights, map_location=device, weights_only=False)
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

    # D8 / SACM-fidelity guard: never silently argmax an untrained IoU head.
    ckpt_cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    iou_weight = ckpt_cfg.get('iou_loss_weight',
                              getattr(args, 'iou_loss_weight', 0.0))
    # The best checkpoint was selected under the training-side selection and
    # threshold.  When the evaluator is left at its defaults, reuse those
    # settings instead of silently evaluating a different decision rule.
    if args.selection == 'auto':
        saved_selection = ckpt_cfg.get('val_selection')
        if saved_selection in ('iou', 'gating'):
            args.selection = saved_selection
            logging.info(f"Using checkpoint val_selection={saved_selection}")
    if args.pred_threshold == 0.0 and 'pred_threshold' in ckpt_cfg:
        args.pred_threshold = float(ckpt_cfg['pred_threshold'])
        logging.info(f"Using checkpoint pred_threshold={args.pred_threshold}")
    if ckpt_cfg.get('no_fov') and not args.no_fov:
        args.no_fov = True
        logging.info("Using checkpoint no_fov=True (official comparison)")
    args.selection = resolve_selection(args.selection, iou_weight)
    if args.fixed_head is not None:
        logging.info(
            f"Fixed-head override: head {args.fixed_head}; "
            f"--selection={args.selection} is ignored for the final mask")
    elif args.selection == 'iou' and float(iou_weight or 0.0) <= 0:
        logging.warning(
            "selection='iou' was requested but the checkpoint has "
            "iou_loss_weight=0 (IoU head untrained); metrics are invalid. "
            "Use --selection auto or gating.")
    if args.selection == 'quality' and float(iou_weight or 0.0) <= 0:
        logging.warning(
            "selection='quality' uses the IoU head as one score term but "
            "the checkpoint has iou_loss_weight=0; compare with gating.")
    logging.info(
        f"Resolved mask selection: {args.selection} "
        f"(iou_loss_weight={iou_weight}, fixed_head={args.fixed_head})")

    eval_input_mode = getattr(args, 'eval_input_mode', 'aspect_pad')
    dataset = TestDataset(
        args.data_root, input_mode=eval_input_mode, image_size=1024)
    logging.info(
        f"Validation dataset size: {len(dataset)} "
        f"(eval_input_mode={eval_input_mode})")

    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    logging.info(f"Mask selection: {args.selection}; TTA: {args.tta}")

    os.makedirs(args.output_dir, exist_ok=True)
    # A failed evaluation must not leave the previous run's summary behind;
    # otherwise aggregate_results.py silently reuses stale numbers.
    for stale in ('results_summary.txt', 'metrics.csv'):
        stale_path = os.path.join(args.output_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # Expensive metrics are opt-in: the AUC sweep costs ~9x skeletonization;
    # Betti matching / Topograph engines cost seconds per image
    metric_keys = METRIC_KEYS + (AUC_KEYS if args.auc else []) + (
        BETTI_MATCHING_KEYS if args.betti_matching else []) + (
        TOPOGRAPH_KEYS if args.topograph_metric else [])

    deep_closing = None
    if getattr(args, 'deep_closing_checkpoint', None):
        from core.postprocess.deep_closing import DeepClosingRefiner
        deep_closing = DeepClosingRefiner.from_checkpoint(
            args.deep_closing_checkpoint,
            device=device,
            resolution=args.deep_closing_resolution,
            dilation_threshold=args.deep_closing_threshold,
            max_iters=args.deep_closing_max_iters,
        )
        logging.info(
            "Deep Closing enabled: checkpoint=%s resolution=%d "
            "dilation_threshold=%.3f max_iters=%s",
            args.deep_closing_checkpoint, deep_closing.resolution,
            deep_closing.dilation_threshold, deep_closing.max_iters)

    evaluator = Evaluator(
        sam, spec, device, criterion, args, metric_keys,
        deep_closing=deep_closing)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(getattr(args, 'seed', 0)) + 1)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=int(getattr(args, 'num_workers', 4)),
        worker_init_fn=seed_worker, generator=eval_generator)
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
    parser.add_argument('--output_dir', type=str, default='results/outputs/tyre', help='Directory to save test results')
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
    parser.add_argument('--use_typed_heads', type=str2bool, default=False,
                        help='Typed structural Stage-2 heads (must match the '
                             'checkpoint; full_typed preset enables it)')

    # Inference options
    parser.add_argument('--selection', type=str, default='auto',
                        choices=['auto', 'iou', 'gating', 'quality'],
                        help="Final mask selection: 'auto' = iou when the checkpoint "
                             "trained the IoU head (iou_loss_weight>0), else gating; "
                             "'iou' = trained IoU head; 'gating' = gating-ranked head 0; "
                             "'quality' = IoU score minus fragmentation penalties")
    parser.add_argument('--fixed_head', type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help='Force one Stage-2 head for every image; overrides '
                             '--selection. Intended for head-calibration audits, '
                             'not for final test tuning.')
    parser.add_argument('--no-fov', dest='no_fov', action='store_true',
                        help='Disable FOV masking to match the official SACM '
                             'adapter / paper-metric prediction protocol')
    parser.add_argument('--pred_threshold', type=float, default=0.0,
                        help='Binarization threshold on logits; 0.0 = probability 0.5 (standard). '
                             'Set 0.5 to replicate the original SACM evaluation convention '
                             '(used by the fidelity gate of the stage-0 protocol).')
    parser.add_argument('--eval_input_mode', type=str, default='aspect_pad',
                        choices=['resize', 'aspect_pad'],
                        help='Test geometry: aspect_pad preserves the source '
                             'aspect ratio and pads; resize is the legacy '
                             'square stretch.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for deterministic evaluation')
    parser.add_argument('--deterministic', type=str2bool, default=True,
                        help='Enable deterministic RNG/kernel settings')
    parser.add_argument('--deterministic_strict', type=str2bool,
                        default=True,
                        help='Fail on operators without deterministic CUDA '
                             'implementations')
    parser.add_argument('--deterministic_threads', type=int, default=1,
                        help='torch CPU threads in deterministic mode')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers; keep fixed across runs')
    parser.add_argument('--empty_guard', action='store_true',
                        help='If the selected mask is empty, fall back to the '
                             'candidate with the largest foreground area before '
                             'computing metrics. Mitigates head-collapse on '
                             'thick/dense domains without using test labels.')
    parser.add_argument('--quality_small_component_weight', type=float,
                        default=0.5,
                        help='quality selection: penalty for small-component rate')
    parser.add_argument('--quality_component_weight', type=float, default=0.05,
                        help='quality selection: penalty for log(1 + n_components)')
    parser.add_argument('--component_metrics', action='store_true',
                        help='Add connected-component/fragmentation metrics to '
                             'metrics.csv and summary (diagnostic extension).')
    parser.add_argument('--component_prune_mode', type=str, default='off',
                        choices=['off', 'absolute', 'relative'],
                        help='Optional component pruning before metrics.')
    parser.add_argument('--component_prune_min_area', type=int, default=0,
                        help='absolute area threshold for component pruning')
    parser.add_argument('--component_prune_min_skeleton', type=int, default=0,
                        help='absolute skeleton-length threshold for pruning')
    parser.add_argument('--component_prune_rel_area', type=float, default=0.0,
                        help='relative area threshold (fraction of image area)')
    parser.add_argument('--component_prune_rel_skeleton', type=float,
                        default=0.0,
                        help='relative skeleton threshold (fraction of max side)')
    parser.add_argument('--component_prune_keep_largest', type=int, default=1,
                        help='always keep this many largest components')
    parser.add_argument('--audit_heads', action='store_true',
                        help='Write head_audit.csv with per-candidate '
                             'Dice/IoU/P/R/component metrics.')
    parser.add_argument('--audit_heads_full', action='store_true',
                        help='Write head_full_audit.csv with the full metric '
                             'suite (clDice/HD95/ASSD/...) for every head. '
                             'Diagnostic-only and slower than --audit_heads.')
    parser.add_argument('--audit_thresholds', nargs='+', type=float,
                        default=None,
                        help='Write threshold_audit.csv for these logit '
                             'thresholds (lightweight metrics only).')
    parser.add_argument('--audit_head_thresholds', nargs='+', type=float,
                        default=None,
                        help='Write head_threshold_audit.csv for every head '
                             'at every given logit threshold (calibration grid).')
    parser.add_argument('--save_candidate_masks', action='store_true',
                        help='Save per-head binary candidate PNGs under '
                             'output_dir/candidates/.')
    parser.add_argument('--save_candidate_probs', action='store_true',
                        help='Save per-head probability maps as float16 '
                             'NPYs under output_dir/candidates/ (for the '
                             'offline topology ranker).')
    parser.add_argument('--nsd_tolerance', type=float, default=1.0,
                        help='Tolerance (pixels) for the surface Dice / NSD metric')
    parser.add_argument('--tta', action='store_true', help='Enable four-way flip test-time augmentation')
    parser.add_argument(
        '--deep_closing_checkpoint', type=str, default=None,
        help='Optional Deep Closing shape-prior checkpoint; applies Deep '
             'Dilation + Simple Component Erosion before metrics.')
    parser.add_argument(
        '--deep_closing_threshold', type=float, default=None,
        help='Override Deep Dilation threshold (checkpoint default: 0.2)')
    parser.add_argument(
        '--deep_closing_resolution', type=int, default=None,
        help='Override shape-prior working resolution (checkpoint default: 512)')
    parser.add_argument(
        '--deep_closing_max_iters', type=int, default=None,
        help='Override Simple Point Erosion iteration cap')
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
                        help='wandb API key (default: WANDB_API_KEY env; empty '
                             'falls back to the wandb login/config store)')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated wandb tags')

    # YAML config overrides (applied after the preset)
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML file (see configs/presets.yaml) applied on top of the preset')

    args = parser.parse_args()
    apply_preset(args, args.preset)
    if args.config is not None:
        from models.sacm_x.configs import apply_yaml_presets
        apply_yaml_presets(args, args.config)
    evaluate(args)
