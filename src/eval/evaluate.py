"""Evaluation CLI entry point (algorithm-agnostic).

Thin wrapper: parses arguments, builds the model via the registry, loads
weights, then delegates the whole evaluation to core.evaluator.Evaluator.

Usage:
    python evaluate.py --algorithm sacm --model sam_l \
        --data_root <root> --trained_weights best_model.pth \
        --output_dir results/name --selection iou --tta

    # Evaluate several datasets with ONE model load (outputs are written to
    # <output_dir>/<NAME> exactly as in the per-dataset mode):
    python evaluate.py --test_datasets DRIVE:data/.../DRIVE \
        CHASEDB1:data/.../CHASEDB1 --trained_weights best_model.pth \
        --output_dir results/name
"""

import argparse
import copy
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# src-layout bootstrap: make the packages importable without an editable
# install (harmless when `pip install -e .` is also in effect).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import models  # noqa: F401  (registers every model family's ModelSpecs)
from core.data import SUPPORTED_EXTENSIONS, TestDataset
from core.evaluator import (
    AUC_KEYS,
    BETTI_MATCHING_KEYS,
    METRIC_KEYS,
    PAPER_METRIC_KEYS,
    TOPOGRAPH_KEYS,
    Evaluator,
    resolve_selection,
)
from core.losses import DiceBCELoss
from core.registry import MODEL_REGISTRY, get_model_spec
from core.reproducibility import configure_determinism, seed_worker
from core.run_artifacts import write_json_atomic, write_prediction_index
from core.wandb_utils import WANDB_API_KEY_DEFAULT, redact_secrets
from models.skein.configs import PRESETS, apply_preset, str2bool


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


def _resolve_metric_keys(args):
    """Return the metric keys to compute for this evaluation."""
    if getattr(args, 'paper_metrics_only', False):
        ignored = [
            flag for flag, enabled in (
                ('--auc', getattr(args, 'auc', False)),
                ('--betti_matching', getattr(args, 'betti_matching', False)),
                ('--topograph_metric', getattr(
                    args, 'topograph_metric', False)),
            ) if enabled
        ]
        if ignored:
            logging.warning(
                "--paper_metrics_only ignores %s; only Dice/IoU/clDice/HD95 "
                "are computed.", ", ".join(ignored))
        if getattr(args, 'audit_heads_full', False):
            logging.warning(
                "--paper_metrics_only also limits head_full_audit to "
                "dice/iou/cldice/hd95.")
        return list(PAPER_METRIC_KEYS)
    return METRIC_KEYS + (AUC_KEYS if args.auc else []) + (
        BETTI_MATCHING_KEYS if args.betti_matching else []) + (
        TOPOGRAPH_KEYS if args.topograph_metric else [])


def _prepare_evaluation(args):
    """Build the model, load the checkpoint and resolve eval settings once."""
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
        use_adapter=getattr(args, 'use_adapter', True),
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_geo_i=args.use_geo_i,
        use_geo_e=args.use_geo_e,
        geo_e_layers=args.geo_e_layers,
        use_coarse_to_fine=args.use_coarse_to_fine,
        use_fusion_v2=args.use_fusion_v2,
        use_multi_depth=args.use_multi_depth,
        use_dynamic_strip=getattr(args, 'use_dynamic_strip', False),
        use_typed_heads=getattr(args, 'use_typed_heads', False),
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
        use_structural_aux=getattr(args, 'use_structural_aux', False),
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
        use_convlora=getattr(args, 'use_convlora', False),
        convlora_rank=getattr(args, 'convlora_rank', 4),
        convlora_alpha=getattr(args, 'convlora_alpha', 32.0),
        convlora_experts=getattr(args, 'convlora_experts', 8),
        use_evp=getattr(args, 'use_evp', False),
        evp_freq_nums=getattr(args, 'evp_freq_nums', 0.25),
        evp_scale_factor=getattr(args, 'evp_scale_factor', 32),
        evp_zero_init=getattr(args, 'evp_zero_init', True),
        use_res_moe=getattr(args, 'use_res_moe', False),
        res_num_experts=getattr(args, 'res_num_experts', 4),
        res_hidden=getattr(args, 'res_hidden', 24),
        res_gate_hidden=getattr(args, 'res_gate_hidden', 32),
        res_gate_size=getattr(args, 'res_gate_size', 512),
        res_gate_mode=getattr(args, 'res_gate_mode', 'learned'),
        res_use_head_disagreement=getattr(
            args, 'res_use_head_disagreement', True),
        res_use_centerline=getattr(args, 'res_use_centerline', True),
        res_gate_input_norm=getattr(args, 'res_gate_input_norm', True),
        res_out_size=getattr(args, 'res_out_size', 512),
        res_head_index=getattr(args, 'res_head_index', 2),
        res_dropout=getattr(args, 'res_dropout', 0.0),
        res_disagreement_tau=getattr(
            args, 'res_disagreement_tau', 0.05),
        res_disagreement_scale=getattr(
            args, 'res_disagreement_scale', 0.10),
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
    from models.sam_network import load_compatible_state_dict
    try:
        load_info = load_compatible_state_dict(
            sam, checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"权重与当前架构不匹配(常见原因:训练/评测的 --preset 或 --geo_e_layers "
            f"不一致;checkpoint 记录的 config = {checkpoint.get('config')}): {e}"
        ) from e
    if load_info['dropped']:
        logging.info(
            "Loaded with %d dropped extra keys (e.g. gate-only residual_moe); "
            "%d/%d target keys loaded",
            len(load_info['dropped']), load_info['loaded'],
            load_info['target'])
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
        if saved_selection in ('iou', 'gating', 'quality'):
            args.selection = saved_selection
            logging.info(f"Using checkpoint val_selection={saved_selection}")
        elif saved_selection == 'fixed':
            saved_head = ckpt_cfg.get('val_fixed_head')
            if args.fixed_head is None and saved_head is not None:
                args.fixed_head = int(saved_head)
                logging.info(
                    "Using checkpoint val_selection=fixed, "
                    f"val_fixed_head={args.fixed_head}")
    if args.selection == 'quality':
        for key in ('quality_small_component_weight',
                    'quality_component_weight'):
            if key in ckpt_cfg:
                setattr(args, key, float(ckpt_cfg[key]))
    if ckpt_cfg.get('val_empty_guard') and not args.empty_guard:
        args.empty_guard = True
        logging.info("Using checkpoint val_empty_guard=True")
    if args.pred_threshold == 0.0 and 'pred_threshold' in ckpt_cfg:
        args.pred_threshold = float(ckpt_cfg['pred_threshold'])
        logging.info(f"Using checkpoint pred_threshold={args.pred_threshold}")
    if ckpt_cfg.get('no_fov') and not args.no_fov:
        args.no_fov = True
        logging.info("Using checkpoint no_fov=True (official comparison)")
    # The final mask is fixed-head when --fixed_head is set; avoid the
    # irrelevant untrained-IoU warning for the unused selection argument.
    if args.fixed_head is not None and args.selection == 'auto':
        args.selection = 'gating'
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

    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    logging.info(f"Mask selection: {args.selection}; TTA: {args.tta}")

    metric_keys = _resolve_metric_keys(args)
    logging.info("Metric suite (%d): %s", len(metric_keys),
                 ", ".join(metric_keys))

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

    return {
        'device': device,
        'spec': spec,
        'sam': sam,
        'criterion': criterion,
        'metric_keys': metric_keys,
        'deep_closing': deep_closing,
    }


def _prediction_index_rows(dataset, output_dir):
    """Map every saved prediction PNG to its image and GT mask."""
    rows = []
    for image_name in dataset.image_files:
        base_name = os.path.splitext(image_name)[0]
        mask_path = None
        for ext in SUPPORTED_EXTENSIONS:
            candidate = os.path.join(dataset.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            continue
        rows.append({
            'dataset': Path(str(dataset.root_dir).rstrip('/')).name,
            'image': image_name,
            # Relative to the per-dataset output dir so the index survives
            # finalize_run moving that directory into the canonical tree.
            'pred_path': f'{image_name}_result.png',
            'gt_path': os.path.abspath(mask_path),
            'image_path': os.path.abspath(os.path.join(
                dataset.image_dir, image_name)),
        })
    return rows


def _write_eval_artifacts(dataset, ctx, args):
    dataset_name = Path(str(args.data_root).rstrip('/')).name
    rows = _prediction_index_rows(dataset, args.output_dir)
    write_prediction_index(
        args.output_dir, dataset_name, rows,
        manifest_extra={
            'data_root': os.path.abspath(str(args.data_root)),
            'image_dir': os.path.abspath(dataset.image_dir),
            'mask_dir': os.path.abspath(dataset.mask_dir),
            'metrics_csv': os.path.abspath(os.path.join(
                args.output_dir, 'metrics.csv')),
            'results_summary': os.path.abspath(os.path.join(
                args.output_dir, 'results_summary.txt')),
            'trained_weights': os.path.abspath(str(args.trained_weights)),
            'model': args.model,
            'preset': args.preset,
            'use_typed_heads': bool(getattr(args, 'use_typed_heads', False)),
            'seed': int(getattr(args, 'seed', 0)),
            'n_images': len(rows),
            'eval_input_mode': getattr(args, 'eval_input_mode', 'aspect_pad'),
            'selection': getattr(args, 'selection', None),
            'fixed_head': getattr(args, 'fixed_head', None),
            'tta': bool(getattr(args, 'tta', False)),
            'empty_guard': bool(getattr(args, 'empty_guard', False)),
            'no_fov': bool(getattr(args, 'no_fov', False)),
            'pred_threshold': float(getattr(args, 'pred_threshold', 0.0)),
            'metric_keys': list(ctx.get('metric_keys', [])),
        })
    logging.info(
        'Prediction index saved to %s',
        os.path.join(args.output_dir, 'prediction_index.csv'))


def _run_dataset(ctx, args):
    """Evaluate one dataset with the prepared model."""
    eval_input_mode = getattr(args, 'eval_input_mode', 'aspect_pad')
    dataset = TestDataset(
        args.data_root, input_mode=eval_input_mode, image_size=1024)
    logging.info(
        f"Validation dataset size: {len(dataset)} "
        f"(eval_input_mode={eval_input_mode})")

    os.makedirs(args.output_dir, exist_ok=True)
    # A failed evaluation must not leave the previous run's summary behind;
    # otherwise aggregate_results.py silently reuses stale numbers.
    for stale in ('results_summary.txt', 'metrics.csv', 'eval_error.txt'):
        stale_path = os.path.join(args.output_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    evaluator = Evaluator(
        ctx['sam'], ctx['spec'], ctx['device'], ctx['criterion'], args,
        ctx['metric_keys'], deep_closing=ctx['deep_closing'])
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(getattr(args, 'seed', 0)) + 1)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=int(getattr(args, 'num_workers', 4)),
        worker_init_fn=seed_worker, generator=eval_generator)
    summary, csv_rows, n_hd95_nan, avg_loss = evaluator.run(
        loader, args.output_dir)

    print("\nTest Results:")
    for k, (mean, std) in summary.items():
        print(f"Average {k.capitalize()}: {mean:.4f} ± {std:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    Evaluator.save_outputs(
        args.output_dir, summary, csv_rows, n_hd95_nan, avg_loss, args)
    try:
        _write_eval_artifacts(dataset, ctx, args)
    except OSError as exc:
        logging.warning('prediction index save failed: %s', exc)
    logging.info(f"Per-image metrics saved to {os.path.join(args.output_dir, 'metrics.csv')}")


def evaluate(args):
    setup_logging()
    ctx = _prepare_evaluation(args)
    _run_dataset(ctx, args)


def _parse_test_datasets(entries):
    """Parse NAME:DIR entries and reject unsafe/duplicate names."""
    specs = []
    seen = set()
    for entry in entries:
        name, sep, root = entry.partition(':')
        name, root = name.strip(), root.strip()
        if not sep or not name or not root:
            raise ValueError(
                f"--test_datasets entries must be NAME:DIR, got {entry!r}")
        if os.path.basename(name) != name or name in ('.', '..'):
            raise ValueError(f"Invalid dataset name {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate dataset name {name!r}")
        seen.add(name)
        specs.append((name, root))
    return specs


def _write_json_atomic(path, payload):
    import json
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def evaluate_many(args):
    """Evaluate every NAME:DIR entry with a single model load."""
    setup_logging()
    specs = _parse_test_datasets(args.test_datasets)
    ctx = _prepare_evaluation(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-dataset status lets the launcher mark individual results even
    # though all datasets share one process.
    status_path = os.path.join(args.output_dir, 'multi_eval_status.json')
    statuses = {
        name: {
            'name': name,
            'root': root,
            'output_dir': os.path.join(args.output_dir, name),
            'status': 'pending',
        }
        for name, root in specs
    }
    _write_json_atomic(status_path, {
        'trained_weights': os.path.abspath(args.trained_weights),
        'metric_keys': list(ctx['metric_keys']),
        'datasets': statuses,
    })

    failures = []
    for name, root in specs:
        ds_args = copy.copy(args)
        ds_args.data_root = root
        ds_args.output_dir = os.path.join(args.output_dir, name)
        statuses[name]['status'] = 'running'
        _write_json_atomic(status_path, {
            'trained_weights': os.path.abspath(args.trained_weights),
            'metric_keys': list(ctx['metric_keys']),
            'datasets': statuses,
        })
        logging.info("=== Evaluating %s (%s) ===", name, root)
        print(f"\n=== Evaluating {name} ({root}) ===", flush=True)
        try:
            _run_dataset(ctx, ds_args)
        except Exception:
            logging.exception("Evaluation failed for dataset %s (%s)",
                              name, root)
            os.makedirs(ds_args.output_dir, exist_ok=True)
            # A partial write must not look like a completed dataset.
            for stale in ('results_summary.txt', 'metrics.csv'):
                stale_path = os.path.join(ds_args.output_dir, stale)
                try:
                    os.remove(stale_path)
                except OSError:
                    pass
            error_path = os.path.join(ds_args.output_dir, 'eval_error.txt')
            with open(error_path, 'w') as f:
                f.write(traceback.format_exc())
            statuses[name]['status'] = 'failed'
            statuses[name]['error'] = error_path
            failures.append(name)
        else:
            statuses[name]['status'] = 'done'
        finally:
            _write_json_atomic(status_path, {
                'trained_weights': os.path.abspath(args.trained_weights),
                'metric_keys': list(ctx['metric_keys']),
                'datasets': statuses,
            })
            # Release per-dataset activations before loading the next split.
            torch.cuda.empty_cache()

    run_manifest = {
        'trained_weights': os.path.abspath(str(args.trained_weights)),
        'metric_keys': list(ctx['metric_keys']),
        'datasets': {
            name: {
                **payload,
                'eval_manifest': os.path.abspath(os.path.join(
                    args.output_dir, name, 'eval_manifest.json')),
                'prediction_index': os.path.abspath(os.path.join(
                    args.output_dir, name, 'prediction_index.csv')),
            }
            for name, payload in statuses.items()
        },
    }
    try:
        write_json_atomic(
            os.path.join(args.output_dir, 'eval_run_manifest.json'),
            run_manifest)
    except OSError as exc:
        logging.warning('eval run manifest save failed: %s', exc)

    n_done = len(specs) - len(failures)
    print(f"\nMulti-dataset evaluation: {n_done}/{len(specs)} datasets done")
    if failures:
        print("Failed datasets: " + ", ".join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root directory (single-dataset mode)')
    parser.add_argument('--test_datasets', nargs='+', default=None,
                        help='Multi-dataset mode: NAME:DIR pairs evaluated '
                             'with ONE model load; each result is written to '
                             '<output_dir>/<NAME>')
    parser.add_argument('--trained_weights', type=str, default='best_model.pth', help='Path to trained weights')
    parser.add_argument('--output_dir', type=str, default='results/outputs/tyre', help='Directory to save test results (multi-dataset mode: output root)')
    parser.add_argument('--use_adapter', type=str2bool, default=True,
                        help='Build SACM adapters (must match checkpoint; '
                             'full_fixed2_lora_evp disables them)')
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
    parser.add_argument('--use_dynamic_strip', type=str2bool, default=False,
                        help='Use deformable (dynamic) strip branches in the '
                             'geometric adapters; must match the checkpoint')
    parser.add_argument('--use_typed_heads', type=str2bool, default=False,
                        help='Typed structural Stage-2 heads (must match the '
                             'checkpoint; full_typed preset enables it)')
    parser.add_argument('--use_tcsu', type=str2bool, default=False,
                        help='PEPA target-conditioned snake upsampling '
                             '(must match the checkpoint; full_pepa enables it)')
    parser.add_argument('--use_tadt', type=str2bool, default=False,
                        help='PEPA target-adaptive threshold (must match the '
                             'checkpoint; full_pepa enables it)')
    parser.add_argument('--tcsu_scale', type=int, default=4,
                        help='TCSU upsampling factor (must match checkpoint)')
    parser.add_argument('--tcsu_mid_channels', type=int, default=32,
                        help='TCSU snake branch width (must match checkpoint)')
    parser.add_argument('--tcsu_max_length', type=int, default=5,
                        help='TCSU maximum odd chain length (must match checkpoint)')
    parser.add_argument('--tadt_hidden', type=int, default=128,
                        help='TADT MLP hidden width (must match checkpoint)')
    parser.add_argument('--tadt_t_min', type=float, default=-4.0,
                        help='TADT minimum logit threshold')
    parser.add_argument('--tadt_t_max', type=float, default=4.0,
                        help='TADT maximum logit threshold')
    parser.add_argument('--use_ranker', type=str2bool, default=False,
                        help='PromptMoE-style candidate ranker (must match the '
                             'checkpoint; full_pepa_rank enables it)')
    parser.add_argument('--ranker_hidden', type=int, default=128,
                        help='Candidate ranker utility MLP width')
    parser.add_argument('--ranker_pair_hidden', type=int, default=128,
                        help='Candidate ranker pairwise MLP width')
    parser.add_argument('--use_detail_stem', type=str2bool, default=False,
                        help='Parallel high-resolution detail stem (must '
                             'match checkpoint; full_pepa_hr enables it)')
    parser.add_argument('--detail_channels', type=int, default=32,
                        help='Detail stem 1/2-resolution width')
    parser.add_argument('--detail_out_channels', type=int, default=32,
                        help='Detail stem output channels')
    parser.add_argument('--high_res_output', type=str2bool, default=False,
                        help='Produce high-resolution Stage-2 masks')
    parser.add_argument('--high_res_size', type=int, default=512,
                        help='High-resolution Stage-2 output size')
    parser.add_argument('--high_res_refine', type=str2bool, default=False,
                        help='Image-guided full-resolution logit refinement')
    parser.add_argument('--high_res_refine_size', type=int, default=1024,
                        help='Output size of the image-guided refinement')
    parser.add_argument('--single_head', type=str2bool, default=False,
                        help='Use one Stage-2 mask head (must match checkpoint; '
                             'full_v2 enables it)')
    parser.add_argument('--use_v3', type=str2bool, default=False,
                        help='Use the v3 dense structural decoder (must match '
                             'checkpoint; full_v3 enables it)')
    parser.add_argument('--use_structural_aux', type=str2bool, default=False,
                        help='Prompt-decoder structural aux heads (must match '
                             'checkpoint; full_v2_skel_aux enables it)')
    parser.add_argument('--v3_feature_dim', type=int, default=128,
                        help='V3 FPN feature width')
    parser.add_argument('--v3_mid_dim', type=int, default=96,
                        help='V3 1/4-resolution feature width')
    parser.add_argument('--v3_fine_dim', type=int, default=64,
                        help='V3 1/2-resolution feature width')
    parser.add_argument('--v3_refine_channels', type=int, default=16,
                        help='V3 1024-refinement hidden width')
    parser.add_argument('--v3_aux_centerline', type=str2bool, default=True,
                        help='V3: centerline head (must match checkpoint)')
    parser.add_argument('--v3_aux_boundary', type=str2bool, default=True,
                        help='V3: boundary head (must match checkpoint)')
    parser.add_argument('--v3_aux_width', type=str2bool, default=True,
                        help='V3: width head (must match checkpoint)')
    parser.add_argument('--use_internal_adapter', type=str2bool, default=True,
                        help='Keep the per-block internal adapters (v3 = False)')
    parser.add_argument('--collect_adapter_features', type=str2bool,
                        default=True,
                        help='Collect external-adapter outputs for decoder '
                             'fusion (v3 = False)')
    parser.add_argument('--use_convlora', type=str2bool, default=False,
                        help='Conv-LoRA qkv branch (must match checkpoint; '
                             'full_fixed2_lora_evp enables it)')
    parser.add_argument('--convlora_rank', type=int, default=4,
                        help='Conv-LoRA low-rank width')
    parser.add_argument('--convlora_alpha', type=float, default=32.0,
                        help='Conv-LoRA residual scaling alpha')
    parser.add_argument('--convlora_experts', type=int, default=8,
                        help='Number of multi-scale conv experts')
    parser.add_argument('--use_evp', type=str2bool, default=False,
                        help='EVP high-frequency prompts (must match '
                             'checkpoint; full_fixed2_lora_evp enables it)')
    parser.add_argument('--evp_freq_nums', type=float, default=0.25,
                        help='EVP FFT high-pass cutoff rate')
    parser.add_argument('--evp_scale_factor', type=int, default=32,
                        help='EVP prompt low-rank divisor')
    parser.add_argument('--evp_zero_init', type=str2bool, default=True,
                        help='Zero-initialise EVP prompt output (must match '
                             'checkpoint)')
    parser.add_argument('--use_res_moe', type=str2bool, default=False,
                        help='Residual MoE on the fixed head (must match '
                             'checkpoint; full_fixed2_resmoe enables it)')
    parser.add_argument('--res_num_experts', type=int, default=4)
    parser.add_argument('--res_hidden', type=int, default=24)
    parser.add_argument('--res_gate_hidden', type=int, default=32)
    parser.add_argument('--res_gate_size', type=int, default=512)
    parser.add_argument('--res_gate_mode', type=str, default='learned',
                        choices=['learned', 'disagreement'])
    parser.add_argument('--res_use_head_disagreement', type=str2bool,
                        default=True)
    parser.add_argument('--res_use_centerline', type=str2bool, default=True)
    parser.add_argument('--res_gate_input_norm', type=str2bool, default=True)
    parser.add_argument('--res_out_size', type=int, default=512)
    parser.add_argument('--res_head_index', type=int, default=2,
                        choices=[0, 1, 2, 3])
    parser.add_argument('--res_dropout', type=float, default=0.0)
    parser.add_argument('--res_disagreement_tau', type=float, default=0.05)
    parser.add_argument('--res_disagreement_scale', type=float, default=0.10)

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
    parser.add_argument('--component_prune_rel_skeleton', type=float, default=0.0,
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
    parser.add_argument('--paper_metrics_only', action='store_true',
                        help='Compute only the four paper-table metrics '
                             '(dice, iou, cldice, hd95).  Values are identical '
                             'to the full suite; the other expensive metrics '
                             'are not computed.')
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
    if args.test_datasets and args.data_root:
        parser.error('--data_root and --test_datasets are mutually exclusive')
    if not args.test_datasets and not args.data_root:
        parser.error('--data_root is required unless --test_datasets is given')
    if args.test_datasets:
        try:
            _parse_test_datasets(args.test_datasets)
        except ValueError as exc:
            parser.error(str(exc))
    apply_preset(args, args.preset)
    if args.config is not None:
        from models.skein.configs import apply_yaml_presets
        apply_yaml_presets(args, args.config)
    if args.test_datasets:
        evaluate_many(args)
    else:
        evaluate(args)
