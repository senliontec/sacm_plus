"""Architecture/training presets for ablation experiments.

Each preset is a complete configuration (all switches explicit), so the
ablation matrix can be reproduced with a single flag, e.g.:

    python train_sam.py --preset sacm      # bug-fixed SACM reproduction
    python train_sam.py --preset stage1    # + coarse-to-fine, deep sup, clDice
    python train_sam.py --preset stage2    # + geometric strip adapters
    python train_sam.py --preset full      # full model (D8 soft supervision)
    python train_sam.py --preset full_wta  # D8 ablation: winner-take-all

CLI flags override nothing: the preset is applied after parsing. For
manual control use --preset none and set every switch explicitly.
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")


FULL = dict(
    use_geo_i=True,
    use_geo_e=True,
    geo_e_layers='all',
    use_coarse_to_fine=True,
    use_fusion_v2=True,
    use_multi_depth=True,
    deep_sup_weight=0.3,
    cl_dice_weight=0.5,
    iou_loss_weight=1.0,
    # ECCV 2024 Skeleton Recall (opt-in; off in the `full` baseline).
    skeleton_recall_weight=0.0,
    skeleton_recall_warmup=10,
    skeleton_recall_ramp=5,
    skeleton_recall_tube=True,
    # D8 fix: every Stage-2 head receives mask gradient via detached
    # uniform + IoU-softmax weights.  The original behaviour is kept as
    # the `full_wta` ablation and in the `sacm` baseline.
    stage2_supervision='soft',
    soft_select_mix=0.5,
    soft_select_tau=1.0,
)

PRESETS = {
    # Bug-fixed SACM reproduction: every new component off (baseline).
    "sacm": dict(
        use_geo_i=False,
        use_geo_e=False,
        use_coarse_to_fine=False,
        use_fusion_v2=False,
        use_multi_depth=False,
        deep_sup_weight=0.0,
        cl_dice_weight=0.0,
        iou_loss_weight=0.0,
        # Explicit D8 baseline: winner-take-all (only gating head 0).
        stage2_supervision='wta',
    ),
    # Stage 1: true coarse-to-fine + M1 deep supervision + clDice loss
    # + IoU head supervision (trained selector).
    "stage1": dict(
        use_geo_i=False,
        use_geo_e=False,
        use_coarse_to_fine=True,
        use_fusion_v2=False,
        use_multi_depth=False,
        deep_sup_weight=0.3,
        cl_dice_weight=0.5,
        iou_loss_weight=1.0,
        stage2_supervision='soft',
    ),
    # Stage 2: + geometric strip adapters (I_geo all layers, E_global on
    # the 20 window-attention layers only).
    "stage2": dict(
        use_geo_i=True,
        use_geo_e=True,
        use_coarse_to_fine=True,
        use_fusion_v2=False,
        use_multi_depth=False,
        deep_sup_weight=0.3,
        cl_dice_weight=0.5,
        iou_loss_weight=1.0,
        stage2_supervision='soft',
    ),
    # Full model: + Fusion v2 (cross-layer interaction + spatial gate)
    # + multi-depth decoder path.
    "full": dict(FULL),
    # D8 candidate-supervision ablations. `full` defaults to soft after
    # the fix; full_wta reproduces the old winner-take-all behaviour.
    # wta = only gating head; allhead = mean over all heads;
    # soft = detached uniform + IoU-softmax (default).
    "full_wta": {**FULL, "stage2_supervision": "wta"},
    "full_allhead": {**FULL, "stage2_supervision": "allhead"},
    "full_soft": {**FULL, "stage2_supervision": "soft"},
    # Full model + Same-Class Neighbor Penalization (SCNP, CVPR 2026):
    # training-only logit transform in front of BCE/Dice, no inference cost.
    "full_scnp": {**FULL, "scnp": True},
    # Full model + Skeleton Recall (ECCV 2024) on the tubed GT skeleton.
    # The weight sweep is a small, fixed set of presets because presets are
    # applied after argparse (CLI flags cannot override them).
    "full_srec": {
        **FULL, "skeleton_recall_weight": 0.2,
        "skeleton_recall_warmup": 0,
    },
    "full_srec_low": {
        **FULL, "skeleton_recall_weight": 0.1,
        "skeleton_recall_warmup": 0,
    },
    "full_srec_high": {
        **FULL, "skeleton_recall_weight": 0.5,
        "skeleton_recall_warmup": 0,
    },
    "full_srec_notube": {
        **FULL, "skeleton_recall_weight": 0.2,
        "skeleton_recall_warmup": 0,
        "skeleton_recall_tube": False,
    },
    # Single-module ablations (full model minus one component).
    "no_geo_i": {**FULL, "use_geo_i": False},
    "no_geo_e": {**FULL, "use_geo_e": False},
    "no_c2f": {**FULL, "use_coarse_to_fine": False},
    "no_fusion_v2": {**FULL, "use_fusion_v2": False},
    "no_multi_depth": {**FULL, "use_multi_depth": False},
    "no_cl": {**FULL, "cl_dice_weight": 0.0},
    "no_ds": {**FULL, "deep_sup_weight": 0.0},
    "no_iou": {**FULL, "iou_loss_weight": 0.0},
    # Layer-placement ablation of A2 (design R2 层子集):
    # E_global only on the first/second half of the window-attention
    # layers (for ViT-L: shallow {0-4, 6-10} / deep {12-16, 18-22};
    # global-attn layers excluded both ways).
    "geo_e_shallow": {**FULL, "geo_e_layers": "shallow"},
    "geo_e_deep": {**FULL, "geo_e_layers": "deep"},
    # No preset: keep CLI values.
    "none": {},
}


def apply_preset(args, preset):
    """Overlay a preset on the parsed argparse namespace."""
    if preset is None or preset == "none":
        return args
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; available: {sorted(PRESETS)}")
    for key, value in PRESETS[preset].items():
        setattr(args, key, value)
    return args


def apply_yaml_presets(args, path):
    """Apply a YAML config on top of the parsed args (after the preset).

    Python ``PRESETS`` is the single source of truth.  The YAML file only
    selects a preset and optionally overrides top-level argparse values:

        preset: full            # selects a preset from Python PRESETS
        use_geo_i: false        # top-level keys override everything
        epochs: 30

    A custom ``presets`` mapping is still accepted for names that do not
    exist in Python PRESETS, but it cannot shadow a built-in preset.

    Requires PyYAML.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("--config requires PyYAML (pip install pyyaml)") from e

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    name = cfg.get('preset', 'full')
    if name != 'none':
        if name in PRESETS:
            apply_preset(args, name)
        elif name in cfg.get('presets', {}):
            for key, value in cfg['presets'][name].items():
                setattr(args, key, value)
        else:
            raise ValueError(
                f"Unknown YAML preset {name!r}; built-ins: "
                f"{sorted(PRESETS)}")
        # Keep the recorded preset consistent with the switches that were
        # actually applied; otherwise wandb/history says stage2 while the
        # YAML selected full.
        args.preset = name

    overrides = {k: v for k, v in cfg.items() if k not in ('preset', 'presets')}
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
