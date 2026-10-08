"""Architecture/training presets for ablation experiments.

Each preset is a complete configuration (all switches explicit), so the
ablation matrix can be reproduced with a single flag, e.g.:

    python train_sam.py --preset sacm      # bug-fixed SACM reproduction
    python train_sam.py --preset stage1    # + coarse-to-fine, deep sup, clDice
    python train_sam.py --preset stage2    # + geometric strip adapters
    python train_sam.py --preset full      # full model (default)

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
    ),
    # Full model: + Fusion v2 (cross-layer interaction + spatial gate)
    # + multi-depth decoder path.
    "full": dict(FULL),
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

    Expected structure (see configs/presets.yaml):

        preset: full            # selects a preset from `presets`
        presets:                # optional mirror of PRESETS
          full: {use_geo_i: true, ...}
        use_geo_i: false        # top-level keys override everything

    Requires PyYAML.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("--config requires PyYAML (pip install pyyaml)") from e

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    name = cfg.get('preset', 'full')
    presets = cfg.get('presets', {})
    if name in presets:
        for key, value in presets[name].items():
            setattr(args, key, value)

    overrides = {k: v for k, v in cfg.items() if k not in ('preset', 'presets')}
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
