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
    use_typed_heads=False,
    deep_sup_weight=0.3,
    cl_dice_weight=0.5,
    iou_loss_weight=1.0,
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
    # Full model + deformable (dynamic) strip branches in the geometric
    # adapters.  Offsets are zero-initialized, so training starts from the
    # exact full-model behaviour and learns local strip deformation.
    "full_dynamic_strip": {**FULL, "use_dynamic_strip": True},
    # D8 candidate-supervision ablations. `full` defaults to soft after
    # the fix; full_wta reproduces the old winner-take-all behaviour.
    # wta = only gating head; allhead = mean over all heads;
    # soft = detached uniform + IoU-softmax (default).
    "full_wta": {**FULL, "stage2_supervision": "wta"},
    "full_allhead": {**FULL, "stage2_supervision": "allhead"},
    "full_soft": {**FULL, "stage2_supervision": "soft"},
    # Typed structural candidates (B prototype): stable head slots
    # thin/blob/boundary/recall, trained with clDice/boundary/Tversky.
    # The selection ranker is deliberately left as the existing IoU head.
    "full_typed": {
        **FULL,
        "use_typed_heads": True,
        "stage2_supervision": "typed",
        "boundary_loss_weight": 0.05,
        "tversky_weight": 0.5,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
    },
    # Full model + Same-Class Neighbor Penalization (SCNP, CVPR 2026):
    # training-only logit transform in front of BCE/Dice, no inference cost.
    "full_scnp": {**FULL, "scnp": True},
    # Full model + PEPA-style post-encoder adapters:
    #   TCSU: target-conditioned snake upsampling residual.
    #   TADT: per-candidate adaptive logit threshold.
    # Opt-in only; the default path remains full.
    "full_pepa": {
        **FULL,
        "use_tcsu": True,
        "use_tadt": True,
        "tadt_consistency_weight": 0.1,
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

# Typed candidates + a component/topology-aware auxiliary loss.  Betti
# matching directly penalizes unmatched components/holes, unlike clDice.
# This is deliberately an ablation preset: keep the weight small and compare
# against `full_typed` before claiming a gain.
PRESETS["full_typed_topo"] = {
    **PRESETS["full_typed"],
    "topology_loss": "betti",
    "topology_loss_weight": 0.05,
    "topology_resolution": 256,
}

# P1 topology variants on top of PEPA.  Separate presets keep each
# mechanism attributable; the combined variant is explicit.
PRESETS["full_pepa_topo"] = {
    **PRESETS["full_pepa"],
    "topology_loss": "topograph",
    "topology_loss_weight": 0.1,
    "topology_resolution": 256,
}
PRESETS["full_pepa_cape"] = {
    **PRESETS["full_pepa"],
    "topology_loss": "cape",
    "topology_loss_weight": 0.1,
    "topology_resolution": 256,
}
PRESETS["full_pepa_skel"] = {
    **PRESETS["full_pepa"],
    "topology_loss": "skeleton_recall",
    "topology_loss_weight": 0.1,
    "topology_resolution": 256,
}
PRESETS["full_pepa_topo_cape"] = {
    **PRESETS["full_pepa"],
    "topology_loss": "topograph_cape",
    "topology_loss_weight": 0.1,
    "topology_resolution": 256,
}
# P2: PromptMoE-style utility + pairwise candidate ranker replacing the
# IoU head for selection.  Selection stays `iou` (the ranker output).
PRESETS["full_pepa_rank"] = {
    **PRESETS["full_pepa"],
    "use_ranker": True,
    "rank_loss_weight": 0.1,
}
PRESETS["full_pepa_all"] = {
    **PRESETS["full_pepa_topo_cape"],
    "use_ranker": True,
    "rank_loss_weight": 0.1,
}
# P3: high-resolution detail stem + 1/2-resolution Stage-2 output.
PRESETS["full_pepa_hr"] = {
    **PRESETS["full_pepa"],
    "use_detail_stem": True,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": True,
    "high_res_refine_size": 1024,
}
PRESETS["full_pepa_all_hr"] = {
    **PRESETS["full_pepa_all"],
    "use_detail_stem": True,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": True,
    "high_res_refine_size": 1024,
}
# Recommended v2: global context from external adapters only, local detail
# from DetailStem, one main head, TADT, and complementary component/skeleton
# topology supervision.  Internal adapters, multi-depth, TCSU, typed heads
# and the four-candidate selector are deliberately disabled.
PRESETS["full_v2"] = {
    **FULL,
    "use_geo_i": False,
    "use_geo_e": True,
    "use_multi_depth": False,
    "use_tcsu": False,
    "use_tadt": True,
    "tadt_consistency_weight": 0.1,
    "use_detail_stem": True,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": True,
    "high_res_refine_size": 1024,
    "single_head": True,
    "topology_loss": "topograph_skeleton",
    "topology_loss_weight": 0.1,
    "topology_resolution": 256,
}

# V3: dense structural decoder.  One adapter mechanism only (external geo),
# no prompt decoder, no candidate heads, no C2F/deep supervision.  The
# mask head is a 512 conv head; centerline/boundary/width are training-only
# auxiliary heads.  Skeleton Recall is evaluated at 512 with max-pooled GT.
PRESETS["full_v3"] = {
    **FULL,
    "use_v3": True,
    "use_geo_i": False,
    "use_geo_e": True,
    "use_internal_adapter": False,
    "collect_adapter_features": False,
    "use_coarse_to_fine": False,
    "use_fusion_v2": False,
    "use_multi_depth": False,
    "use_typed_heads": False,
    "deep_sup_weight": 0.0,
    "cl_dice_weight": 0.1,
    "iou_loss_weight": 0.5,
    # Thin-foreground balance: BCE 0.7 leaves a ~27:1 background gradient
    # at 1% foreground, which traps a randomly initialised decoder in the
    # all-background solution.  Dice/Tversky dominate in v3.
    "bce_weight": 0.3,
    "dice_weight": 0.7,
    "tversky_weight": 0.5,
    # The v3 decoder is trained from scratch (no SAM mask-decoder warm
    # start), so it gets the adapter LR rather than the 1e-4 new-module LR.
    "new_module_lr": 3e-4,
    "stage2_supervision": "soft",
    "init_stage2_from_stage1": False,
    "use_detail_stem": True,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": True,
    "high_res_refine_size": 1024,
    "single_head": True,
    "use_tcsu": False,
    "use_tadt": False,
    "use_ranker": False,
    "use_dynamic_strip": False,
    # Skeleton Recall is bounded in [0,1]; raw Topograph grows with the
    # predicted component count (random logits ~1.6e3), so it is kept as
    # a separate ablation with a much smaller weight.
    "topology_loss": "skeleton_recall",
    "topology_loss_weight": 0.2,
    "topology_resolution": 512,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "v3_aux_centerline": True,
    "v3_aux_boundary": True,
    "v3_aux_width": True,
    "v3_centerline_weight": 0.2,
    "v3_boundary_weight": 0.1,
    "v3_width_weight": 0.05,
    "v3_area_weight": 0.05,
}


# Transition ablation: keep the v2 prompt decoder but use the fixed
# (max-pooled, 512) Skeleton Recall instead of Topograph.  Isolates the
# topology-target fix from the v3 architecture change.
PRESETS["full_v2_skel"] = {
    **PRESETS["full_v2"],
    "use_tadt": False,
    "topology_loss": "skeleton_recall",
    "topology_loss_weight": 0.2,
    "topology_resolution": 512,
    "tversky_weight": 0.3,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
}

# Hybrid: keep the pretrained prompt decoder (fast few-shot convergence)
# and add the v3 centerline/boundary/width supervision on its 1/2-res
# features.  No candidate selector, no dense decoder rewrite.
PRESETS["full_v2_skel_aux"] = {
    **PRESETS["full_v2_skel"],
    "use_structural_aux": True,
    "v3_centerline_weight": 0.2,
    "v3_boundary_weight": 0.1,
    "v3_width_weight": 0.05,
    "v3_area_weight": 0.05,
}


# Back to the proven full/fixed2 architecture.  Keep the 4 candidate heads,
# FusionV2, multi-depth path and C2F; add only the validated structural
# supervision (Skeleton Recall@512 + Tversky + training-only aux heads).
# fixed head 2 is selected by the runner, not by the model.
PRESETS["full_fixed2_aux"] = {
    **FULL,
    "use_structural_aux": True,
    # Direct supervision of the deployment head (fixed2), not only the
    # soft mixture of four near-identical heads.
    "fixed_head_loss_weight": 0.5,
    "fixed_supervision_head": 2,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": False,
    "topology_loss": "skeleton_recall",
    "topology_loss_weight": 0.2,
    "topology_resolution": 512,
    "tversky_weight": 0.3,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "v3_centerline_weight": 0.2,
    "v3_boundary_weight": 0.1,
    "v3_width_weight": 0.05,
    "v3_area_weight": 0.05,
}

# Training-only SCNP on top of fixed2_aux (CVPR 2026).  No inference cost.
PRESETS["full_fixed2_scnp"] = {
    **PRESETS["full_fixed2_aux"],
    "scnp": True,
}

# Topology-aware residual MoE on the fixed deployment head (head 2).
# FineSAM++-style gate + deterministic residual experts; the coarse fixed2
# decoder and all SACM components stay unchanged.
PRESETS["full_fixed2_resmoe"] = {
    **PRESETS["full_fixed2_aux"],
    "use_res_moe": True,
    "res_num_experts": 4,
    "res_hidden": 24,
    "res_gate_hidden": 32,
    "res_gate_size": 512,
    "res_gate_mode": "learned",
    "res_use_head_disagreement": True,
    "res_use_centerline": True,
    "res_gate_input_norm": True,
    "res_out_size": 512,
    "res_head_index": 2,
    "res_dropout": 0.0,
    "res_disagreement_tau": 0.05,
    "res_disagreement_scale": 0.10,
    "res_loss_weight": 0.5,
    "res_gate_loss_weight": 0.3,
    "res_sparsity_weight": 0.02,
    "res_warmup_epochs": 30,
    "res_ramp_epochs": 30,
}

# Robust fallback: no learned gate; the four-head disagreement gates the
# residual experts.  Useful for Unseen-domain checks.
PRESETS["full_fixed2_resmoe_dg"] = {
    **PRESETS["full_fixed2_resmoe"],
    "res_gate_mode": "disagreement",
    "res_gate_loss_weight": 0.0,
}

# Same fixed2 base plus the high-resolution detail path.  Needs ~24GB.
PRESETS["full_fixed2_hr_aux"] = {
    **PRESETS["full_fixed2_aux"],
    "use_detail_stem": True,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": True,
    "high_res_refine_size": 1024,
}

# Hybrid SACM + Conv-LoRA/EVP: retain SACM's external geometric adapter and
# FusionV2 (global cross-layer context), drop the duplicate block-internal
# adapter, and add official-style Conv-LoRA qkv residuals plus EVP FFT prompts.
# The fixed2 decoder, C2F, multi-depth path, structural aux heads and Skeleton
# Recall supervision are unchanged.
PRESETS["full_fixed2_lora_evp"] = {
    **PRESETS["full_fixed2_aux"],
    "use_adapter": True,
    "use_geo_i": False,
    "use_geo_e": True,
    "use_internal_adapter": False,
    "collect_adapter_features": True,
    "use_fusion_v2": True,
    "use_multi_depth": True,
    "use_convlora": True,
    "convlora_rank": 4,
    "convlora_alpha": 32.0,
    "convlora_experts": 8,
    "use_evp": True,
    "evp_freq_nums": 0.25,
    "evp_scale_factor": 32,
    "evp_zero_init": True,
    "moe_loss_weight": 1.0,
    # Keep the comparison against full_fixed2_aux clean: no DetailStem,
    # no 1024 refine.  EVP is the high-frequency branch in this ablation.
    "use_detail_stem": False,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": False,
}

# Single-component controls for the CVPR-time ablation matrix.
PRESETS["full_fixed2_evp"] = {
    **PRESETS["full_fixed2_aux"],
    "use_adapter": True,
    "use_internal_adapter": True,
    "collect_adapter_features": True,
    "use_convlora": False,
    "moe_loss_weight": 0.0,
    "use_evp": True,
    "evp_freq_nums": 0.25,
    "evp_scale_factor": 32,
    "evp_zero_init": True,
    "use_detail_stem": False,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": False,
}

PRESETS["full_fixed2_convlora"] = {
    **PRESETS["full_fixed2_aux"],
    "use_adapter": True,
    "use_internal_adapter": True,
    "collect_adapter_features": True,
    "use_evp": False,
    "use_convlora": True,
    "convlora_rank": 4,
    "convlora_alpha": 32.0,
    "convlora_experts": 8,
    "moe_loss_weight": 1.0,
    "use_detail_stem": False,
    "high_res_output": True,
    "high_res_size": 512,
    "high_res_refine": False,
}

# Replacement-style PEFT ablation: Conv-LoRA/EVP replace the SACM adapter
# stack entirely (no internal/external adapters, no FusionV2).  The decoder
# remains fixed2 + C2F + multi-depth + structural aux.
PRESETS["full_fixed2_peft_lora_evp"] = {
    **PRESETS["full_fixed2_lora_evp"],
    "use_adapter": False,
    "use_geo_i": False,
    "use_geo_e": False,
    "use_internal_adapter": False,
    "collect_adapter_features": False,
    "use_fusion_v2": False,
    "use_multi_depth": True,
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
