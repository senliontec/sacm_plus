"""Task/supervision groups for the baseline zoo.

The 20-dataset protocol mixes methods that do not answer the same question:

* prompt-free supervised segmentation;
* support-based few-shot segmentation (needs labeled supports);
* oracle-prompted segmentation (uses test GT prompts);
* semi-supervised training (uses extra unlabeled images);
* zero-shot foundation-model evaluation;
* appendix/project extensions;
* paper-level reconstructions of missing official code.

Putting these rows in one table is not a fair comparison.  The registry keeps
the model metadata, while this module supplies the task grouping used by the
report and by paper-table filters.  A model id appears in exactly one primary
group; orthogonal properties live in ``flags_for``.
"""

from __future__ import annotations

GROUP_LABELS = {
    'prompt_free': 'Prompt-free supervised',
    'proposed': 'Proposed method',
    'support': 'Support-based few-shot',
    'oracle': 'Oracle-prompted reference',
    'semi_supervised': 'Semi-supervised',
    'zero_shot': 'Zero-shot foundation model',
    'appendix': 'Appendix / project extension',
    'reconstructed': 'Paper-level reconstruction',
}

_PROMPT_FREE = frozenset({
    'unet', 'cs2net', 'bcunet', 'maskvsc', 'nnunet2d', 'nnwnet',
    'jtfn', 'dscnet', 'flowsdf', 'medcagd', 's2mnet', 'csunet', 'liot',
    'dconnnet', 'scrwkv', 'kprism', 'segdino', 'segdino_v2',
    'sam_adapter', 'topolora', 'samed', 'convlora', 'blo_sam', 'cwsam',
    'tpsegformer', 'aco_moe_lora',
    'sacm_official',
})

_SUPPORT = frozenset({'persam', 'matcher', 'fssam', 'sansa', 'rsrm'})
_ORACLE = frozenset({'ucs', 'sam_med2d', 'sam_octa'})
_SEMI = frozenset({'knowsam'})
_ZERO_SHOT = frozenset({'c2p'})
_APPENDIX = frozenset({'sam2_unet', 'sam_vit_l_unet'})
_RECONSTRUCTED = frozenset({'a2sam', 'dka'})

_TRAINING_FREE = frozenset({'persam', 'matcher', 'rsrm', 'c2p'})
_PROJECT_EXTENSION = frozenset({'sam2_unet', 'sam_vit_l_unet'})
_ANALYSIS_VARIANT = frozenset()

_GROUP_NAMES = (
    ('prompt_free', _PROMPT_FREE),
    ('support', _SUPPORT),
    ('oracle', _ORACLE),
    ('semi_supervised', _SEMI),
    ('zero_shot', _ZERO_SHOT),
    ('appendix', _APPENDIX),
    ('reconstructed', _RECONSTRUCTED),
)


def group_for(model: str) -> str:
    """Return the primary task group for a model id."""
    model = str(model)
    if model == 'sacm_stc':
        return 'proposed'
    for name, members in _GROUP_NAMES:
        if model in members:
            return name
    # Unknown ids must be classified explicitly before they appear in a
    # grouped paper table; callers can still display them under this label.
    return 'unclassified'


def group_label(model: str) -> str:
    group = group_for(model)
    return GROUP_LABELS.get(group, 'Unclassified')


def flags_for(model: str) -> dict[str, bool]:
    """Orthogonal properties that are useful in table captions."""
    model = str(model)
    return {
        'training_free': model in _TRAINING_FREE,
        'project_extension': model in _PROJECT_EXTENSION,
        'analysis_variant': model in _ANALYSIS_VARIANT,
        'reconstructed': model in _RECONSTRUCTED,
        'oracle_prompted': model in _ORACLE,
        'support_based': model in _SUPPORT,
        'semi_supervised': model in _SEMI,
        'zero_shot': model in _ZERO_SHOT,
    }


def validate_groups(models) -> list[str]:
    """Return ids that have no explicit primary group."""
    return sorted({str(model) for model in models
                   if group_for(model) == 'unclassified'})
