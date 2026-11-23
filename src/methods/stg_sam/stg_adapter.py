"""STG-SAM: structure-tensor-guided SAM decoder.

The scaffold, recipe and inference are the proven GFA/BCTD-SAM pipeline;
the only new mechanism is a deterministic multi-scale structure tensor
(coherence and doubled-angle orientation) that is patch-embedded and added
to the SAM RGB patch tokens.  The stem and the usual adapters/decoder are
trained on the 18 source images; no target data is used.

This tests the hypothesis that an explicit, domain-agnostic elongated-
structure prior can reduce unseen-domain over-segmentation and preserve
continuity, without adding a data-hungry prompt-conditioned head.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path('/home/dev/mywork/project/segment/sacm_plus')
sys.path.insert(0, str(REPO / 'src'))

from baseline_zoo.runtime import main_for  # noqa: E402
from methods.stg_sam.adapter import CoSNAAdapter  # noqa: E402


class STGAdapter(CoSNAAdapter):
    id = 'stg_sam'
    display_name = 'STG-SAM structure-tensor guidance'
    official_source = (
        'BCTD-SAM scaffold (copied) + multi-scale structure-tensor input')
    official_entrypoint = (
        'Gated four-candidate decoder plus a learned patch-embedding stem '
        'for coherence/orientation maps')
    deviations = tuple(CoSNAAdapter.deviations) + (
        'A deterministic multi-scale structure tensor (coherence, cos2θ, '
        'sin2θ at three scales, per-image z-scored) is patch-embedded and '
        'added to the official SAM RGB patch tokens; the stem is trained '
        'with the ordinary adapters and decoder.',
        'The tensor maps are computed from the input image only; they use '
        'no target labels and no dataset-specific parameters.',
        'All other training and inference settings are identical to the '
        'GFA scaffold control.',
    )

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--structure-tensor', type=int, choices=[0, 1],
                            default=1,
                            help='enable the multi-scale structure-tensor '
                                 'side input')

    def build_model(self, args):
        self.structure_tensor = int(getattr(args, 'structure_tensor', 1))
        return super().build_model(args)

    def configure_trainable(self, model):
        super().configure_trainable(model)
        stem = getattr(model.image_encoder, 'structure_stem', None)
        if stem is not None:
            for parameter in stem.parameters():
                parameter.requires_grad_(True)

    def save_config(self, args):
        config = super().save_config(args)
        config['structure_tensor'] = self.structure_tensor
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        if 'structure_tensor' in config:
            args.structure_tensor = int(config['structure_tensor'])


if __name__ == '__main__':
    raise SystemExit(main_for(STGAdapter()))
