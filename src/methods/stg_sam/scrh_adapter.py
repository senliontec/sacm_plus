"""SCrH-SAM: SAM decoder with SCRWKV-inspired structure mixing.

The base scaffold and SAM code are owned by this package.  The only change
is an optional trainable StructureMix module (AMCM + GBST + DSCD) on the
decoder token grid, trained end-to-end from the official SAM weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path('/home/dev/mywork/project/segment/sacm_plus')
sys.path.insert(0, str(REPO / 'src'))

from baseline_zoo.runtime import main_for  # noqa: E402
from methods.lsa_sam.adapter import CoSNAAdapter  # noqa: E402
from methods.lsa_sam.structure_mix import StructureMix  # noqa: E402


class SCrHAdapter(CoSNAAdapter):
    id = 'cosna'
    display_name = 'SCrH-SAM (screening)'

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--structure-mix', type=int, choices=[0, 1],
                            default=1)
        parser.add_argument('--structure-amcm', type=int, default=1)
        parser.add_argument('--structure-gbst', type=int, default=1)
        parser.add_argument('--structure-dscd', type=int, default=1)

    def build_model(self, args):
        model = super().build_model(args)
        self.structure_mix = int(getattr(args, 'structure_mix', 1))
        if self.structure_mix:
            model.mask_decoder.structure_mix = StructureMix(
                model.mask_decoder.transformer_dim)
        return model

    def save_config(self, args):
        config = super().save_config(args)
        config['structure_mix'] = self.structure_mix
        config['structure_amcm'] = int(getattr(args, 'structure_amcm', 1))
        config['structure_gbst'] = int(getattr(args, 'structure_gbst', 1))
        config['structure_dscd'] = int(getattr(args, 'structure_dscd', 1))
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        for key in ('structure_mix', 'structure_amcm', 'structure_gbst',
                    'structure_dscd'):
            if key in config:
                setattr(args, key, config[key])


if __name__ == '__main__':
    raise SystemExit(main_for(SCrHAdapter()))
