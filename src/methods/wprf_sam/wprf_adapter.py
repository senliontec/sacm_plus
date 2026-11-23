"""WPRF-SAM: SAM scaffold trained with Widest-Path Reachability Fields.

The backbone, decoder, losses and inference are the proven GFA/BCTD-SAM
scaffold.  The only new training signal is WPRF (2026): an affinity head on
the encoder embedding plus Max-Min reachability losses that route gradients
to connectivity bottlenecks.  At inference only the union mask is emitted;
the affinity head adds no cost.

Deviation summary: graph stride 8 at 1024 input, 16 sources per image
instead of 32, and the paper's bottleneck-aware segmentation term is
omitted because the scaffold already has BCE/Dice, soft-clDice and skeleton
recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path('/home/dev/mywork/project/segment/sacm_plus')
sys.path.insert(0, str(REPO / 'src'))

from baseline_zoo.runtime import main_for  # noqa: E402
from methods.wprf_sam.adapter import CoSNAAdapter  # noqa: E402
from methods.wprf_sam.wprf import AffinityHead, WPRFLoss  # noqa: E402


class WPRFAdapter(CoSNAAdapter):
    id = 'wprf_sam'
    display_name = 'WPRF-SAM reachability'
    official_source = (
        'BCTD-SAM scaffold + Widest-Path Reachability Fields (2026)')
    official_entrypoint = (
        'Gated four-candidate decoder trained with Max-Min edge and '
        'reachability losses on a skeleton support graph')
    deviations = tuple(CoSNAAdapter.deviations) + (
        'An affinity head on the encoder embedding predicts directed '
        '8-neighbour link logits; differentiable Max-Min DP turns them into '
        'k-step widest-path reachability values.',
        'Edge and reachability losses (weights 1.0 each, scales '
        '{1,2,4,8,16}) are added to the existing scaffold loss; inference '
        'uses only the union mask, so the head adds no test-time cost.',
        'Graph stride is 8 at 1024 input, sources per image are 16 (paper: '
        '4 at 512 and 32 sources); the bottleneck-aware segmentation term '
        'is omitted because the scaffold already uses clDice and skeleton '
        'recall.',
    )

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--wprf', type=int, choices=[0, 1], default=1)
        parser.add_argument('--wprf-edge-weight', type=float, default=1.0)
        parser.add_argument('--wprf-reach-weight', type=float, default=1.0)
        parser.add_argument('--wprf-stride', type=int, default=8)
        parser.add_argument('--wprf-sources', type=int, default=16)
        parser.add_argument('--wprf-scales', type=str,
                            default='1 2 4 8 16')
        parser.add_argument('--wprf-warmup-epochs', type=int,
                            default=0)

    def build_model(self, args):
        self.wprf = int(getattr(args, 'wprf', 1))
        self.wprf_edge_weight = float(
            getattr(args, 'wprf_edge_weight', 1.0))
        self.wprf_reach_weight = float(
            getattr(args, 'wprf_reach_weight', 1.0))
        self.wprf_stride = int(getattr(args, 'wprf_stride', 8))
        self.wprf_sources = int(getattr(args, 'wprf_sources', 16))
        self.wprf_scales = str(getattr(args, 'wprf_scales', '1 2 4 8 16'))
        self._wprf_warmup_epochs = int(
            getattr(args, 'wprf_warmup_epochs', 0))
        model = super().build_model(args)
        if self.wprf:
            model.wprf_affinity = AffinityHead(
                model.prompt_encoder.embed_dim)
            self._wprf_loss = WPRFLoss(
                stride=self.wprf_stride,
                sources=self.wprf_sources,
                scales=tuple(int(v) for v in self.wprf_scales.split()),
                edge_weight=self.wprf_edge_weight,
                reach_weight=self.wprf_reach_weight,
            )
            self._wprf_enabled = True
        else:
            self._wprf_loss = None
            self._wprf_enabled = False
        return model

    def configure_trainable(self, model):
        super().configure_trainable(model)
        head = getattr(model, 'wprf_affinity', None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(True)

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'wprf': self.wprf,
            'wprf_edge_weight': self.wprf_edge_weight,
            'wprf_reach_weight': self.wprf_reach_weight,
            'wprf_stride': self.wprf_stride,
            'wprf_sources': self.wprf_sources,
            'wprf_scales': self.wprf_scales,
            'wprf_warmup_epochs': self._wprf_warmup_epochs,
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        for key in ('wprf', 'wprf_edge_weight', 'wprf_reach_weight',
                    'wprf_stride', 'wprf_sources', 'wprf_scales',
                    'wprf_warmup_epochs'):
            if key in config:
                setattr(args, key, config[key])


if __name__ == '__main__':
    raise SystemExit(main_for(WPRFAdapter()))
