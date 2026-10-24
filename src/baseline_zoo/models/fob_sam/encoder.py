"""ResNet-101 feature encoder from the official FoB/SAM implementation.

Vendored from ``baselines/FoB_SAM/models/encoder.py`` (Apache-2.0).  The
official code hard-codes absolute checkpoint paths; this copy accepts a
checkpoint path (or ``None`` for smoke tests) and robustly matches both
plain ``resnet101`` keys and the ``backbone.*``-prefixed DeepLabV3 COCO
checkpoint used by the paper.
"""

import os

import torch
import torch.nn as nn
import torchvision


class Res101Encoder(nn.Module):
    """ResNet-101 backbone used by FoB's background-prompt generator.

    ``replace_stride_with_dilation=[True, True, False]`` keeps the layer3
    feature map at 1/4 resolution (64x64 for 256x256 inputs), matching the
    official model's 64x64 positional embedding.
    """

    def __init__(self, replace_stride_with_dilation=None,
                 pretrained_weights='COCO'):
        super().__init__()
        if pretrained_weights in (None, '', 'none'):
            self.pretrained_weights = None
        elif isinstance(pretrained_weights, (str, os.PathLike)):
            self.pretrained_weights = torch.load(
                os.fspath(pretrained_weights), map_location='cpu',
                weights_only=False)
        else:
            # Caller passed an already-loaded state dict (smoke tests).
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(
            pretrained=False,
            replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for name, module in _model.named_children():
            self.backbone[name] = module

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def forward(self, x):
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)

        x = self.backbone['maxpool'](x)
        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        x = self.backbone['layer3'](x)

        feature = self.reduce1(x)
        x = self.backbone['layer4'](x)
        t = self.backbone['avgpool'](x)
        t = torch.flatten(t, 1)
        t = self.backbone['fc'](t)
        t = self.reduce1d(t)
        return feature, t

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if self.pretrained_weights is None:
            return

        state = self.pretrained_weights
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise TypeError(
                'ResNet-101 pretrained weights must be a state dict or a '
                'checkpoint containing one')

        current = self.state_dict()
        loaded, skipped = 0, 0
        for key, value in state.items():
            # DeepLabV3 COCO checkpoints use ``backbone.<key>``; plain
            # torchvision ResNet checkpoints use ``<key>``.  Try both.
            candidates = (key, f'backbone.{key}', key.removeprefix('backbone.'))
            for candidate in candidates:
                if candidate in current and current[candidate].shape == value.shape:
                    current[candidate] = value
                    loaded += 1
                    break
            else:
                skipped += 1
        self.load_state_dict(current)
        if loaded == 0:
            raise RuntimeError(
                'ResNet-101 checkpoint did not match any model tensors; '
                'expected a DeepLabV3 COCO or torchvision ResNet-101 state '
                'dict')
        self.loaded_pretrained = loaded
        self.skipped_pretrained = skipped
