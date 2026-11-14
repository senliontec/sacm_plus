"""TubeToken-SAM adapter: structure-token post-decode control for DCD.

Only a new post-decode path is added.  DCD's image encoder, prompt encoder,
four candidates, Stage-1 gating, and fixed candidate-3 deployment are
unchanged, so the deployed candidate keeps exactly the same physical role.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from baseline_zoo.adapters.dcd import DCDAdapter

from .deployed_head import (
    DeployedStructureHead,
    SelfPromptGenerator,
    StructuralPromptGenerator,
)
from .model import StructureControl


class TubeTokenAdapter(DCDAdapter):
    id = 'tube_token'
    display_name = 'TubeToken-SAM'
    official_source = 'DCD + structure-token post-decode control'
    official_entrypoint = (
        'DCD fixed3 recipe + image-derived structure tokens controlling '
        'candidate 3 residual, high-resolution reconstruction and decision')
    deviations = DCDAdapter.deviations + (
        'A 128-token image-derived structure field is added after decoding '
        'to candidate 3 only; it never changes the Stage-1 gating order or '
        'the physical identity of the fixed deployment slot.',
        'The native structure residual is added before DCD topology losses, '
        'so the unchanged DCD objective trains the tokenizer.',
        'A lightweight structure-guided high-resolution residual is applied '
        'only to candidate 3 and trained by the unchanged dense BCE+Dice.',
        'Final deployment uses a skeleton-calibrated threshold '
        '(0.5 - beta*S); background far from structure keeps 0.5.  No '
        'per-dataset or test-time threshold tuning is used.',
    )

    def __init__(self):
        super().__init__()
        self.deploy_mode = 'legacy'
        self.prompt_mode = 'off'
        self.prompt_scale_init = 0.1
        self.self_prompt_mode = 'off'
        self.self_prompt_scale_init = 0.1
        self.self_prompt_coarse_weight = 0.25
        self.train_aug = 'none'
        self.aux_eval_off = False
        self.self_prompt_blend = 0.0
        self.self_prompt_gate = False
        self._clean_images = None
        self.head_feature_dim = 32
        self.head_coarse_size = 512
        self.head_render_size = 1024
        self.head_snake_kernel = 5
        self.head_render_channels = 16
        self.structure_mode = 'post'
        self.structure_tokens = 128
        self.structure_grid_h = 8
        self.structure_grid_w = 16
        self.structure_render_size = 128
        self.structure_highres_size = 1024
        self.structure_residual_weight = 0.1
        self.structure_residual_prior = 0.5
        self.structure_decision_weight = 0.1
        self._structure_cache = None

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--structure-mode', choices=['post', 'off'], default='post',
            help='post-decode structure control (post) or plain DCD (off)')
        parser.add_argument('--structure-tokens', type=int, default=128)
        parser.add_argument('--structure-grid-h', type=int, default=8)
        parser.add_argument('--structure-grid-w', type=int, default=16)
        parser.add_argument('--structure-render-size', type=int, default=128)
        parser.add_argument(
            '--structure-highres-size', type=int, default=1024)
        parser.add_argument(
            '--structure-residual-weight', type=float, default=0.1)
        parser.add_argument(
            '--structure-residual-prior', type=float, default=0.5)
        parser.add_argument(
            '--structure-decision-weight', type=float, default=0.1)
        parser.add_argument(
            '--structure-eval-highres-mult', type=float, default=1.0,
            help='eval-only multiplier on the high-resolution residual')
        parser.add_argument(
            '--structure-eval-residual-mult', type=float, default=1.0,
            help='eval-only multiplier on the native residual')
        parser.add_argument(
            '--structure-eval-decision-weight', type=float, default=None,
            help='eval-only override of the skeleton-calibrated threshold')
        parser.add_argument(
            '--guided-upsample', action='store_true',
            help='eval-only image-guided refinement of the deployed logits')
        parser.add_argument(
            '--tta-flip', action='store_true',
            help='eval-only horizontal-flip test-time averaging')
        parser.add_argument(
            '--tile-inference', action='store_true',
            help='eval-only native-resolution tiling for large images')
        parser.add_argument('--tile-size', type=int, default=1024)
        parser.add_argument('--tile-stride', type=int, default=768)
        parser.add_argument(
            '--tile-min-size', type=int, default=3000,
            help='tile only when the original long side reaches this size')
        parser.add_argument(
            '--deploy-mode', choices=['legacy', 'head'], default='legacy',
            help='legacy post-decode residual or the deployed structure '
                 'head trained directly by the DCD dense/topology losses')
        parser.add_argument(
            '--prompt-mode', choices=['off', 'structure'], default='off',
            help='add a learned image-derived dense mask prompt before the '
                 'SAM mask decoder; trained end-to-end by the DCD losses')
        parser.add_argument(
            '--prompt-scale-init', type=float, default=0.1,
            help='initial scale of the structural dense prompt')
        parser.add_argument(
            '--self-prompt-mode', choices=['off', 'on'], default='off',
            help='second SAM decoder pass prompted by the coarse prediction')
        parser.add_argument('--self-prompt-scale-init', type=float,
                            default=0.1)
        parser.add_argument('--self-prompt-coarse-weight', type=float,
                            default=0.25,
                            help='weight of the coarse dense loss in the '
                                 'self-prompt cascade')
        parser.add_argument(
            '--train-aug', choices=['none', 'mild', 'strong'],
            default='none',
            help='structure-preserving training augmentation')
        parser.add_argument(
            '--aux-eval-off', action='store_true',
            help='ablation: deploy the base candidate without the head or '
                 'self-prompt refinement')
        parser.add_argument(
            '--self-prompt-blend', type=float, default=0.0,
            help='inference blend weight of the coarse branch logits with '
                 'the self-prompt refined logits (0 = refined only)')
        parser.add_argument(
            '--self-prompt-gate', action='store_true',
            help='inference: use the refined branch only where the coarse '
                 'branch is uncertain')
        parser.add_argument('--head-feature-dim', type=int, default=32)
        parser.add_argument('--head-coarse-size', type=int, default=512)
        parser.add_argument('--head-render-size', type=int, default=1024)
        parser.add_argument('--head-snake-kernel', type=int, default=5)
        parser.add_argument(
            '--head-render-channels', type=int, default=16)

    def _read_structure_args(self, args):
        self.structure_mode = str(
            getattr(args, 'structure_mode', 'post'))
        self.structure_tokens = int(
            getattr(args, 'structure_tokens', 128))
        self.structure_grid_h = int(
            getattr(args, 'structure_grid_h', 8))
        self.structure_grid_w = int(
            getattr(args, 'structure_grid_w', 16))
        if self.structure_grid_h * self.structure_grid_w != self.structure_tokens:
            raise SystemExit(
                'structure-tokens must equal structure-grid-h * '
                'structure-grid-w')
        if self.structure_mode not in ('post', 'off'):
            raise SystemExit(f'unsupported structure-mode: '
                             f'{self.structure_mode!r}')
        self.structure_render_size = int(
            getattr(args, 'structure_render_size', 128))
        self.structure_highres_size = int(
            getattr(args, 'structure_highres_size', 1024))
        self.structure_residual_weight = float(
            getattr(args, 'structure_residual_weight', 0.1))
        self.structure_residual_prior = float(
            getattr(args, 'structure_residual_prior', 0.5))
        self.structure_decision_weight = float(
            getattr(args, 'structure_decision_weight', 0.1))
        self.structure_eval_highres_mult = float(
            getattr(args, 'structure_eval_highres_mult', 1.0))
        self.structure_eval_residual_mult = float(
            getattr(args, 'structure_eval_residual_mult', 1.0))
        self.structure_eval_decision_weight = getattr(
            args, 'structure_eval_decision_weight', None)
        self.guided_upsample = bool(
            getattr(args, 'guided_upsample', False))
        self.tta_flip = bool(getattr(args, 'tta_flip', False))
        self.tile_inference = bool(
            getattr(args, 'tile_inference', False))
        self.tile_size = int(getattr(args, 'tile_size', 1024))
        self.tile_stride = int(getattr(args, 'tile_stride', 768))
        self.tile_min_size = int(
            getattr(args, 'tile_min_size', 3000))
        self.deploy_mode = str(
            getattr(args, 'deploy_mode', 'legacy'))
        if self.deploy_mode not in ('legacy', 'head'):
            raise SystemExit(
                f'unsupported deploy-mode: {self.deploy_mode!r}')
        self.prompt_mode = str(
            getattr(args, 'prompt_mode', 'off'))
        if self.prompt_mode not in ('off', 'structure'):
            raise SystemExit(
                f'unsupported prompt-mode: {self.prompt_mode!r}')
        self.prompt_scale_init = float(
            getattr(args, 'prompt_scale_init', 0.1))
        self.self_prompt_mode = str(
            getattr(args, 'self_prompt_mode', 'off'))
        if self.self_prompt_mode not in ('off', 'on'):
            raise SystemExit(
                f'unsupported self-prompt-mode: {self.self_prompt_mode!r}')
        self.self_prompt_scale_init = float(
            getattr(args, 'self_prompt_scale_init', 0.1))
        self.self_prompt_coarse_weight = float(
            getattr(args, 'self_prompt_coarse_weight', 0.25))
        self.train_aug = str(getattr(args, 'train_aug', 'none'))
        if self.train_aug not in ('none', 'mild', 'strong'):
            raise SystemExit(
                f'unsupported train-aug: {self.train_aug!r}')
        self.aux_eval_off = bool(
            getattr(args, 'aux_eval_off', False))
        self.self_prompt_blend = float(
            getattr(args, 'self_prompt_blend', 0.0))
        self.self_prompt_gate = bool(
            getattr(args, 'self_prompt_gate', False))
        self.head_feature_dim = int(
            getattr(args, 'head_feature_dim', 32))
        self.head_coarse_size = int(
            getattr(args, 'head_coarse_size', 512))
        self.head_render_size = int(
            getattr(args, 'head_render_size', 1024))
        self.head_snake_kernel = int(
            getattr(args, 'head_snake_kernel', 5))
        self.head_render_channels = int(
            getattr(args, 'head_render_channels', 16))

    def build_model(self, args):
        model = super().build_model(args)
        self._read_structure_args(args)
        if self.deploy_mode == 'head':
            model.deployed_head = DeployedStructureHead(
                adapter_dim=1024,
                feature_dim=self.head_feature_dim,
                coarse_size=self.head_coarse_size,
                render_size=self.head_render_size,
                snake_kernel=self.head_snake_kernel,
                render_channels=self.head_render_channels,
            )
        elif self.structure_mode == 'post':
            model.structure_control = StructureControl(
                adapter_dim=1024,
                feature_dim=64,
                grid_h=self.structure_grid_h,
                grid_w=self.structure_grid_w,
                render_size=self.structure_render_size,
                highres_size=self.structure_highres_size,
                residual_weight=self.structure_residual_weight,  # noqa: E501
                residual_prior=self.structure_residual_prior,
                decision_weight=self.structure_decision_weight,
            )
        if self.prompt_mode == 'structure':
            model.structural_prompt = StructuralPromptGenerator(
                scale_init=self.prompt_scale_init)
        if self.self_prompt_mode == 'on':
            model.self_prompt = SelfPromptGenerator(
                scale_init=self.self_prompt_scale_init)
        return model

    def save_config(self, args):
        config = super().save_config(args)
        config.update({
            'deploy_mode': self.deploy_mode,
            'prompt_mode': self.prompt_mode,
            'prompt_scale_init': float(self.prompt_scale_init),
            'self_prompt_mode': self.self_prompt_mode,
            'self_prompt_scale_init': float(
                self.self_prompt_scale_init),
            'self_prompt_coarse_weight': float(
                self.self_prompt_coarse_weight),
            'train_aug': self.train_aug,
            'head_feature_dim': int(self.head_feature_dim),
            'head_coarse_size': int(self.head_coarse_size),
            'head_render_size': int(self.head_render_size),
            'head_snake_kernel': int(self.head_snake_kernel),
            'head_render_channels': int(self.head_render_channels),
            'structure_mode': self.structure_mode,
            'structure_tokens': int(self.structure_tokens),
            'structure_grid_h': int(self.structure_grid_h),
            'structure_grid_w': int(self.structure_grid_w),
            'structure_render_size': int(self.structure_render_size),
            'structure_highres_size': int(self.structure_highres_size),
            'structure_residual_weight': float(
                self.structure_residual_weight),
            'structure_residual_prior': float(
                self.structure_residual_prior),
            'structure_decision_weight': float(
                self.structure_decision_weight),
        })
        return config

    def restore_config(self, args, config):
        super().restore_config(args, config)
        for key in (
                'deploy_mode', 'prompt_mode', 'prompt_scale_init',
                'self_prompt_mode', 'self_prompt_scale_init',
                'self_prompt_coarse_weight', 'train_aug',
                'head_feature_dim', 'head_coarse_size',
                'head_render_size', 'head_snake_kernel',
                'head_render_channels',
                'structure_mode', 'structure_tokens', 'structure_grid_h',
                'structure_grid_w', 'structure_render_size',
                'structure_highres_size', 'structure_residual_weight',
                'structure_residual_prior', 'structure_decision_weight'):
            if key in config:
                setattr(args, key, config[key])

    def _forward_structured(self, model, images, return_stage1=False):
        images_norm = self._sam_preprocess(model, images)
        batch = images_norm.shape[0]
        prompt_images = images
        if (not getattr(self, 'prediction_mode', False)
                and self._clean_images is not None):
            prompt_images = self._clean_images
        sparse, dense = model.prompt_encoder(
            points=None, boxes=None, masks=None)
        if (self.prompt_mode == 'structure'
                and hasattr(model, 'structural_prompt')):
            prompt, prompt_scale = model.structural_prompt(prompt_images)
            _, dense_prompt = model.prompt_encoder(
                points=None, boxes=None, masks=prompt)
            dense = dense + prompt_scale * dense_prompt
        sparse = sparse.expand(batch, -1, -1)
        dense = dense.expand(batch, -1, -1, -1)
        encoder_output = model.image_encoder(
            images_norm, return_features=True)
        if len(encoder_output) == 3:
            embeddings, _, adapter_features = encoder_output
        else:
            embeddings, _ = encoder_output
            adapter_features = None
        out = model.mask_decoder.predict_masks(
            image_embeddings=embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            adapter_features=adapter_features,
            return_stage1=return_stage1,
        )
        if return_stage1:
            stage2, iou, stage1 = out
        else:
            stage2, iou = out
            stage1 = None
        stage2_base = stage2
        structure = None
        cache = None
        if (self.self_prompt_mode == 'on'
                and hasattr(model, 'self_prompt')):
            coarse = stage2
            prompt, prompt_scale = model.self_prompt(
                coarse, prompt_images)
            _, dense_prompt = model.prompt_encoder(
                points=None, boxes=None, masks=prompt)
            dense_second = dense + prompt_scale * dense_prompt
            out2 = model.mask_decoder.predict_masks(
                image_embeddings=embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense_second,
                adapter_features=adapter_features,
                return_stage1=return_stage1,
            )
            if return_stage1:
                stage2, iou, _ = out2
            else:
                stage2, iou = out2
            cache = {
                'stage2_coarse': coarse,
                'self_prompt': prompt,
                'self_prompt_scale': prompt_scale.detach(),
            }
        if (self.deploy_mode == 'head'
                and hasattr(model, 'deployed_head')):
            stage2, stage2_high, head_cache = model.deployed_head(
                adapter_features, images, stage2)
            head_cache['stage2_high'] = stage2_high
            if cache is None:
                cache = {}
            cache.update(head_cache)
        elif (self.structure_mode == 'post'
                and hasattr(model, 'structure_control')):
            residual_mult = (
                self.structure_eval_residual_mult
                if getattr(self, 'prediction_mode', False) else 1.0)
            stage2, structure, structure_cache = model.structure_control(
                adapter_features, images, stage2,
                residual_mult=residual_mult)
            if structure_cache is not None:
                if cache is None:
                    cache = {}
                cache.update(structure_cache)
        if cache is not None:
            cache.setdefault('stage2_base', stage2_base)
        self._structure_cache = cache
        return stage2, iou, stage1, structure, cache

    def _forward(self, model, images, return_stage1=False):
        stage2, iou, stage1, _, _ = self._forward_structured(
            model, images, return_stage1=return_stage1)
        if return_stage1:
            return stage2, iou, stage1
        return stage2, iou

    def begin_prediction(self):
        super().begin_prediction()
        self._clean_images = None

    def prepare_train_batch(self, samples, rng):
        images, masks = super().prepare_train_batch(samples, rng)
        if self.train_aug == 'none':
            self._clean_images = None
            return images, masks
        # Keep a clean view for the structural prompt; the SAM decoder still
        # sees the augmented view.  Geometry is shared.
        clean = images
        if rng.random() < 0.5:
            images = torch.flip(images, dims=[3])
            clean = torch.flip(clean, dims=[3])
            masks = torch.flip(masks, dims=[3])
        if rng.random() < 0.5:
            images = torch.flip(images, dims=[2])
            clean = torch.flip(clean, dims=[2])
            masks = torch.flip(masks, dims=[2])
        if self.train_aug == 'strong':
            k = int(rng.randrange(4))
            if k:
                images = torch.rot90(images, k, dims=[2, 3])
                clean = torch.rot90(clean, k, dims=[2, 3])
                masks = torch.rot90(masks, k, dims=[2, 3])
            if rng.random() < 0.2:
                images = torch.nn.functional.avg_pool2d(
                    images, 3, 1, 1)
        # Mild photometric variation for cross-domain robustness.
        if rng.random() < 0.4:
            mean = images.mean(dim=(1, 2, 3), keepdim=True)
            contrast = 0.8 + 0.4 * rng.random()
            brightness = 0.8 + 0.4 * rng.random()
            images = ((images - mean) * contrast + mean) * brightness
        if rng.random() < 0.3:
            gamma = 0.8 + 0.4 * rng.random()
            images = images.clamp(0.0, 1.0).pow(gamma)
        if rng.random() < 0.2:
            images = images + torch.randn_like(images) * 0.01
        self._clean_images = clean
        return images.clamp(0.0, 1.0), masks

    def train_step(self, model, batch):
        images, masks = batch
        stage2, _, stage1, structure, cache = self._forward_structured(
            model, images, return_stage1=True)
        stage2_native = stage2
        high = stage2
        if (self.deploy_mode == 'head' and cache is not None):
            high = cache['stage2_high']
        elif (self.structure_mode == 'post' and structure is not None
                and cache is not None):
            high = model.structure_control.highres(
                stage2, images, structure, cache['evidence'],
                uncertainty=cache.get('uncertainty'))
        masks = F.interpolate(
            masks, (1024, 1024), mode='bilinear', align_corners=False)
        stage1 = F.interpolate(
            stage1, (1024, 1024), mode='bilinear', align_corners=False)
        if high.shape[-1] != 1024:
            high = F.interpolate(
                high, (1024, 1024), mode='bilinear',
                align_corners=False)

        bce_weight = float(self.args.bce_weight)
        dice_weight = float(self.args.dice_weight)
        head_indices = self._stage2_loss_indices(high.shape[1])
        head_losses = {
            k: self._mask_criterion(high[:, k:k + 1], masks,
                                    bce_weight, dice_weight)
            for k in head_indices
        }
        loss = self._aggregate_head_losses(
            head_losses, head_indices, self._deployment_anchor_index())
        if (self.self_prompt_mode == 'on' and cache is not None
                and 'stage2_coarse' in cache):
            coarse = F.interpolate(
                cache['stage2_coarse'], (1024, 1024), mode='bilinear',
                align_corners=False)
            coarse_losses = {
                k: self._mask_criterion(
                    coarse[:, k:k + 1], masks, bce_weight, dice_weight)
                for k in head_indices
            }
            coarse_loss = self._aggregate_head_losses(
                coarse_losses, head_indices,
                self._deployment_anchor_index())
            loss = loss + self.self_prompt_coarse_weight * coarse_loss

        if self.deep_sup_weight > 0:
            ds_losses = [
                self._mask_criterion(stage1[:, k:k + 1], masks,
                                     bce_weight, dice_weight)
                for k in range(stage1.shape[1])
            ]
            loss = loss + self.deep_sup_weight * torch.stack(ds_losses).mean()

        # In head mode the deployed 1024 output is also the source of the
        # topology and role losses, so the high-resolution renderer is
        # trained by the unchanged DCD structural objective rather than by
        # the dense loss alone.  Legacy mode keeps the native 256 tensor.
        topology_source = (
            high if self.deploy_mode == 'head' else stage2_native)
        loss = loss + self._topology_loss(topology_source, masks)
        if self.specialized_heads:
            loss = loss + self._specialized_head_loss(
                topology_source, masks)
        if self.head_fusion:
            fused = F.interpolate(
                self._fused_heads(model, topology_source),
                (1024, 1024), mode='bilinear', align_corners=False)
            loss = loss + self.fusion_loss_weight * self._mask_criterion(
                fused, masks, bce_weight, dice_weight)
        return loss

    def _deploy_logits(self, model, prepared):
        stage2, _, _, structure, cache = self._forward_structured(
            model, prepared, return_stage1=False)
        if (self.aux_eval_off and cache is not None
                and 'stage2_base' in cache):
            base = cache['stage2_base']
            if self.head_fusion:
                logits = self._fused_heads(model, base)
            else:
                index = self._selected_stage2_index()
                logits = base[:, index:index + 1]
            return logits, None, cache
        if (self.self_prompt_gate and cache is not None
                and 'stage2_coarse' in cache):
            coarse = cache['stage2_coarse']
            if self.head_fusion:
                refined_logits = self._fused_heads(model, stage2)
                coarse_logits = self._fused_heads(model, coarse)
            else:
                index = self._selected_stage2_index()
                refined_logits = stage2[:, index:index + 1]
                coarse_logits = coarse[:, index:index + 1]
            p = torch.sigmoid(coarse_logits)
            gate = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)
            logits = (1.0 - gate) * coarse_logits + gate * refined_logits
            return logits, None, cache
        if (self.self_prompt_blend > 0.0 and cache is not None
                and 'stage2_coarse' in cache):
            weight = min(1.0, max(0.0, self.self_prompt_blend))
            coarse = cache['stage2_coarse']
            if self.head_fusion:
                refined_logits = self._fused_heads(model, stage2)
                coarse_logits = self._fused_heads(model, coarse)
            else:
                index = self._selected_stage2_index()
                refined_logits = stage2[:, index:index + 1]
                coarse_logits = coarse[:, index:index + 1]
            logits = ((1.0 - weight) * refined_logits
                      + weight * coarse_logits)
            return logits, None, cache
        if self.deploy_mode == 'head' and cache is not None:
            logits = cache['stage2_high']
            if self.head_fusion:
                logits = self._fused_heads(model, logits)
            else:
                index = self._selected_stage2_index()
                logits = logits[:, index:index + 1]
            return logits, None, cache
        final_mode = bool(
            getattr(self, 'prediction_mode', False)
            and self.structure_mode == 'post'
            and structure is not None
            and cache is not None)
        if final_mode:
            logits = model.structure_control.highres(
                stage2, prepared, structure, cache['evidence'],
                highres_mult=self.structure_eval_highres_mult,
                uncertainty=cache.get('uncertainty'))
            if self.head_fusion:
                logits = self._fused_heads(model, logits)
            else:
                index = self._selected_stage2_index()
                logits = logits[:, index:index + 1]
        else:
            if self.head_fusion:
                logits = self._fused_heads(model, stage2)
            else:
                index = self._selected_stage2_index()
                logits = stage2[:, index:index + 1]
        return logits, structure, cache

    def _predict_tiled(self, model, sample):
        from baseline_zoo.data import load_rgb
        image = load_rgb(sample.image_path)
        width, height = image.size
        if max(width, height) <= self.tile_size:
            return None
        if max(width, height) < self.tile_min_size:
            return None
        import numpy as np
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        tensor = tensor.to(self.device)
        canvas = torch.zeros(1, height, width, device=self.device)
        weight = torch.zeros(1, height, width, device=self.device)
        size = self.tile_size
        stride = max(1, min(self.tile_stride, size))
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y1, x1 = min(y + size, height), min(x + size, width)
                if y1 - y < 16 or x1 - x < 16:
                    continue
                patch = torch.zeros(3, size, size, device=self.device)
                patch[:, :y1 - y, :x1 - x] = tensor[:, y:y1, x:x1]
                logits, _, _ = self._deploy_logits(
                    model, patch.unsqueeze(0))
                logits = F.interpolate(
                    logits, size=(size, size), mode='bilinear',
                    align_corners=False)[0, 0]
                canvas[0, y:y1, x:x1] += logits[:y1 - y, :x1 - x]
                weight[0, y:y1, x:x1] += 1.0
        canvas = canvas / weight.clamp_min(1.0)
        threshold = float(
            getattr(self.args, 'deploy_threshold', 0.5))
        return (torch.sigmoid(canvas)[0].cpu().numpy() > threshold)

    def predict_sample(self, model, prepared, sample):
        if self.prediction_mode and self.tile_inference:
            tiled = self._predict_tiled(model, sample)
            if tiled is not None:
                return tiled
        logits, structure, cache = self._deploy_logits(model, prepared)
        if self.prediction_mode and self.tta_flip:
            flipped = torch.flip(prepared, dims=[3])
            flip_logits, _, _ = self._deploy_logits(model, flipped)
            logits = 0.5 * (
                logits + torch.flip(flip_logits, dims=[3]))
        if (getattr(self, 'prediction_mode', False)
                and self.guided_upsample
                and self.deploy_mode != 'head'):
            logits = self._guided_refine(logits, prepared)
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(
            logits, size=(height, width), mode='bilinear',
            align_corners=False)
        probs = torch.sigmoid(logits)[0, 0]
        threshold = float(
            getattr(self.args, 'deploy_threshold', 0.5))
        if (getattr(self, 'prediction_mode', False)
                and structure is not None):
            u = (4.0 * probs * (1.0 - probs)).clamp(0.0, 1.0)
            t = model.structure_control.decision_threshold(
                structure, (height, width), base=threshold,
                weight=self.structure_eval_decision_weight,
                uncertainty=u.view(1, 1, height, width))[0, 0]
            return (probs > t).cpu().numpy()
        return (probs > threshold).cpu().numpy()

    @staticmethod
    def _guided_refine(logits: torch.Tensor, image: torch.Tensor,
                       radius: int = 4, eps: float = 1e-2) -> torch.Tensor:
        """Edge-aware guided-filter refinement at the decoder resolution."""
        if logits.shape[-2:] != image.shape[-2:]:
            image = F.interpolate(
                image, size=logits.shape[-2:], mode='bilinear',
                align_corners=False)
        guide = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2]
                 + 0.114 * image[:, 2:3])
        mean = guide.mean(dim=(2, 3), keepdim=True)
        std = guide.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        guide = (guide - mean) / std
        k = 2 * radius + 1

        def box(x):
            return F.avg_pool2d(
                x, k, 1, radius, count_include_pad=False)

        mean_i = box(guide)
        mean_p = box(logits)
        corr_i = box(guide * guide)
        corr_ip = box(guide * logits)
        var_i = corr_i - mean_i.square()
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return box(a) * guide + box(b)

    @torch.no_grad()
    def predict_candidates(self, model, prepared, sample):
        if not sample.mask_path:
            raise ValueError('candidate diagnostics require a GT mask')
        stage2, _, _, _, _ = self._forward_structured(
            model, prepared, return_stage1=False)
        if stage2.shape[1] != 4:
            raise RuntimeError(
                f'expected four Stage-2 candidates, got {stage2.shape[1]}')
        with Image.open(sample.mask_path) as mask:
            width, height = mask.size
        logits = F.interpolate(
            stage2, size=(height, width), mode='bilinear',
            align_corners=False)
        threshold = float(getattr(self.args, 'deploy_threshold', 0.5))
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return probs > threshold
