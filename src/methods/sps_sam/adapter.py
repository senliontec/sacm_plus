"""Runtime adapter for SP-SAM (self-prompting SAM).

Training (source only, 18 images):
  * a proposal head learns the GT-skeleton click heatmap at 256x256;
  * the official SAM promptable decoder is trained so a click returns the
    GT inside a 128-pixel disk around the click (a local structure patch);
  * several point masks are aggregated by predicted IoU and supervised
    against the union of the clicks' local patches.

Local targets are essential: with the full connected component as the
target, every foreground click has the same target on one-component GT and
the decoder learns to ignore the click (v1 prompt collapse).

Inference uses only target images: top-k proposal peaks are clicked, each
peak is decoded, and the masks are IoU-weighted.  No target labels, no
gradient updates, no pseudo-labels.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

REPO = Path('/home/dev/mywork/project/segment/sacm_plus')
sys.path.insert(0, str(REPO / 'src'))

from baseline_zoo.data import load_mask  # noqa: E402
from baseline_zoo.runtime import BaselineAdapter, main_for  # noqa: E402

from methods.sps_sam.model import (  # noqa: E402
    SelfPromptSAM, build_model, select_peaks)
from methods.sps_sam.topology_losses import SoftclDiceLoss  # noqa: E402


class SPSAdapter(BaselineAdapter):
    id = 'sps_sam'
    display_name = 'SP-SAM self-prompting'
    official_source = 'SAM (official ViT-L weights) promptable decoder'
    official_entrypoint = (
        'Prompt-proposal head + point-conditioned official SAM decoder + '
        'IoU-weighted prompt aggregation')
    input_size = 1024
    official_lr = 3e-4
    official_weight_decay = 0.01
    train_batch_size = 1
    selection = 'dice'
    deviations = (
        'The image encoder starts from the official SAM ViT-L checkpoint; '
        'only the per-block adapters, the prompt encoder, the official '
        'stage-1 mask-decoder weights and a new proposal head are trained.',
        'The decoder path is official SAM single-mask decoding: prompt '
        'encoder point embedding, two-way transformer, mask token 0 and the '
        'stage-1 hypernetwork; the SACM stage-2/candidate machinery is not '
        'used.',
        'Training teaches per-click local targets: a click returns the GT '
        'inside a 128-pixel disk around the click.  Full-component targets '
        'were tried first and collapsed to a prompt-invariant global mask.',
        'Inference aggregates K=64 label-free proposal peaks with '
        'predicted-IoU weights; no hand-written guard, consensus gate, '
        'TCSU residual, or component filter is applied.',
        'Images are resized to 1024x1024 for the screen (no tiling); GT '
        'masks use bilinear resize with a 0.25 threshold to preserve thin '
        'structures, and training targets are max-pooled to the loss '
        'resolution.',
    )

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '--sam-checkpoint', type=Path,
            default=REPO / 'checkpoints' / 'sam_vit_l_0b3195.pth')
        parser.add_argument('--adapter-dim-ratio', type=float, default=0.1)
        parser.add_argument('--sp-adapters', type=int, choices=[0, 1],
                            default=1)
        parser.add_argument('--sp-prompt-encoder', type=int, choices=[0, 1],
                            default=1)
        parser.add_argument('--sp-train-k', type=int, default=8)
        parser.add_argument('--sp-neg-k', type=int, default=2)
        parser.add_argument('--sp-deploy-k', type=int, default=64)
        parser.add_argument('--sp-nms-radius', type=int, default=2,
                            help='NMS radius in 256x256 heatmap cells')
        parser.add_argument('--sp-local-radius', type=int, default=128,
                            help='radius (1024 frame) of the per-click '
                                 'local target disk')
        parser.add_argument('--sp-min-click-dist', type=int, default=64,
                            help='minimum distance between structure clicks')
        parser.add_argument('--sp-loss-res', type=int, default=512)
        parser.add_argument('--sp-proposal-weight', type=float, default=0.2)
        parser.add_argument('--sp-point-weight', type=float, default=0.5)
        parser.add_argument('--sp-agg-weight', type=float, default=1.0)
        parser.add_argument('--sp-cl-weight', type=float, default=0.1)
        parser.add_argument('--sp-iou-weight', type=float, default=0.05)
        parser.add_argument('--sp-neg-weight', type=float, default=0.5)
        parser.add_argument('--sp-proposal-pos-weight', type=float,
                            default=2.0)
        parser.add_argument('--sp-bce-pos-weight', type=float, default=2.0)
        parser.add_argument('--sp-sample-skeleton', type=int, choices=[0, 1],
                            default=1)
        parser.add_argument('--sp-rot90', type=int, choices=[0, 1], default=1)
        parser.add_argument('--sp-photo', type=int, choices=[0, 1], default=1)
        parser.add_argument('--sp-gt-threshold', type=float, default=0.25)

    def save_config(self, args: argparse.Namespace) -> dict:
        return {
            'sam_checkpoint': str(args.sam_checkpoint),
            'adapter_dim_ratio': float(args.adapter_dim_ratio),
            'sp_adapters': int(args.sp_adapters),
            'sp_prompt_encoder': int(args.sp_prompt_encoder),
            'sp_train_k': int(args.sp_train_k),
            'sp_neg_k': int(args.sp_neg_k),
            'sp_deploy_k': int(args.sp_deploy_k),
            'sp_nms_radius': int(args.sp_nms_radius),
            'sp_local_radius': int(args.sp_local_radius),
            'sp_min_click_dist': int(args.sp_min_click_dist),
            'sp_loss_res': int(args.sp_loss_res),
            'sp_proposal_weight': float(args.sp_proposal_weight),
            'sp_point_weight': float(args.sp_point_weight),
            'sp_agg_weight': float(args.sp_agg_weight),
            'sp_cl_weight': float(args.sp_cl_weight),
            'sp_iou_weight': float(args.sp_iou_weight),
            'sp_neg_weight': float(args.sp_neg_weight),
            'sp_proposal_pos_weight': float(args.sp_proposal_pos_weight),
            'sp_bce_pos_weight': float(args.sp_bce_pos_weight),
            'sp_sample_skeleton': int(args.sp_sample_skeleton),
            'sp_rot90': int(args.sp_rot90),
            'sp_photo': int(args.sp_photo),
            'sp_gt_threshold': float(args.sp_gt_threshold),
        }

    def restore_config(self, args: argparse.Namespace,
                       config: dict) -> None:
        for key, value in config.items():
            if key == 'sam_checkpoint':
                setattr(args, key, Path(value))
            else:
                setattr(args, key, value)

    def build_model(self, args: argparse.Namespace) -> torch.nn.Module:
        model = build_model(args)
        self._cl_loss = SoftclDiceLoss(num_iter=5)
        return model

    def configure_trainable(self, model: SelfPromptSAM) -> None:
        args = self.args
        # Recompute frozen-block activations: the ViT-L forward/backward at
        # 1024 does not fit an 11 GiB card otherwise.
        model.sam.image_encoder.grad_checkpointing = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        enc = 'sam.image_encoder.'
        adapter_keys = (f'{enc}external_adapters.', f'{enc}adapter_norms.')
        decoder_skip = ('output_hypernetworks_mlps_stage2',
                        'adapter_weight_proj', 'adapter_weight_ffn',
                        'adapter_aggregator')
        for name, parameter in model.named_parameters():
            if name.startswith('proposal.'):
                parameter.requires_grad_(True)
            elif name.startswith('sam.prompt_encoder.'):
                parameter.requires_grad_(
                    bool(getattr(args, 'sp_prompt_encoder', 1)))
            elif name.startswith('sam.mask_decoder.'):
                if not any(key in name for key in decoder_skip):
                    parameter.requires_grad_(True)
            elif any(name.startswith(key) for key in adapter_keys):
                parameter.requires_grad_(
                    bool(getattr(args, 'sp_adapters', 1)))

    def make_optimizer(self, model, args):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay)

    # ---- training data ------------------------------------------------
    def _augment(self, image: np.ndarray, mask: np.ndarray, rng: random.Random):
        if rng.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        if rng.random() < 0.5:
            image = image[::-1]
            mask = mask[::-1]
        if getattr(self.args, 'sp_rot90', 1) and rng.random() < 0.5:
            k = rng.randrange(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if getattr(self.args, 'sp_photo', 1):
            image = image.astype(np.float32)
            image = image * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
            if rng.random() < 0.3:
                image = image + np.random.default_rng(
                    rng.randrange(2 ** 31)).normal(
                        0, 4.0, image.shape).astype(np.float32)
            image = np.clip(image, 0, 255)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    @staticmethod
    def _resize(image: np.ndarray, mask: np.ndarray,
                threshold: float) -> tuple[np.ndarray, np.ndarray]:
        pil = Image.fromarray(image.astype(np.uint8)).resize(
            (1024, 1024), Image.BILINEAR)
        image = np.asarray(pil, dtype=np.float32) / 255.0
        m = Image.fromarray(mask.astype(np.float32) * 255.0).resize(
            (1024, 1024), Image.BILINEAR)
        mask = np.asarray(m, dtype=np.float32) / 255.0 > threshold
        return image, mask

    @staticmethod
    def _local_target(mask: np.ndarray, x: float, y: float,
                      radius: int) -> np.ndarray:
        """GT inside a disk centred at the click (local structure patch)."""
        height, width = mask.shape
        x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
        target = np.zeros_like(mask, dtype=np.float32)
        target[y0:y1, x0:x1] = mask[y0:y1, x0:x1] & disk
        return target

    def _sample_points(self, mask: np.ndarray, rng: random.Random):
        args = self.args
        train_k = int(getattr(args, 'sp_train_k', 8))
        neg_k = min(int(getattr(args, 'sp_neg_k', 2)), max(train_k - 1, 0))
        pos_k = max(train_k - neg_k, 1)
        radius = int(getattr(args, 'sp_local_radius', 128))
        min_dist = int(getattr(args, 'sp_min_click_dist', 64))
        skel = skeletonize(mask) if getattr(
            args, 'sp_sample_skeleton', 1) else mask
        if not skel.any():
            skel = mask
        ys, xs = np.nonzero(skel)
        if len(ys) == 0:
            coords = np.zeros((train_k, 2), np.float32)
            labels = np.zeros((train_k,), np.int64)
            targets = np.zeros((train_k, *mask.shape), np.float32)
            return coords, labels, targets
        picks: list[tuple[int, int]] = []
        for _ in range(600):
            if len(picks) == pos_k:
                break
            index = rng.randrange(len(ys))
            x, y = int(xs[index]), int(ys[index])
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2
                   for px, py in picks):
                picks.append((x, y))
        while len(picks) < pos_k:
            index = rng.randrange(len(ys))
            picks.append((int(xs[index]), int(ys[index])))
        neg_picks: list[tuple[int, int]] = []
        if neg_k > 0:
            dist = distance_transform_edt(~mask)
            far = dist >= max(min_dist, radius)
            if far.any():
                ny, nx = np.nonzero(far)
            else:
                ny, nx = np.nonzero(~mask)
            if len(ny) > 0:
                for j in rng.sample(range(len(ny)), min(neg_k, len(ny))):
                    neg_picks.append((int(nx[j]), int(ny[j])))
        coords, labels, targets = [], [], []
        entries = ([(x, y, 1) for x, y in picks]
                   + [(x, y, 0) for x, y in neg_picks])
        for x, y, label_value in entries:
            # The proposal heatmap localises a click to a 4-pixel cell at
            # 1024; jitter by one cell so inference-time maxima stay on the
            # training manifold.
            jx = min(max(x + rng.uniform(-4.0, 4.0), 0.0),
                     mask.shape[1] - 1.0)
            jy = min(max(y + rng.uniform(-4.0, 4.0), 0.0),
                     mask.shape[0] - 1.0)
            coords.append([jx, jy])
            labels.append(label_value)
            targets.append(self._local_target(mask, jx, jy, radius))
        return (np.asarray(coords, np.float32),
                np.asarray(labels, np.int64),
                np.asarray(targets, np.float32))

    def prepare_train_batch(self, samples, rng: random.Random):
        sample = samples[0]
        image = np.asarray(
            Image.open(sample.image_path).convert('RGB'))
        mask = load_mask(Path(sample.mask_path)).astype(np.uint8)
        image, mask = self._augment(image, mask, rng)
        image, mask = self._resize(
            image, mask, float(getattr(self.args, 'sp_gt_threshold', 0.25)))
        coords, labels, targets = self._sample_points(mask, rng)
        res = int(getattr(self.args, 'sp_loss_res', 512))
        skel = skeletonize(mask) if mask.any() else mask
        skel256 = F.adaptive_max_pool2d(
            torch.from_numpy(skel[None, None].astype(np.float32)),
            (256, 256))
        band = F.max_pool2d(skel256, kernel_size=3, stride=1, padding=1)
        tgt = F.adaptive_max_pool2d(
            torch.from_numpy(targets[:, None]), (res, res))[:, 0][None]
        agg = tgt.amax(dim=1, keepdim=True)
        return {
            'image': torch.from_numpy(image).permute(2, 0, 1)[None].float(),
            'skel': band.float(),
            'coords': torch.from_numpy(coords)[None],
            'labels': torch.from_numpy(labels)[None],
            'targets': tgt,
            'agg': agg,
        }

    # ---- training -----------------------------------------------------
    @staticmethod
    def _dice(prob: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
        num = 2.0 * (prob * target).sum(dim=(-1, -2)) + eps
        den = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + eps
        return num / den

    def train_step(self, model: SelfPromptSAM, batch) -> torch.Tensor:
        device = self.device
        image = batch['image'].to(device)
        skel = batch['skel'].to(device)
        coords = batch['coords'].to(device)
        labels = batch['labels'].to(device)
        targets = batch['targets'].to(device)
        # Aggregate target: the union of the clicks' local patches.  Using
        # the full GT here would reward a prompt-invariant global mask,
        # which is exactly the v1 failure.
        gt = batch['agg'].to(device)
        args = self.args

        features = model.encode(image)
        prop_logits = model.proposal(features)
        prop_loss = F.binary_cross_entropy_with_logits(
            prop_logits, skel,
            pos_weight=torch.tensor(
                float(getattr(args, 'sp_proposal_pos_weight', 2.0)),
                device=device))
        masks, iou = model.decode_points(features, coords, labels)
        b, p = coords.shape[0], coords.shape[1]
        logits = F.interpolate(
            masks[:, :, 0].reshape(b * p, 1, *masks.shape[-2:]),
            size=targets.shape[-2:], mode='bilinear',
            align_corners=False).view(b, p, *targets.shape[-2:])
        prob = torch.sigmoid(logits)
        with torch.no_grad():
            dice_t = self._dice(prob, targets).clamp(0.0, 1.0)
        iou_loss = F.mse_loss(torch.sigmoid(iou[:, :, 0]), dice_t)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor(
                float(getattr(args, 'sp_bce_pos_weight', 2.0)),
                device=device)).mean(dim=(-1, -2))
        point_loss = (bce + 1.0 - self._dice(prob, targets)).mean()

        # Weighted union: one confident prompt may contribute a local patch
        # on its own.  Normalising by the best prompt keeps the aggregation
        # scale-invariant (a plain mean would erase patches covered by fewer
        # than K/2 prompts).
        weight = torch.sigmoid(iou[:, :, 0])
        weight = weight / weight.max(dim=1, keepdim=True).values.clamp_min(
            1e-6)
        agg = (prob * weight[..., None, None]).amax(dim=1, keepdim=True)
        agg_logits = torch.logit(agg.clamp(1e-4, 1.0 - 1e-4))
        agg_loss = (F.binary_cross_entropy_with_logits(
            agg_logits, gt, pos_weight=torch.tensor(
                float(getattr(args, 'sp_bce_pos_weight', 2.0)),
                device=device))
            + 1.0 - self._dice(agg, gt).mean())
        cl_loss = self._cl_loss(agg_logits, gt)

        return (float(getattr(args, 'sp_proposal_weight', 0.2)) * prop_loss
                + float(getattr(args, 'sp_point_weight', 0.5)) * point_loss
                + float(getattr(args, 'sp_agg_weight', 1.0)) * agg_loss
                + float(getattr(args, 'sp_cl_weight', 0.1)) * cl_loss
                + float(getattr(args, 'sp_iou_weight', 0.05)) * iou_loss)

    # ---- inference ----------------------------------------------------
    def prepare_eval_sample(self, sample, device):
        image = np.asarray(
            Image.open(sample.image_path).convert('RGB'))
        height, width = image.shape[:2]
        pil = Image.fromarray(image).resize((1024, 1024), Image.BILINEAR)
        tensor = (torch.from_numpy(
            np.asarray(pil, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)[None].to(device))
        return {'image': tensor, 'shape': (height, width)}

    @torch.no_grad()
    def predict_sample(self, model: SelfPromptSAM, prepared, sample):
        device = self.device
        image = prepared['image']
        shape = prepared['shape']
        features = model.encode(image)
        heat = torch.sigmoid(model.proposal(features))[0, 0]
        cells = select_peaks(
            heat, int(getattr(self.args, 'sp_deploy_k', 16)),
            int(getattr(self.args, 'sp_nms_radius', 2)))
        if cells.numel() == 0:
            return np.zeros(shape, np.float32)
        scale = 1024.0 / heat.shape[-1]
        coords = (cells * scale)[None].to(device)
        labels = torch.ones(
            (1, coords.shape[1]), dtype=torch.int64, device=device)
        masks, iou = model.decode_points(features, coords, labels)
        prob = torch.sigmoid(masks[:, :, 0])[0]
        weight = torch.sigmoid(iou[:, :, 0])[0]
        weight = weight / weight.max().clamp_min(1e-6)
        agg = (prob * weight[:, None, None]).amax(dim=0)
        agg = F.interpolate(
            agg[None, None], size=shape, mode='bilinear',
            align_corners=False)[0, 0]
        return agg.cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    raise SystemExit(main_for(SPSAdapter()))
