"""Topology-aware residual Mixture-of-Experts (FineSAM++-style).

The module is inserted after the fixed Stage-2 head.  A lightweight gate
predicts *where* the coarse prediction is wrong; several deterministic
residual experts predict *how* to correct it.  The correction is added in
logit space and only inside the gate:

    z = z0 + m * sum_j w_j * r_j

Design differences from FineSAM++ (NeurIPS 2025, "Gate to the Vessel"):

* Gate targets are topology-aware (missed skeleton tubes, spurious
  components, dilated boundary errors), not only raw pixel error.
* Experts use deterministic, curvilinear-specific degradations (opening,
  closing, threshold jitter) at both train and test time.  This removes
  FineSAM++'s train/test degradation mismatch.
* The gate may also consume the disagreement of the four Stage-2 heads,
  which the network already computes, and the training-only centerline aux
  logits.
* The residual loss can be Dice/BCE plus Skeleton Recall instead of MSE.

The module is deliberately small: four depthwise-separable experts and a
light gate.  The last expert layer is zero-initialised, so enabling the
module starts from the exact coarse prediction.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import LayerNorm2d


# Deterministic expert preprocessing.  Every expert sees the same coarse
# probability map transformed in a fixed way, both during training and at
# inference.  The experts therefore learn to invert a known transform
# instead of memorising a random train-time degradation.
_EXPERT_KINDS = ('open', 'close', 'threshold_low', 'threshold_high')


def _resize(x: torch.Tensor, size: Tuple[int, int],
            mode: str = 'bilinear') -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    if mode == 'nearest':
        return F.interpolate(x, size=size, mode='nearest')
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def _resize_target_topology(target: torch.Tensor,
                            size: int) -> torch.Tensor:
    """Resize a mask for topology supervision.

    Downsampling must use max-pooling: a 1-px line resized with bilinear
    becomes <0.5 and disappears at the skeleton threshold.
    """
    if target.shape[-1] == size and target.shape[-2] == size:
        return target
    if target.shape[-1] > size:
        return F.adaptive_max_pool2d(target, (size, size))
    return F.interpolate(target, size=(size, size), mode='nearest')


def _max_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _min_filter(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return 1.0 - _max_filter(1.0 - x, k=k)


def _open(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return _max_filter(_min_filter(x, k=k), k=k)


def _close(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return _min_filter(_max_filter(x, k=k), k=k)


def degrade_coarse(prob: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply one deterministic expert-specific degradation."""
    if kind == 'open':
        return _open(prob)
    if kind == 'close':
        return _close(prob)
    if kind == 'threshold_low':
        return (prob > 0.35).to(prob.dtype)
    if kind == 'threshold_high':
        return (prob > 0.65).to(prob.dtype)
    if kind == 'erode':
        return _min_filter(prob)
    if kind == 'dilate':
        return _max_filter(prob)
    raise ValueError(f'unknown expert degradation {kind!r}')


def build_topo_gate_target(
    coarse_logits: torch.Tensor,
    target: torch.Tensor,
    band: int = 2,
    dilate: int = 2,
) -> torch.Tensor:
    """Build the gate pseudo-label.

    For binary GT, ``|p - y| > 0.5`` is exactly the hard misclassification
    mask at threshold 0.5.  We extend it with three structural terms:

    * ``missed``   : GT pixels farther than ``band`` px from any prediction
                     (fragmented / omitted thin structures);
    * ``spurious`` : predicted pixels farther than ``band`` px from any GT
                     (isolated components / false positives);
    * ``boundary`` : errors inside the dilated GT boundary band.

    The union is dilated once so the gate learns a local error region, not
    a 1-px sparse mask.
    """
    target = target.to(coarse_logits.device)
    if coarse_logits.shape[-2:] != target.shape[-2:]:
        target = _resize_target_topology(
            target, coarse_logits.shape[-1])
    pred = coarse_logits.detach() > 0
    gt = target.detach() > 0.5
    err = (pred != gt)

    k = 2 * int(band) + 1
    gt_dil = _max_filter(gt.to(coarse_logits.dtype), k=k) > 0.5
    pred_dil = _max_filter(pred.to(coarse_logits.dtype), k=k) > 0.5
    gt_er = _min_filter(gt.to(coarse_logits.dtype), k=3) > 0.5

    missed = gt & ~pred_dil
    spurious = pred & ~gt_dil
    boundary = (gt_dil & ~gt_er) & err

    union = err | missed | spurious | boundary
    kd = 2 * int(dilate) + 1
    union = _max_filter(union.to(coarse_logits.dtype), k=kd) > 0.5
    return union.to(coarse_logits.dtype)


def _soft_dice(prob: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def _expert_kinds(num_experts: int) -> Tuple[str, ...]:
    if num_experts <= 0:
        raise ValueError('res_num_experts must be positive')
    kinds: List[str] = []
    for i in range(num_experts):
        kinds.append(_EXPERT_KINDS[i % len(_EXPERT_KINDS)])
    return tuple(kinds)


class _ResidualExpert(nn.Module):
    """Depthwise-separable multi-dilation residual expert."""

    def __init__(self, in_channels: int, hidden: int = 24,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            LayerNorm2d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    hidden, hidden, kernel_size=3, padding=d,
                    dilation=d, groups=hidden, bias=False),
                nn.Conv2d(hidden, hidden, kernel_size=1),
                LayerNorm2d(hidden),
                nn.GELU(),
            )
            for d in (1, 2, 4)
        ])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        # Identity start: the residual is exactly zero at step 0.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(self.drop(h))


class _GateNet(nn.Module):
    """Lightweight spatial gate + global soft routing."""

    def __init__(self, in_channels: int, num_experts: int,
                 hidden: int = 32, gate_size: int = 256) -> None:
        super().__init__()
        self.gate_size = int(gate_size)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3,
                      stride=2, padding=1),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            LayerNorm2d(hidden),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.route_head = nn.Linear(hidden, num_experts)
        # Sparse start: sigmoid(-2) ~= 0.12; residual is zero anyway.
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -2.0)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _resize(x, (self.gate_size, self.gate_size))
        feat = self.body(x)
        m_logits = self.mask_head(feat)
        route = self.route_head(feat.mean(dim=(2, 3)))
        return m_logits, route


class TopoResidualMoE(nn.Module):
    """Gate + deterministic residual experts on one Stage-2 head.

    Args:
        num_experts: number of residual experts (2-4 recommended).
        hidden: expert hidden width.
        gate_hidden: gate hidden width.
        gate_size: gate working resolution (experts run at ``out_size``).
        gate_mode: ``learned`` (FineSAM++-style) or ``disagreement``
            (training-free head-variance gate).
        use_head_disagreement: feed the four-head std to the gate.
        use_centerline: feed the training-only centerline aux logits.
        gate_input_norm: instance-normalise the image channel for the gate.
        out_size: residual working resolution (default 512).
        head_index: Stage-2 slot corrected by this module.
    """

    def __init__(
        self,
        num_experts: int = 4,
        hidden: int = 24,
        gate_hidden: int = 32,
        gate_size: int = 512,
        gate_mode: str = 'learned',
        use_head_disagreement: bool = True,
        use_centerline: bool = True,
        gate_input_norm: bool = True,
        out_size: int = 512,
        head_index: int = 2,
        dropout: float = 0.0,
        disagreement_tau: float = 0.05,
        disagreement_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if gate_mode not in ('learned', 'disagreement'):
            raise ValueError(
                f'gate_mode must be learned|disagreement, got {gate_mode!r}')
        self.num_experts = int(num_experts)
        self.hidden = int(hidden)
        self.gate_hidden = int(gate_hidden)
        self.gate_size = int(gate_size)
        self.gate_mode = gate_mode
        self.use_head_disagreement = bool(use_head_disagreement)
        self.use_centerline = bool(use_centerline)
        self.gate_input_norm = bool(gate_input_norm)
        self.out_size = int(out_size)
        self.head_index = int(head_index)
        self.disagreement_tau = float(disagreement_tau)
        self.disagreement_scale = float(disagreement_scale)
        self.expert_kinds = _expert_kinds(self.num_experts)
        # Controlled by the trainer.  0 makes the module an exact identity
        # during warm-up; 0<scale<1 ramps the residual correction in.
        self.res_scale: float = 1.0

        gate_in = 3 + 1
        if self.use_head_disagreement:
            gate_in += 1
        if self.use_centerline:
            gate_in += 1
        self.gate = None
        if self.gate_mode == 'learned':
            self.gate = _GateNet(
                gate_in, self.num_experts,
                hidden=self.gate_hidden, gate_size=self.gate_size)

        self.experts = nn.ModuleList([
            _ResidualExpert(4, hidden=self.hidden, dropout=dropout)
            for _ in range(self.num_experts)
        ])

        self.last_coarse_logits: Optional[torch.Tensor] = None
        self.last_gate_logits: Optional[torch.Tensor] = None
        self.last_gate_mask: Optional[torch.Tensor] = None
        self.last_routing: Optional[torch.Tensor] = None
        self.last_target_rate: float = 0.0
        self.experts_frozen: bool = False

    @property
    def has_learned_gate(self) -> bool:
        return self.gate is not None

    def freeze_experts(self) -> None:
        """Freeze the residual experts at an exactly-zero correction.

        Gate-only training needs the learned gate (and its localization
        signal) but no pixel residual.  The main mask loss would otherwise
        train the experts through ``fixed_head_loss_weight`` even when both
        repair loss terms are zero, making the checkpoint's head 2 differ
        from the coarse head that the component-repair evaluator consumes.
        Zeroing the final expert convolution makes the residual exactly
        zero, and freezing every expert parameter keeps it there.
        """
        for param in self.experts.parameters():
            param.requires_grad = False
        with torch.no_grad():
            for expert in self.experts:
                expert.out.weight.zero_()
                expert.out.bias.zero_()
        self.experts_frozen = True

    def _image_channels(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 3:
            return image
        if image.shape[1] > 3:
            return image[:, :3]
        return image.repeat(1, 3, 1, 1)

    @staticmethod
    def _match_batch(x: Optional[torch.Tensor],
                     batch: int) -> Optional[torch.Tensor]:
        if x is None or x.shape[0] == batch:
            return x
        if batch % x.shape[0] != 0:
            raise ValueError(
                f'auxiliary batch {x.shape[0]} does not divide {batch}')
        return x.repeat_interleave(batch // x.shape[0], dim=0)

    def _gate_input(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        all_head_logits: Optional[torch.Tensor],
        aux: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        size = (self.gate_size, self.gate_size)
        img = _resize(self._image_channels(image), size)
        if self.gate_input_norm:
            mean = img.mean(dim=(2, 3), keepdim=True)
            std = img.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
            img = (img - mean) / std
        # Gate conditioning must not backprop into the coarse model or the
        # candidate heads: otherwise the gate loss could "fix" its own
        # input instead of learning to localize errors.
        parts = [img, _resize(coarse_logits.detach(), size)]
        if self.use_head_disagreement:
            heads = self._match_batch(
                all_head_logits, coarse_logits.shape[0])
            if heads is not None and heads.shape[1] > 1:
                probs = torch.sigmoid(heads.detach())
                disagreement = probs.std(dim=1, keepdim=True)
            else:
                disagreement = torch.zeros_like(coarse_logits)
            parts.append(_resize(disagreement, size))
        if self.use_centerline:
            center = None
            if aux is not None:
                center = aux.get('centerline')
            center = self._match_batch(center, coarse_logits.shape[0])
            if center is None:
                center = torch.zeros_like(coarse_logits)
            parts.append(torch.sigmoid(_resize(center.detach(), size)))
        return torch.cat(parts, dim=1)

    def _disagreement_mask(
        self,
        all_head_logits: Optional[torch.Tensor],
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        heads = self._match_batch(
            all_head_logits, coarse_logits.shape[0])
        if heads is None or heads.shape[1] <= 1:
            return torch.zeros_like(coarse_logits)
        probs = torch.sigmoid(heads.detach())
        disagreement = probs.std(dim=1, keepdim=True)
        # Clamp, not sigmoid: zero disagreement must produce an exactly
        # zero gate.  sigmoid(-tau/scale) is ~0.38 and would apply a
        # correction even when all four heads agree.
        m = torch.clamp(
            (disagreement - self.disagreement_tau) /
            max(self.disagreement_scale, 1e-6), min=0.0, max=1.0)
        return m

    def forward(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        all_head_logits: Optional[torch.Tensor] = None,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return corrected logits for ``head_index``.

        ``coarse_logits`` is [B, 1, H, W]; ``all_head_logits`` is
        [B, 4, H, W] (used only for the disagreement channel).  ``aux`` is
        the decoder's auxiliary dict, if available.
        """
        if image is None or coarse_logits is None:
            return coarse_logits
        if self.res_scale <= 0:
            # Strict warm-up: do not build/apply the residual pathway at all.
            self.last_coarse_logits = None
            self.last_gate_logits = None
            self.last_gate_mask = None
            self.last_routing = None
            return coarse_logits
        if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
            raise ValueError(
                f'coarse_logits must be [B,1,H,W], got '
                f'{tuple(coarse_logits.shape)}')
        # Multi-prompt inference calls the decoder with B images and K
        # prompt replicas (K = B * repeats).  Repeat each image to the
        # mask batch; do not silently broadcast a wrong batch.
        if image.shape[0] != coarse_logits.shape[0]:
            if coarse_logits.shape[0] % image.shape[0] != 0:
                raise ValueError(
                    f'image batch {image.shape[0]} does not divide coarse '
                    f'batch {coarse_logits.shape[0]}')
            image = image.repeat_interleave(
                coarse_logits.shape[0] // image.shape[0], dim=0)
        orig_size = coarse_logits.shape[-2:]

        res_size = (self.out_size, self.out_size)
        if coarse_logits.shape[-2:] != res_size:
            coarse = _resize(coarse_logits, res_size)
        else:
            coarse = coarse_logits
        image_r = _resize(self._image_channels(image), res_size)

        if self.gate_mode == 'learned':
            gate_in = self._gate_input(
                image, coarse, all_head_logits, aux)
            m_logits, route = self.gate(gate_in)
            m_logits = _resize(m_logits, res_size)
            gate_mask = torch.sigmoid(m_logits)
        else:
            m_logits = None
            route = None
            gate_mask = _resize(
                self._disagreement_mask(all_head_logits, coarse),
                res_size)

        gate_mask = gate_mask * float(self.res_scale)

        if route is None:
            weights = torch.full(
                (coarse.shape[0], self.num_experts),
                1.0 / float(self.num_experts),
                device=coarse.device, dtype=coarse.dtype)
        else:
            weights = F.softmax(route, dim=-1)

        # Experts condition on the coarse mask as data; they must not
        # backprop into the coarse head through their input.
        prob = torch.sigmoid(coarse.detach())
        residuals = []
        for j, expert in enumerate(self.experts):
            degraded = degrade_coarse(prob, self.expert_kinds[j])
            residuals.append(expert(
                torch.cat([image_r, degraded], dim=1)))
        residual = torch.stack(residuals, dim=1)  # [B,J,1,H,W]
        mixed = (residual * weights.view(
            weights.shape[0], weights.shape[1], 1, 1, 1)).sum(dim=1)
        out = coarse + gate_mask * mixed
        if out.shape[-2:] != orig_size:
            out = _resize(out, orig_size)

        self.last_coarse_logits = coarse.detach()
        self.last_gate_logits = m_logits
        self.last_gate_mask = gate_mask.detach()
        self.last_routing = weights.detach()
        return out

    def gate_loss(self, target: torch.Tensor) -> torch.Tensor:
        """BCE + Dice on the topology-aware gate target."""
        if self.last_gate_logits is None or self.last_coarse_logits is None:
            if self.last_gate_mask is None:
                return torch.zeros((), device=target.device)
            return torch.zeros((), device=self.last_gate_mask.device)
        target_res = _resize_target_topology(
            target, self.last_coarse_logits.shape[-1])
        gt = build_topo_gate_target(self.last_coarse_logits, target_res)
        gt_gate = F.adaptive_max_pool2d(
            gt, self.last_gate_logits.shape[-2:])
        self.last_target_rate = float(gt_gate.mean().detach().item())
        pos = gt_gate.mean().detach().clamp_min(1e-4)
        pos_weight = ((1.0 - pos) / pos).clamp(min=0.5, max=50.0)
        bce = F.binary_cross_entropy_with_logits(
            self.last_gate_logits, gt_gate,
            pos_weight=pos_weight)
        dice = _soft_dice(torch.sigmoid(self.last_gate_logits), gt_gate)
        return bce + dice

    def sparsity_loss(self) -> torch.Tensor:
        # Use the live logits so the L1 penalty trains the gate.  The
        # detached last_gate_mask is kept only for logging.
        if self.last_gate_logits is not None:
            return torch.sigmoid(self.last_gate_logits).mean()
        if self.last_gate_mask is not None:
            return self.last_gate_mask.mean()
        return torch.zeros((), device=self.experts[0].out.weight.device)

    def extra_repr(self) -> str:
        return (
            f'num_experts={self.num_experts}, hidden={self.hidden}, '
            f'gate_mode={self.gate_mode}, gate_size={self.gate_size}, '
            f'out_size={self.out_size}, head_index={self.head_index}, '
            f'kinds={self.expert_kinds}')
