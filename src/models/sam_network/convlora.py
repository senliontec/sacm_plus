"""Conv-LoRA residual branch for SAM's ViT qkv projection.

The branch follows the official AutoGluon Conv-LoRA computation::

    y = W0 x + B * (A x + MoE_conv(A x)) * alpha / rank

with a noisy top-1 gate, full 3x3 convolutional experts at multiple
upsampling ratios, and the load-balancing loss returned by the gate.  The
pretrained qkv is kept intact and frozen; the branch is added as a residual
so base SAM checkpoints still load without key remapping.

The branch also accepts SAM's window-partitioned attention input, i.e.
``[B * n_windows, window, window, C]``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MoEGate(nn.Module):
    """Noisy top-k gate from the official Conv-LoRA implementation."""

    def __init__(self, d: int, num_experts: int = 4, top_k: int = 1,
                 noisy_gating: bool = True) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.noisy_gating = bool(noisy_gating)
        self.w_gate = nn.Parameter(
            torch.zeros(d, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(
            torch.zeros(d, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))
        if self.top_k > self.num_experts:
            raise ValueError('top_k must be <= num_experts')

    def forward(self, feats: torch.Tensor):
        batch_size = feats.shape[0]
        feats_s = self.gap(feats).view(batch_size, -1)
        clean_logits = feats_s @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_s @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if (self.noisy_gating and self.top_k < self.num_experts
                and self.training):
            load = self._prob_in_top_k(
                clean_logits, logits, noise_stddev, top_logits).sum(0)
        else:
            load = (gates > 0).sum(0)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        return gates, loss * 1e-2

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev,
                       noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k)
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf(
            (clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)


class SparseDispatcher:
    """Dispatch/combine utility from Shazeer et al. / official Conv-LoRA."""

    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[
            index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(
            gates_exp, 1, self._expert_index)

    def dispatch(self, inp: torch.Tensor):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates: bool = True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(
                self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            expert_out[-1].size(2),
            expert_out[-1].size(3),
            requires_grad=True,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()


class ConvLoRABranch(nn.Module):
    """Official-style Conv-LoRA residual branch for qkv.

    Input/output shape is ``[B, H, W, 3 * dim]`` (normal qkv output), with
    window-attention inputs flattened into the leading batch dimension.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 4,
        alpha: float = 32.0,
        num_experts: int = 8,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError('Conv-LoRA rank must be positive')
        if num_experts <= 0:
            raise ValueError('Conv-LoRA needs at least one expert')
        self.dim = int(dim)
        self.rank = int(rank)
        self.num_experts = int(num_experts)
        self.scaling = float(alpha) / float(rank)
        self.last_moe_loss: torch.Tensor | None = None

        self.down = nn.Linear(dim, rank, bias=False)
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            conv = nn.Conv2d(rank, rank, 3, padding=1, bias=True)
            nn.init.zeros_(conv.bias)
            self.experts.append(nn.Sequential(conv, nn.GELU()))
        self.upsample_ratios = list(range(1, self.num_experts + 1))
        self.gate = MoEGate(rank, self.num_experts, top_k=1)
        self.up = nn.Linear(rank, dim * 3, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 3
        if squeeze:
            b, n, c = x.shape
            h = w = int(round(n ** 0.5))
            if h * w != n:
                raise ValueError(
                    f'cannot reshape {n} tokens into a square feature map')
            x = x.reshape(b, h, w, c)
        else:
            b, h, w, _ = x.shape

        low = self.down(x)                         # [B,H,W,r]
        feat = low.permute(0, 3, 1, 2).contiguous()  # [B,r,H,W]
        gates, moe_loss = self.gate(feat)
        self.last_moe_loss = moe_loss

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(feat)
        expert_outputs = []
        for index, expert in enumerate(self.experts):
            current = expert_inputs[index]
            if current.numel() == 0:
                continue
            ratio = self.upsample_ratios[index]
            if ratio != 1:
                current = F.interpolate(
                    current, scale_factor=ratio, mode='bicubic')
            current = expert(current)
            if ratio != 1:
                current = F.interpolate(
                    current, size=(h, w), mode='bicubic')
            expert_outputs.append(current)
        combined = dispatcher.combine(
            expert_outputs, multiply_by_gates=False)

        # Official Conv-LoRA keeps the low-rank identity path and adds the
        # convolutional experts on top of it.
        lora_res = feat + combined
        out = self.up(
            lora_res.permute(0, 2, 3, 1).contiguous()) * self.scaling
        if squeeze:
            out = out.reshape(b, n, -1)
        return out
