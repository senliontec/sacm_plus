"""Conv-LoRA layers, adapted from the official AutoGluon implementation.

Source: ``3rd/autogluon/multimodal/src/autogluon/multimodal/models/
adaptation_layers.py`` (Apache-2.0 AutoGluon; the LoRA base is MIT from
Microsoft's loralib).  Only the classes needed by this port are copied.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float,
                 merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class ConvLoRALinear(nn.Linear, LoRALayer):
    """Linear layer plus Conv-LoRA's gated multi-scale convolutional experts.

    ``qkv`` is frozen and the low-rank branch is added alongside it.  The
    forward returns ``(output, moe_loss)`` exactly like the official layer;
    the caller aggregates ``moe_loss`` over all injected layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        conv_lora_expert_num: int | None = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.last_moe_loss: torch.Tensor | None = None

        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

            topk = 1
            self.lora_moe_gating = MoEGate(
                M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(
                range(1, conv_lora_expert_num + 1))
            for _ in self.upsample_ratios:
                expert = nn.Conv2d(
                    in_channels=r, out_channels=r, kernel_size=3,
                    stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(
                    nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        moe_loss = None
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                batch, length, channels = lora_res.size()
                height = width = int(math.sqrt(length))
                lora_res = lora_res.reshape(batch, height, width, channels)
            else:
                height, width = lora_res.size()[1:3]

            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for index in range(self.num_experts):
                if len(expert_inputs[index]) == 0:
                    continue
                ratio = self.upsample_ratios[index]
                current = expert_inputs[index]
                if ratio != 1:
                    current = F.interpolate(
                        current, scale_factor=ratio, mode='bicubic')
                current = self.lora_moe_experts[index](current)
                if ratio != 1:
                    current = F.interpolate(
                        current, size=(int(height), int(width)),
                        mode='bicubic')
                expert_outputs.append(current)

            combined = dispatcher.combine(
                expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + combined

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(batch, length, channels)
            result = result + (lora_res @ self.lora_B.T) * self.scaling

        self.last_moe_loss = moe_loss
        return result, moe_loss


class MoEGate(nn.Module):
    """Noisy top-k gate from the official Conv-LoRA implementation."""

    def __init__(self, d, M=4, K=1, noisy_gating=True):
        super().__init__()
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.noisy_gating = noisy_gating
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        batch_size = feats.shape[0]
        feats_s = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_s @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_s @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(
                clean_logits, noisy_logits, noise_stddev,
                top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = loss * loss_coef
        return gates, loss

    @staticmethod
    def _gates_to_load(gates):
        return (gates > 0).sum(0)

    @staticmethod
    def cv_squared(x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev,
                       noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k)
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)
                                / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)
                                 / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)


class SparseDispatcher:
    """Dispatch/combine utility copied from Shazeer et al. / official Conv-LoRA."""

    def __init__(self, num_experts, gates):
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

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(
                self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
