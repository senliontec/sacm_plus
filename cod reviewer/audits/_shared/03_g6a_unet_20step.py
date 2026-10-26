#!/usr/bin/env python3
"""G6a pilot: 20 optimizer steps with a shared U-Net state and batch.

This probe isolates the model/loss/optimizer path from the official training
recipe. Both sides use the same real 3-shot training batches, the same loaded
state_dict and a shared AdamW configuration; the official RMSprop recipe is a
G3 deviation and is not being claimed here. It is not G6b or G6c.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(ROOT / '3rd' / 'Pytorch-UNet'))

from baseline_zoo.adapters.registry import load_adapter  # noqa: E402
from baseline_zoo.adapters.unet import dice_loss as port_dice  # noqa: E402
from baseline_zoo.data import load_split_samples  # noqa: E402
from baseline_zoo.models.unet import UNet as PortUNet  # noqa: E402
from unet.unet_model import UNet as OffUNet  # noqa: E402
from utils.dice_score import dice_loss as off_dice  # noqa: E402


def tensor_sha(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def max_grad_diff(left: nn.Module, right: nn.Module):
    worst = 0.0
    missing = []
    for (name_l, p_l), (name_r, p_r) in zip(
            left.named_parameters(), right.named_parameters()):
        if name_l != name_r:
            missing.append((name_l, name_r))
            continue
        if p_l.grad is None and p_r.grad is None:
            continue
        if p_l.grad is None or p_r.grad is None:
            missing.append(name_l)
            continue
        worst = max(worst, float((p_l.grad - p_r.grad).abs().max()))
    return worst, missing


def max_param_diff(left: nn.Module, right: nn.Module) -> float:
    return max(
        float((p_l.detach() - p_r.detach()).abs().max())
        for p_l, p_r in zip(left.parameters(), right.parameters()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--input-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--out', type=Path,
                        default=REPO / 'cod reviewer' / 'audits' / 'unet'
                        / '03_g6a_20step.json')
    args = parser.parse_args()
    steps = max(int(args.steps), 20)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    adapter = load_adapter('unet')
    adapter.input_size = int(args.input_size)
    adapter.device = torch.device('cpu')
    samples = load_split_samples(REPO / 'data/sacm_3shot_seed0', 'train')
    batches = []
    for index in range(0, min(len(samples), 8), int(args.batch_size)):
        chunk = samples[index:index + int(args.batch_size)]
        if len(chunk) != int(args.batch_size):
            continue
        batches.append(adapter.prepare_train_batch(
            chunk, random.Random(1000 + index)))
    if not batches:
        raise SystemExit('no training batches could be staged')

    off = OffUNet(n_channels=3, n_classes=1)
    port = PortUNet(n_channels=3, n_classes=1)
    state = off.state_dict()
    load = port.load_state_dict(state, strict=True)
    off.to(device).train()
    port.to(device).train()
    opt_off = torch.optim.AdamW(
        off.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_port = torch.optim.AdamW(
        port.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    rows = []
    worst_loss = worst_grad = worst_param = worst_forward = 0.0
    missing = []
    for step in range(steps):
        images, masks = batches[step % len(batches)]
        images = images.to(device)
        masks = masks.to(device)
        opt_off.zero_grad(set_to_none=True)
        opt_port.zero_grad(set_to_none=True)
        logits_off = off(images)
        logits_port = port(images)
        loss_off = bce(logits_off, masks) + off_dice(
            torch.sigmoid(logits_off.squeeze(1)), masks.squeeze(1))
        loss_port = bce(logits_port, masks) + port_dice(logits_port, masks)
        loss_off.backward()
        loss_port.backward()
        grad_diff, missing_now = max_grad_diff(off, port)
        missing.extend(missing_now)
        opt_off.step()
        opt_port.step()
        param_diff = max_param_diff(off, port)
        forward_diff = float(
            (logits_off.detach() - logits_port.detach()).abs().max())
        loss_diff = abs(float(loss_off.detach()) - float(loss_port.detach()))
        lr_off = opt_off.param_groups[0]['lr']
        lr_port = opt_port.param_groups[0]['lr']
        worst_loss = max(worst_loss, loss_diff)
        worst_grad = max(worst_grad, grad_diff)
        worst_param = max(worst_param, param_diff)
        worst_forward = max(worst_forward, forward_diff)
        rows.append({
            'step': step + 1,
            'batch_index': step % len(batches),
            'loss_official': float(loss_off.detach()),
            'loss_port': float(loss_port.detach()),
            'loss_absdiff': loss_diff,
            'grad_maxdiff': grad_diff,
            'param_maxdiff': param_diff,
            'forward_maxdiff': forward_diff,
            'lr_official': lr_off,
            'lr_port': lr_port,
        })

    passed = (
        worst_loss <= args.atol
        and worst_grad <= args.atol
        and worst_param <= args.atol
        and worst_forward <= args.atol
        and not missing
        and all(row['lr_official'] == row['lr_port'] for row in rows)
    )
    payload = {
        'meta': {
            'gate': 'G6a',
            'purpose': 'U-Net 20-step training-path parity pilot',
            'device': str(device),
            'steps': steps,
            'batch_size': int(args.batch_size),
            'input_size': int(args.input_size),
            'batch_sha256': tensor_sha(
                [t for batch in batches for t in batch]),
            'optimizer': 'AdamW (shared, path probe only)',
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'loss': 'BCEWithLogits + official/port global Dice',
            'atol': args.atol,
            'note': 'Shared AdamW isolates path parity; official RMSprop is a '
                    'G3 recipe deviation. This is not G6c.',
        },
        'state': {
            'load_state_dict': load,
            'keys_equal': list(off.state_dict()) == list(port.state_dict()),
            'param_count_official': sum(p.numel() for p in off.parameters()),
            'param_count_port': sum(p.numel() for p in port.parameters()),
        },
        'max_loss_absdiff': worst_loss,
        'max_grad_maxdiff': worst_grad,
        'max_param_maxdiff': worst_param,
        'max_forward_maxdiff': worst_forward,
        'grad_missing': missing,
        'passed': bool(passed),
        'steps_detail': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({
        'out': str(args.out),
        'passed': payload['passed'],
        'max_loss_absdiff': worst_loss,
        'max_grad_maxdiff': worst_grad,
        'max_param_maxdiff': worst_param,
        'max_forward_maxdiff': worst_forward,
        'state': payload['state'],
    }, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
