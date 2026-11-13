"""Local fidelity probes for UNet/MaskVSC/JTFN/DSCNet/nnWNet/SegDINO.

Read-only w.r.t. the repo.  No downloads, no training beyond one loss/grad step.
"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/home/dev/mywork/project/segment')
PROJ = ROOT / 'sacm_plus'
THIRD = ROOT / '3rd'
sys.path.insert(0, str(PROJ / 'src'))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = {}


def state_check(off, port, name):
    so, sp = off.state_dict(), port.state_dict()
    keys_equal = list(so.keys()) == list(sp.keys())
    shapes_equal = keys_equal and all(so[k].shape == sp[k].shape for k in so)
    try:
        port.load_state_dict(so)
        load_ok = True
    except Exception:
        load_ok = False
    out = {
        'keys_equal': keys_equal,
        'shapes_equal': shapes_equal,
        'param_count_off': sum(p.numel() for p in off.parameters()),
        'param_count_port': sum(p.numel() for p in port.parameters()),
        'buffer_count_off': sum(b.numel() for b in off.buffers()),
        'buffer_count_port': sum(b.numel() for b in port.buffers()),
        'official_loads_port_state': load_ok,
    }
    OUT[name] = out
    return out


def load_both(off, port):
    port.load_state_dict(off.state_dict())
    off.eval(); port.eval()
    return off, port


def maxdiff(a, b):
    return float((a - b).abs().max().item())


def grad_diff(m1, m2):
    worst = 0.0
    missing = []
    both_none = 0
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if n1 != n2:
            missing.append((n1, n2))
            continue
        if p1.grad is None and p2.grad is None:
            both_none += 1
            continue
        if p1.grad is None or p2.grad is None:
            missing.append(n1)
            continue
        worst = max(worst, float((p1.grad - p2.grad).abs().max().item()))
    return worst, missing, both_none


def run_unet():
    from baseline_zoo.models.unet import UNet as PortUNet
    from baseline_zoo.adapters.unet import dice_loss as port_dice
    sys.path.insert(0, str(THIRD / 'Pytorch-UNet'))
    from unet.unet_model import UNet as OffUNet
    from utils.dice_score import dice_loss as off_dice
    torch.manual_seed(11)
    off = OffUNet(3, 1)
    port = PortUNet(3, 1)
    state_check(off, port, 'unet_state')
    load_both(off, port)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 3, 64, 64, generator=g)
    t = (torch.rand(2, 1, 64, 64, generator=g) > 0.75).float()
    with torch.no_grad():
        yo, yp = off(x), port(x)
    OUT['unet_forward_maxdiff'] = maxdiff(yo, yp)
    crit = nn.BCEWithLogitsLoss()
    off.zero_grad(); lo = off(x); lo = crit(lo, t) + off_dice(torch.sigmoid(lo.squeeze(1)), t.squeeze(1))
    lo.backward()
    port.zero_grad(); lp = port(x); lp = crit(lp, t) + port_dice(lp, t)
    lp.backward()
    OUT['unet_loss_off'] = float(lo.item()); OUT['unet_loss_port'] = float(lp.item())
    OUT['unet_loss_absdiff'] = abs(float(lo.item()) - float(lp.item()))
    gd, miss, both_none = grad_diff(off, port)
    OUT['unet_grad_maxdiff'] = gd; OUT['unet_grad_missing'] = miss; OUT['unet_grad_both_none'] = both_none


def run_maskvsc():
    sys.path.insert(0, str(THIRD / 'MaskVSC'))
    from model.UNet import UNet as OffUNet
    from lib.Loss import Connect_Loss as OffLoss
    sys.path.pop(0)
    from baseline_zoo.models.maskvsc import UNet as PortUNet, Connect_Loss as PortLoss
    torch.manual_seed(13)
    off = OffUNet(3, 1); port = PortUNet(3, 1)
    state_check(off, port, 'maskvsc_state')
    off = off.to(DEV); port = port.to(DEV)
    load_both(off, port)
    g = torch.Generator(device='cpu').manual_seed(9)
    x = torch.randn(2, 3, 64, 64, generator=g).to(DEV)
    t = (torch.rand(2, 1, 64, 64, generator=g) > 0.9).float().to(DEV)
    with torch.no_grad():
        yo, yp = off(x), port(x)
    OUT['maskvsc_forward_maxdiff'] = maxdiff(yo, yp)
    bce = nn.BCELoss()
    off.zero_grad(); lo = bce(off(x), t) + OffLoss()(off(x), t); lo.backward()
    port.zero_grad()
    lp = bce(port(x), t) + PortLoss(mode='release')(port(x), t)
    lp.backward()
    OUT['maskvsc_loss_off'] = float(lo.item()); OUT['maskvsc_loss_port'] = float(lp.item())
    OUT['maskvsc_loss_absdiff'] = abs(float(lo.item()) - float(lp.item()))
    gd, miss, both_none = grad_diff(off, port)
    OUT['maskvsc_grad_maxdiff'] = gd; OUT['maskvsc_grad_missing'] = miss; OUT['maskvsc_grad_both_none'] = both_none


def run_jtfn():
    sys.path.insert(0, str(THIRD / 'JTFN'))
    import models.jtfn as off_mod
    sys.path.pop(0)
    from baseline_zoo.models.jtfn import build_jtfn
    torch.manual_seed(17)
    off = off_mod.JTFN('base64', [True] * 5, [True] * 4, [True] * 4, 1, 3)
    port = build_jtfn('base64', 3)
    state_check(off, port, 'jtfn_state')
    load_both(off, port)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 3, 64, 64, generator=g)
    with torch.no_grad():
        oo, op = off({'img': x}), port({'img': x})
    diffs = []
    for i in range(3):
        for kind in ('output_mask', 'output_boundary'):
            for k, (a, b) in enumerate(zip(oo[f'step_{i}_{kind}'], op[f'step_{i}_{kind}'])):
                diffs.append(maxdiff(a, b))
    OUT['jtfn_forward_maxdiff_all_outputs'] = max(diffs)
    m = (torch.rand(1, 1, 64, 64, generator=g) > 0.85).float()
    b = (m - F.max_pool2d(m, 3, 1, 1)).abs()
    b = (b >= 0.5).float()
    bd = {'anno_mask': m, 'anno_boundary': b}
    off.zero_grad(); lo = off.compute_objective(off({'img': x}), bd)['total_loss']; lo.backward()
    port.zero_grad(); lp = port.compute_objective(port({'img': x}), bd)['total_loss']; lp.backward()
    OUT['jtfn_loss_off'] = float(lo.item()); OUT['jtfn_loss_port'] = float(lp.item())
    OUT['jtfn_loss_absdiff'] = abs(float(lo.item()) - float(lp.item()))
    gd, miss, both_none = grad_diff(off, port)
    OUT['jtfn_grad_maxdiff'] = gd; OUT['jtfn_grad_missing'] = miss; OUT['jtfn_grad_both_none'] = both_none


def run_dscnet():
    official_dir = THIRD / 'DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet'
    sys.path.insert(0, str(official_dir))
    from S3_DSCNet import DSCNet as OffDSCNet
    from S3_Loss import cross_loss as OffLoss
    sys.path.pop(0)
    from baseline_zoo.models.dscnet import build_dscnet, cross_loss as PortLoss
    torch.manual_seed(19)
    off = OffDSCNet(1, 1, 9, 1, True, DEV, 16, 1).to(DEV)
    port = build_dscnet(1, 1, 16, 9).to(DEV)
    state_check(off, port, 'dscnet_state')
    load_both(off, port)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 1, 64, 64, generator=g).to(DEV)
    t = (torch.rand(1, 1, 64, 64, generator=g) > 0.8).float().to(DEV)
    with torch.no_grad():
        yo, yp = off(x), port(x)
    OUT['dscnet_forward_maxdiff'] = maxdiff(yo, yp)
    off.zero_grad(); lo = OffLoss()(t, off(x)); lo.backward()
    port.zero_grad(); lp = PortLoss()(t, port(x)); lp.backward()
    OUT['dscnet_loss_off'] = float(lo.item()); OUT['dscnet_loss_port'] = float(lp.item())
    OUT['dscnet_loss_absdiff'] = abs(float(lo.item()) - float(lp.item()))
    gd, miss, both_none = grad_diff(off, port)
    OUT['dscnet_grad_maxdiff'] = gd; OUT['dscnet_grad_missing'] = miss; OUT['dscnet_grad_both_none'] = both_none


def run_nnwnet():
    nnroot = Path('/tmp/nnwnet_probe')
    for sub in ('raw', 'preprocessed', 'results'):
        (nnroot / sub).mkdir(parents=True, exist_ok=True)
    os.environ['nnUNet_raw'] = str(nnroot / 'raw')
    os.environ['nnUNet_preprocessed'] = str(nnroot / 'preprocessed')
    os.environ['nnUNet_results'] = str(nnroot / 'results')
    os.environ['nnUNet_compile'] = '0'
    spec = importlib.util.spec_from_file_location(
        'off_nnwnet', THIRD / 'nnWNet/nnUNetTrainer_WNet2D.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from baseline_zoo.models.nnwnet.trainer import WNet2D as PortWNet
    torch.manual_seed(23)
    off = mod.WNet2D(3, 2, True)
    port = PortWNet(3, 2, True)
    state_check(off, port, 'nnwnet_state')
    load_both(off, port)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 3, 64, 64, generator=g)
    with torch.no_grad():
        yo, yp = off(x), port(x)
    OUT['nnwnet_forward_num_outputs'] = [len(yo), len(yp)]
    OUT['nnwnet_forward_maxdiff_all_outputs'] = max(maxdiff(a, b) for a, b in zip(yo, yp))
    # loss/grad deliberately not probed: loss is inherited from nnU-Net trainer.
    OUT['nnwnet_official_file_imports'] = True


def run_segdino():
    from baseline_zoo.models.segdino import find_dinov3_checkpoint, build_segdino
    ckpt = find_dinov3_checkpoint('small', PROJ / 'checkpoints')
    OUT['segdino_local_ckpt'] = str(ckpt) if ckpt else None
    if ckpt:
        OUT['segdino_ckpt_bytes'] = int(Path(ckpt).stat().st_size)
        try:
            model = build_segdino('small', str(ckpt), False)
            OUT['segdino_port_build_with_ckpt'] = True
            OUT['segdino_port_param_count'] = sum(p.numel() for p in model.parameters())
            bb = model.backbone
            OUT['segdino_timm_has_get_intermediate_layers'] = hasattr(bb, 'get_intermediate_layers')
            OUT['segdino_timm_embed_dim'] = int(getattr(bb, 'embed_dim', -1))
        except BaseException as exc:
            OUT['segdino_port_build_with_ckpt'] = f'FAIL: {type(exc).__name__}: {exc}'
    # DPT head structure parity with a dummy backbone (official hub repo absent).
    class Dummy(nn.Module):
        embed_dim = 384
    sys.path.insert(0, str(THIRD / 'segdino'))
    import dpt as off_dpt
    sys.path.pop(0)
    from baseline_zoo.models.segdino.dpt import DPT as PortDPT
    off = off_dpt.DPT(encoder_size='small', nclass=1, backbone=Dummy())
    port = PortDPT(encoder_size='small', nclass=1, backbone=Dummy())
    so, sp = off.state_dict(), port.state_dict()
    OUT['segdino_head_keys_equal'] = list(so.keys()) == list(sp.keys())
    OUT['segdino_head_shapes_equal'] = OUT['segdino_head_keys_equal'] and all(
        so[k].shape == sp[k].shape for k in so)
    OUT['segdino_head_param_count'] = sum(p.numel() for p in off.head.parameters())


def run_bcunet():
    spec = importlib.util.spec_from_file_location(
        'bcu_unet_parts', THIRD / 'BCU-Net/code/core/unet_parts.py')
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    try:
        up.OutConv(32, 2)
        OUT['bcunet_outconv_construct'] = 'OK'
    except BaseException as exc:
        OUT['bcunet_outconv_construct'] = f'{type(exc).__name__}: {exc}'
    OUT['bcunet_outconv_first_line'] = Path(
        THIRD / 'BCU-Net/code/core/unet_parts.py').read_text().splitlines()[98].strip()
    OUT['bcunet_official_model_construct'] = 'NOT ATTEMPTED: pretrained backbone would download; OutConv already fails'


def main():
    torch.set_grad_enabled(True)
    for name, fn in [('unet', run_unet), ('maskvsc', run_maskvsc),
                     ('jtfn', run_jtfn), ('dscnet', run_dscnet),
                     ('nnwnet', run_nnwnet), ('segdino', run_segdino),
                     ('bcunet', run_bcunet)]:
        try:
            fn()
        except BaseException as exc:
            OUT[name + '_ERROR'] = f'{type(exc).__name__}: {exc}'
            OUT[name + '_TRACEBACK'] = traceback.format_exc()[-2500:]
    OUT['device'] = str(DEV)
    OUT['torch'] = torch.__version__
    print(json.dumps(OUT, indent=2))


if __name__ == '__main__':
    main()
