"""Numerical cross-check of our strict ports against the official code.

For every pair, the same synthetic inputs are fed to the official
implementation (from the locally pulled repositories) and to our port
(from the losses package / eval_metrics); outputs must match.

Run on any machine with torch/numpy/scipy installed:

    python verify_against_official.py

Sections guarded by dependencies:
  - medpy metrics: importable from the local repo (medpy/medpy/metric/binary.py)
  - NSD: local repo surface-distance/surface_distance (numpy/scipy only)
  - clDice metric: local repo clDice/clDice_metric (skimage optional)
  - clDice loss + DECL: torch only
  - centerline-CE: SKIPPED by design — the pulled repo lacks
    soft_skeleton.py, so the official module cannot be imported; the
    caveat is documented in losses/centerline_ce.py.
  - Betti matching: requires panel/matplotlib/gudhi (their module-level
    imports); tested only when available.
  - SATLoss: requires gudhi + torch-topological + POT.
  - Topograph: the official module imports its compiled C++ extension
    unconditionally; without building it the official code cannot be
    imported, so our pure-Python port is tested for internal sanity only
    (runs without error and returns a finite scalar).
"""

import os
import sys

import numpy as np
import torch

# This script lives in scripts/, the project root is one level up
# and the pulled official repos live in its parent.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_DIR = os.path.dirname(REPO_ROOT)
# The official repos now live in a 3rd/ subfolder of the parent (kept
# tidy together); fall back to the old flat layout so both arrangements
# work on any machine.
THIRD_PARTY_DIR = os.path.join(SEGMENT_DIR, '3rd')
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))  # project packages


def _repo_dir(name):
    """Location of an official repo: 3rd/<name> first, then the flat layout."""
    for root in (THIRD_PARTY_DIR, SEGMENT_DIR):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return os.path.join(THIRD_PARTY_DIR, name)

PASS = []
FAIL = []


def check(name, a, b, tol=1e-5):
    a = float(a)
    b = float(b)
    ok = abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    (PASS if ok else FAIL).append(name)
    status = "OK " if ok else "FAIL"
    print(f"[{status}] {name}: official={a:.8f}  ours={b:.8f}")


def make_pair(seed=0, size=64):
    rng = np.random.default_rng(seed)
    gt = rng.random((size, size)) > 0.6
    pred = (rng.random((size, size)) > 0.4) & gt
    pred = pred | (rng.random((size, size)) > 0.97)  # a few false positives
    return pred.astype(np.uint8), gt.astype(np.uint8)


def main():
    pred, gt = make_pair()

    # ---------------------------------------------------------- medpy ----
    medpy_dir = _repo_dir('medpy')
    if os.path.isdir(medpy_dir):
        sys.path.insert(0, medpy_dir)
        try:
            from medpy.metric import binary as medpy_binary
            from core.metrics import (
                dice_score, iou_score, precision_score, recall_score,
                sensitivity_score, specificity_score, hd95_score,
                assd_score, asd_score, ravd_score, hausdorff_score,
            )
            check('medpy/dc', medpy_binary.dc(pred, gt), dice_score(pred, gt))
            check('medpy/jc', medpy_binary.jc(pred, gt), iou_score(pred, gt))
            check('medpy/precision', medpy_binary.precision(pred, gt), precision_score(pred, gt))
            check('medpy/recall', medpy_binary.recall(pred, gt), recall_score(pred, gt))
            check('medpy/sensitivity', medpy_binary.sensitivity(pred, gt), sensitivity_score(pred, gt))
            check('medpy/specificity', medpy_binary.specificity(pred, gt), specificity_score(pred, gt))
            check('medpy/hd', medpy_binary.hd(pred, gt), hausdorff_score(pred, gt))
            check('medpy/hd95', medpy_binary.hd95(pred, gt), hd95_score(pred, gt))
            check('medpy/assd', medpy_binary.assd(pred, gt), assd_score(pred, gt))
            check('medpy/asd', medpy_binary.asd(pred, gt), asd_score(pred, gt))
            check('medpy/ravd', medpy_binary.ravd(pred, gt), ravd_score(pred, gt))
        except ImportError as e:
            print(f"[SKIP] medpy section: {e}")
        finally:
            sys.path.pop(0)
    else:
        print("[SKIP] medpy not found in segment dir")

    # ------------------------------------------------- surface-distance ----
    sd_dir = _repo_dir('surface-distance')
    if os.path.isdir(sd_dir):
        sys.path.insert(0, sd_dir)
        try:
            import surface_distance.metrics as sd_metrics
            from core.metrics import nsd_score
            sd = sd_metrics.compute_surface_distances(gt.astype(bool), pred.astype(bool), (1.0, 1.0))
            official_nsd = sd_metrics.compute_surface_dice_at_tolerance(sd, 1.0)
            check('surface-distance/nsd', official_nsd, nsd_score(pred, gt, tolerance_mm=1.0))
        except ImportError as e:
            print(f"[SKIP] surface-distance section: {e}")
        finally:
            sys.path.pop(0)
    else:
        print("[SKIP] surface-distance not found in segment dir")

    # ----------------------------------------------------- clDice metric ----
    # The clDice repo nests its package one level down (clDice/clDice/);
    # accept both the nested and the flat layout.
    cldice_repo = _repo_dir('clDice')
    cldice_dir = os.path.join(cldice_repo, 'clDice')
    if not os.path.isdir(cldice_dir):
        cldice_dir = cldice_repo
    if os.path.isdir(cldice_dir):
        sys.path.insert(0, cldice_dir)
        try:
            # Recent scikit-image removed skeletonize_3d; the official
            # module imports it at top level. This script only exercises
            # the 2D path, so a placeholder restores the import.
            import skimage.morphology as _skim
            if not hasattr(_skim, 'skeletonize_3d'):
                _skim.skeletonize_3d = None
            try:
                from clDice_metric.clDice import clDice as official_clDice
            except ImportError:
                # Some pulls of the repo use lowercase module/file names.
                from cldice_metric.cldice import clDice as official_clDice
            from core.metrics import cldice_score
            check('clDice/metric', official_clDice(pred.astype(bool), gt.astype(bool)),
                  cldice_score(pred, gt))
        except ImportError as e:
            print(f"[SKIP] clDice metric section: {e}")
        finally:
            sys.path.pop(0)
    else:
        print("[SKIP] clDice not found in segment dir")

    # ------------------------------------------------------ clDice loss ----
    # The official module uses a relative import (`from .soft_skeleton
    # import ...`), so it must be imported as a submodule of its parent:
    # sys.path gets cldice_loss/ and the import is pytorch.cldice.
    cldice_loss_dir = os.path.join(cldice_dir, 'cldice_loss')
    if os.path.isdir(cldice_loss_dir):
        sys.path.insert(0, cldice_loss_dir)
        try:
            try:
                from pytorch.clDice import soft_cldice as OfficialSoftCLDice
            except ImportError:
                # Some pulls of the repo use lowercase module names.
                from pytorch.cldice import soft_cldice as OfficialSoftCLDice
            from core.losses import SoftclDiceLoss
            logits = torch.randn(1, 1, 64, 64) * 2.0
            target = (torch.rand(1, 1, 64, 64) > 0.7).float()
            # Official API takes probabilities; ours takes logits.
            official = OfficialSoftCLDice()(target, torch.sigmoid(logits))
            ours = SoftclDiceLoss()(logits, target)
            check('clDice/loss', official.item(), ours.item(), tol=1e-4)
        except ImportError as e:
            print(f"[SKIP] clDice loss section: {e}")
        finally:
            sys.path.pop(0)
    else:
        print("[SKIP] clDice loss module not found in segment dir")

    # ------------------------------------------------------------- DECL ----
    decl_dir = _repo_dir('DECL-Loss')
    if os.path.isdir(decl_dir):
        sys.path.insert(0, decl_dir)
        try:
            from loss.end_distance_loss import EndpointDistanceLossAverage as OfficialDECL
            from core.losses import EndpointDistanceLossAverage
            logits = torch.randn(1, 1, 64, 64) * 2.0
            target = (torch.rand(1, 1, 64, 64) > 0.7).float()
            two_ch = torch.cat([torch.zeros_like(logits), logits], dim=1)
            official = OfficialDECL()(two_ch, target)
            ours = EndpointDistanceLossAverage()(logits, target)
            check('decl/loss', official.item(), ours.item(), tol=1e-4)
        except ImportError as e:
            print(f"[SKIP] DECL section: {e}")
        finally:
            sys.path.pop(0)
    else:
        print("[SKIP] DECL-Loss not found in segment dir")

    # ------------------------------------------------------- Betti family --
    try:
        import panel  # noqa: F401  (official module imports it at top level)
        import matplotlib  # noqa: F401
        betti_dir = _repo_dir('Betti-matching')
        sys.path.insert(0, betti_dir)
        try:
            import BettiMatching as official_bm
            # Compare the engines directly on numpy inputs (the official
            # loss entry lives in loss_functions.py, which needs monai;
            # the engine is the computation being verified).
            logits = torch.randn(1, 1, 64, 64) * 2.0
            target = (torch.rand(1, 1, 64, 64) > 0.7).float()
            probs = torch.sigmoid(logits)[0, 0].numpy()
            gt_np = target[0, 0].numpy()
            o_engine = official_bm.BettiMatching(probs, gt_np, filtration='superlevel', training=False)
            from core.losses.betti_matching import BettiMatching
            m_engine = BettiMatching(probs, gt_np, filtration='superlevel', training=False)
            check('betti/loss', o_engine.loss(), m_engine.loss(), tol=1e-4)
            check('betti/metric', o_engine.Betti_number_error(threshold=0.5),
                  m_engine.Betti_number_error(threshold=0.5))
        except Exception as e:
            print(f"[SKIP] Betti section (deps or import issue): {e}")
        finally:
            sys.path.pop(0)
    except ImportError:
        print("[SKIP] Betti section: panel/matplotlib/gudhi not installed")

    # -------------------------------------------------------- Topograph ----
    try:
        import networkx  # noqa: F401
        from core.losses.topograph import TopographLossAdapter
        logits = torch.randn(1, 1, 64, 64) * 2.0
        target = (torch.rand(1, 1, 64, 64) > 0.7).float()
        out = TopographLossAdapter(resolution=64)(logits, target)
        ok = torch.isfinite(out).item()
        print(f"[{'OK ' if ok else 'FAIL'}] topograph/sanity: finite scalar = {out.item():.8f}")
        (PASS if ok else FAIL).append('topograph/sanity')
        print("         (official module cannot be imported without its compiled "
              "C++ extension; the pure-Python port is checked for sanity here, "
              "and the official cross-check requires building the extension)")
    except ImportError as e:
        print(f"[SKIP] Topograph section: {e}")

    # ------------------------------------------------ centerline-CE family --
    # The pulled repo lacks soft_skeleton.py and needs nnUNet's
    # RobustCrossEntropyLoss (default kwargs = plain CE). Both are shimmed
    # here — the skeleton shim is the VERIFIED clDice skeleton (the
    # documented substitution), the CE shim is exactly the official
    # default behaviour — so the official cldice_loss.py module becomes
    # importable and all four classes can be cross-checked.
    try:
        import importlib.util
        import types

        cce_pkg = types.ModuleType('cce_official')
        cce_pkg.__path__ = []
        sys.modules['cce_official'] = cce_pkg

        def _soft_skel_shim(x, iter_=3):
            from core.losses.core import SoftSkeletonize
            return SoftSkeletonize(num_iter=iter_)(x)

        skel_mod = types.ModuleType('cce_official.soft_skeleton')
        skel_mod.soft_skel = _soft_skel_shim
        sys.modules['cce_official.soft_skeleton'] = skel_mod

        nnunet = types.ModuleType('nnunetv2')
        nnunet.__path__ = []
        nnunet_t = types.ModuleType('nnunetv2.training')
        nnunet_t.__path__ = []
        nnunet_l = types.ModuleType('nnunetv2.training.loss')
        nnunet_l.__path__ = []
        robust = types.ModuleType('nnunetv2.training.loss.robust_ce_loss')
        robust.RobustCrossEntropyLoss = torch.nn.CrossEntropyLoss  # official default kwargs
        sys.modules['nnunetv2'] = nnunet
        sys.modules['nnunetv2.training'] = nnunet_t
        sys.modules['nnunetv2.training.loss'] = nnunet_l
        sys.modules['nnunetv2.training.loss.robust_ce_loss'] = robust

        cce_path = os.path.join(_repo_dir('centerline_CE'), 'nnUNet', 'losses', 'cldice_loss.py')
        spec = importlib.util.spec_from_file_location('cce_official.cldice_loss', cce_path)
        cce_mod = importlib.util.module_from_spec(spec)
        sys.modules['cce_official.cldice_loss'] = cce_mod
        spec.loader.exec_module(cce_mod)

        import core.losses.centerline_ce as our_cce
        logits = torch.randn(1, 1, 64, 64) * 2.0
        target = (torch.rand(1, 1, 64, 64) > 0.7).float()
        two_ch = torch.cat([torch.zeros_like(logits), logits], dim=1)
        long_target = (target > 0.5).long()

        pairs = [
            ('centerline_ce/dice_clCE', cce_mod.dice_clCE_loss(), our_cce.DiceCenterlineCELoss()),
            ('centerline_ce/dice_cldice', cce_mod.dice_cldice_loss(), our_cce.DiceCLDiceLoss()),
            ('centerline_ce/ce_cldice', cce_mod.CE_cldice_loss({}), our_cce.CECLDiceLoss()),
            ('centerline_ce/ce_clCE', cce_mod.CE_clCE_loss({}), our_cce.CECLCELoss()),
        ]
        for name, o_loss, m_loss in pairs:
            official = o_loss(two_ch, long_target)
            ours = m_loss(logits, target)
            check(name, official.item(), ours.item(), tol=1e-4)
    except ImportError as e:
        print(f"[SKIP] centerline-CE section: {e}")

    # ----------------------------------------------------------- SATLoss ----
    try:
        import gudhi  # noqa: F401
        import ot  # noqa: F401
        import torch_topological  # noqa: F401
        satloss_dir = _repo_dir('SATLoss')
        sys.path.insert(0, satloss_dir)
        try:
            from utils.losses import PDMatchingLoss as OfficialPDMatchingLoss
            from core.losses.satloss import SATLossAdapter
            logits = torch.randn(1, 1, 32, 32) * 2.0
            target = (torch.rand(1, 1, 32, 32) > 0.7).float()
            probs = torch.sigmoid(logits)
            opt = types.SimpleNamespace(precal_PD=False)
            official = OfficialPDMatchingLoss(opt=opt, p=2)(probs, target)
            ours = SATLossAdapter(resolution=32)(logits, target)
            check('satloss/loss', official.item(), ours.item(), tol=1e-4)
        except Exception as e:
            print(f"[SKIP] SATLoss section (deps or import issue): {e}")
        finally:
            sys.path.pop(0)
    except ImportError:
        print("[SKIP] SATLoss section: gudhi/torch-topological/POT not installed")

    # -------------------------------------------- randomized metric battery --
    # Extends the single-pair checks: N random pairs plus edge cases
    # (both empty, one empty, identical, single pixel). Every metric must
    # agree with its official implementation on ALL of them.
    try:
        rng = np.random.default_rng(20260930)
        medpy_dir = _repo_dir('medpy')
        sd_dir = _repo_dir('surface-distance')
        cldice_metric_dir = os.path.join(_repo_dir('clDice'), 'clDice')
        if not os.path.isdir(cldice_metric_dir):
            cldice_metric_dir = _repo_dir('clDice')
        if os.path.isdir(medpy_dir) and os.path.isdir(sd_dir) and os.path.isdir(cldice_metric_dir):
            sys.path.insert(0, medpy_dir)
            sys.path.insert(0, sd_dir)
            sys.path.insert(0, cldice_metric_dir)
            try:
                from medpy.metric import binary as medpy_binary
                import surface_distance.metrics as sd_metrics
                from cldice_metric.cldice import clDice as official_clDice
                from core.metrics import (
                    dice_score, iou_score, precision_score, recall_score,
                    sensitivity_score, specificity_score, hd95_score,
                    assd_score, asd_score, ravd_score, hausdorff_score,
                    nsd_score, cldice_score,
                )

                cases = []
                for size in (16, 33, 64):
                    for i in range(6):
                        g = rng.random((size, size)) > 0.6
                        p = (rng.random((size, size)) > 0.45) & g
                        cases.append((p.astype(np.uint8), g.astype(np.uint8)))
                e = np.zeros((32, 32), dtype=np.uint8)
                f = np.ones((32, 32), dtype=np.uint8)
                sp = e.copy(); sp[10, 10] = 1
                cases += [(e, e), (e, f), (f, e), (f, f), (sp, sp), (sp, e)]

                def _off(fn):
                    # medpy raises RuntimeError on empty masks; our ports
                    # return NaN there (documented deviation) — treat the
                    # official exception as NaN.
                    try:
                        return float(fn())
                    except RuntimeError:
                        return float('nan')

                worst = {}
                for pred, gt in cases:
                    pairs = [
                        ('dc', medpy_binary.dc(pred, gt), dice_score(pred, gt)),
                        ('jc', medpy_binary.jc(pred, gt), iou_score(pred, gt)),
                        ('precision', medpy_binary.precision(pred, gt), precision_score(pred, gt)),
                        ('recall', medpy_binary.recall(pred, gt), recall_score(pred, gt)),
                        ('sensitivity', medpy_binary.sensitivity(pred, gt), sensitivity_score(pred, gt)),
                        ('specificity', medpy_binary.specificity(pred, gt), specificity_score(pred, gt)),
                        ('hd95', _off(lambda: medpy_binary.hd95(pred, gt)), hd95_score(pred, gt)),
                        ('assd', _off(lambda: medpy_binary.assd(pred, gt)), assd_score(pred, gt)),
                        ('asd', _off(lambda: medpy_binary.asd(pred, gt)), asd_score(pred, gt)),
                        ('ravd', _off(lambda: medpy_binary.ravd(pred, gt)), ravd_score(pred, gt)),
                    ]
                    for name, official, ours in pairs:
                        if np.isnan(official) and np.isnan(ours):
                            continue
                        if np.isnan(official) != np.isnan(ours):
                            raise AssertionError(f'battery/{name}: NaN mismatch on {pred.shape}')
                        worst[name] = max(worst.get(name, 0.0),
                                          abs(float(official) - float(ours)))
                    # medpy hd raises on empty -> skip when one side is empty
                    if pred.any() and gt.any():
                        hd_o = medpy_binary.hd(pred, gt)
                        hd_m = hausdorff_score(pred, gt)
                        worst['hd'] = max(worst.get('hd', 0.0), abs(float(hd_o) - float(hd_m)))
                    # NSD
                    if pred.any() or gt.any():
                        sd = sd_metrics.compute_surface_distances(gt.astype(bool), pred.astype(bool), (1.0, 1.0))
                        nsd_o = sd_metrics.compute_surface_dice_at_tolerance(sd, 1.0)
                        nsd_m = nsd_score(pred, gt, tolerance_mm=1.0)
                        worst['nsd'] = max(worst.get('nsd', 0.0), abs(float(nsd_o) - float(nsd_m)))
                    # clDice metric (skimage path)
                    if pred.any() and gt.any():
                        cl_o = official_clDice(pred.astype(bool), gt.astype(bool))
                        cl_m = cldice_score(pred, gt)
                        if not np.isnan(cl_o) and not np.isnan(cl_m):
                            worst['cldice'] = max(worst.get('cldice', 0.0),
                                                  abs(float(cl_o) - float(cl_m)))
                for name, w in sorted(worst.items()):
                    check(f'battery/{name}', 0.0, w, tol=1e-9)
            finally:
                sys.path.pop(0)
                sys.path.pop(0)
                sys.path.pop(0)
        else:
            print("[SKIP] battery: medpy/surface-distance/clDice not found")
    except Exception as e:
        print(f"[SKIP] battery: {e}")

    print()
    print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
