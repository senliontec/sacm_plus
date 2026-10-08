"""Euler-refine loss (topo_loss_2d) — strict vendor of the official repo.

Official source: Universal-Topology-Refinement, the official PyTorch
implementation of "Universal Topology Refinement for Medical Image
Segmentation with Polynomial Feature Synthesis" (Liu Li, Hanchun Wang,
Matthew Baugh, Qiang Ma, Weitong Zhang, Cheng Ouyang, Daniel Rueckert,
Bernhard Kainz; MICCAI 2024). Locally pulled at
3rd/Universal-Topology-Refinement.

Vendored (bodies verbatim, only irrelevant imports / dead debug helpers /
unused instrumentation removed — the complete deviation list is below):

  loss.py:754-1218        topo_loss_2d, gudhi branch only
      compute_dgm_force_new   loss.py:760
      pre_process             loss.py:802
      check_point_exist       loss.py:822
      reidx_f_gudhi           loss.py:881
      get_info_gudhi          loss.py:922
      get_topo_loss           loss.py:966
      forward                 loss.py:1105 (2D path, cgm_dims=None)
  polynomial.py:9-172     get_background_index_, generate_multi_gauss_mask,
                          random_mask
  polynomial.py:174-329   Polynomial
  utils/utils.py:32-40    norm_ten (pulled into polynomial.py by its
                          `from utils.utils import *`)

The official 2D entry point is process_batch.py:405
(`Topo_loss = topo_loss_2d(package='gudhi')`), fed with a single-channel
`pred_softmax = Sigmoid()(pred)` (process_batch.py:284) and the foreground
`labelmap_onehot`. The loss reads the persistence pairs of the birth/death
points of the predicted map (Euler-characteristic / cubical homology,
`get_info_gudhi`), decides which topological features must be fixed or
removed against those of the ground truth (`compute_dgm_force_new`) and
returns a weighted L2 map pulling the probabilities at those critical points
towards the reference values. `Polynomial` / `random_mask` are the paper's
polynomial feature synthesis (README: "Polynomial perturbation"), i.e. the
synthetic-mask generator of the same refinement pipeline.

Dependencies: gudhi only (the only import the vendored code needs). The
classes are always defined; registration ('euler_refine') is skipped when
gudhi is missing, so the package imports cleanly.

API bridge: the official loss consumes PROBABILITY maps [B, 1, H, W]. Our
adapter builds the project's 2-channel bridge `cat([zeros, logits])` —
softmax(...)[:, 1] == sigmoid(logits) EXACTLY — and feeds the foreground
channel (== sigmoid(logits), the official `pred_softmax`) to the official
computation, which indexes channel 0. `resolution` (256 by default) mirrors
the official `topo_size=256` tiling: gudhi cubical persistence at 1024 per
training iteration is expensive, so inputs of a different spatial size are
bilinearly resized first (same policy as losses/satloss.py). The official
computation itself is unchanged.

Deviation list (complete)
-------------------------
1. `import cripser` (loss.py:9, unconditional upstream) is NOT vendored and
   the `package='cripser'` branch of `get_topo_loss` is NOT ported. Our
   `topo_loss_2d` asserts `package == 'gudhi'` and is always constructed
   with the default `package='gudhi'` (the cripser 4-connectivity path also
   cannot run upstream: it leaves `info_gt` undefined — NameError — while
   the gudhi branch is the one documented in
   process_batch.py:405's comment).
2. `reidx_f_gudhi`: the official 2D-branch assertion
   `assert (reidx_0 == re_idx).all()` compares a shape-(2,) array with the
   shape-(3,) scratch array `re_idx`, so it raises ValueError on *every* 2D
   call (verified on numpy 2.2.6) — the published 2D gudhi path cannot run
   as-is. We compare against `re_idx[:len(ori_shape)]`, which is the
   assertion's evident intent (C-order index check); the rest of the method
   (and its first assertion) is verbatim.
3. The `cgm_dims` (multiclass) branch of `forward` (loss.py:1107-1181) is
   NOT ported: it spawns `multiprocessing` pools, needs the debug helpers
   `get_arg_map_range`/`save_nii` that write to hard-coded
   `/vol/biomedic3/...` paths, and it calls
   `self.get_topo_loss(input, batch, cgm_idx)` with 3 arguments although the
   method takes 4 (latent upstream bug). The official 2D entry point calls
   the loss with `cgm_dims=None`; a non-None `cgm_dims` here raises
   NotImplementedError.
4. Unreachable debug helpers dropped: `get_prediction_map` (loss.py:829),
   `save_B_D_points` (loss.py:841, also uses `np.int`, removed in
   numpy>=1.24), `save_nii` (loss.py:1075) and `get_arg_map_range`
   (loss.py:1094). None of them is reachable from `forward`/`get_topo_loss`
   (all call sites upstream are commented out); they only write NIfTI files
   under hard-coded cluster paths via SimpleITK.
5. Unused instrumentation dropped: the `t1/t2/t3` timing stamps and the
   `deta`/`rank` "top ranked ph length" lines of `get_topo_loss` — their
   only consumers were commented-out debug prints. `betti` (returned by the
   official tuple) is kept.
6. `random_mask` uses the official clDice `soft_skel(x, iter_=3)`; we call
   the project's verified `core.SoftSkeletonize(num_iter=3)` instead — the
   same substitution losses/decl.py makes, algorithmically identical to the
   official 2D branch.
7. polynomial.py imports dropped: matplotlib / mpl_toolkits / scipy.special /
   `from utils.utils import *` (that module imports cripser unconditionally)
   / `from cldice import ...`. `norm_ten` is copied verbatim from
   utils/utils.py:32-40 because polynomial.py needs it. `Polynomial's
   hermite path keeps `np.math.factorial` verbatim (removed in
   numpy>=1.24); it is unreachable with the default `basis_type='legendre'`,
   which is what the official pipeline uses.
8. Adapter-level bridge (not a change to the vendored math): the official
   `forward` returns a 5-tuple `(loss_topo, betti, num_points_updating,
   weight_map, ref_map)`; the vendored class returns the same tuple, and
   `EulerRefineLoss` (the registered entry) returns the scalar
   `loss_topo.sum()`. With the official `cgm_dims=None` default that is
   exactly the upstream `loss_topo_all[0]` (weighted by 0.001 in
   process_batch.py:406), as required by the loss-registry contract.
   Bilinear resize to `resolution` is applied when the input differs, while
   the gt is resized with nearest-neighbour to keep it a binary {0, 1} mask:
   the official `compute_dgm_force_new` asserts that the gt persistence
   values are 0/1, an invariant a bilinear mask resize breaks (the official
   code never resizes — it only tiles the map into `topo_size` crops).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import SoftSkeletonize
from .registry import register

try:
    import gudhi as gd
    _EULER_REFINE_AVAILABLE = True
except ImportError:
    gd = None
    _EULER_REFINE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vendored from loss.py:754 (= 3rd/Universal-Topology-Refinement/loss.py)
# ---------------------------------------------------------------------------
class topo_loss_2d(nn.Module):
    """Vendored official 2D topology (Euler characteristic) refinement loss.

    Only the gudhi branch is ported (deviation 1); the methods below are the
    official ones, with the deviations listed in the module docstring.
    """

    def __init__(self, package='gudhi'):
        super(topo_loss_2d, self).__init__()
        '''package = gudhi (8-connectivity in 2d) or cripser (4-connectivity in 2d) '''
        assert package == 'gudhi', (
            "topo_loss_2d: only the gudhi path is ported (the official "
            "cripser path is not vendored, see the module docstring)"
        )
        self.package = package

    def compute_dgm_force_new(self, lh_dgm, gt_dgm, pers_thresh=0, pers_thresh_perfect=0.99, do_return_perfect=False):
        idx_fix_holes = {}
        idx_remove_holes = {}

        for dim in list(lh_dgm.keys()):
            idx_fix_holes.update({dim: []})
            idx_remove_holes.update({dim: []})
            dim_int = int(dim)
            lh_pers = abs(lh_dgm[dim][:, 1] - lh_dgm[dim][:, 0])
            lh_pers_idx_ranked = np.argsort(lh_pers)[::-1]
            lh_n = len(lh_pers)

            if dim in gt_dgm.keys():
                gt_pers = abs(gt_dgm[dim][:, 1] - gt_dgm[dim][:, 0])
                if 0 in gt_pers:        # all background segmentation
                    gt_n = 0    #len(gt_pers) - len(np.where(gt_pers == 0))
                else:
                    gt_n = len(gt_pers)
                    assert np.array_equal(gt_pers, np.ones(gt_n))
            else:
                gt_pers = None
                gt_n = 0

            '''the number of likelihood complex > gt: some of them fixed and some of them removed '''
            if lh_n > gt_n:
                N_holes_2_fix = gt_n
                N_holes_2_remove = lh_n - gt_n

                idx_fix_holes.update({dim: lh_pers_idx_ranked[0:gt_n]})
                idx_remove_holes.update({dim: lh_pers_idx_ranked[gt_n::]})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix
                assert len(idx_remove_holes[dim]) == N_holes_2_remove
            elif lh_n <= gt_n:
                N_holes_2_fix = lh_n
                N_holes_2_remove = 0
                idx_fix_holes.update({dim: lh_pers_idx_ranked})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix

        return idx_fix_holes, idx_remove_holes

    # def recover(self, dim_eff):

    def pre_process(self, info):
        '''info in shape [dim, b, d, b_x, b_y. b_z, d_x, d_y, d_z]'''

        revised_row_array = np.where(info[:, 2] > 1)[0]
        for row in revised_row_array:
            info[row, 2] = 1

        dim_eff_all = np.unique(info[:, 0])
        pd_gt_1 = {}
        bcp_gt_1 = {}
        dcp_gt_1 = {}

        for dim_eff in dim_eff_all:
            idx = info[:, 0] == dim_eff
            pd_gt_1.update({str(int(dim_eff)): info[idx][:, 1:3]})
            bcp_gt_1.update({str(int(dim_eff)): info[idx][:, 3:6]})
            dcp_gt_1.update({str(int(dim_eff)): info[idx][:, 6::]})

        return pd_gt_1, bcp_gt_1, dcp_gt_1

    def check_point_exist(self, r_max, p):
        result = True
        for i, point in enumerate(list(p)):
            if point < 0 or point >= r_max[i]:
                result = False
        return result

    def reidx_f_gudhi(self, idx, ori_shape):
        '''given the original shape of a map, convert a gudhi flat index back
        to C-order coordinates (official 2D branch)'''
        re_idx = np.zeros(3, dtype=np.uint16)
        reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
        if len(ori_shape) == 3:
            div_0 = ori_shape[1] * ori_shape[2]
            re_idx[0] = int(idx // div_0)
            mod_0 = idx % div_0

            re_idx[1] = int(mod_0 // ori_shape[2])  # updated on 23/09/02 from ori_shape[1] to ori_shape[2]
            re_idx[2] = int(mod_0 % ori_shape[2])
            if idx != re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[2] + re_idx[
                2]:  # updated on 23/09/02 from re_idx[1] * ori_shape[1] to re_idx[1] * ori_shape[2]
                print('hold on, wrong reidx')
            assert (
                        reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'
        elif len(ori_shape) == 2:
            re_idx[0] = int(idx // ori_shape[1])
            re_idx[1] = int(idx % ori_shape[1])
            assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
            # Official asserts `(reidx_0 == re_idx).all()`, comparing a
            # shape-(2,) array against the shape-(3,) scratch array: that
            # raises ValueError on every 2D call (deviation 2). The intended
            # C-order check is against the first len(ori_shape) entries.
            assert (
                        reidx_0 == re_idx[:len(ori_shape)]).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'
        return re_idx  # re_idx

    def get_info_gudhi(self, map):
        cc = gd.CubicalComplex(dimensions=map.shape[::-1], top_dimensional_cells=1 - map.flatten())
        ph = cc.persistence()
        # betti_2 = cc.persistent_betti_numbers(from_value=1, to_value=0)
        x = cc.cofaces_of_persistence_pairs()

        '''3.1 get birth and death point coordinate from gudhi, and generate info array'''
        info_gudhi = np.zeros((len(ph), 9))
        # x will lack one death point where the filtration is inf
        '''3.1.1 manually write the inf death point'''
        reidx_birth_0 = self.reidx_f_gudhi(x[1][0][0], map.shape)
        if len(map.shape) == 2:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1]]
        elif len(map.shape) == 3:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2]]

        info_gudhi[0, :] = [0, birth_filtration, 1,
                            reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2],
                            0, 0, 0]
        idx_row = 1
        for dim in range(len(x[0])):
            for idx in range(x[0][dim].shape[0]):
                idx_brith, idx_death = x[0][dim][idx]
                reidx_birth = self.reidx_f_gudhi(idx_brith, map.shape)
                reidx_death = self.reidx_f_gudhi(idx_death, map.shape)

                if len(map.shape) == 2:
                    if reidx_birth[0]>=map.shape[0] or reidx_birth[1]>=map.shape[1] or reidx_death[0]>=map.shape[0] or reidx_death[1]>=map.shape[1]:
                        print('hold')
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1]]
                elif len(map.shape) == 3:
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1], reidx_birth[2]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1], reidx_death[2]]
                else:
                    assert False, 'wrong input dimension!'

                info_gudhi[idx_row, :] = [dim, birth_filtration, death_filtration,
                                          reidx_birth[0], reidx_birth[1], reidx_birth[2],
                                          reidx_death[0], reidx_death[1], reidx_death[2]]
                idx_row += 1

        return info_gudhi

    def get_topo_loss(self, map, gt, batch, cgm_idx):

        betti = np.zeros((1, 3))
        # Official: `if self.package == 'cripser': cripser.computePH(1 - map,
        # maxdim=2)` — the cripser branch is not ported (deviation 1). Only
        # the gudhi branch defines info_gt, which pre_process below needs.
        info_lh = self.get_info_gudhi(map)
        info_gt = self.get_info_gudhi(gt)

        pd_lh, bcp_lh, dcp_lh = self.pre_process(info_lh)
        pd_gt, bcp_gt, dcp_gt = self.pre_process(info_gt)
        for i in range(3):
            betti[0, i] = len(pd_lh[str(i)]) if str(i) in list(pd_lh.keys()) else 0

        idx_holes_to_fix, idx_holes_to_remove = self.compute_dgm_force_new(pd_lh, pd_gt, pers_thresh=0)
        '''
        topo_cp value map:
        0： background
        1, 2, 3：point to fix (born: force it to 0)
        4, 5, 6: point to fix (death: force it to 1)
        7, 8, 9: point to remove (born: force it to the prob of death)
        10,11,12: point to remove (death: force it to the prob of born)
        '''
        topo_size = list(map.shape)
        topo_cp_weight_map = np.zeros(map.shape)
        topo_cp_ref_map = np.zeros(map.shape)

        for dim in ['0', '1']:
            dim_int = int(dim)
            if dim in list(idx_holes_to_fix.keys()):
                for hole_indx in idx_holes_to_fix[dim]:
                    if self.check_point_exist(topo_size, bcp_lh[dim][hole_indx][0:2]):
                        coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_bb = (coor_b[0], coor_b[1])
                        topo_cp_weight_map[coor_bb] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_bb] = 0

                    if self.check_point_exist(topo_size, dcp_lh[dim][hole_indx][0:2]):
                        coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_dd = (coor_d[0], coor_d[1])
                        topo_cp_weight_map[coor_dd] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_dd] = 1

            no_1 = 0
            no_2 = 0
            np_3 = 0
            if dim in list(idx_holes_to_remove.keys()):
                for hole_indx in idx_holes_to_remove[dim]:
                    coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_bb = (coor_b[0], coor_b[1])

                    coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_dd = (coor_d[0], coor_d[1])

                    b_exists = self.check_point_exist(topo_size, bcp_lh[dim][hole_indx][0:2])
                    d_exists = self.check_point_exist(topo_size, dcp_lh[dim][hole_indx][0:2])
                    if b_exists and d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_bb] = map[coor_dd]
                        topo_cp_ref_map[coor_dd] = map[coor_bb]
                        no_1 = no_1 + 1
                    elif b_exists and not d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_ref_map[coor_bb] = 1
                        no_2 = no_2 + 1

                    elif not b_exists and d_exists:
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_dd] = 0
                        no_3 = no_3 + 1

        return topo_cp_weight_map, topo_cp_ref_map, betti, batch, cgm_idx

    def forward(self, map, gt, device, cgm_dims=None, topo_size=256):
        '''for the gt should be betti (0,0,1)'''
        if cgm_dims is not None:
            # The official multiclass branch (loss.py:1107-1181) is not ported
            # (deviation 3).
            raise NotImplementedError(
                "topo_loss_2d: the multiclass cgm_dims branch of the official "
                "forward (loss.py:1107) is not ported; call with cgm_dims=None "
                "as the official 2D entry point process_batch.py:405 does."
            )
        topo_cp_weight_map = np.zeros(map.shape)
        topo_cp_ref_map = np.zeros(map.shape)
        betti = np.zeros((map.shape[0], 3))
        for batch in range(map.shape[0]):
            for y in range(0, map.shape[2], topo_size):
                for x in range(0, map.shape[3], topo_size):
                    map_patch = map[batch, 0,
                                y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])]
                    gt_patch = gt[batch, 0,
                                y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])]
                    topo_cp_weight_map[batch, 0,
                    y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])], \
                    topo_cp_ref_map[batch, 0,
                    y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])], bb, _, __ \
                        = self.get_topo_loss(map_patch.cpu().detach().numpy(), gt_patch.cpu().detach().numpy(), batch, 0)
                    betti[batch, :] = bb

        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float32).to(device)
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float32).to(device)
        cgm_dims = [0]
        cgm_no = 1
        loss_topo = torch.zeros(cgm_no, dtype=torch.float32).to(device)
        num_points_updating = 0

        for cgm_idx, cgm_dim in enumerate(cgm_dims):
            idx = np.s_[:, cgm_idx, ...]
            idx_for_map = np.s_[:, cgm_dim, ...]
            loss_topo[cgm_idx] = 1 / (map.shape[0] * map.shape[2] * map.shape[3]) * (
                        ((map[idx_for_map] * topo_cp_weight_map[idx]) - topo_cp_ref_map[idx]) ** 2).sum()
            num_points_updating += topo_cp_weight_map[idx].sum()
        betti_return = betti.mean(axis=0)

        return loss_topo, betti_return, num_points_updating / (
                    map.shape[0] * map.shape[2] * map.shape[3] * len(cgm_dims)), topo_cp_weight_map, topo_cp_ref_map


# ---------------------------------------------------------------------------
# Polynomial feature synthesis, vendored from polynomial.py of the same repo
# (module docstring, deviations 6-7).
# ---------------------------------------------------------------------------
# Official `soft_skel(x, iter_=3)` from the repo's cldice.py: the project's
# verified 2D soft-skeletonizer, same algorithm (deviation 6).
_soft_skel_3 = SoftSkeletonize(num_iter=3)


def norm_ten(pred_ce, binary=False):
    # Vendored verbatim from 3rd/Universal-Topology-Refinement/utils/utils.py:32
    mmax = pred_ce.max()
    mmin = pred_ce.min()
    pred_ce = (pred_ce - mmin)/ (mmax - mmin+ 0.001)
    if binary:
        pred_ce[pred_ce>=0.5] == 1
        pred_ce[pred_ce<0.5] == 0
    return pred_ce


def get_background_index_(x, no_idx_select=10, label=0):
    # polynomial.py:9
    all_coordinate = torch.nonzero(x == label, as_tuple=False)
    random_index = []
    all_coordinate_idx = all_coordinate.shape[0]
    selected_coordinate_idx = np.random.randint(0, all_coordinate_idx, no_idx_select)
    center_coor_list = [all_coordinate[x] for x in selected_coordinate_idx]
    return center_coor_list


def generate_multi_gauss_mask(x, center_coor_list, var):
    # polynomial.py:17
    dim = len(x.shape) - 2
    if dim ==3:
        xx, yy, zz = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[3]), np.arange(x.shape[4]), indexing='ij')
    elif dim ==2:
        xx, yy,  = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[3]), indexing='ij')
    multi_gaussian_map = torch.zeros_like(x)
    binary_mask_for_weighted_loss = torch.zeros_like(x)
    for i, coor in enumerate(center_coor_list):
        if dim == 3:
            coor_x, coor_y, coor_z = [int(coor[2]), int(coor[3]), int(coor[4])]
            distances_squared = ((xx - coor_x) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (yy - coor_y) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (zz - coor_z) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2))
        elif dim == 2:
            coor_x, coor_y = [int(coor[2]), int(coor[3])]
            distances_squared = ((xx - coor_x) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) +
                                 (yy - coor_y) ** 2 / (2 * float(np.random.normal(loc=var, scale=0.5, size=1)) ** 2) )

        gaussian_map = np.exp(-distances_squared).astype(np.float32)

        # Normalize the values to the range [0, 1]
        gaussian_map = (gaussian_map - np.min(gaussian_map)) / (np.max(gaussian_map) - np.min(gaussian_map))

        mask_1 = torch.from_numpy(gaussian_map[np.newaxis, np.newaxis, ...]).to(x.device)
        mask_1 = mask_1 * np.random.randint(10)/10

        multi_gaussian_map += mask_1

    binary_mask_for_weighted_loss[multi_gaussian_map>=0.05] = 1

    return multi_gaussian_map, binary_mask_for_weighted_loss


def random_mask(x, map_size, inpaint_type, poly, order, isCL=True, var=5, idx=None):
    # polynomial.py:52 — the paper's polynomial perturbation (README).
    dim = int(len(x.shape) - 2)
    if isCL:
        x_skel = _soft_skel_3(x)
    else:
        x_skel = x
    x_np = x_skel.cpu().detach().numpy()
    foreground_coords = np.column_stack(np.where(x_np != 0))
    if len(foreground_coords) == 0:
        # print(ValueError("Label map does not have any foreground."))
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    if idx is not None:
        rand_index = idx * len(foreground_coords)//11
    else:
        rand_index = np.random.randint(0, len(foreground_coords))
    if dim ==3:
        center_x, center_y, center_z = foreground_coords[rand_index][2:5]
    elif dim ==2:
        center_x, center_y = foreground_coords[rand_index][2:4]
    delta = map_size
    start_x = center_x - delta // 2
    start_y = center_y - delta // 2
    start_x = max(0, min(start_x, x_np.shape[2] - delta))
    start_y = max(0, min(start_y, x_np.shape[3] - delta))
    if dim == 3:
        start_z = center_z - delta // 2
        start_z = max(0, min(start_z, x_np.shape[4] - delta))
        slicing_foreground = np.s_[:, :, start_x:start_x+delta, start_y:start_y+delta, start_z:start_z+delta]
    elif dim == 2:
        slicing_foreground = np.s_[:, :, start_x:start_x + delta, start_y:start_y + delta]

    if inpaint_type in ['polynomial_bi', 'polynomial_bi_binary']:
        # get the coordinate of background first, in order to mask on background later
        x_background = 1 - x
        background_coordinate = torch.nonzero(x_background == 1, as_tuple=False)
        random_index = background_coordinate[np.random.randint(0, background_coordinate.shape[0])]
        if dim == 3:
            center_x_1, center_y_1, center_z_1 = [random_index[2], random_index[3], random_index[4],]
        elif dim == 2:
            center_x_1, center_y_1 = [random_index[2], random_index[3], ]
        center_x_1 = int(center_x_1)
        center_y_1 = int(center_y_1)
        start_x_1 =  center_x_1 - delta//2
        start_y_1 =  center_y_1- delta//2
        start_x_1 = max(0, min(start_x_1, x_np.shape[2] - delta))
        start_y_1 = max(0, min(start_y_1, x_np.shape[3] - delta))
        if dim == 3:
            center_z_1 = int(center_z_1)
            start_z_1 = center_z_1 - delta//2
            start_z_1 = max(0, min(start_z_1, x_np.shape[4] - delta))
            slicing_background = np.s_[:, :, start_x_1:start_x_1 + delta, start_y_1:start_y_1 + delta,
                                 start_z_1:start_z_1 + delta]
        elif dim == 2:
            slicing_background = np.s_[:, :, start_x_1:start_x_1 + delta, start_y_1:start_y_1 + delta]

    '''remove poly on foreground, marked with 'polynomial_xxx' '''
    map, coeffs = poly.get_mask(order=order)
    map = 1.5 * map - 0.5
    map[map<0] =0
    mask = torch.zeros_like(x)
    mask[slicing_foreground] = 1
    x_masked = x.clone().detach()
    x_masked[slicing_foreground] = \
        x_masked[slicing_foreground] * torch.from_numpy(map[np.newaxis, np.newaxis, ...]).to(x_masked.device)

    '''add poly in background, marked with '_bi' '''
    if inpaint_type in ['polynomial_bi', 'polynomial_bi_binary']:
        map_2, coeffs_2 = poly.get_mask(order=order, p_low=0.05, p_high=0.5)
        map_2 = 1.5 * map_2 - 0.5
        map_2[map_2 < 0] = 0
        mask[slicing_background] = 0.1
        x_masked[slicing_background] += \
            (1 - x_masked[slicing_background]) * torch.from_numpy(map_2[np.newaxis, np.newaxis, ...]).to(x_masked.device)

    '''add gauss dots in background, marked with '_gauss' '''
    if inpaint_type in ['polynomial_gauss', 'polynomial_gauss_binary',]:
        no_idx_select= np.random.randint(1, 10)
        center_coor_list = get_background_index_(x, no_idx_select=no_idx_select, label=0)
        multi_gaussian_map, binary_mask_for_weighted_loss = generate_multi_gauss_mask(x, center_coor_list, var=3)

        x_masked = x_masked + multi_gaussian_map
        x_masked[x_masked>1] = 1
        mask[binary_mask_for_weighted_loss == 1] = 1

    '''remove square boxes on foreground, marked with '_binary' '''
    if inpaint_type in ['polynomial_binary', 'polynomial_gauss_binary', 'polynomial_bi_binary']:
        if isCL:
            x_skel = _soft_skel_3(x)
        else:
            x_skel = x
        x_np = x_skel.cpu().detach().numpy()
        foreground_coords = np.column_stack(np.where(x_np != 0))
        # if len(foreground_coords) == 0:
        #     raise ValueError("Label map does not have any foreground.")
        if idx is not None:
            rand_index = idx * len(foreground_coords)//15
        else:
            rand_index = np.random.randint(0, len(foreground_coords))

        if dim == 3:
            center_x_2, center_y_2, center_z_2 = foreground_coords[rand_index][2:5]
        elif dim == 2:
            center_x_2, center_y_2 = foreground_coords[rand_index][2:4]
        delta_1 = 30
        start_x_2 = center_x_2 - delta_1 //2
        start_y_2 = center_y_2 - delta_1 // 2
        start_x_2 = max(0, min(start_x_2, x_np.shape[2] - delta_1))
        start_y_2 = max(0, min(start_y_2, x_np.shape[3] - delta_1))
        if dim == 3:
            start_z_2 = center_z_2 - delta_1 // 2
            start_z_2 = max(0, min(start_z_2, x_np.shape[4] - delta_1))
            slicing_foreground_1 = np.s_[:,:, start_x_2:start_x_2+delta_1, start_y_2:start_y_2+delta_1, start_z_2:start_z_2+delta_1]
        elif dim == 2:
            slicing_foreground_1 = np.s_[:,:, start_x_2:start_x_2+delta_1, start_y_2:start_y_2+delta_1]

        x_masked[slicing_foreground_1] = 0
        mask[slicing_foreground_1] = 1

    return x_masked, mask, x_skel


class Polynomial:
    # polynomial.py:174 — the polynomial basis / random-mask synthesiser of
    # the paper (basis_type: 'legendre' (default), 'chebyshev' or 'hermite').
    def __init__(self, map_size, order=10, dim=2, basis_type='legendre'):
        """
        Initialize the polynomial with its type and coefficients for each dimension.
        :param basis_type: String, the type of polynomial ("hermite", "legendre", etc.)
        :param coefficients: List of lists, coefficients for the polynomial in each dimension.
        """
        self.basis_type = basis_type
        self.dimensions = dim
        self.order = order
        self.map_size = map_size

        self.basis = {}
        if dim ==2:
            self.init_2d_basis(self.order)
        elif dim ==3:
            self.init_3d_basis(self.order)
        else:
            assert 'WRONG INPUT DIMENSION'

    def legendre_polynomial(self, n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            return ((2 * n - 1) * x * self.legendre_polynomial(n - 1, x) - (n - 1) * self.legendre_polynomial(n - 2, x)) / n

    def chebyshev_first_kind_iter(self, n, x):
        x = np.array(x, dtype=np.float64)
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            T0 = np.ones_like(x)
            T1 = x
            for _ in range(2, n+1):
                T2 = 2 * x * T1 - T0
                T0, T1 = T1, T2
            return T1

    def hermite_polynomial_physicist(self, n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * self.hermite_polynomial_physicist(n - 1, x) - 2 * (n - 1) * self.hermite_polynomial_physicist(n - 2, x)

    def hermite_functions(self,n, x):
        # Official uses `np.math.factorial`, removed in numpy>=1.24 (deviation
        # 7): kept verbatim, unreachable with the default legendre basis.
        return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**-0.5 * self.hermite_polynomial_physicist(n, x) * np.exp(-x**2 / 2)

    def hermite_functions_nomalized(self,n, x):
        """ scale the x axis to [-1, 1] , the scale should be the 95 precent of the nth hermite function"""
        scale = 2.5 * np.sqrt(n+1)
        x = x * scale
        return self.hermite_functions(n, x)

    def init_2d_basis(self, order):
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        X, Y = np.meshgrid(x, y)

        choice = np.eye(order, dtype=int).tolist()
        for i in range(order):
            for j in range(order):
                # Create a polynomial with random coefficients
                basis = self.evaluate([X, Y], [choice[i], choice[j]])
                key_name = str(i) + '-' + str(j)
                self.basis.update({
                    key_name: basis
                })

    def init_3d_basis(self, order):
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        z = np.linspace(-1, 1, self.map_size)
        X, Y, Z = np.meshgrid(x, y, z)

        choice = np.eye(order, dtype=int).tolist()
        for i in range(order):
            for j in range(order):
                for k in range(order):
                    # Create a polynomial with random coefficients
                    basis = self.evaluate(points=[X, Y, Z], coefficients=[choice[i], choice[j], choice[k]])
                    key_name = str(i) + '-' + str(j) + '-' + str(k)
                    self.basis.update({
                        key_name: basis
                    })

    def evaluate_basis(self, n, x):
        """
        Evaluate the n-th polynomial of the given basis type at points x.
        :param n: Integer, the order of the polynomial.
        :param x: Array of points at which to evaluate the polynomial.
        :return: Evaluated polynomial at points x.
        """
        if self.basis_type == "hermite":
            return self.hermite_functions_nomalized(n, x)
        elif self.basis_type == "legendre":
            return self.legendre_polynomial(n, x)
        elif self.basis_type == 'chebyshev':
            return self.chebyshev_first_kind_iter(n, x)
        else:
            raise ValueError("Unsupported basis type")

    def evaluate(self, points, coefficients):
        """
        Evaluate the polynomial at given points in space.
        :param points: Array-like, the points in space at which to evaluate the polynomial.
                       Should have the same number of dimensions as there are sets of coefficients.
        :return: The polynomial evaluated at the given points.
        """
        if len(points) != self.dimensions:
            raise ValueError("Points dimensionality does not match the polynomial dimensions")

        result = np.ones_like(points[0])
        for dim in range(self.dimensions):
            dim_result = np.zeros_like(points[dim])
            for order, coeff in enumerate(coefficients[dim]):
                dim_result += coeff * self.evaluate_basis(order, points[dim])
            result *= dim_result

        return result

    def get_mask(self, order, coeffs_type='gauss', p_low=0.05, p_high=0.95):
        FLAG= True
        while FLAG:
            if coeffs_type == 'gauss':
                if self.dimensions ==2:
                    coeffs = np.random.randn(order, order)
                    r = np.zeros((self.map_size, self.map_size))

                    for i in range(order):
                        for j in range(order):
                            key_name = str(i) + '-' + str(j)
                            r += coeffs[i, j] * self.basis[key_name]

                elif self.dimensions ==3:
                    coeffs = np.random.randn(order, order, order)
                    r = np.zeros((self.map_size, self.map_size, self.map_size))

                    for i in range(order):
                        for j in range(order):
                            for k in range(order):
                                key_name = str(i) + '-' + str(j) + '-' + str(k)
                                r += coeffs[i, j, k] * self.basis[key_name]

            r = norm_ten(r)
            p = norm_ten(r, binary=True).sum() / (self.map_size ** self.dimensions)
            if p <= p_high and p >= p_low:
                FLAG = False

        return r, coeffs


# ---------------------------------------------------------------------------
# Registered adapter
# ---------------------------------------------------------------------------
class EulerRefineLoss(nn.Module):
    """Registered adapter: 2-channel bridge + optional `resolution` resize.

    The official computation (network -> persistence -> force maps -> weighted
    L2) is unchanged; only the scalar extraction from the official 5-tuple and
    the interpolation policy are ours (module docstring, deviation 8)."""

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.loss = topo_loss_2d(package='gudhi')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: [B, 1, H, W] raw logits; target: [B, 1, H, W] in [0, 1]."""
        # Bridge to the official probability input: the official network is
        # single-channel (pred_softmax = Sigmoid()(pred), num_classes == 1),
        # and softmax(cat([zeros, logits]))[:, 1] == sigmoid(logits) EXACTLY,
        # so the foreground channel we forward is bit-identical to the
        # official sigmoid output.
        probs = F.softmax(torch.cat([torch.zeros_like(logits), logits], dim=1), dim=1)[:, 1:2]
        if self.resolution is not None and probs.shape[-1] != self.resolution:
            probs = F.interpolate(probs, size=(self.resolution, self.resolution),
                                  mode='bilinear', align_corners=False)
            # The gt stays a BINARY {0, 1} mask: the official
            # compute_dgm_force_new asserts that the gt persistence values are
            # 0/1, which bilinear interpolation breaks (deviation 8).
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='nearest')
        loss_topo, _betti, _num_points, _weight_map, _ref_map = self.loss(
            probs, target.float(), probs.device)
        # cgm_dims=None upstream -> a single foreground channel; the official
        # training loop uses loss_topo_all[0] (process_batch.py:406).
        return loss_topo.sum()


if _EULER_REFINE_AVAILABLE:
    register('euler_refine')(EulerRefineLoss)
