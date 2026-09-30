"""SATLoss — strict vendor of the official repository (ICCVW 2025,
"Topology-Preserving Image Segmentation with Spatial-Aware Persistent
Feature Matching").

Vendored verbatim from the official repo:
  utils/cubical_complex.py  (CubicalComplex, adapted from
                             torch_topological)
  utils/PDMatching.py       (SpatialAware_WassersteinDistance)
  utils/losses.py           (PDMatchingLoss)

Dependencies: gudhi, torch-topological, POT (ot). The classes are always
defined; registration ('satloss') is skipped when a dependency is
missing, so the package imports cleanly.

API bridge: the official loss consumes PROBABILITY maps [N, 1, H, W];
our adapter applies sigmoid to our logits, evaluates at `resolution`
(256 by default — gudhi's cubical persistence at 1024 is expensive per
training iteration), and returns the scalar loss. The official
computation is unchanged.
"""

import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

try:
    import gudhi  # noqa: F401
    from torch_topological.nn import PersistenceInformation
    from torch_topological.utils import wrap_if_not_iterable
    import ot  # noqa: F401
    _SATLOSS_AVAILABLE = True
except ImportError:
    PersistenceInformation = None
    wrap_if_not_iterable = None
    _SATLOSS_AVAILABLE = False


class CubicalComplex(nn.Module):
    def __init__(self, superlevel=False, dim=2):
        super().__init__()
        self.superlevel = superlevel
        self.dim = dim

    def forward(self, x):
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)
        else:
            dims = len(x.shape) - 2

        if dims == 0:
            return self._forward(x)
        elif dims == 1:
            return [
                self._forward(x_) for x_ in x
            ]
        elif dims == 2:
            return [
                    [self._forward(x__) for x__ in x_] for x_ in x
            ]

    def _forward(self, x):
        if self.superlevel:
            x = -x

        cubical_complex = gudhi.CubicalComplex(
            dimensions=x.shape,
            top_dimensional_cells=x.flatten()
        )

        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)

        persistence_information = [
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            ) for dim in range(0, max_dim)
        ]

        return persistence_information

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs, dim)

    def _create_tensors_from_pairs(self, x, pairs, dim):
        xs = x.shape

        creators = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 0], xs)
                ),
                dtype=torch.long
        )
        destroyers = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 1], xs)
                ),
                dtype=torch.long
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)))

        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )


class SpatialAware_WassersteinDistance(torch.nn.Module):
    def __init__(self, p=torch.inf, q=1):
        super().__init__()
        self.p = p
        self.q = q

    def _project_to_diagonal(self, diagram):
        x = diagram[:, 0]
        y = diagram[:, 1]

        return 0.5 * torch.stack(((x + y), (x + y)), 1)

    def _distance_to_diagonal(self, diagram):
        return torch.linalg.vector_norm(
            diagram - self._project_to_diagonal(diagram),
            self.p,
            dim=1
        )

    def _make_distance_matrix(self, D1, D2, C1, C2):
        dist_D11 = self._distance_to_diagonal(D1)
        dist_D22 = self._distance_to_diagonal(D2)

        PD_dist = torch.cdist(D1, D2, p=self.p)

        Spatial_dist = torch.cdist(C1, C2, p=self.p)
        Spatial_dist = torch.clamp(Spatial_dist, 0.05, 1)

        Weighted_dist = Spatial_dist * PD_dist

        upper_blocks = torch.hstack((Weighted_dist, dist_D11[:, None]))
        lower_blocks = torch.cat(
            (dist_D22, torch.tensor(0, device=dist_D22.device).unsqueeze(0))
        )
        M = torch.vstack((upper_blocks, lower_blocks))

        M = M.pow(self.q)

        return M

    def forward(self, X, Y, H, W):
        total_cost = 0.0

        X = wrap_if_not_iterable(X)
        Y = wrap_if_not_iterable(Y)

        for pers_info in zip(X, Y):
            D1 = pers_info[0].diagram
            D2 = pers_info[1].diagram

            C1 = pers_info[0].pairing[:,:2].float()
            C2 = pers_info[1].pairing[:,:2].float()

            C1[:, 0] /= H
            C2[:, 0] /= H
            C1[:, 1] /= W
            C2[:, 1] /= W

            n = len(D1)
            m = len(D2)

            dist = self._make_distance_matrix(D1, D2, C1, C2)

            a = torch.ones(n + 1, device=dist.device)
            b = torch.ones(m + 1, device=dist.device)

            a[-1] = m
            b[-1] = n

            total_cost += ot.emd2(a, b, dist)

        return total_cost.pow(1.0 / self.q)


class PDMatchingLoss(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        self.getPersistentInfo = CubicalComplex(dim=2)
        self.criterion = SpatialAware_WassersteinDistance(p=p)
        self.precal_PD = opt.precal_PD
        self.PD_target = {}
        self.pad_dims = (1, 1, 1, 1)

    def _pad_to_square(self, x1, x2, H, W):
        margin = abs(H - W)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H > W:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if x1 is not None:
            x1 = F.pad(x1, paddings, "constant", 0.0)
        if x2 is not None:
            x2 = F.pad(x2, paddings, "constant", 0.0)

        return x1, x2

    def _pre_compute_PD(self, target, img_names):
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, _, H, W = target.size()

        if H != W:
            _, padded_target = self._pad_to_square(None, padded_target, H, W)

        padded_target = torch.clamp(padded_target, min=0.0, max=1.0)
        padded_target = 1.0 - padded_target

        for i in range(N):
            img = padded_target[i,0,:,:].unsqueeze(0).unsqueeze(0)
            self.PD_target[img_names[i]] = self.getPersistentInfo(img)

    def forward(self, input, target, img_names=None):
        N, C, H, W = input.size()
        assert input.size() == target.size()
        assert input.device == target.device
        assert C == 1

        self.device = input.device

        input = input.to(torch.float32)
        target = target.to(torch.float32)

        padded_input = F.pad(input, self.pad_dims, mode='constant', value=1)
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, C, H, W = input.size()

        if H != W:
            input, target = self._pad_to_square(padded_input, padded_target, H, W)

        input = torch.clamp(input, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        input = 1.0 - input
        target = 1.0 - target

        pi_x = self.getPersistentInfo(input)
        if self.precal_PD:
            pi_y = [self.PD_target[img_names[0]][0]]
            for idx in range(1, N):
                pi_y.append(self.PD_target[img_names[idx]][0])
        else:
            pi_y = self.getPersistentInfo(target)

        for i in range(N):
            pd_x_0 = pi_x[i][0][0]
            pd_y_0 = pi_y[i][0][0]

            pd_x_1 = pi_x[i][0][1]
            pd_y_1 = pi_y[i][0][1]

            wd_0 = self.criterion(pd_x_0, pd_y_0, H, W)
            wd_1 = self.criterion(pd_x_1, pd_y_1, H, W)

            loss += (wd_0 + wd_1)

        loss /= N

        return loss


class SATLossAdapter(nn.Module):
    """Registered adapter: sigmoid bridge + optional 256-resolution
    evaluation (gudhi persistence per training iteration at 1024 is
    expensive). The official computation is unchanged."""

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution
        opt = types.SimpleNamespace(precal_PD=False)
        self.loss = PDMatchingLoss(opt=opt, p=2)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if self.resolution is not None and probs.shape[-1] != self.resolution:
            probs = F.interpolate(probs, size=(self.resolution, self.resolution),
                                  mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
        return self.loss(probs, target)


if _SATLOSS_AVAILABLE:
    register('satloss')(SATLossAdapter)
