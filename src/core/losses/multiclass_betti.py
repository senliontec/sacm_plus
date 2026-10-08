"""HuTopo — Multiclass Dice + Wasserstein matching (official port).

Official source: the multiclass-BettiMatching repository (HuTopo; the
topology-aware loss of the "Multiclass Betti Matching" line of work).

    losses/hutopo.py
        MulticlassDiceWassersteinLoss   l. 20-81   (the 'HuTopo' entry of
                                                    config.LOSS.USE_LOSS)
        MulticlassWassersteinLoss       l. 83-128  (softmax / one-vs-rest,
                                                    per-channel flattening)
        WassersteinLoss                 l. 131-239 (barcode computation +
                                                    gudhi Wasserstein matching)
    losses/utils.py
        FiltrationType                  l. 12-15
        DiceType                        l. 22-24
        convert_to_one_vs_rest          l. 26-51   (one-vs-max)
    losses/dice_losses.py
        Multiclass_CLDice               l. 14-...  NOT re-ported: the project
                                                    already vendors that very
                                                    class (same lineage) in
                                                    core.losses.topograph and
                                                    reuses it as-is.

Vendoring of the engine
-----------------------
The official module imports the *compiled* pybind module ``betti_matching``
(nstucki/Betti-Matching-3D, C++) for ``compute_barcode``. This project
vendors the reference pure-Python implementation of the same algorithm in
``core.losses.betti_matching`` (UnionFind / BoundaryMatrix /
CubicalPersistence / ...), so the glue :func:`compute_barcode` below only
re-implements the mechanical part of the C++ binding (cells -> numpy
coordinate arrays) on top of ``CubicalPersistence`` — no engine code is
duplicated. ``gudhi.wasserstein.wasserstein_distance`` (the official
matching and cost) is used verbatim; it is a hard dependency of the glue,
so registration is skipped gracefully when gudhi is missing.

Deviations from the official file (exhaustive list)
---------------------------------------------------
1.  Engine: ``betti_matching.compute_barcode`` (C++, Betti-Matching-3D) ->
    :func:`compute_barcode` here, a thin wrapper over the vendored
    ``CubicalPersistence``. Both compute the persistence of the
    V-construction of the map in the *sublevel* reading (a cell's filtration
    value is the MAX over the voxels it touches — the C++ engine's
    ``CubicalGridComplex::getBirth``), and both omit the essential interval
    ("Note that we do not output the essential interval", Betti-Matching-3D
    introduction). MULTICLASS is realised one-vs-rest upstream: the official
    ``WassersteinLoss`` inverts the maps for ``FiltrationType.SUPERLEVEL``
    (the default) before calling the engine — that inversion is kept
    verbatim, so the filtration convention is unchanged.
2.  Coordinate/value recovery: the official code indexes the spatial tensor
    with the engine's birth/death coordinates. The vendored engine addresses
    cells on its doubled ``(2m-1) x (2n-1)`` V-construction grid, so the
    wrapper stores for every pair endpoint the input PIXEL that realises the
    cell's filtration value — the C++ ``getParentVoxel`` rule (max over the
    touched pixels, ties resolved to the larger coordinate). Indexing the
    tensor with these coordinates reproduces the engine's cell values
    exactly and keeps the official, differentiable gather.
3.  2D only: the vendored engine is 2D (``CubicalPersistence``); the
    official code also accepts 1D/3D volumes.
4.  Zero-persistence pairs are dropped (``valid='positive'``, the vendored
    engine's training default: strictly positive persistence); this is also
    what removes the single essential (birth, +inf) interval of dimension 0.
5.  ``Multiclass_CLDice`` / ``convert_to_one_vs_rest`` / ``DiceType`` come
    from ``core.losses.topograph`` (line-identical classes from the same
    lineage) instead of ``losses/dice_losses.py`` / ``losses/utils.py``.
6.  ``_wasserstein_loss`` reads ``num_pairs_by_dim`` — the (older)
    ``BarcodeResult`` layout referenced by the official file; the current
    Betti-Matching-3D renamed that field to ``num_pairs``. The wrapper
    reproduces the older layout.
7.  The ``monai.data.meta_tensor.MetaTensor`` unwrapping of the official
    ``_wasserstein_loss`` is dropped: our bridge produces plain tensors, for
    which the official calls are no-ops.
8.  The official ``forward`` takes ``alpha`` per call (``train.py`` ramps it
    up with an exponential warm-up schedule over ``ALPHA_WARMUP_EPOCHS``);
    the registered adapter exposes it as a constructor argument, the default
    being the official call-time default (0.5).
9.  Bridge, single channel (see :class:`HutopoLossAdapter`): ``[B, 1, H, W]``
    logits are turned into the official multi-class one-hot form with
    ``cat([zeros, logits])`` + one-hot target, i.e. ``softmax(x)[:, 1] ==
    sigmoid(logits)`` exactly (the same bridge as ``core.losses.decl``).

Performance note
----------------
The vendored engine is pure Python. Measured on this machine for a single
random 2D map (one ``CubicalPersistence``, both dimensions): 64^2 -> 0.18 s,
128^2 -> 1.2 s, 256^2 -> 12.7 s. At 256^2 a barcode holds ~13k dimension-0
bars, so the official gudhi matching allocates a ~1.4 GB cost matrix per
dimension. The official configuration trains on 200x200 crops
(configs/example_config.yaml: DATA.IMG_SIZE), which is the same order. The
registered adapter evaluates at ``resolution`` (default 256, the decoder
output resolution of the other engine-based adapters) with ``logits`` and
``target`` bilinearly resized; pass a smaller ``resolution`` (e.g. 128) to
keep a training iteration affordable.
"""

import enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .betti_matching import CubicalPersistence
from .registry import register
from .topograph import DiceType, Multiclass_CLDice, convert_to_one_vs_rest

try:
    from gudhi import wasserstein
    _HAS_GUDHI = True
except ImportError:  # pragma: no cover - environment without gudhi
    wasserstein = None
    _HAS_GUDHI = False

# Official losses/hutopo.py sets this module-level flag from
# WassersteinLoss.compute_wasserstein_loss; declared here so the verbatim
# `global` statement below is well formed.
ENCOUNTERED_NONCONTIGUOUS = False


class FiltrationType(enum.Enum):
    """Official losses/utils.py, l. 12-15."""

    SUPERLEVEL = "superlevel"
    SUBLEVEL = "sublevel"
    BOTHLEVELS = "bothlevels"


class BarcodeResult:
    """Stand-in for ``betti_matching.return_types.BarcodeResult``.

    Mirrors the layout the official ``hutopo.py`` relies on (l. 195-202,
    222-223): ``birth_coordinates`` / ``death_coordinates`` are lists indexed
    by homology dimension, each an ``(n_pairs, n_dimensions)`` int64 array,
    and ``num_pairs_by_dim`` is the per-dimension pair count.
    """

    __slots__ = ("birth_coordinates", "death_coordinates", "num_pairs_by_dim")

    def __init__(self, birth_coordinates, death_coordinates, num_pairs_by_dim):
        self.birth_coordinates = birth_coordinates
        self.death_coordinates = death_coordinates
        self.num_pairs_by_dim = num_pairs_by_dim


def compute_barcode(image):
    """Barcode of a 2D map — replacement for the C++ ``compute_barcode``.

    Reuses the vendored engine (no engine code is duplicated): the
    persistence pairs come from ``CubicalPersistence`` (V-construction,
    sublevel reading, strictly positive persistence), and the essential
    interval is dropped, exactly like the official C++ binding. Accepts one
    map or a list of maps (the official batch overload used by
    ``WassersteinLoss.compute_wasserstein_loss``, l. 183-184).

    Args:
        image: ``(H, W)`` array-like (a list of such maps for the batch form).

    Returns:
        BarcodeResult, or a list of them for the batch form. Pair endpoints
        are PIXEL coordinates of `image` (see deviation 2 of the module
        docstring), so ``image[coords.T]`` yields the engine's cell values.
    """
    if isinstance(image, (list, tuple)):
        return [compute_barcode(one) for one in image]

    image = np.ascontiguousarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"compute_barcode supports 2D maps, got shape {image.shape}")

    cp = CubicalPersistence(image, filtration='sublevel', construction='V',
                            valid='positive')
    m, n = image.shape

    # Cell -> pixel that realises the cell's filtration value. On the doubled
    # V-construction grid a vertex (even, even) belongs to one pixel and an
    # edge (mixed parity) to the two pixels it separates; the engine-values
    # the cell with the maximum of them (C++ getParentVoxel, ties to the
    # larger coordinate).
    x = np.arange(cp.M)[:, None]
    y = np.arange(cp.N)[None, :]
    i0, j0 = x // 2, y // 2
    i1 = np.minimum(i0 + (x % 2), m - 1)
    j1 = np.minimum(j0 + (y % 2), n - 1)
    take_second = image[i1, j1] >= image[i0, j0]
    owner_i = np.where(take_second, i1, i0)
    owner_j = np.where(take_second, j1, j0)

    birth_coordinates, death_coordinates, num_pairs_by_dim = [], [], []
    for dim in (0, 1):
        # Drop the essential interval (death index is +inf).
        pairs = [(birth, death) for birth, death in cp.intervals[dim]
                 if np.isfinite(death)]
        if pairs:
            births = np.array([pair[0] for pair in pairs], dtype=np.int64)
            deaths = np.array([pair[1] for pair in pairs], dtype=np.int64)
            coords_birth = np.stack([owner_i.ravel()[births],
                                     owner_j.ravel()[births]], axis=1)
            coords_death = np.stack([owner_i.ravel()[deaths],
                                     owner_j.ravel()[deaths]], axis=1)
        else:
            coords_birth = np.zeros((0, 2), dtype=np.int64)
            coords_death = np.zeros((0, 2), dtype=np.int64)
        birth_coordinates.append(coords_birth)
        death_coordinates.append(coords_death)
        num_pairs_by_dim.append(len(pairs))

    return BarcodeResult(birth_coordinates, death_coordinates,
                         np.array(num_pairs_by_dim, dtype=np.int64))


class WassersteinLoss(torch.nn.modules.loss._Loss):
    """Official losses/hutopo.py, l. 131-239 (verbatim)."""

    def __init__(
        self,
        filtration_type: FiltrationType = FiltrationType.SUPERLEVEL,
        num_processes=1,
    ) -> None:
        super().__init__()
        self.filtration_type = filtration_type
        self.num_processes = num_processes

    def forward(self, input, target):
        wasserstein_losses = self.compute_wasserstein_loss(input, target)

        dic = {
            'losses': wasserstein_losses,
        }
        loss: torch.Tensor = torch.mean(torch.concatenate(wasserstein_losses))
        return loss, dic

    def compute_wasserstein_loss(self, prediction, target):
        if self.filtration_type == FiltrationType.SUPERLEVEL:
            # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
            prediction = 1 - prediction
            target = 1 - target
        if self.filtration_type == FiltrationType.BOTHLEVELS:
            # Just duplicate the number of elements in the batch, once with sublevel, once with superlevel
            prediction = torch.concat([prediction, 1 - prediction])
            target = torch.concat([target, 1 - target])

        split_indices = np.arange(self.num_processes, prediction.shape[0], self.num_processes)
        predictions_list_numpy = np.split(prediction.detach().cpu().numpy().astype(np.float64), split_indices)
        targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)

        losses = []

        current_instance_index = 0
        for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
            predictions_cpu_batch, targets_cpu_batch = list(predictions_cpu_batch.squeeze(1)), list(targets_cpu_batch.squeeze(1))
            if not (all(a.data.contiguous for a in predictions_cpu_batch) and all(a.data.contiguous for a in targets_cpu_batch)):
                print("WARNING! Non-contiguous arrays encountered. Shape:", predictions_cpu_batch[0].shape)
                global ENCOUNTERED_NONCONTIGUOUS
                ENCOUNTERED_NONCONTIGUOUS = True
            predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
            targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

            # Official: betti_matching.compute_barcode(predictions + targets)
            barcodes_batch = compute_barcode(
                predictions_cpu_batch + targets_cpu_batch)
            barcodes_predictions, barcodes_targets = barcodes_batch[:len(barcodes_batch)//2], barcodes_batch[len(barcodes_batch)//2:]

            for barcode_prediction, barcode_target in zip(barcodes_predictions, barcodes_targets):
                losses.append(self._wasserstein_loss(prediction[current_instance_index].squeeze(0), target[current_instance_index].squeeze(0), barcode_prediction, barcode_target))
                current_instance_index += 1

        return losses

    def _wasserstein_loss(self, prediction, target,
                          barcode_result_prediction, barcode_result_target):
        (prediction_birth_coordinates, prediction_death_coordinates, target_birth_coordinates, target_death_coordinates) = (
            [torch.tensor(array, device=prediction.device, dtype=torch.long) if array.strides[-1] > 0 else torch.zeros(0, len(prediction.shape), device=prediction.device, dtype=torch.long)
            for array in [barcode_result_prediction.birth_coordinates, barcode_result_prediction.death_coordinates,
                            barcode_result_target.birth_coordinates, barcode_result_target.death_coordinates]])

        # (M, 2) tensor of persistence pairs for prediction
        prediction_pairs = torch.stack([
            prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [prediction_birth_coordinates, prediction_death_coordinates]
        ], dim=1)
        # (M, 2) tensor of persistence pairs for target
        target_pairs = torch.stack([
            target[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [target_birth_coordinates, target_death_coordinates]
        ], dim=1)

        losses_matched_by_dim = []
        losses_unmatched_by_dim = []

        for prediction_pairs_dim, target_pairs_dim in zip(
            torch.split(prediction_pairs, barcode_result_prediction.num_pairs_by_dim.tolist()),
            torch.split(target_pairs, barcode_result_target.num_pairs_by_dim.tolist())
        ):
            _, matching = wasserstein.wasserstein_distance(prediction_pairs_dim.detach().cpu(), target_pairs_dim.detach().cpu(),
                                                           matching=True, keep_essential_parts=False)
            matching = torch.tensor(matching.reshape(-1, 2), device=prediction.device, dtype=torch.long)

            matched_pairs = matching[(matching[:, 0] >= 0) & (matching[:, 1] >= 0)]
            loss_matched = ((prediction_pairs_dim[matched_pairs[:, 0]] - target_pairs_dim[matched_pairs[:, 1]])**2).sum()
            prediction_pairs_unmatched = prediction_pairs_dim[matching[matching[:, 1] == -1][:, 0]]
            target_pairs_unmatched = target_pairs_dim[matching[matching[:, 0] == -1][:, 1]]
            loss_unmatched = 0.5*(((prediction_pairs_unmatched[:, 0] - prediction_pairs_unmatched[:, 1])**2).sum()
                                  + ((target_pairs_unmatched[:, 0] - target_pairs_unmatched[:, 1])**2).sum())

            losses_matched_by_dim.append(loss_matched)
            losses_unmatched_by_dim.append(loss_unmatched)

        return (sum(losses_matched_by_dim) + sum(losses_unmatched_by_dim)).reshape(1)


class MulticlassWassersteinLoss(_Loss):
    """Official losses/hutopo.py, l. 83-128 (verbatim)."""

    def __init__(self,
                 filtration_type: FiltrationType = FiltrationType.SUPERLEVEL,
                 num_processes: int = 1,
                 convert_to_one_vs_rest: bool = True,
                 softmax: bool = False,
                 ignore_background: bool = False,
                 ) -> None:
        super().__init__()
        if not softmax and not convert_to_one_vs_rest:
            raise ValueError("If softmax is False, convert_to_one_vs_rest must be True")
        if softmax and convert_to_one_vs_rest:
            raise ValueError("If softmax is True, convert_to_one_vs_rest must be False. Softmax is already handled by one vs rest")

        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.ignore_background = ignore_background

        self.WassersteinLoss = WassersteinLoss(
            filtration_type=filtration_type,
            num_processes=num_processes,
        )

    def forward(self, prediction, target):
        if self.softmax:
            prediction = torch.softmax(prediction, dim=1)

        if self.convert_to_one_vs_rest:
            prediction = convert_to_one_vs_rest(prediction.clone())

        if self.ignore_background:
            prediction = prediction[:, 1:]
            target = target[:, 1:]

        # Flatten out channel dimension to treat each channel as a separate instance
        prediction = torch.flatten(prediction, start_dim=0, end_dim=1).unsqueeze(1)
        converted_target = torch.flatten(target, start_dim=0, end_dim=1).unsqueeze(1)

        # Compute Wasserstein loss
        wasserstein_loss, losses = self.WassersteinLoss(prediction, converted_target)

        return wasserstein_loss, losses


class MulticlassDiceWassersteinLoss(_Loss):
    """Official losses/hutopo.py, l. 20-81 (verbatim; the Dice part reuses
    the project's Multiclass_CLDice, see deviation 5 of the module
    docstring)."""

    def __init__(self,
                 filtration_type: FiltrationType = FiltrationType.SUPERLEVEL,
                 dice_type: DiceType = DiceType.CLDICE,
                 num_processes: int = 1,
                 convert_to_one_vs_rest: bool = False,
                 cldice_alpha: float = 0.5,
                 ignore_background: bool = False,
                 ) -> None:
        super().__init__()

        if dice_type == DiceType.DICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=not convert_to_one_vs_rest,
                include_background=True,
                smooth=1e-5,
                alpha=0.0,
                convert_to_one_vs_rest=convert_to_one_vs_rest,
                batch=True
            )
        elif dice_type == DiceType.CLDICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=not convert_to_one_vs_rest,
                include_background=True,
                smooth=1e-5,
                alpha=cldice_alpha,
                iter_=5,
                convert_to_one_vs_rest=convert_to_one_vs_rest,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")

        self.MulticlassWassersteinloss = MulticlassWassersteinLoss(
            filtration_type=filtration_type,
            num_processes=num_processes,
            convert_to_one_vs_rest=convert_to_one_vs_rest,
            softmax=not convert_to_one_vs_rest,
            ignore_background=ignore_background,
        )

    def forward(self, prediction, target, alpha: float = 0.5):
        # Compute multiclass Wasserstein losses
        if alpha > 0:
            wasserstein_loss, losses = self.MulticlassWassersteinloss(prediction, target)
            losses = {"single_matches": losses}
        else:
            wasserstein_loss = torch.zeros(1, device=prediction.device)
            losses = {}

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)

        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["wasserstein"] = alpha * wasserstein_loss.item()

        return dice_loss + alpha * wasserstein_loss, losses


class HutopoLossAdapter(nn.Module):
    """Registered adapter for the HuTopo loss: single-channel bridge +
    `resolution` evaluation + scalar return.

    The official loss is multi-class one-vs-rest. Our segmentation head is
    single channel, so the bridge builds the official multi-class form by
    stacking a zero background logit in front of ours: with the official
    default ``convert_to_one_vs_rest=False`` the loss applies a softmax over
    the channel dimension, and ``softmax([0, logits])[:, 1] ==
    sigmoid(logits)`` exactly (the bridge of ``core.losses.decl``). The
    target is one-hot for the same reason. The official computation
    downstream of that is unchanged.

    Args:
        resolution: side length the maps are bilinearly resized to before the
            pure-Python persistence engine runs (default 256 — see the
            performance note of the module docstring).
        alpha: weight of the Wasserstein term vs. the Dice term (official
            call-time default 0.5).
        **kwargs: forwarded to MulticlassDiceWassersteinLoss
            (filtration_type, dice_type, num_processes, convert_to_one_vs_rest,
            cldice_alpha, ignore_background).
    """

    def __init__(self, resolution=256, alpha=0.5, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.alpha = alpha
        self.loss = MulticlassDiceWassersteinLoss(**kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: [B, 1, H, W] raw logits; target: [B, 1, H, W] in [0, 1]."""
        if self.resolution is not None and logits.shape[-1] != self.resolution:
            logits = F.interpolate(logits, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
        two_ch_logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        two_ch_target = torch.cat([1.0 - target, target], dim=1)
        loss, _dic = self.loss(two_ch_logits, two_ch_target, alpha=self.alpha)
        return loss


if _HAS_GUDHI:
    register('hutopo')(HutopoLossAdapter)
