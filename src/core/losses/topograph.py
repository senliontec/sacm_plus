"""Topograph losses & metric — strict vendor of the official repository
("Topograph: An efficient graph-based framework for strictly topology
preserving image segmentation").

Vendored verbatim from Topograph/losses/topograph.py, losses/utils.py
(loss-relevant parts), losses/dice_losses.py, losses/exact_topograph.py
and metrics/topograph.py (core, without the monai wrapper). Documented
deviations:

1. The official module imports a compiled C++ extension (`import
   Topograph`) for the relabel-mask creation (use_c=True). We vendor the
   PURE-PYTHON path only (use_c=False); the C import is guarded, and
   `create_relabel_masks_c` raises if the extension is unavailable.

2. The official losses consume [B, C, H, W] multi-class logits and a
   one-hot target. The adapters bridge single-channel logits via
   cat([zeros, logits]) (softmax foreground == sigmoid(logits) exactly)
   and evaluate at `resolution` (256 by default — the per-image networkx
   graph construction is CPU-heavy; the official code trains on small
   patches).

3. `fill_adj_matr` official body uses starred subscripts
   (`adj_matrix[*h_edges] = True`), which need Python 3.11. The vendored
   copy writes `adj_matrix[tuple(h_edges)] = True` — for numpy the two
   forms are identical — so the project's Python >= 3.10 floor holds.

Requires: networkx, scipy (topograph/dice_topograph/topograph_error);
cv2 (exact_topograph). Python >= 3.10.
"""

import enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from scipy.ndimage import label as scipy_label
from scipy.cluster.hierarchy import DisjointSet

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import Topograph as _TopographC  # noqa: F401
    _HAS_C_EXTENSION = True
except ImportError:
    _HAS_C_EXTENSION = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

from .registry import register
from .centerline_ce import _soft_skel_clce


# ---------------------------------------------------------------------------
# Verbatim from losses/utils.py (loss-relevant parts only)
# ---------------------------------------------------------------------------

class AggregationType(enum.Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"


class ThresholdDistribution(enum.Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"


def new_compute_diffs(paired_img_batch: torch.Tensor):
    h_diff = paired_img_batch[:,:-1, :] - paired_img_batch[:,1:, :]
    v_diff = paired_img_batch[:,:, :-1] - paired_img_batch[:,:, 1:]
    h_diff = h_diff != 0
    v_diff = v_diff != 0
    return h_diff, v_diff


def new_compute_diag_diffs(paired_img_batch: torch.Tensor, th: int = 11):
    weight = torch.tensor([[1, -1], [-1, 1]], device=paired_img_batch.device).unsqueeze(0).unsqueeze(0)
    diag_connections = F.conv2d(paired_img_batch.unsqueeze(1).float(), weight.float()).squeeze(1)
    diagr = diag_connections > th
    diagl = diag_connections < -th
    special_case_r = torch.logical_or(diag_connections == 7, diag_connections == 4)
    special_case_l = torch.logical_or(diag_connections == -7, diag_connections == -4)
    return diagr, diagl, special_case_r, special_case_l


def fill_adj_matr(adj_matrix, h_edges, v_edges):
    adj_matrix[tuple(h_edges)] = True
    adj_matrix[tuple(h_edges[::-1])] = True
    adj_matrix[tuple(v_edges)] = True
    adj_matrix[tuple(v_edges[::-1])] = True
    np.fill_diagonal(adj_matrix, False)
    return adj_matrix


def compute_diffs_2d(labelled_components):
    """Verbatim from Topograph/losses/utils.py compute_diffs (renamed)."""
    h_diff = labelled_components[:-1, :] - labelled_components[1:, :]
    v_diff = labelled_components[:, :-1] - labelled_components[:, 1:]
    return h_diff, v_diff


def smoothing_5x5(preds, gts):
    """Verbatim from Topograph/losses/utils.py (smoothing_5x5)."""
    intersection_filter = torch.ones((9,9), device=preds.device)
    rest_filter = torch.ones((7,7), device=preds.device)

    rest_filter = F.pad(rest_filter, (1,1,1,1), value=0)

    intersection = torch.logical_and(preds, gts)
    pred_fg = torch.logical_and(preds, ~intersection)
    gt_fg = torch.logical_and(gts, ~intersection)

    smooth_intersect = F.conv_transpose2d(intersection.unsqueeze(1).float(), intersection_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_intersect[smooth_intersect > 0] = 3

    smooth_pred = F.conv_transpose2d(pred_fg.unsqueeze(1).float(), rest_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_pred[smooth_pred > 0] = 1

    smooth_gt = F.conv_transpose2d(gt_fg.unsqueeze(1).float(), rest_filter.unsqueeze(0).unsqueeze(0), stride=5, padding=2)[:,0]
    smooth_gt[smooth_gt > 0] = 2

    smooth_img = smooth_pred + smooth_gt + smooth_intersect
    smooth_img[smooth_img > 3] = 3

    return smooth_img


# ---------------------------------------------------------------------------
# Verbatim from losses/topograph.py (pure-Python path)
# ---------------------------------------------------------------------------

def reverse_pairing(pairing: int) -> tuple[int, int]:
    match pairing:
        case 0: return 0, 0
        case 1: return 1, 0
        case 2: return 0, 1
        case 3: return 1, 1
        case _: return -1, -1


def label_regions(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paired_img = (pred + 2 * gt)
    masked_imgs = np.eye(4)[paired_img].transpose(2, 0, 1).astype(np.int32)

    cc_result = map(scipy_label, masked_imgs)

    all_labels = np.zeros(pred.shape, dtype=np.int32)
    label_counter = 0

    gt_labels = []
    pred_labels = []

    for inters_class, (labeled_regions, num_nodes) in enumerate(cc_result):
        all_labels += labeled_regions + (masked_imgs[inters_class] * label_counter)
        label_counter += num_nodes

        pred_class, gt_class = reverse_pairing(inters_class)

        pred_labels.append(np.zeros((num_nodes)) + pred_class)
        gt_labels.append(np.zeros((num_nodes)) + gt_class)

    pred_labels = np.concatenate(pred_labels)
    gt_labels = np.concatenate(gt_labels)

    if all_labels.max() > 0:
        all_labels -= 1

    return all_labels, pred_labels, gt_labels


def rag(labelled_regions, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl):
    max_label = labelled_regions.max()

    if max_label == 0:
        edges = np.empty((2, 0))
    else:
        h_edges = np.stack([labelled_regions[1:, :][h_diff], labelled_regions[:-1, :][h_diff]])
        v_edges = np.stack([labelled_regions[:, 1:][v_diff], labelled_regions[:, :-1][v_diff]])

        adj = np.zeros((max_label+1, max_label+1), dtype=bool)
        special_adj = np.zeros((max_label+1, max_label+1), dtype=bool)
        adj = fill_adj_matr(adj, h_edges, v_edges)

        dr_edges = np.stack([labelled_regions[:-1, :-1][diagr], labelled_regions[1:, 1:][diagr]])
        dl_edges = np.stack([labelled_regions[:-1, 1:][diagl], labelled_regions[1:, :-1][diagl]])
        special_dr_edges = np.stack([labelled_regions[:-1, :-1][special_diagr], labelled_regions[1:, 1:][special_diagr]])
        special_dl_edges = np.stack([labelled_regions[:-1, 1:][special_diagl], labelled_regions[1:, :-1][special_diagl]])
        adj = fill_adj_matr(adj, dr_edges, dl_edges)
        special_adj = fill_adj_matr(special_adj, special_dr_edges, special_dl_edges)

        edges = np.stack(np.nonzero(adj))
        special_edges = np.stack(np.nonzero(special_adj))

    return edges, special_edges


def contract_graph(graph):
    same_nodes = DisjointSet(graph.nodes)

    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue

        cur_node_cluster = same_nodes[node]

        for neighbor in graph[node]:
            if neighbor < node or graph[node][neighbor].get('special', False):
                continue
            if graph.nodes[neighbor]['predicted_classes'] == graph.nodes[node]['predicted_classes'] and graph.nodes[neighbor]['gt_classes'] == graph.nodes[node]['gt_classes']:
                nbr_cluster = same_nodes[neighbor]

                if nbr_cluster != cur_node_cluster:
                    same_nodes.merge(cur_node_cluster, nbr_cluster)

    for cluster in same_nodes.subsets():
        if len(cluster) == 1:
            continue

        first_node = cluster.pop()

        graph.nodes[first_node]['contracted_nodes'] = cluster

        for node in cluster:
            nx.contracted_nodes(graph, first_node, node, self_loops=False, copy=False)

    return graph


def identify_clusters(graph):
    pred_cluster = DisjointSet(graph.nodes)
    gt_cluster = DisjointSet(graph.nodes)

    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue

        cur_pred_cluster = pred_cluster[node]
        cur_gt_cluster = gt_cluster[node]

        for neighbor in graph[node]:
            if neighbor < node:
                continue
            if graph.nodes[neighbor]['predicted_classes'] == 1 and graph.nodes[node]['predicted_classes'] == 1:
                pred_nbr_cluster = pred_cluster[neighbor]

                if pred_nbr_cluster != cur_pred_cluster:
                    pred_cluster.merge(cur_pred_cluster, pred_nbr_cluster)

            if graph.nodes[neighbor]['gt_classes'] == 1 and graph.nodes[node]['gt_classes'] == 1:
                gt_nbr_cluster = gt_cluster[neighbor]

                if gt_nbr_cluster != cur_gt_cluster:
                    gt_cluster.merge(cur_gt_cluster, gt_nbr_cluster)

    for cluster in pred_cluster.subsets():
        node = cluster.pop()
        root = pred_cluster[node]
        graph.nodes[node]['pred_cluster'] = root

        for node in cluster:
            graph.nodes[node]['pred_cluster'] = root

    for cluster in gt_cluster.subsets():
        node = cluster.pop()
        root = gt_cluster[node]
        graph.nodes[node]['gt_cluster'] = root

        for node in cluster:
            graph.nodes[node]['gt_cluster'] = root

    return graph


def create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl):
    labelled_regions, predicted_classes, gt_classes = label_regions(argmax_pred, argmax_gt)

    if labelled_regions.max() == 0:
        graph = nx.Graph()
        graph.add_node(0)
        edge_index = torch.tensor([[],[]])
        special_edge_index = torch.tensor([[],[]])
    else:
        edge_index, special_edge_index = rag(labelled_regions, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl)

    graph = nx.Graph()
    graph.add_edges_from(edge_index.T)
    graph.add_edges_from(special_edge_index.T, special=True)

    for node in graph.nodes:
        graph.nodes[node]['predicted_classes'] = predicted_classes[node]
        graph.nodes[node]['gt_classes'] = gt_classes[node]

    graph.graph['predicted_classes'] = predicted_classes

    graph = contract_graph(graph)

    graph = identify_clusters(graph)

    return graph, labelled_regions


def get_critical_nodes(graph):
    critical_nodes = []
    cluster_lengths = []

    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])

        fg_nbr_clusters = set()
        correct_bg_nbrs_count = 0
        counter_class_str = "gt_cluster" if graph.nodes[node]['predicted_classes'] == 1 else "pred_cluster"

        for nbr in all_nbrs:
            if graph[node][nbr].get('special', False):
                continue

            nbr_gt_class = graph.nodes[nbr]['gt_classes']

            if nbr_gt_class == 0 and graph.nodes[nbr]['predicted_classes'] == 0:
                correct_bg_nbrs_count += 1
                if correct_bg_nbrs_count > 1:
                    break
            else:
                fg_nbr_clusters.add(graph.nodes[nbr][counter_class_str])

        if correct_bg_nbrs_count != 1 or len(fg_nbr_clusters) != 1:
            critical_nodes.append(node)
            if "contracted_nodes" in graph.nodes[node]:
                critical_nodes += graph.nodes[node]["contracted_nodes"]
                cluster_lengths.append(len(graph.nodes[node]["contracted_nodes"]) + 1)
            else:
                cluster_lengths.append(1)
            continue

    return critical_nodes, cluster_lengths


def get_critical_nbrs(graph):
    """Verbatim from losses/topograph.py get_critical_nbrs (including the
    official 'gt_visisted' attribute typo)."""
    error_count = 0

    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])

        fg_nbr_clusters = set()
        bg_nbrs = set()
        counter_class_str = "gt_cluster" if graph.nodes[node]['predicted_classes'] == 1 else "pred_cluster"
        class_str = "gt_visisted" if graph.nodes[node]['predicted_classes'] == 1 else "pred_visited"

        for nbr in all_nbrs:
            if graph[node][nbr].get('special', False):
                continue

            nbr_gt_class = graph.nodes[nbr]['gt_classes']

            if nbr_gt_class == 0 and graph.nodes[nbr]['predicted_classes'] == 0:
                bg_nbrs.add(nbr)
            else:
                fg_nbr_clusters.add(graph.nodes[nbr][counter_class_str])

        if len(bg_nbrs) == 1 and len(fg_nbr_clusters) == 1:
            continue

        if len(bg_nbrs) == 0:
            error_count += 1
        elif len(bg_nbrs) > 1:
            seen_nodes = 0
            for error_node in bg_nbrs:
                if not class_str in graph.nodes[error_node]:
                    graph.nodes[error_node][class_str] = True
                else:
                    seen_nodes += 1

            error_count += len(bg_nbrs) - max(seen_nodes, 1)

        if len(fg_nbr_clusters) == 0:
            error_count += 1
        elif len(fg_nbr_clusters) > 1:
            seen_nodes = 0
            for error_node in fg_nbr_clusters:
                if not class_str in graph.nodes[error_node]:
                    graph.nodes[error_node][class_str] = True
                else:
                    seen_nodes += 1

            error_count += len(fg_nbr_clusters) - max(seen_nodes, 1)

    return error_count


def create_relabel_masks(critical_node_list, cluster_lengths, all_labels):
    region_error_infos = []
    remaining_nodes_in_cluster = 0
    i = 0
    cluster_counter = -1

    while i < len(critical_node_list):
        cluster_counter += 1
        node_set = [critical_node_list[i]]
        i += 1
        remaining_nodes_in_cluster = cluster_lengths[cluster_counter] - 1

        while remaining_nodes_in_cluster > 0:
            node_set.append(critical_node_list[i])
            i += 1
            remaining_nodes_in_cluster -= 1

        relabel_mask = np.isin(all_labels, node_set)

        index_relabel_mask = np.nonzero(relabel_mask)

        region_error_infos.append(index_relabel_mask)

    return region_error_infos


def create_relabel_masks_c(critical_node_list, cluster_lengths, all_labels):
    if not _HAS_C_EXTENSION:
        raise RuntimeError(
            "Topograph C++ extension not available; use use_c=False "
            "(pure-Python path, identical loss computation)."
        )
    all_labels = np.asfortranarray(all_labels).astype(np.int32)
    critical_nodes = np.asfortranarray(critical_node_list).astype(np.int32)
    cluster_lengths = np.asfortranarray(cluster_lengths).astype(np.int32)

    relabel_indices = _TopographC.get_relabel_indices(all_labels, critical_nodes, cluster_lengths)

    return relabel_indices


def _single_sample_class_loss(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl, sample_no, use_c=False):
    graph, labelled_regions = create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl)

    critical_nodes, cluster_lengths = get_critical_nodes(graph)

    if use_c:
        error_region_infos = create_relabel_masks_c(critical_nodes, cluster_lengths, labelled_regions)
    else:
        error_region_infos = create_relabel_masks(critical_nodes, cluster_lengths, labelled_regions)

    return error_region_infos, sample_no


def single_sample_class_loss(args: dict):
    return _single_sample_class_loss(**args)


def _single_sample_class_metric_exact(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl, sample_no):
    graph, labelled_regions = create_graph(argmax_pred, argmax_gt, h_diff, v_diff, diagr, diagl, special_diagr, special_diagl)

    error_count = get_critical_nbrs(graph)

    return error_count, sample_no


def single_sample_class_metric_exact(args: dict):
    return _single_sample_class_metric_exact(**args)


class TopographLoss(_Loss):
    def __init__(self,
                 softmax=True,
                 num_processes=1,
                 include_background=True,
                 use_c=False,
                 sphere=False,
                 eight_connectivity=True,
                 aggregation=AggregationType.MEAN,
                 thres_distr=ThresholdDistribution.NONE,
                 thres_var=0.0,
        ):
        super(TopographLoss, self).__init__()
        self.softmax = softmax
        self.num_processes = num_processes
        self.include_background = include_background
        self.use_c = use_c
        self.sphere = sphere
        self.eight_connectivity = eight_connectivity
        self.thres_distr = thres_distr
        self.thres_var = thres_var
        self.aggregation = aggregation

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        num_classes = input.shape[1]

        if self.softmax:
            input = F.softmax(input, dim=1)

        num_classes = input.shape[1]

        single_calc_inputs = []
        relabel_masks = []
        skip_index = 0 if self.include_background else 1

        if self.thres_distr != ThresholdDistribution.NONE:
            match self.thres_distr:
                case ThresholdDistribution.UNIFORM:
                    thres_noise = torch.rand(size=[input.shape[0], 1, 1], requires_grad=False, device=input.device) * (self.thres_var / (num_classes - 1))
                case ThresholdDistribution.GAUSSIAN:
                    thres_noise = torch.randn(size=[input.shape[0], 1, 1], requires_grad=False, device=input.device) * (self.thres_var / (num_classes - 1))

            input_detached = input.detach().clone()

            noise_class = torch.randint(0, num_classes, (input.shape[0],), device=input.device)

            neg_noise = (thres_noise / (num_classes - 1))

            input_detached[:, noise_class] += thres_noise + neg_noise

            input_detached -= neg_noise.unsqueeze(1)

            modified_input = input_detached
        else:
            modified_input = input

        argmax_preds = torch.argmax(modified_input, dim=1)
        argmax_gts = torch.argmax(target, dim=1)

        if self.sphere:
            argmax_preds = F.pad(argmax_preds, (1, 1, 1, 1), value=0)
            argmax_gts = F.pad(argmax_gts, (1, 1, 1, 1), value=0)

        for class_index in range(skip_index, num_classes):
            bin_preds = torch.zeros_like(argmax_preds)
            bin_gts = torch.zeros_like(argmax_gts)
            bin_preds[argmax_preds == class_index] = 1
            bin_gts[argmax_gts == class_index] = 1

            paired_imgs = bin_preds + 2 * bin_gts

            diag_val_1, diag_val_2 = (-4, 16) if self.eight_connectivity else (16, -4)

            paired_imgs[paired_imgs==0] = diag_val_1
            paired_imgs[paired_imgs==3] = diag_val_2

            h_diff, v_diff = new_compute_diffs(paired_imgs)
            diagr, diagl, special_diag_r, special_diag_l = new_compute_diag_diffs(paired_imgs, th=7)

            bin_preds = bin_preds.cpu().numpy()
            bin_gts = bin_gts.cpu().numpy()
            h_diff = h_diff.cpu().numpy()
            v_diff = v_diff.cpu().numpy()
            diagr = diagr.cpu().numpy()
            diagl = diagl.cpu().numpy()
            special_diag_r = special_diag_r.cpu().numpy()
            special_diag_l = special_diag_l.cpu().numpy()

            for i in range(input.shape[0]):
                single_calc_input = {
                    "argmax_pred": bin_preds[i],
                    "argmax_gt": bin_gts[i],
                    "h_diff": h_diff[i],
                    "v_diff": v_diff[i],
                    "diagr": diagr[i],
                    "diagl": diagl[i],
                    "special_diagr": special_diag_r[i],
                    "special_diagl": special_diag_l[i],
                    "sample_no": i,
                    "use_c": self.use_c,
                }
                single_calc_inputs.append(single_calc_input)

        relabel_masks = []

        relabel_masks = map(single_sample_class_loss, single_calc_inputs)

        g_loss = torch.tensor(0.0, device=input.device)

        for region_error_infos, sample_no in relabel_masks:
            for region_indices in region_error_infos:
                if self.sphere:
                    region_indices = torch.tensor(region_indices)
                    region_indices -= 1

                if self.aggregation != AggregationType.CE:
                    class_indices = argmax_preds[sample_no, region_indices[0], region_indices[1]]
                    nominator = input[sample_no,class_indices,region_indices[0], region_indices[1]]

                match self.aggregation:
                    case AggregationType.MEAN:
                        g_loss += nominator.mean()
                    case AggregationType.RMS:
                        g_loss += torch.sqrt((nominator**2).mean())
                    case AggregationType.SUM:
                        g_loss += nominator.sum()
                    case AggregationType.MAX:
                        g_loss += nominator.max()
                    case AggregationType.MIN:
                        g_loss += nominator.min()
                    case AggregationType.CE:
                        masked_input = input[sample_no, :, region_indices[0], region_indices[1]].unsqueeze(0)
                        masked_target = target[sample_no, :, region_indices[0], region_indices[1]].unsqueeze(0)
                        g_loss += F.cross_entropy(masked_input, masked_target, reduction='mean')
                    case _:
                        raise ValueError(f"Invalid aggregation type: {self.aggregation}")

        g_loss /= (input.shape[0] * (num_classes - skip_index))

        return g_loss


class TopographLossAdapter(nn.Module):
    """Registered adapter: single-channel bridge, 256 resolution, scalar
    return. Requires networkx + scipy + Python >= 3.10."""

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.loss = TopographLoss(softmax=True, num_processes=1,
                                  include_background=True, use_c=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resolution is not None and logits.shape[-1] != self.resolution:
            logits = F.interpolate(logits, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
        two_ch_logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        one_hot_target = torch.cat([1.0 - target, target], dim=1)
        return self.loss(two_ch_logits, one_hot_target)


class DiceType(enum.Enum):
    DICE = "dice"
    CLDICE = "cldice"


def convert_to_one_vs_rest(prediction):
    """Verbatim from Topograph/losses/utils.py (one-vs-max strategy)."""
    converted_prediction = torch.zeros_like(prediction)

    for channel in range(prediction.shape[1]):
        channel_logits = prediction[:, channel].unsqueeze(1)

        rest_logits = torch.max(prediction[:, torch.arange(prediction.shape[1]) != channel], dim=1).values.unsqueeze(1)

        converted_prediction[:, channel] = torch.softmax(torch.cat([rest_logits, channel_logits], dim=1), dim=1)[:, 1]

    return converted_prediction


class Multiclass_CLDice(nn.Module):
    """Verbatim from Topograph/losses/dice_losses.py. soft_skel is the
    verified official clDice skeleton (their utils.soft_skel is the same
    algorithm). The unreachable tail after the first return is dropped."""

    def __init__(self, weights=[], iter_=3, alpha=0.5, smooth=1e-5, sigmoid=False, softmax=False, include_background=False, convert_to_one_vs_rest=False, batch=False):
        super(Multiclass_CLDice, self).__init__()
        if int(sigmoid) + int(softmax) + int(convert_to_one_vs_rest) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, convert_to_one_vs_rest=True].")

        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.batch = batch
        self.weights = torch.tensor(weights)

    def forward(self, input, target):
        if (len(self.weights) > 0 and (
            (self.include_background and len(self.weights) != input.shape[1]) or
            (not self.include_background and len(self.weights) != input.shape[1] - 1))
        ):
            raise ValueError(f"Number of class weights ({len(self.weights)}) must match the number of classes ({input.shape[1]}).")
        elif len(self.weights) > 0:
            self.weights = self.weights.to(input.device)
            if self.batch:
                weights = self.weights.unsqueeze(0)
                weights = weights.expand(input.shape[0], -1)

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                raise ValueError("softmax=True, but the number of channels for the prediction is 1.")
            else:
                input = torch.softmax(input, 1)

        if not self.include_background and n_pred_ch == 1:
            raise ValueError("single channel prediction, `include_background=False` is not a valid combination.")

        if self.convert_to_one_vs_rest:
            input = convert_to_one_vs_rest(input)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        dic = {}

        if self.alpha > 0:
            starting_class = 1 if self.include_background else 0

            pred_skeletons = _soft_skel_clce(input[:, starting_class:].float(), self.iter)
            target_skeletons = _soft_skel_clce(target[:, starting_class:].float(), self.iter)

            tprec = (torch.sum(torch.multiply(pred_skeletons, target[:, starting_class:]), dim=reduce_axis)+self.smooth)/(torch.sum(pred_skeletons, dim=reduce_axis)+self.smooth)
            tsens = (torch.sum(torch.multiply(target_skeletons, input[:, starting_class:]), dim=reduce_axis)+self.smooth)/(torch.sum(target_skeletons, dim=reduce_axis)+self.smooth)
            cl_dice = torch.mean(1.- 2.0*(tprec*tsens)/(tprec+tsens))

            if len(self.weights) > 0:
                cl_dice = torch.multiply(cl_dice, weights[starting_class:])
        else:
            cl_dice = torch.zeros(size=[1], device=input.device)

        intersection = torch.sum(target * input, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o

        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        if len(self.weights) > 0:
            dice = torch.multiply(dice, weights)

        dice = torch.mean(dice)
        cl_dice = torch.mean(cl_dice)

        loss = (1 - self.alpha) * dice + self.alpha * cl_dice

        dic = {}
        dic['dice'] = (1 - self.alpha) * dice
        dic['cldice'] = self.alpha * cl_dice
        return loss, dic


class DiceTopographLoss(nn.Module):
    """Verbatim from Topograph/losses/topograph.py (DiceTopographLoss).
    Combines Multiclass_CLDice with TopographLoss; returns (loss, dict)."""

    def __init__(self,
                 softmax: bool = True,
                 dice_type: DiceType = DiceType.CLDICE,
                 num_processes: int = 1,
                 cldice_alpha: float = 0.5,
                 include_background: bool = True,
                 use_c=False,
                 sphere=False,
                 eight_connectivity=True,
                 aggregation=AggregationType.MEAN,
                 thres_distr=ThresholdDistribution.NONE,
                 thres_var=0.0) -> None:
        super().__init__()
        if dice_type == DiceType.DICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=softmax,
                include_background=True,
                smooth=1e-5,
                alpha=0.0,
                convert_to_one_vs_rest=False,
                batch=True,
            )
        elif dice_type == DiceType.CLDICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=softmax,
                include_background=include_background,
                smooth=1e-5,
                alpha=cldice_alpha,
                iter_=5,
                convert_to_one_vs_rest=False,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")

        self.TopographLoss = TopographLoss(
            softmax=softmax,
            num_processes=num_processes,
            include_background=include_background,
            use_c=use_c,
            sphere=sphere,
            eight_connectivity=eight_connectivity,
            aggregation=aggregation,
            thres_var=thres_var,
            thres_distr=thres_distr
        )

    def forward(self,
                prediction,
                target,
                alpha: float = 0.5
                ) -> tuple:
        losses = {}
        if alpha > 0:
            topograph_loss = self.TopographLoss(prediction, target)
        else:
            topograph_loss = torch.zeros(1, device=prediction.device)

        dice_loss, dic = self.DiceLoss(prediction, target)

        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["topograph_loss"] = alpha * topograph_loss

        return dice_loss + alpha * topograph_loss, losses


class DiceTopographLossAdapter(nn.Module):
    """Registered adapter: single-channel bridge (softmax fg == sigmoid),
    one-hot target, 256-resolution evaluation, scalar return."""

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.loss = DiceTopographLoss(softmax=True, use_c=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resolution is not None and logits.shape[-1] != self.resolution:
            logits = F.interpolate(logits, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
        two_ch_logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        one_hot_target = torch.cat([1.0 - target, target], dim=1)
        loss, _dic = self.loss(two_ch_logits, one_hot_target)
        return loss


# ---------------------------------------------------------------------------
# exact_topograph.py (verbatim; renamed helpers to avoid collisions)
# ---------------------------------------------------------------------------

def exact_simple_rag(labelled_components: np.ndarray):
    """Verbatim from exact_topograph.py simple_rag (renamed)."""
    max_label = labelled_components.max()
    max_label = int(max_label)

    if max_label == 0:
        edges = np.empty((2, 0))
    else:
        h_diff, v_diff = compute_diffs_2d(labelled_components)

        h_edges = np.stack([labelled_components[1:, :][h_diff != 0], labelled_components[:-1, :][h_diff != 0]])
        v_edges = np.stack([labelled_components[:, 1:][v_diff != 0], labelled_components[:, :-1][v_diff != 0]])

        adj = np.zeros((max_label+1, max_label+1), dtype=bool)

        h_edges = h_edges.astype(int)
        v_edges = v_edges.astype(int)

        adj = fill_adj_matr(adj, h_edges, v_edges)

        edges = np.stack(np.nonzero(adj))

    return edges


def exact_get_critical_nodes(graph):
    """Verbatim from exact_topograph.py get_critical_nodes (renamed)."""
    critical_nodes = []
    cluster_lengths = []
    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue

        all_nbrs = list(graph[node])
        correct_predicted_neighbors = []
        for nbr in all_nbrs:
            nbr_gt_class = graph.nodes[nbr]['gt_classes']
            if graph.nodes[nbr]['predicted_classes'] == nbr_gt_class:
                correct_predicted_neighbors.append((nbr, nbr_gt_class))

        if len(correct_predicted_neighbors) != 2 or correct_predicted_neighbors[0][1] == correct_predicted_neighbors[1][1]:
            critical_nodes.append(node)
            cluster_lengths.append(1)
            continue

    return critical_nodes, cluster_lengths


def exact_label_regions_paired_img(paired_img):
    """Verbatim from exact_topograph.py label_regions_paired_img
    (cv2.connectedComponents, 4-connectivity)."""
    labelled_regions = np.zeros_like(paired_img, dtype="uint16")

    intersection = paired_img == 3
    FF_num_labels, FF_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    labelled_regions[FF_regions > 0] = FF_regions[FF_regions > 0]

    intersection = paired_img == 0
    BB_num_labels, BB_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    labelled_regions[BB_regions > 0] = BB_regions[BB_regions > 0] + FF_num_labels

    intersection = paired_img == 2
    FB_num_labels, FB_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    labelled_regions[FB_regions > 0] = FB_regions[FB_regions > 0] + FF_num_labels + BB_num_labels

    intersection = paired_img == 1
    BF_num_labels, BF_regions = cv2.connectedComponents(intersection.astype("uint8"), connectivity=4)
    labelled_regions[BF_regions > 0] = BF_regions[BF_regions > 0] + FF_num_labels + BB_num_labels + FB_num_labels

    predicted_classes = np.zeros(FF_num_labels + BB_num_labels + FB_num_labels + BF_num_labels)
    gt_classes = np.zeros(FF_num_labels + BB_num_labels + FB_num_labels + BF_num_labels)

    predicted_classes[:FF_num_labels] = 1
    gt_classes[:FF_num_labels] = 1

    predicted_classes[FF_num_labels:FF_num_labels + BB_num_labels] = 0
    gt_classes[FF_num_labels:FF_num_labels + BB_num_labels] = 0

    predicted_classes[FF_num_labels + BB_num_labels:FF_num_labels + BB_num_labels + FB_num_labels] = 0
    gt_classes[FF_num_labels + BB_num_labels:FF_num_labels + BB_num_labels + FB_num_labels] = 1

    predicted_classes[FF_num_labels + BB_num_labels + FB_num_labels:] = 1
    gt_classes[FF_num_labels + BB_num_labels + FB_num_labels:] = 0

    return labelled_regions, predicted_classes, gt_classes


def exact_create_graph(paired_img):
    """Verbatim from exact_topograph.py create_graph (renamed)."""
    labelled_region, predicted_classes, gt_classes = exact_label_regions_paired_img(paired_img)

    edge_index = exact_simple_rag(labelled_region)

    graph = nx.Graph()
    graph.add_edges_from(edge_index.T)

    graph.graph["predicted_classes"] = predicted_classes[1:]
    for node in graph.nodes:
        graph.nodes[node]['predicted_classes'] = predicted_classes[node-1]
        graph.nodes[node]['gt_classes'] = gt_classes[node-1]

    return graph, labelled_region


def _single_sample_loss_exact(paired_img, sample_no, use_c=False):
    binary_graph, labelled_regions = exact_create_graph(paired_img)
    critical_node_list, cluster_lengths = exact_get_critical_nodes(binary_graph)
    filter = torch.zeros((5,5))
    filter[filter.shape[0]//2,filter.shape[0]//2] = 1
    original_region_labels = F.conv2d(torch.tensor(labelled_regions).unsqueeze(0).unsqueeze(0).float(), filter.unsqueeze(0).unsqueeze(0).float(), stride=filter.shape[0]).squeeze(0).squeeze(0).int().numpy()
    if use_c:
        relabel_indices = create_relabel_masks_c(critical_node_list, cluster_lengths, original_region_labels)
    else:
        relabel_indices = create_relabel_masks(critical_node_list, cluster_lengths, original_region_labels)
    return relabel_indices, sample_no


def single_sample_loss_exact(args: dict):
    return _single_sample_loss_exact(**args)


class ExactTopographLoss(_Loss):
    """Verbatim from exact_topograph.py (ExactTopographLoss, use_c=False
    path). Depends on cv2 (4-connectivity connected components) and the
    smoothing_5x5 thickening. Requires networkx + cv2."""

    def __init__(self, softmax=True, num_processes=1, include_background=True, use_c=False, sphere=False):
        super(ExactTopographLoss, self).__init__()
        self.softmax = softmax
        self.num_processes = num_processes
        self.include_background = include_background
        self.use_c = use_c
        self.sphere = sphere

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        num_classes = input.shape[1]

        if self.softmax:
            input = F.softmax(input, dim=1)

        num_classes = input.shape[1]

        argmax_preds = torch.argmax(input.clone().detach(), dim=1)
        argmax_gts = torch.argmax(target.clone().detach(), dim=1)

        if self.sphere:
            argmax_preds = F.pad(argmax_preds, (1, 1, 1, 1), value=0)
            argmax_gts = F.pad(argmax_gts, (1, 1, 1, 1), value=0)

        single_calc_inputs = []
        relabel_masks = []
        skip_index = 0 if self.include_background else 1

        for class_index in range(skip_index, num_classes):
            bin_preds = torch.zeros_like(argmax_preds)
            bin_gts = torch.zeros_like(argmax_gts)
            bin_preds[argmax_preds == class_index] = 1
            bin_gts[argmax_gts == class_index] = 1
            smooth_paired_imgs = smoothing_5x5(bin_preds, bin_gts).cpu().numpy()

            del bin_preds
            del bin_gts

            for i in range(input.shape[0]):
                single_calc_input = {
                    "paired_img": smooth_paired_imgs[i],
                    "sample_no": i,
                    "use_c": self.use_c,
                }
                single_calc_inputs.append(single_calc_input)

        relabel_masks = []

        relabel_masks = map(single_sample_loss_exact, single_calc_inputs)

        g_loss = torch.tensor(0.0, device=input.device)
        for region_error_infos, sample_no in relabel_masks:
            for region_indices in region_error_infos:
                class_indices = argmax_preds[sample_no, region_indices[0], region_indices[1]]
                if self.sphere:
                    region_indices = torch.tensor(region_indices)
                    region_indices -= 1

                nominator = input[sample_no,class_indices,region_indices[0], region_indices[1]]
                num_pixels = len(class_indices)
                g_loss += (nominator / num_pixels).sum()

        g_loss /= (input.shape[0] * (num_classes - skip_index))

        return g_loss


class ExactTopographLossAdapter(nn.Module):
    """Registered adapter: single-channel bridge, 256 resolution, scalar
    return. Requires networkx + cv2; use_c=False (pure-Python path)."""

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.loss = ExactTopographLoss(softmax=True, use_c=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resolution is not None and logits.shape[-1] != self.resolution:
            logits = F.interpolate(logits, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
        two_ch_logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        one_hot_target = torch.cat([1.0 - target, target], dim=1)
        return self.loss(two_ch_logits, one_hot_target)


# ---------------------------------------------------------------------------
# Topograph metric (metrics/topograph.py core, without the monai wrapper)
# ---------------------------------------------------------------------------

def topograph_error_metric(pred_bin, gt_bin, eight_connectivity=True):
    """Topograph topological error count for a single binary pair
    (2D numpy masks). Mirrors TopographMetric._compute_tensor for one
    sample/one class; the monai accumulation wrapper is replaced by a
    plain call — the computation itself is the official code.
    """
    paired_imgs = (pred_bin.astype(np.int64) + 2 * gt_bin.astype(np.int64))
    diag_val_1, diag_val_2 = (-4, 16) if eight_connectivity else (16, -4)

    paired_imgs[paired_imgs == 0] = diag_val_1
    paired_imgs[paired_imgs == 3] = diag_val_2

    paired_t = torch.tensor(paired_imgs).unsqueeze(0)
    h_diff, v_diff = new_compute_diffs(paired_t)
    diagr, diagl, special_diag_r, special_diag_l = new_compute_diag_diffs(paired_t, th=7)

    args = {
        "argmax_pred": pred_bin.astype(np.int64),
        "argmax_gt": gt_bin.astype(np.int64),
        "h_diff": h_diff[0].numpy(),
        "v_diff": v_diff[0].numpy(),
        "diagr": diagr[0].numpy(),
        "diagl": diagl[0].numpy(),
        "special_diagr": special_diag_r[0].numpy(),
        "special_diagl": special_diag_l[0].numpy(),
        "sample_no": 0,
    }
    error_count, _ = single_sample_class_metric_exact(args)
    return int(error_count)


# Register only when the hard dependency (networkx) is importable — mirroring
# the graceful-degradation pattern of satloss.py.
if nx is not None:
    register('topograph')(TopographLossAdapter)
    register('dice_topograph')(DiceTopographLossAdapter)
    if _HAS_CV2:
        register('exact_topograph')(ExactTopographLossAdapter)
