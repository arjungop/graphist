"""Quantify the edge-geometry defects in GrapHist's message passing.

The ACM block (src/train/models/acm_gin.py) aggregates as a weighted mean whose
weight is the z-scored Euclidean distance between two cells:

    message(x_j) = edge_weight * x_j
    deg          = scatter_sum(edge_weight)
    out          = out / deg

This script measures three consequences of that choice on real graphs, using the
released normalization statistics so the numbers correspond to what the published
checkpoint actually saw during pre-training:

  1. Virtual-node dominance. AddVirtualNode injects a raw-micron constant (22.5)
     *after* NormalizeData has z-scored the edge weights. We report the share of
     each node's aggregation mass that the virtual node alone accounts for, and
     the same share under the correctly normalized constant.

  2. Signed-degree cancellation. z-scored weights are signed, so `deg` is a sum
     of positive and negative terms and can cancel toward zero. The released
     guard, masked_fill_(deg_inv == inf, 0), catches only an exactly-zero degree.
     We report the real-edge-only degree distribution to show how close to zero
     it gets once the 22.5 constant no longer dominates the sum.

  3. Distance-as-affinity inversion. Larger weight means *farther apart*, so a
     node's most distant neighbour receives the largest share of the aggregation.
     We report the mass-weighted mean neighbour distance against the plain mean.

Run from the repository root:

    python -m src.analysis.edge_geometry_diagnostic \
        --graphs_path graph-pannuke/data \
        --norm_json_path normalization.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch_geometric.transforms import Compose, ToUndirected

from src.train.utils import GraphDataset, NormalizeData, set_random_seed

# The constant hardcoded in AddVirtualNode, in raw micrometres.
RELEASED_VIRTUAL_WEIGHT = 22.5


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--norm_json_path", type=str, required=True)
    parser.add_argument("--num_graphs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def summarize(name, values):
    """Return a percentile summary of a 1-D array."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "metric": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "p01": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main(args):
    set_random_seed(args.seed)

    with open(args.norm_json_path) as f:
        scale_vals = json.load(f)

    edge_mean = float(scale_vals["edge_attr"]["mean"][0])
    edge_std = float(scale_vals["edge_attr"]["std"][0])
    corrected_virtual_weight = (RELEASED_VIRTUAL_WEIGHT - edge_mean) / edge_std

    print(f"Pre-training edge weight mean / std : {edge_mean:.4f} / {edge_std:.4f} um")
    print(f"Released virtual-node weight        : {RELEASED_VIRTUAL_WEIGHT:.4f}")
    print(f"Correctly normalized equivalent     : {corrected_virtual_weight:.4f}")
    print(f"Overstatement factor                : "
          f"{RELEASED_VIRTUAL_WEIGHT / corrected_virtual_weight:.1f}x")
    print()

    all_paths = sorted(glob.glob(os.path.join(args.graphs_path, "**", "*.pt"),
                                 recursive=True))
    if not all_paths:
        raise SystemExit(f"No .pt graphs found under {args.graphs_path}")

    rng = np.random.default_rng(args.seed)
    if args.num_graphs < len(all_paths):
        sel = rng.choice(len(all_paths), size=args.num_graphs, replace=False)
        all_paths = [all_paths[i] for i in sorted(sel)]
    print(f"Sampling {len(all_paths)} graphs from {args.graphs_path}\n")

    # Deliberately excludes AddVirtualNode: we add the virtual mass analytically
    # so that the released and corrected constants can be compared on identical
    # graphs without rebuilding the edge index twice.
    transforms = Compose([ToUndirected(), NormalizeData(scale_vals)])
    ds = GraphDataset(all_paths, transform=transforms)

    released_share = []
    corrected_share = []
    real_degree = []
    released_total_degree = []
    mean_neighbour_um = []
    massw_neighbour_um = []
    all_weights = []

    for i in range(len(ds)):
        data = ds[i]
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            continue

        w = data.edge_attr.view(-1).to(torch.float64)
        dst = data.edge_index[1]
        num_nodes = data.num_nodes
        all_weights.append(w.numpy())

        # Real-edge weighted degree, exactly as scatter() computes it in acm_gin.
        deg = torch.zeros(num_nodes, dtype=torch.float64).scatter_add_(0, dst, w)

        # Every real node gains exactly one virtual edge, so the virtual node's
        # contribution to that node's degree is the constant itself.
        released_deg = deg + RELEASED_VIRTUAL_WEIGHT
        corrected_deg = deg + corrected_virtual_weight

        released_share.append(
            (RELEASED_VIRTUAL_WEIGHT / released_deg.abs().clamp(min=1e-12)).numpy()
        )
        corrected_share.append(
            (corrected_virtual_weight / corrected_deg.abs().clamp(min=1e-12)).numpy()
        )
        real_degree.append(deg.numpy())
        released_total_degree.append(released_deg.numpy())

        # Inversion: recover raw micrometres and compare a plain neighbour mean
        # against the mean the aggregation actually applies.
        raw_um = w * edge_std + edge_mean
        mass = w.abs()
        mass_sum = torch.zeros(num_nodes, dtype=torch.float64).scatter_add_(0, dst, mass)
        weighted_um = torch.zeros(num_nodes, dtype=torch.float64).scatter_add_(
            0, dst, mass * raw_um
        )
        plain_um = torch.zeros(num_nodes, dtype=torch.float64).scatter_add_(0, dst, raw_um)
        counts = torch.zeros(num_nodes, dtype=torch.float64).scatter_add_(
            0, dst, torch.ones_like(raw_um)
        )
        valid = (counts > 0) & (mass_sum > 1e-12)
        mean_neighbour_um.append((plain_um[valid] / counts[valid]).numpy())
        massw_neighbour_um.append((weighted_um[valid] / mass_sum[valid]).numpy())

    released_share = np.concatenate(released_share)
    corrected_share = np.concatenate(corrected_share)
    real_degree = np.concatenate(real_degree)
    released_total_degree = np.concatenate(released_total_degree)
    mean_neighbour_um = np.concatenate(mean_neighbour_um)
    massw_neighbour_um = np.concatenate(massw_neighbour_um)
    all_weights = np.concatenate(all_weights)

    results = {
        "edge_mean_um": edge_mean,
        "edge_std_um": edge_std,
        "released_virtual_weight": RELEASED_VIRTUAL_WEIGHT,
        "corrected_virtual_weight": corrected_virtual_weight,
        "overstatement_factor": RELEASED_VIRTUAL_WEIGHT / corrected_virtual_weight,
        "num_graphs": len(all_paths),
        "summaries": [
            summarize("virtual share of degree (released, 22.5)", released_share),
            summarize("virtual share of degree (corrected)", corrected_share),
            summarize("real-edge-only weighted degree", real_degree),
            summarize("total weighted degree (released)", released_total_degree),
            summarize("plain mean neighbour distance (um)", mean_neighbour_um),
            summarize("mass-weighted neighbour distance (um)", massw_neighbour_um),
        ],
    }

    print("=" * 78)
    print(f"{'metric':<44}{'median':>10}{'mean':>10}{'p01':>10}")
    print("=" * 78)
    for s in results["summaries"]:
        print(f"{s['metric']:<44}{s['median']:>10.4f}{s['mean']:>10.4f}{s['p01']:>10.4f}")
    print("=" * 78)

    frac_dominated = float((released_share > 0.9).mean())
    frac_dominated_corrected = float((corrected_share > 0.9).mean())
    print(f"\nNodes whose aggregation is >90% virtual (released) : {frac_dominated:.1%}")
    print(f"Nodes whose aggregation is >90% virtual (corrected): "
          f"{frac_dominated_corrected:.1%}")

    # Sign-cancellation hazard, which the 22.5 constant currently masks.
    near_zero = float((np.abs(real_degree) < 1.0).mean())
    negative = float((real_degree < 0).mean())
    print(f"\nReal-edge degree within +/-1.0 of zero            : {near_zero:.1%}")
    print(f"Real-edge degree strictly negative                : {negative:.1%}")
    print(f"Smallest |real-edge degree|                        : "
          f"{np.abs(real_degree).min():.2e}")
    print(f"Smallest |total degree| as released                : "
          f"{np.abs(released_total_degree).min():.4f}")

    inversion = float(np.mean(massw_neighbour_um - mean_neighbour_um))
    print(f"\nMass-weighted minus plain neighbour distance      : {inversion:+.3f} um")
    print("(positive means aggregation is biased toward more distant cells)")

    # The sign of the message weight is what makes the inversion pathological
    # rather than merely a reweighting: an edge shorter than the pre-training
    # mean carries a negative z-score, so the closest cells contribute their
    # features negated.
    frac_negative_edges = float((all_weights < 0).mean())
    print(f"\nEdges with a negative message weight              : "
          f"{frac_negative_edges:.1%}")
    for d_um in (5.0, 10.0, 40.0):
        print(f"  weight of a {d_um:>4.0f} um edge                        : "
              f"{(d_um - edge_mean) / edge_std:+.3f}")

    results["frac_negative_edge_weight"] = frac_negative_edges
    results["summaries"].append(summarize("edge message weight", all_weights))

    results["frac_nodes_over_90pct_virtual_released"] = frac_dominated
    results["frac_nodes_over_90pct_virtual_corrected"] = frac_dominated_corrected
    results["frac_real_degree_near_zero"] = near_zero
    results["frac_real_degree_negative"] = negative
    results["min_abs_real_degree"] = float(np.abs(real_degree).min())
    results["min_abs_released_degree"] = float(np.abs(released_total_degree).min())
    results["inversion_um"] = inversion

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main(build_args())
