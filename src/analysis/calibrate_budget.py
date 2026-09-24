"""Measure real training throughput, then solve for a step budget that fits.

The 10-hour question cannot be answered by arithmetic on the paper's reported
50 GPU-hours, because that figure describes a different machine, a different
software stack and a corpus staged differently on disk. This script measures the
throughput that the sweep will actually run at and then inverts it: given a
wall-clock allowance and a number of arms, it reports the `--max_steps` value
that makes the sweep fit.

Run it on the machine that will run the sweep, against the staged corpus:

    python src/analysis/calibrate_budget.py \
        --metadata_csv_path /data/graph-tcga-brca/metadata.csv \
        --norm_json_path /data/graph-tcga-brca/normalization.json \
        --budget_hours 10 --num_arms 4

Throughput is measured after a warmup so that CUDA autotuning, page-cache
population and worker spawn are excluded from the steady-state rate.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from models import build_model  # noqa: E402
from utils import (  # noqa: E402
    AddVirtualNode,
    GraphDataset,
    NormalizeData,
    seed_worker,
    set_random_seed,
    split_data,
)

# Reserved for staging, calibration, evaluation and slack. Everything outside
# the pre-training arms themselves comes out of the allowance before the step
# budget is solved.
DEFAULT_OVERHEAD_HOURS = 2.0


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv_path", type=str, required=True)
    parser.add_argument("--norm_json_path", type=str, required=True)

    parser.add_argument("--budget_hours", type=float, default=10.0)
    parser.add_argument("--num_arms", type=int, default=4)
    parser.add_argument("--overhead_hours", type=float, default=DEFAULT_OVERHEAD_HOURS)

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--measure_steps", type=int, default=50)

    # Kept identical to the sweep's own settings, since throughput depends on
    # all of them.
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_hidden", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--replace_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    # Model knobs build_model expects; defaults mirror pretrain.py.
    parser.add_argument("--encoder", type=str, default="acm_gin")
    parser.add_argument("--decoder", type=str, default="acm_gin")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--node_pooling", type=str, default="mean")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--residual", type=bool, default=None)
    parser.add_argument("--attn_drop", type=float, default=0.1)
    parser.add_argument("--in_drop", type=float, default=0.2)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--batchnorm", type=bool, default=False)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=3)
    parser.add_argument("--concat_hidden", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main(args):
    set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(args.norm_json_path) as f:
        scale_vals = json.load(f)

    data_df = pd.read_csv(args.metadata_csv_path)
    train_mask, _, _ = split_data(data_df)
    train_paths = data_df[train_mask]["graph_path"].tolist()
    print(f"Train graphs staged: {len(train_paths):,}")

    transforms = Compose([ToUndirected(), NormalizeData(scale_vals), AddVirtualNode()])
    ds = GraphDataset(train_paths, transform=transforms)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    args.num_features = int(ds[0].num_features)
    args.num_edge_features = int(ds[0].edge_attr.size(1))
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    total_steps = args.warmup_steps + args.measure_steps
    graphs_seen = 0
    t_start = None

    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        batch = batch.to(device)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Start the clock only after warmup, and synchronise so the measurement
        # covers completed GPU work rather than queued launches.
        if step == args.warmup_steps - 1:
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.time()
        elif step >= args.warmup_steps:
            graphs_seen += batch.num_graphs

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t_start

    measured_steps = args.measure_steps
    steps_per_s = measured_steps / elapsed
    graphs_per_s = graphs_seen / elapsed

    print("\n" + "=" * 70)
    print("MEASURED THROUGHPUT")
    print("=" * 70)
    print(f"Steps measured            : {measured_steps}")
    print(f"Wall clock                : {elapsed:.1f} s")
    print(f"Throughput                : {steps_per_s:.2f} steps/s")
    print(f"                          : {graphs_per_s:,.0f} graphs/s")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory           : {peak:.1f} GB")

    train_hours = args.budget_hours - args.overhead_hours
    if train_hours <= 0:
        raise SystemExit("Overhead consumes the whole allowance; nothing left to train.")
    per_arm_s = (train_hours * 3600) / args.num_arms
    max_steps = int(per_arm_s * steps_per_s)
    graphs_per_arm = max_steps * args.batch_size
    epochs_equiv = graphs_per_arm / max(len(train_paths), 1)

    # The paper's own budget, for context on how far below it we are running.
    paper_presentations = 100 * 11_149_500

    print("\n" + "=" * 70)
    print("SOLVED BUDGET")
    print("=" * 70)
    print(f"Total allowance           : {args.budget_hours:.1f} h")
    print(f"Reserved for overhead     : {args.overhead_hours:.1f} h")
    print(f"Training time             : {train_hours:.1f} h across {args.num_arms} arms")
    print(f"Per arm                   : {per_arm_s / 3600:.2f} h")
    print()
    print(f"  --max_steps {max_steps}")
    print(f"  --max_seconds {per_arm_s * 1.15:.0f}")
    print()
    print(f"Graph presentations/arm   : {graphs_per_arm:,}")
    print(f"Passes over staged corpus : {epochs_equiv:.1f}")
    print(f"Fraction of paper budget  : {100 * graphs_per_arm / paper_presentations:.2f}%")
    print("=" * 70)
    print(
        "\nUse --max_steps for every arm so all arms see identical data volume;\n"
        "--max_seconds is set 15% above the expected per-arm time purely as a\n"
        "safety net, and should not normally fire."
    )


if __name__ == "__main__":
    main(build_args())
