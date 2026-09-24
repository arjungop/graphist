# Phase 2 — The gap, and how we close it

Phase 1 established that GrapHist's published numbers reproduce. This document
states the methodological gap we intend to fix, the evidence that it is real, and
the experiment that settles it.

Reproduce every number below with:

```bash
python -m src.analysis.edge_geometry_diagnostic \
    --graphs_path graph-pannuke/data \
    --norm_json_path normalization.json \
    --num_graphs 1500 \
    --output_json src/analysis/edge_geometry_pannuke.json
```

---

## 1. The thesis

**GrapHist's message passing is driven by an edge geometry whose sign, scale and
normalization are all incorrect, and the model trains only because one bug
accidentally suppresses the others.**

The ACM block (`src/train/models/acm_gin.py:68-96`) aggregates as

```python
message(x_j) = edge_weight * x_j
deg          = scatter_sum(edge_weight)
out          = out / deg
```

where `edge_weight` is the **z-scored Euclidean distance in micrometres** between
two cells. Three consequences follow, all measured on real PanNuke graphs using
the released pre-training statistics (mean 18.627 um, std 12.735 um).

### 1.1 The majority of cell-cell messages are sign-inverted

A z-scored distance is negative whenever two cells are closer than the corpus
mean of 18.6 um. Most adjacent cells are:

| Edge length | Message weight |
|---|---:|
| 5 um | **-1.070** |
| 10 um | **-0.677** |
| 40 um | +1.678 |

**74.5% of all edges carry a negative message weight.** The closest, most
biologically relevant neighbours do not merely receive less weight — they
contribute their features *negated*. Distance is being consumed as though it were
affinity, which inverts the spatial prior the method rests on.

The dataset card acknowledges this directly, recommending users "experiment with
edge weights, such as using their inverse (`1/distance`) or negative distance."
The paper never does, and the released checkpoint does not.

### 1.2 The weighted degree is negative for most nodes and can vanish

Because the weights are signed, `deg` is a sum of mixed-sign terms:

- **74.2%** of nodes have a strictly negative real-edge weighted degree
- **21.5%** fall within ±1.0 of zero
- the smallest observed **|degree| is 3.4e-04**, a ~3000x amplification if divided by

The released guard, `masked_fill_(deg_inv == inf, 0)`, catches only an
exactly-zero degree. It never fires here, and it cannot catch near-cancellation.

### 1.3 The virtual node is load-bearing, not merely mis-scaled

`AddVirtualNode` (`src/train/utils.py:76`) hardcodes `22.5` — a raw-micron
constant — and the pipeline is ordered `ToUndirected -> NormalizeData ->
AddVirtualNode`, so it is injected **after** z-scoring. The correctly normalized
value is `(22.5 - 18.627) / 12.735 = 0.304`, so the constant is **74x too large**.

Its effect on aggregation:

| | Released (22.5) | Corrected (0.304) |
|---|---:|---:|
| Nodes >90% of aggregation mass from virtual node | **91.9%** | 8.0% |
| Median total weighted degree | 20.72 | — |
| Smallest observed \|total degree\| | 13.998 | — |

So for 92% of nodes, the "local cellular neighbourhood" the paper's argument
depends on contributes under a tenth of the aggregation. The virtual node's share
of the degree actually **exceeds 1.0** (median 1.086), because the real
neighbourhood sums to a *negative* number that the constant then overshoots.

**This is the key insight.** The 22.5 constant is the only thing keeping the
denominator positive and bounded away from zero. It is silently acting as a
stabilizer for the two defects above. A naive one-line fix of `22.5 -> 0.304`
would expose the near-zero degrees in 1.2 and is expected to **destabilize
training** — which is precisely why this is a methodological contribution and not
a typo report.

---

## 2. What is now known to be reachable

The pre-training corpus is published, contrary to our earlier assumption:

| Artifact | Detail |
|---|---|
| `ogutsevda/graph-tcga-brca` | 249 tar shards, **269 GB**, 11,149,500 graphs |
| `metadata.csv` (2.36 GB) | `graph_path, sample_id, wsi_x, wsi_y, label, split` |
| `normalization.json` | the same statistics validated in phase 1 |

`pretrain.py` consumes `--metadata_csv_path` and `--norm_json_path` directly, and
the corpus ships the authors' **own `split` column**, so unlike NuCLS no split
reconstruction is needed. Re-pre-training is therefore fully reachable on the
A100. The paper reports 50 GPU-hours per run.

---

## 3. The experiment

Four pre-training arms at **identical budget, seed and split**, each evaluated
with the phase-1 probes (PanNuke pan-cancer and breast-only, NuCLS main and
super).

| Arm | Virtual node | Edge weight | Tests |
|---|---|---|---|
| **A** control | 22.5 raw | z-scored distance | Reproduces the published model at our budget |
| **B** | 0.304 + magnitude clamp | z-scored distance | Does correcting scale alone help or destabilize? |
| **C** | consistent w/ affinity | affinity (`-d` or `1/d`) | Does fixing the sign inversion help? |
| **D** | 0.304 + clamp | affinity | Both corrections |

Arm A is **not optional**: at any budget below the paper's full run, the released
checkpoint is not a valid comparator, so we need a same-budget control.

Prediction, stated in advance so it is falsifiable: **B underperforms A** (the
stabilizer is removed without fixing what it was masking) while **D outperforms
both**. If instead A ≈ D, the finding is that the paper's stated mechanism is not
what drives its performance — which is a publishable negative result and equally
satisfies the contribution requirement.

### Required code changes

1. **Parameterize the virtual node.** Add `--virtual_node_weight` (default 22.5 to
   preserve arm A) threaded into `AddVirtualNode`, applied in normalized space.
2. **Parameterize edge semantics.** Add `--edge_transform {raw,neg,inv}` applied
   inside the transform pipeline before normalization.
3. **Replace the degree guard** in `acm_gin.py:69-70` with a magnitude clamp, so
   near-zero degrees cannot amplify. Required for arms B and D.
4. **Mirror all three in `generate_embs.py`**, whose transform pipeline must match
   the arm's pre-training pipeline exactly or the embeddings are invalid.

### Open risks

- Arms B and D may diverge in training. The clamp threshold is a hyperparameter
  we may need to tune; budget one pilot run at reduced epochs before committing.
- 269 GB must be staged on the A100 host. The local machine has 24 GB free and
  cannot hold the corpus.

---

## 4. Fitting the sweep into a 10-hour window

### The budget is enforced, not predicted

`pretrain.py` now takes `--max_steps` and `--max_seconds`. Every arm is given the
**same `--max_steps`**, which is what keeps the arms comparable: they see an
identical number of optimizer steps no matter how fast each happens to run.
`--max_seconds` is a safety net set ~15% above the expected per-arm time, so one
misbehaving arm cannot eat the whole allowance. This makes the 10-hour figure a
property of the design rather than a forecast.

`--run_name` is also new and is **required** for a sweep: the run name was
previously hardcoded to a placeholder, so all four arms would have overwritten one
another's checkpoints.

### Calibrate before committing

Throughput cannot be inferred from the paper's 50 GPU-hours. Run this on the
sweep machine, against the staged corpus, and it solves for `--max_steps`:

```bash
python src/analysis/calibrate_budget.py \
    --metadata_csv_path /data/graph-tcga-brca/metadata.csv \
    --norm_json_path   /data/graph-tcga-brca/normalization.json \
    --budget_hours 10 --num_arms 4
```

### Where the time actually goes

For reference, the paper's own run is **1.115 billion graph presentations**
(100 epochs x 11,149,500 graphs) in 50 A100-hours — a sustained **6,194
graphs/s**. Allowing 2 h for staging, calibration, evaluation and slack leaves
8 h across 4 arms, i.e. **2 h per arm**:

| Sustained rate | Graphs/arm | Fraction of paper budget |
|---:|---:|---:|
| 6,000/s | 43 M | 3.9% |
| 15,000/s | 108 M | 9.7% |
| 30,000/s | 216 M | 19.4% |

Arm A will therefore **not** reproduce the paper's downstream numbers, and is not
meant to — it is the same-budget internal control. The exact reproduction already
exists in the phase-1 report.

Evaluation is cheap and not a risk: phase 1 measured the full PanNuke + NuCLS
probe suite at roughly 4 minutes on a CPU laptop. Skip BRACS during the sweep
(10 m 34 s per arm) and run it only for the winning arm.

### The real bottleneck is I/O, not the GPU

Each graph is a separate ~12 KB `.pt` file that `GraphDataset.get()` opens and
unpickles individually, so a batch of 2048 costs 2048 file opens. Measured
locally at **2,932 graphs/s** single-process on warm cache — only about 2x below
the rate the paper's entire run averaged. With 11.1 M files read in shuffled
order the page cache will not help, so a naive run is I/O-bound and the A100 will
idle.

Two consequences for the 10-hour plan:

1. **Do not stage the 269 GB corpus inside the 10 hours.** Downloading 249 shards
   and unpacking ~11 M small files is plausibly a multi-hour job on its own. Run
   it out of band, ahead of the window.
2. **Train from a packed subset, not from loose files.** Collating ~2 M graphs
   (about 45 shards) into contiguous tensors makes batch assembly a slice instead
   of thousands of syscalls. At ~30 nodes per graph that is roughly 35 GB, which
   fits comfortably in host RAM and very nearly on the 80 GB A100 itself. This is
   the single highest-leverage change for making the window achievable.

## 5. Still outstanding from phase 1

Independent of the gap work, these remain open:

1. **Report the label-alignment defect upstream** — it silently corrupts
   cell-level results for anyone running the shipped defaults.
2. **Request the NuCLS `splits/` directory** from the authors; it is the only
   thing between our NuCLS numbers and an exact reproduction.
3. **TCGA-BRCA `metadata.csv` carries `sample_id`, `wsi_x`, `wsi_y` and a label**,
   which makes the WSI-level column reachable and, with GDC clinical data, the
   survival analysis too. This is a second contribution avenue if the gap work
   lands early.
