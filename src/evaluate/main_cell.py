import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from utils import set_random_seed


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cell-level logistic regression evaluation"
    )

    # Split mode
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["csv", "folder", "group"],
        required=True,
        help="'csv': single emb dir + fold CSVs (NuCLS). "
        "'folder': separate fold directories (PanNuke). "
        "'group': single emb dir, grouped k-fold via --group_regex.",
    )

    # csv mode args
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=None,
        help="Directory with all .npz embeddings (csv mode)",
    )
    parser.add_argument(
        "--split_csv_dir",
        type=str,
        default=None,
        help="Directory with fold_X_train.csv / fold_X_test.csv (csv mode)",
    )

    # folder mode args
    parser.add_argument(
        "--fold_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Fold directories (folder mode), e.g. fold1/ fold2/ fold3/",
    )

    # Common args
    parser.add_argument(
        "--num_folds",
        type=int,
        default=None,
        help="Number of folds (auto-detected if not set)",
    )
    parser.add_argument(
        "--tissue_filter",
        type=str,
        default=None,
        help="Keep only files whose basename starts with this prefix, e.g. 'Breast' "
        "for the PanNuke breast-only subset reported in Table 5.",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        choices=["nucls_super"],
        help="Aggregate fine-grained labels into a coarser set. 'nucls_super' "
        "collapses the 7 NuCLS main classes into the 4 super classes.",
    )
    parser.add_argument(
        "--group_regex",
        type=str,
        default=None,
        help="With --split_mode group: regex whose first capture group extracts a "
        "grouping key (e.g. a slide or hospital id) from each filename.",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--class_weight", type=str, default="balanced")

    return parser.parse_args()


# ---------- Metrics ----------


def macro_ovr_scores(y_true, y_proba, clf_classes):
    """Macro one-vs-rest AUROC and AUPRC that tolerate degenerate folds.

    sklearn's built-in multi_class="ovr" requires every class to be present in
    both the training set (so it has a probability column) and the test set.
    Grouped or hospital-based folds break that assumption for rare classes -- in
    NuCLS, "Other" (587 cells) and "Tumor (mitotic)" (216 cells) are absent from
    whole folds. We therefore average the per-class scores over the classes that
    are well defined in this fold, which is exactly what sklearn's macro average
    computes when no class is missing.
    """
    aurocs, auprcs = [], []
    for j, c in enumerate(clf_classes):
        pos = (y_true == c).astype(int)
        if pos.sum() == 0 or pos.sum() == pos.shape[0]:
            continue
        aurocs.append(roc_auc_score(pos, y_proba[:, j]))
        auprcs.append(average_precision_score(pos, y_proba[:, j]))
    if not aurocs:
        return float("nan"), float("nan")
    return float(np.mean(aurocs)), float(np.mean(auprcs))


# ---------- Label maps ----------

# NuCLS ships only the 7-class "main" labels. The paper also reports the 4-class
# "super" task, which the NuCLS authors define by merging main classes. Verified
# against Figure A.6 of the paper: this mapping reproduces all four super-class
# counts exactly (23369 / 17261 / 10893 / 587).
NUCLS_SUPER = {
    6: 0,  # Tumor (non-mitotic) -> Tumor Any
    5: 0,  # Tumor (mitotic)     -> Tumor Any
    0: 1,  # Lymphocyte          -> sTIL
    4: 1,  # Plasma cell         -> sTIL
    2: 2,  # Non-TIL/Non-MQ str. -> Non-TIL stromal
    1: 2,  # Macrophage          -> Non-TIL stromal
    3: 3,  # Other               -> Other
}

LABEL_MAPS = {"nucls_super": NUCLS_SUPER}


def apply_label_map(y, name):
    if name is None:
        return y
    mapping = LABEL_MAPS[name]
    out = np.empty_like(y)
    for src, dst in mapping.items():
        out[y == src] = dst
    return out


def filter_paths(paths, tissue_filter):
    if tissue_filter is None:
        return paths
    return [p for p in paths if os.path.basename(p).startswith(tissue_filter)]


# ---------- Loading helpers ----------


def load_cell_embeddings_from_dir(emb_dir, tissue_filter=None):
    """Load all .npz files from a directory."""
    xs, ys = [], []
    all_npz_files = filter_paths(
        sorted(glob.glob(os.path.join(emb_dir, "*.npz"))), tissue_filter
    )

    for fp in tqdm(all_npz_files, desc=f"Loading {emb_dir}"):
        data = np.load(fp)
        emb = np.asarray(data["embedding"])
        lbl = np.asarray(data["labels"])

        if lbl.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Label length {lbl.shape[0]} != num cells {emb.shape[0]} in {fp}"
            )

        xs.append(emb)
        ys.append(lbl)

    X = np.vstack(xs)
    y = np.concatenate(ys)
    return X, y


def load_cell_embeddings_by_slides(emb_dir, slide_ids):
    """Load .npz files whose filename starts with one of the given slide IDs."""
    xs, ys = [], []
    slide_id_set = set(slide_ids)
    all_npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))

    files_to_load = []
    for fp in all_npz_files:
        # Slide ID is the first underscore-delimited token
        file_slide_id = os.path.basename(fp).split("_")[0]
        if file_slide_id in slide_id_set:
            files_to_load.append(fp)

    for fp in tqdm(sorted(files_to_load), desc="Loading embs"):
        data = np.load(fp)
        emb = np.asarray(data["embedding"])
        lbl = np.asarray(data["labels"])

        if lbl.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Label length {lbl.shape[0]} != num cells {emb.shape[0]} in {fp}"
            )

        xs.append(emb)
        ys.append(lbl)

    X = np.vstack(xs)
    y = np.concatenate(ys)
    return X, y


# ---------- Train & evaluate ----------


def train_and_eval_cell_level(
    X_train,
    y_train,
    X_test,
    y_test,
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=0,
):

    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    # Pipeline: Standardize → Logistic Regression
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)  # shape: [N, num_classes]

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # AUROC & AUPRC (macro one-vs-rest, robust to classes missing from a fold)
    auroc, auprc = macro_ovr_scores(y_test, y_proba, clf.named_steps["logreg"].classes_)

    # Print results
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {f1_macro:.4f}")
    print(f"  Balanced Accuracy : {balanced_acc:.4f}")
    print(f"  AUROC (macro OvR) : {auroc:.4f}")
    print(f"  AUPRC (macro OvR) : {auprc:.4f}")

    seen = set(clf.named_steps["logreg"].classes_.tolist())
    missing_train = sorted(set(classes.tolist()) - seen)
    missing_test = sorted(set(classes.tolist()) - set(np.unique(y_test).tolist()))
    if missing_train:
        print(f"  NOTE: classes absent from train split: {missing_train}")
    if missing_test:
        print(f"  NOTE: classes absent from test split : {missing_test}")

    cm_test = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"  Confusion matrix:\n{cm_test}\n")

    result = {
        "num_train_cells": int(X_train.shape[0]),
        "num_test_cells": int(X_test.shape[0]),
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "balanced_accuracy": float(balanced_acc),
        "auroc_macro_ovr": float(auroc),
        "auprc_macro_ovr": float(auprc),
        "classification_report": report_dict,
        "classes_missing_from_train": [int(c) for c in missing_train],
        "classes_missing_from_test": [int(c) for c in missing_test],
    }

    return result


# ---------- Cross-validation runners ----------


def run_csv_mode(args):
    assert args.emb_dir is not None
    assert args.split_csv_dir is not None

    # Auto-detect number of folds
    if args.num_folds is None:
        fold_files = glob.glob(os.path.join(args.split_csv_dir, "fold_*_train.csv"))
        args.num_folds = len(fold_files)
    print(f"Running {args.num_folds}-fold CV (csv mode)\n")

    results = []
    for i in range(1, args.num_folds + 1):
        print(f"--- Fold {i}/{args.num_folds}: test on fold {i} ---")

        train_csv = os.path.join(args.split_csv_dir, f"fold_{i}_train.csv")
        test_csv = os.path.join(args.split_csv_dir, f"fold_{i}_test.csv")

        train_slides = pd.read_csv(train_csv)["slide_name"].tolist()
        test_slides = pd.read_csv(test_csv)["slide_name"].tolist()

        X_train, y_train = load_cell_embeddings_by_slides(args.emb_dir, train_slides)
        X_test, y_test = load_cell_embeddings_by_slides(args.emb_dir, test_slides)

        result = train_and_eval_cell_level(
            X_train,
            y_train,
            X_test,
            y_test,
            C=args.C,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
        result["fold"] = i
        results.append(result)

    return results


def run_group_mode(args):
    """Grouped k-fold over a single embedding directory.

    Used for NuCLS: the dataset card documents a `splits/` directory of fold
    CSVs, but it is not present in the published repository, so the paper's exact
    fold assignment cannot be recovered. NuCLS folds are hospital-based, and for
    TCGA slides the two-character tissue source site code is the hospital, so we
    group on that and report it as an approximation of the original split.
    """
    from sklearn.model_selection import GroupKFold

    assert args.emb_dir is not None
    assert args.group_regex is not None
    num_folds = args.num_folds or 5

    paths = filter_paths(
        sorted(glob.glob(os.path.join(args.emb_dir, "*.npz"))), args.tissue_filter
    )
    pat = re.compile(args.group_regex)

    xs, ys, gs = [], [], []
    for fp in tqdm(paths, desc="Loading embs"):
        m = pat.search(os.path.basename(fp))
        if m is None:
            raise ValueError(f"--group_regex did not match {os.path.basename(fp)}")
        data = np.load(fp)
        emb, lbl = np.asarray(data["embedding"]), np.asarray(data["labels"])
        if lbl.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Label length {lbl.shape[0]} != num cells {emb.shape[0]} in {fp}"
            )
        xs.append(emb)
        ys.append(lbl)
        gs.append(np.full(emb.shape[0], m.group(1)))

    X = np.vstack(xs)
    y = apply_label_map(np.concatenate(ys), args.label_map)
    groups = np.concatenate(gs)
    print(
        f"Running {num_folds}-fold GroupKFold (group mode) over "
        f"{len(paths)} patches, {X.shape[0]:,} cells, "
        f"{len(np.unique(groups))} groups, {len(np.unique(y))} classes\n"
    )

    results = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=num_folds).split(X, y, groups), 1):
        print(f"--- Fold {i}/{num_folds} ---")
        result = train_and_eval_cell_level(
            X[tr],
            y[tr],
            X[te],
            y[te],
            C=args.C,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
        result["fold"] = i
        result["n_test_groups"] = int(len(np.unique(groups[te])))
        results.append(result)

    return results


def run_folder_mode(args):
    assert args.fold_dirs is not None and len(args.fold_dirs) >= 2
    num_folds = len(args.fold_dirs)
    print(f"Running {num_folds}-fold CV (folder mode)\n")

    results = []
    for i in range(num_folds):
        test_dir = args.fold_dirs[i]
        train_dirs = [d for j, d in enumerate(args.fold_dirs) if j != i]

        print(f"--- Fold {i+1}/{num_folds}: test on {test_dir} ---")

        # Load training data from all other folds
        X_tr_list, y_tr_list = [], []
        for d in train_dirs:
            X_f, y_f = load_cell_embeddings_from_dir(d, args.tissue_filter)
            X_tr_list.append(X_f)
            y_tr_list.append(y_f)
        X_train = np.vstack(X_tr_list)
        y_train = apply_label_map(np.concatenate(y_tr_list), args.label_map)

        X_test, y_test = load_cell_embeddings_from_dir(test_dir, args.tissue_filter)
        y_test = apply_label_map(y_test, args.label_map)

        result = train_and_eval_cell_level(
            X_train,
            y_train,
            X_test,
            y_test,
            C=args.C,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
        result["test_fold"] = test_dir
        result["train_folds"] = train_dirs
        results.append(result)

    return results


# ---------- Main ----------


def main():
    args = parse_arguments()

    set_random_seed(args.seed)
    print(f"Seed: {args.seed}")

    if args.split_mode == "csv":
        results = run_csv_mode(args)
    elif args.split_mode == "group":
        results = run_group_mode(args)
    else:
        results = run_folder_mode(args)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary
    print("\n===== SUMMARY =====")
    summary = {}
    for key, label in [
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("auroc_macro_ovr", "AUROC"),
        ("auprc_macro_ovr", "AUPRC"),
        ("accuracy", "Accuracy"),
    ]:
        v = np.array([r[key] for r in results]) * 100
        summary[key] = {"mean": float(v.mean()), "std": float(v.std())}
        print(f"  {label:<18} {v.mean():6.2f} +/- {v.std():.2f}")
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Results saved to {results_path}")
    print("===================\n")


if __name__ == "__main__":
    main()
