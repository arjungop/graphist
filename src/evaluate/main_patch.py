import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, balanced_accuracy_score

from utils import set_random_seed, macro_ovr_scores


def load_embs(emb_dir):
    embs = {}
    for file in tqdm(os.listdir(emb_dir), desc="Loading embs"):
        if file.endswith(".npy"):
            sample_id = file.replace(".npy", "")
            embs[sample_id] = np.load(os.path.join(emb_dir, file))
    return embs


def manual_standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std
    return train_scaled, test_scaled


def main(dataset, seed, metadata_csv, embs_path, save_dir):

    set_random_seed(seed)
    print("Set random seed to", seed)

    # Load embs and metadata
    emb_dict = load_embs(embs_path)
    data_df = pd.read_csv(metadata_csv)

    # Filter by available embs
    data_df = data_df[data_df["sample_id"].isin(emb_dict)]

    # Split train/test
    train_df = data_df[data_df["split"] == "train"]
    test_df = data_df[data_df["split"] == "test"]

    # Get embs and labels for train/test
    X_train = np.stack([emb_dict[sid] for sid in train_df["sample_id"]])
    y_train = train_df["label"].values
    X_test = np.stack([emb_dict[sid] for sid in test_df["sample_id"]])
    y_test = test_df["label"].values

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print("Label classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Normalization
    X_train_scaled, X_test_scaled = manual_standardize(X_train, X_test)

    # Initialize logistic regression model
    clf = LogisticRegression(
        max_iter=2500, solver="lbfgs", class_weight="balanced", random_state=seed
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf,
        X_train_scaled,
        y_train_enc,
        cv=skf,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
        n_jobs=-1,
    )

    print(f"\n=== Cross-Validation Performance on {dataset} ===")
    for metric in scores:
        if metric.startswith("test_"):
            mean = scores[metric].mean()
            std = scores[metric].std()
            print(f"{metric[5:]}: {mean:.4f} ± {std:.4f}")

    # Train on the full training set and evaluate on the test set
    clf.fit(X_train_scaled, y_train_enc)
    y_pred_enc = clf.predict(X_test_scaled)

    # Classification report
    target_names = le.classes_
    report_dict = classification_report(
        y_test_enc, y_pred_enc, target_names=target_names, output_dict=True
    )

    acc = report_dict["accuracy"]
    f1 = report_dict["macro avg"]["f1-score"]

    # Balanced accuracy, AUROC and AUPRC are reported by the paper but were not
    # computed by the original script.
    y_proba = clf.predict_proba(X_test_scaled)
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred_enc)
    auroc, auprc = macro_ovr_scores(y_test_enc, y_proba, list(clf.classes_))

    print(f"\n  Macro F1          : {f1:.4f}")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print(f"  AUROC (macro OvR) : {auroc:.4f}")
    print(f"  AUPRC (macro OvR) : {auprc:.4f}")
    print(f"  Accuracy          : {acc:.4f}")

    # Save results
    results = {
        "dataset": dataset,
        "cross_validation": {
            "accuracy_mean": round(scores["test_accuracy"].mean(), 4),
            "accuracy_std": round(scores["test_accuracy"].std(), 4),
            "f1_macro_mean": round(scores["test_f1_macro"].mean(), 4),
            "f1_macro_std": round(scores["test_f1_macro"].std(), 4),
        },
        "test_set": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "auroc_macro_ovr": round(auroc, 4),
            "auprc_macro_ovr": round(auprc, 4),
        },
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="CSV with sample_id, label and split columns.",
    )
    parser.add_argument(
        "--embs_path", type=str, required=True, help="Directory of {sample_id}.npy"
    )
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset, args.seed, args.metadata_csv, args.embs_path, args.save_dir)
