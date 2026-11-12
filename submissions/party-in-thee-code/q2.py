import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from baseline import ROOT, SEED, load_data, load_split
from q1 import make_q1_model, fit_timed

SIZES = [500, 1000, 2000, 5000, 8000]


def nested_subsets(train):
    rng = np.random.default_rng(SEED)
    per_class = {
        label: rng.permutation(train.index[train.label == label].to_numpy())
        for label in [0, 1]
    }
    subsets = {}
    for n in SIZES:
        chosen = np.concatenate([per_class[0][:n // 2], per_class[1][:n // 2]])
        # Keep the shared training order; at 8,000 rows it matches Q1 exactly.
        subset = train.loc[train.index.isin(chosen)].copy()
        if len(subset) != n or subset.label.value_counts().to_dict() != {0: n // 2, 1: n // 2}:
            raise ValueError("Unexpected subset size or class balance.")
        subsets[n] = subset
    for small, large in zip(SIZES[:-1], SIZES[1:]):
        if not set(subsets[small].id).issubset(set(subsets[large].id)):
            raise ValueError("Training subsets are not nested.")
    return subsets


def plot_curve(results, output):
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(results.training_rows, results.training_auroc,
            color="#2367a3", marker="o", linewidth=1.8, label="Training")
    ax.plot(results.training_rows, results.validation_auroc,
            color="#c65b28", marker="s", linewidth=1.8, label="Validation")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("AUROC")
    ax.set_xticks(SIZES)
    ax.set_xlim(150, 8350)
    ax.set_ylim(0.50, 1.02)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Character TF-IDF + linear SVM (C = 1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "q2_learning_curve.pdf", bbox_inches="tight")
    fig.savefig(output / "q2_learning_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_q2(data_path=ROOT / "data" / "train.csv", project_root=ROOT):
    project_root = Path(project_root)
    split_dir = project_root / "splits"
    split_files = [split_dir / f for f in ["train_ids.csv", "validation_ids.csv", "split_manifest.json"]]
    if not all(f.exists() for f in split_files):
        raise FileNotFoundError("The original shared split files are required.")
    split_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in split_files}
    train, val = load_split(load_data(data_path), split_dir)
    subsets = nested_subsets(train)
    output = project_root / "results" / "q2"
    (output / "subset_ids").mkdir(parents=True, exist_ok=True)
    rows, saved_scores = [], []

    with threadpool_limits(limits=1):
        for n in SIZES:
            subset = subsets[n]
            if set(subset.id) & set(val.id):
                raise ValueError("A validation ID appeared in training.")
            model = make_q1_model("char_svm", 1.0)
            # Both TF-IDF and SVM are fitted again for this subset only.
            seconds = fit_timed(model, subset)
            train_score = model.decision_function(subset.text)
            val_score = model.decision_function(val.text)
            if not np.isfinite(train_score).all() or not np.isfinite(val_score).all():
                raise ValueError("Nonfinite decision scores.")
            rows.append({
                "training_rows": n,
                "training_auroc": float(roc_auc_score(subset.label, train_score)),
                "validation_auroc": float(roc_auc_score(val.label, val_score)),
                "fit_time_seconds": seconds,
                "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
                "human_training_rows": n // 2,
                "ai_training_rows": n // 2,
                "validation_rows": len(val),
            })
            subset[["id"]].to_csv(output / "subset_ids" / f"train_ids_{n}.csv", index=False)
            for name, frame, score in [("training", subset, train_score), ("validation", val, val_score)]:
                pred = frame[["id", "label"]].rename(columns={"label": "true_label"}).copy()
                pred["training_rows"] = n
                pred["partition"] = name
                pred["decision_score"] = score
                saved_scores.append(pred)
            print(f"n={n}: train={rows[-1]['training_auroc']:.6f}, "
                  f"val={rows[-1]['validation_auroc']:.6f}, fit={seconds:.3f}s", flush=True)

    results = pd.DataFrame(rows)
    results["generalization_gap"] = results.training_auroc - results.validation_auroc
    results.to_csv(output / "learning_curve.csv", index=False)
    pd.concat(saved_scores, ignore_index=True).to_csv(output / "curve_scores.csv", index=False)
    plot_curve(results, output)
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in split_files}
    if after != split_hashes:
        raise RuntimeError("The shared split files changed.")

    model = make_q1_model("char_svm", 1.0)
    config = {
        "model": "Character TF-IDF + Linear SVM", "C": 1.0,
        "analyzer": "char", "ngram_range": [3, 5],
        "min_df": 2, "max_df": 1.0, "max_features": 50000,
        "sublinear_tf": True, "smooth_idf": True, "norm": "l2",
        "lowercase": True, "svm_dual": "auto", "svm_max_iter": 5000,
        "svm_tol": 1e-4, "seed": SEED, "sizes": SIZES,
        "subset_sampling": "Seeded per-class permutations; balanced nested prefixes; shared training order retained.",
        "tfidf_fitting": "Refitted separately on each training subset; validation texts never used for fitting.",
        "auc_score_type": "Uncalibrated decision_function, as in the Q1 SVM comparison. No probability calibration is needed for this ranking metric.",
        "timing_scope": "Full pipeline fit; excludes loading, predictions and plotting; one BLAS thread.",
        "split_file_sha256": split_hashes,
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    q1_path = project_root / "results" / "q1" / "model_comparison.csv"
    if q1_path.exists():
        reference = pd.read_csv(q1_path).query("family == 'char_svm'").iloc[0]
        last = results.iloc[-1]
        if not np.allclose([last.training_auroc, last.validation_auroc],
                           [reference.training_auroc, reference.validation_auroc], atol=1e-10, rtol=0):
            raise RuntimeError("The 8,000-row endpoint differs from the recorded Q1 SVM result.")
    print(results.to_string(index=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    args = parser.parse_args()
    run_q2(args.train)


if __name__ == "__main__":
    main()
