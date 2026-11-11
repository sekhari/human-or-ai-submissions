"""Q5 reproducible ensemble comparison from three held-out prediction files.

No Kaggle labels or text data are required: reads only local cached validation
prediction tables. Do NOT commit row-level predictions to the public repo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    prob_cols = [s for s in ("probability", "p_ai", "m1_prob", "model1_p_ai", "model2_p_ai", "model3_p_ai") if s in df.columns]
    label_cols = [s for s in ("label", "true_label") if s in df.columns]
    if len(prob_cols) != 1 or len(label_cols) != 1:
        raise ValueError(f"{path}: need exactly one probability and truth-label column")
    df = df[["id", label_cols[0], prob_cols[0]]].copy()
    df.columns = ["id", "label", model_name]
    if len(df) != 2000 or not df.id.is_unique:
        raise ValueError(f"{path}: expected 2,000 unique validation IDs")
    if not df.label.isin([0, 1]).all() or not df[model_name].between(0, 1).all():
        raise ValueError(f"{path}: invalid truth labels or probabilities")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model1", type=Path, required=True, help="Model 1 validation predictions CSV")
    parser.add_argument("--model2", type=Path, required=True, help="Model 2 validation predictions CSV")
    parser.add_argument("--model3", type=Path, required=True, help="Model 3 validation predictions CSV")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.outdir}")
    args.outdir.mkdir(parents=True)

    inputs = [load(getattr(args, f"model{i}"), f"m{i}") for i in (1, 2, 3)]
    if not (set(inputs[0].id) == set(inputs[1].id) == set(inputs[2].id)):
        raise ValueError("Different validation ID memberships: do not ensemble")
    joined = inputs[0].merge(inputs[1], on=["id", "label"], validate="one_to_one")
    joined = joined.merge(inputs[2], on=["id", "label"], validate="one_to_one")
    if len(joined) != 2000 or joined.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Labels disagree or partitions are unbalanced")
    joined = joined.sort_values("id", key=lambda x: x.astype(int)).reset_index(drop=True)
    y = joined.label.to_numpy()
    p1, p2, p3 = (joined[f"m{i}"].to_numpy() for i in (1, 2, 3))

    # First six rows represent standalone and prespecified simple averaging.
    # Last two are exploratory unequal weights, tuned on this same holdout.
    mixtures = [
        ("Model 1", np.array([1., 0., 0.]), False),
        ("Model 2 (baseline)", np.array([0., 1., 0.]), False),
        ("Model 3", np.array([0., 0., 1.]), False),
        ("M1+M2 equal", np.array([.5, .5, 0.]), False),
        ("M2+M3 equal", np.array([0., .5, .5]), False),
        ("M1+M2+M3 equal", np.array([1/3, 1/3, 1/3]), False),
        ("M2 75% + M3 25%", np.array([0., .75, .25]), True),
        ("M1 20% + M2 60% + M3 20%", np.array([.2, .6, .2]), True),
    ]
    base_auc = roc_auc_score(y, p2)
    rows = []
    local_preds = joined[["id", "label"]].copy()
    for name, weights, exploratory in mixtures:
        prob = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
        auc = roc_auc_score(y, prob)
        # Benchmarks *only* NumPy aggregation of cached probabilities;
        # model training/feature extraction/prediction and AUROC are excluded.
        times_ns = []
        for _ in range(1000):
            t0 = perf_counter_ns()
            _ = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
            times_ns.append(perf_counter_ns() - t0)
        rows.append({"Configuration": name, "Validation AUROC": auc,
                     "Difference vs M2": auc - base_auc,
                     "Combination runtime median ms": float(np.median(times_ns)) / 1e6,
                     "Exploratory weights": exploratory})
        local_preds[name] = prob
    pd.DataFrame(rows).to_csv(args.outdir / "ensemble_summary.csv", index=False)
    local_preds.to_csv(args.outdir / "aligned_validation_predictions_LOCAL_ONLY.csv", index=False)
    (args.outdir / "README_LOCAL_ONLY.txt").write_text(
        "Per-row labels and predictions are validation-only; do not commit them to GitHub.\n")
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    print("Output:", args.outdir)

if __name__ == "__main__":
    main()
