"""Generalization and error analysis of the final model (report Q4).

The final model, improve_all_blend, is refit on the 8,000-row training partition exactly as in
experiments.improve_lm and scored on the fixed 2,000-row validation set. Outputs:

- gap.csv: training AUROC (in-sample, and 5-fold out-of-fold within the training partition) against
  validation AUROC.
- subgroups.csv: validation AUROC by word-count bin, blank-line presence and topic domain, with
  bootstrap 95% intervals. The 10% RoBERTa rank blend is scored too.
- predictions.csv: validation scores with the observable attributes used for error inspection.
- errors.csv: false positives (human scored as AI) and false negatives (AI scored as human).

Errors are read from the final blend score, the same score submitted to Kaggle, with threshold 0.5.
That score is an average percentile rank, not a calibrated probability, so 0.5 labels the top-ranked
half of the validation set as AI, which matches its 50% AI rate. `p_ai` is kept for reference: the
probability output of the blend's gradient-boosting component.

Usage:
    python -m experiments.error_analysis    # writes results/error_analysis/
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments import improve_lm
from experiments.improve import fold_design
from experiments.model_comparison import bootstrap_ci
from experiments.robustness import load_domains
from src.data import load_split
from src.models import SEED

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "error_analysis"
NEURAL_VAL = ROOT / "results" / "curvature_roberta" / "single_validation_predictions.csv"
LENGTH_BINS = [0, 150, 250, np.inf]
LENGTH_LABELS = ["<150 words", "150-250 words", ">250 words"]
THRESHOLD = 0.5
CURLY = "[\u2018\u2019\u201c\u201d]"  # typographic quotes and apostrophes


def blend_parts(Xf, y, Xe):
    """Return (final blend score, P(AI) from the bagged boosting component)."""
    cols = list(Xf.columns)  # all_blend keeps every feature group
    boost = improve_lm.bagged_booster(Xf[cols], y, Xe[cols], max_features=improve_lm.COLSAMPLE)
    trees = improve_lm.extra_trees(Xf[cols], y, Xe[cols])
    return (rankdata(boost) + rankdata(trees)) / (2 * len(Xe)), boost


def auroc_row(group, name, y, s):
    if len(np.unique(y)) < 2:
        return {"grouping": group, "subgroup": name, "count": len(y), "n_ai": int(y.sum()),
                "auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    low, high = bootstrap_ci(y, s)
    return {"grouping": group, "subgroup": name, "count": len(y), "n_ai": int(y.sum()),
            "auroc": roc_auc_score(y, s), "ci_low": low, "ci_high": high}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, val, labeled = load_domains()
    labeled = labeled.reset_index(drop=True)
    unlabeled = load_split("unlabeled")
    in_val = labeled.id.isin(val.id)
    train, val = labeled[~in_val].reset_index(drop=True), labeled[in_val].reset_index(drop=True)
    frame = pd.concat([labeled[["id", "text"]], unlabeled[["id", "text"]]], ignore_index=True)
    feats = improve_lm.all_features(frame, unlabeled.text.tolist())
    y_train, y_val = train.label.astype(int).values, val.label.astype(int).values

    # Train/validation gap. Xf already holds out-of-fold text scores, so in-sample fit is the usual
    # "training AUROC"; the 5-fold score shows how the model does on unseen rows of the same partition.
    Xf, Xe, _ = fold_design(feats, train, val, unlabeled)
    val_blend, val_prob = blend_parts(Xf, y_train, Xe)
    train_blend, _ = blend_parts(Xf, y_train, Xf)
    oof = np.zeros(len(train))
    for fit_idx, ev_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, y_train):
        fit, ev = train.iloc[fit_idx], train.iloc[ev_idx]
        f_X, e_X, _ = fold_design(feats, fit, ev, unlabeled)
        oof[ev_idx] = blend_parts(f_X, y_train[fit_idx], e_X)[0]
    gap = pd.DataFrame([
        {"split": "train (in-sample, 8000)", "auroc": roc_auc_score(y_train, train_blend)},
        {"split": "train (5-fold out-of-fold, 8000)", "auroc": roc_auc_score(y_train, oof)},
        {"split": "validation (2000)", "auroc": roc_auc_score(y_val, val_blend)},
        {"split": "validation, boosting component P(AI)", "auroc": roc_auc_score(y_val, val_prob)},
    ])
    gap.to_csv(OUT_DIR / "gap.csv", index=False)

    neural = pd.read_csv(NEURAL_VAL).set_index("id").loc[val.id, "neural"].values
    rank10 = 0.9 * val_blend + 0.1 * rankdata(neural, method="average") / len(neural)
    v = feats.loc[val.id]
    pred = pd.DataFrame({
        "id": val.id.values, "label": y_val, "blend": val_blend, "p_ai": val_prob, "rank10": rank10,
        "words": val.text.str.split().str.len().values, "blank_lines": v.blank_lines.values,
        "domain": val.domain.values, "gpt2_nll_mean": v.gpt2_nll_mean.values,
        "curly_quotes": val.text.str.contains(CURLY).values,
    })
    pred["length_bin"] = pd.cut(pred.words, LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    pred.to_csv(OUT_DIR / "predictions.csv", index=False)

    rows = []
    groupings = {"length": "length_bin", "blank line": pred.blank_lines.gt(0).map({True: "has blank line",
                                                                                  False: "no blank line"}),
                 "typography": pred.curly_quotes.map({True: "curly quotes", False: "straight quotes only"}),
                 "domain": "domain"}
    for group, key in groupings.items():
        keys = pred[key] if isinstance(key, str) else key
        for name, part in pred.groupby(keys, observed=True):
            for model in ("blend", "rank10"):
                rows.append({**auroc_row(group, name, part.label.values, part[model].values), "model": model})
    pd.DataFrame(rows).to_csv(OUT_DIR / "subgroups.csv", index=False)

    fp = pred[(pred.label == 0) & (pred.blend >= THRESHOLD)].sort_values("blend", ascending=False)
    fn = pred[(pred.label == 1) & (pred.blend < THRESHOLD)].sort_values("blend")
    errors = pd.concat([fp.assign(error="FP"), fn.assign(error="FN")])
    errors.to_csv(OUT_DIR / "errors.csv", index=False)

    print(gap.round(5).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nFP={len(fp)} FN={len(fn)} at blend-score threshold {THRESHOLD}")
    print(errors.drop(columns=["rank10"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
