"""Phase 2: tune and compare models on the fixed validation set (report Q1.2, Q1.3).

Protocol: hyperparameters are chosen by 5-fold CV inside the 8,000-row training partition only.
The chosen configuration is refit on those 8,000 rows and scored once on the fixed validation set.

Usage:
    python -m experiments.model_comparison                    # all models
    python -m experiments.model_comparison --models word_lr   # re-run some, keep the other rows

Outputs (results/model_comparison/):
    comparison.csv, comparison.md   one row per model: CV, train and validation AUROC, timings
    best_params.json                chosen settings, read by experiments.make_submission
    tuning_<model>.csv              every candidate with its mean/std CV AUROC
    val_predictions.csv             validation scores per model (for error analysis and ensembling)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models import MODELS, SEED, build_tuned, scores
from src.split import load_train_val

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "model_comparison"


def bootstrap_ci(y, s, n=1000, seed=SEED):
    """95% percentile bootstrap interval for AUROC."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    aucs = [roc_auc_score(y[i], s[i]) for i in idx]
    return np.percentile(aucs, [2.5, 97.5])


def tune(spec, X, y):
    """Return (best_params, best_cv_auroc, cv_std); an empty grid gets a plain 5-fold CV score."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(spec.build(), spec.grid or [{}], scoring="roc_auc", cv=cv, n_jobs=-1, refit=False)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_fit_time"]
    res[keep].sort_values("mean_test_score", ascending=False).to_csv(OUT_DIR / f"tuning_{spec.name}.csv", index=False)
    return search.best_params_, search.best_score_, res.std_test_score[search.best_index_]


def evaluate(spec, train, val):
    X_train, y_train = spec.inputs(train), train.label.astype(int).values
    y_val = val.label.astype(int).values

    t0 = time.time()
    params, cv_auc, cv_std = tune(spec, X_train, y_train)
    tune_time = time.time() - t0

    model = build_tuned(spec.name, params)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    val_scores = scores(model, spec.inputs(val))
    predict_time = time.time() - t0
    stripped_scores = scores(model, spec.inputs(val, layout_stripped=True))
    low, high = bootstrap_ci(y_val, val_scores)

    row = {
        "model": spec.name,
        "representation": spec.representation,
        "cv_auroc": cv_auc,
        "cv_std": cv_std,
        "train_auroc": roc_auc_score(y_train, scores(model, X_train)),
        "val_auroc": roc_auc_score(y_val, val_scores),
        "val_ci_low": low,
        "val_ci_high": high,
        "val_auroc_layout_stripped": roc_auc_score(y_val, stripped_scores),
        "fit_time_s": fit_time,
        "predict_val_time_s": predict_time,
        "tune_time_s": tune_time,
        "candidates": len(pd.read_csv(OUT_DIR / f"tuning_{spec.name}.csv")),
    }
    return row, params, val_scores


def merge_csv(path, new, key):
    """Replace rows for re-run models and keep the rest, so partial re-runs lose nothing."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val = load_train_val()
    params_path = OUT_DIR / "best_params.json"
    best_params = json.loads(params_path.read_text()) if params_path.exists() else {}
    preds_path = OUT_DIR / "val_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else val[["id", "label"]].copy()

    rows = []
    for name in args.models:
        print(f"--- {name}", flush=True)
        row, params, val_scores = evaluate(MODELS[name], train, val)
        print(f"    val AUROC {row['val_auroc']:.4f}  params {params}  ({row['tune_time_s']:.0f}s tuning)", flush=True)
        rows.append(row)
        best_params[name] = params
        preds[name] = val_scores

    table = merge_csv(OUT_DIR / "comparison.csv", pd.DataFrame(rows), "model").sort_values("val_auroc", ascending=False)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    (OUT_DIR / "comparison.md").write_text(table.round(4).to_markdown(index=False) + "\n")
    params_path.write_text(json.dumps(best_params, indent=2, default=list) + "\n")
    preds.to_csv(preds_path, index=False)
    print(table.drop(columns="representation").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
