"""Step 4 - train logistic regression and gradient boosting on the LM features.

Features: the 11 LM features from extract_features.py (n_tokens, id, label and max_length are
excluded). Optional: --use-n-tokens (ablation) and --flags (has_zero_width, has_lookalike,
n_ellipsis from normalize.raw_flags on the RAW text). n_em_dash is never used.

The 8,000 / 2,000 split comes from split.get_split. The scaler is fit on the 8,000 train rows only.
Model selection is by validation AUROC only.

Outputs (under --out-dir, default: this folder; all gitignored):
  results/results.csv        one appended row per config (train AUROC, val AUROC, fit time)
  preds/val/<name>.csv       id,label  (label = predicted P(class 1))
  preds/test/<name>.csv      id,label

Usage:
  python train.py --features features/train_qwen0.5B.parquet --test-features features/test_qwen0.5B.parquet
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from normalize import raw_flags
from split import get_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261023

LM_FEATURES = [
    "base_lp_mean", "base_lp_std", "instr_lp_mean", "instr_lp_std",
    "base_logrank_mean", "instr_logrank_mean",
    "base_entropy_mean", "instr_entropy_mean",
    "bino_ppl", "bino_xppl", "binoculars",
]
FLAG_FEATURES = ["has_zero_width", "has_lookalike", "n_ellipsis"]  # n_em_dash deliberately absent

LR_GRID = {"C": [0.01, 0.1, 1, 10]}
GB_GRID = {"max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300]}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_feature_file(path):
    df = pd.read_parquet(path)
    missing = [c for c in LM_FEATURES + ["id", "max_length"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    lengths = df["max_length"].unique()
    if len(lengths) != 1:
        raise ValueError(f"{path} mixes max_length values {lengths}")
    return df


def attach_features(frame, feats, feature_cols, what):
    """Left-join feature columns onto frame (id,text,label[,...]) and fail loudly on gaps."""
    cols = ["id"] + [c for c in feature_cols if c in feats.columns]
    out = frame.merge(feats[cols], on="id", how="left", validate="one_to_one")
    bad = out[feature_cols].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} rows have no features (e.g. ids {out.loc[bad, 'id'].head(3).tolist()})")
    return out


def add_flags(frame):
    flags = pd.DataFrame([raw_flags(t) for t in frame["text"].astype(str)])
    out = frame.copy()
    out["has_zero_width"] = flags["has_zero_width"].to_numpy()
    out["has_lookalike"] = flags["has_lookalike"].to_numpy()
    out["n_ellipsis"] = flags["n_ellipsis"].to_numpy()
    return out


def feature_set_name(args):
    return "lm" + ("+ntok" if args.use_n_tokens else "") + ("+flags" if args.flags else "")


def configs():
    for C in LR_GRID["C"]:
        yield f"lr_C{C}", "logreg", {"C": C}
    for d, lr, n in itertools.product(GB_GRID["max_depth"], GB_GRID["learning_rate"], GB_GRID["n_estimators"]):
        yield f"gb_d{d}_lr{lr}_n{n}", "gboost", {"max_depth": d, "learning_rate": lr, "n_estimators": n}


def make_model(kind, params, seed):
    if kind == "logreg":
        return LogisticRegression(C=params["C"], max_iter=5000, random_state=seed)
    return GradientBoostingClassifier(random_state=seed, **params)


def write_preds(path, ids, scores):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"id": ids, "label": scores}).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=os.path.join(HERE, "data", "train.csv"), help="path to train.csv")
    ap.add_argument("--test", default=os.path.join(HERE, "data", "test.csv"), help="path to test.csv (for --flags)")
    ap.add_argument("--features", default=os.path.join(HERE, "features", "train_qwen0.5B.parquet"))
    ap.add_argument("--test-features", default=os.path.join(HERE, "features", "test_qwen0.5B.parquet"))
    ap.add_argument("--out-dir", default=HERE, help="where results/ and preds/ are written")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--use-n-tokens", action="store_true", help="ablation: add n_tokens")
    ap.add_argument("--flags", action="store_true", help="add has_zero_width, has_lookalike, n_ellipsis")
    args = ap.parse_args()
    set_seeds(args.seed)

    cols = LM_FEATURES + (["n_tokens"] if args.use_n_tokens else []) + (FLAG_FEATURES if args.flags else [])
    fset = feature_set_name(args)

    # ---- data: split from split.py, features joined on id ----
    train_df, val_df = get_split(args.seed, args.train)
    feats = load_feature_file(args.features)
    test_feats = load_feature_file(args.test_features)
    if feats["max_length"].iloc[0] != test_feats["max_length"].iloc[0]:
        raise ValueError("train and test features were extracted with different max_length")
    test_df = pd.read_csv(args.test)
    if args.flags:
        train_df, val_df, test_df = add_flags(train_df), add_flags(val_df), add_flags(test_df)
    lm_cols = [c for c in cols if c not in FLAG_FEATURES]
    train_df = attach_features(train_df, feats, lm_cols, "train split")
    val_df = attach_features(val_df, feats, lm_cols, "val split")
    test_df = attach_features(test_df, test_feats, lm_cols, "test")
    assert len(train_df) == 8000 and len(val_df) == 2000

    # ---- standardize on the 8,000 train rows only ----
    scaler = StandardScaler().fit(train_df[cols].to_numpy(dtype=float))
    X_tr, X_val, X_te = (scaler.transform(d[cols].to_numpy(dtype=float)) for d in (train_df, val_df, test_df))
    y_tr, y_val = train_df["label"].to_numpy(), val_df["label"].to_numpy()

    results_path = os.path.join(args.out_dir, "results", "results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    max_length = int(feats["max_length"].iloc[0])
    scored = []
    for name, kind, params in configs():
        model = make_model(kind, params, args.seed)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        p_tr, p_val, p_te = (model.predict_proba(X)[:, 1] for X in (X_tr, X_val, X_te))
        row = {
            "run_id": run_id, "feature_set": fset, "config": name, "model": kind,
            "params": repr(params), "train_auroc": roc_auc_score(y_tr, p_tr),
            "val_auroc": roc_auc_score(y_val, p_val), "train_time_s": train_time,
            "n_features": len(cols), "max_length": max_length, "seed": args.seed,
        }
        pd.DataFrame([row]).to_csv(results_path, mode="a", index=False, header=not os.path.exists(results_path))
        pred_name = f"{fset}__{name}"
        write_preds(os.path.join(args.out_dir, "preds", "val", f"{pred_name}.csv"), val_df["id"], p_val)
        write_preds(os.path.join(args.out_dir, "preds", "test", f"{pred_name}.csv"), test_df["id"], p_te)
        scored.append(row)
        print(f"{fset:<12} {name:<22} train_auroc={row['train_auroc']:.4f} "
              f"val_auroc={row['val_auroc']:.4f} fit={train_time:.1f}s", flush=True)

    best = max(scored, key=lambda r: r["val_auroc"])  # selection on val AUROC only
    print(f"\nbest by val AUROC: {best['config']} ({best['val_auroc']:.4f})  feature_set={fset}")


if __name__ == "__main__":
    main()
