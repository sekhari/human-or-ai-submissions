"""Blending / stacking utilities shared by scripts/blend.py and the run_all report.

All methods work on out-of-fold (OOF) predictions. Runs may come from different
CV layouts (e.g. 5-fold vs holdout); we only use the rows where *every* run has
an OOF prediction, which keeps the evaluation honest (each OOF value was made
by a model that never saw that row).

Methods
  rank_mean   : average of per-model rank-normalised scores (robust, no fitting)
  greedy      : Caruana-style forward selection with replacement on rank scores,
                maximising AUROC; its CV score is computed with nested CV
  stack_lr    : logistic regression on logit(OOF probabilities); CV-evaluated
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .utils import auroc


def rank01(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_runs(run_dirs) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Return (oof matrix on common rows, test matrix, y on common rows, names)."""
    oofs, tests, names = [], [], []
    for r in map(Path, run_dirs):
        if not (r / "oof.csv").exists() or not (r / "test_pred.csv").exists():
            continue
        o = pd.read_csv(r / "oof.csv").sort_values("id").reset_index(drop=True)
        t = pd.read_csv(r / "test_pred.csv").sort_values("id").reset_index(drop=True)
        oofs.append(o)
        tests.append(t)
        names.append(r.name)
    if not names:
        raise ValueError("No runs with both oof.csv and test_pred.csv")
    mask = np.ones(len(oofs[0]), bool)
    for o in oofs:
        mask &= o["fold"].to_numpy() >= 0
    y = oofs[0]["label"].to_numpy()[mask]
    O = pd.DataFrame({n: o["pred"].to_numpy()[mask] for n, o in zip(names, oofs)})
    T = pd.DataFrame({n: t["label"].to_numpy() for n, t in zip(names, tests)})
    T.index = tests[0]["id"].to_numpy()
    return O, T, y, names


def rank_mean(O: pd.DataFrame, T: pd.DataFrame, cols=None, weights=None):
    cols = list(cols or O.columns)
    w = np.ones(len(cols)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    oof = sum(wi * rank01(O[c]) for wi, c in zip(w, cols))
    test = sum(wi * rank01(T[c]) for wi, c in zip(w, cols))
    return oof, test


def cdf_mean(O: pd.DataFrame, T: pd.DataFrame, cols=None, weights=None):
    """Like rank_mean, but test scores are mapped through each model's OOF-score CDF instead of being
    ranked within test, so the blend doesn't depend on test's class/domain mix. On OOF rows it equals
    rank_mean."""
    cols = list(cols or O.columns)
    w = np.ones(len(cols)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    cdf = {c: (lambda x, ref=np.sort(O[c].to_numpy()): np.searchsorted(ref, x, side="right") / len(ref))
           for c in cols}
    oof = sum(wi * cdf[c](O[c].to_numpy()) for wi, c in zip(w, cols))
    test = sum(wi * cdf[c](T[c].to_numpy()) for wi, c in zip(w, cols))
    return oof, test


def greedy_weights(O: pd.DataFrame, y, n_iter: int = 30) -> dict:
    """Forward selection with replacement (Caruana et al., 2004) on rank scores."""
    R = {c: rank01(O[c]) for c in O.columns}
    chosen, cur = [], np.zeros(len(y))
    for _ in range(n_iter):
        best, best_auc = None, -1
        for c, r in R.items():
            a = auroc(y, (cur * len(chosen) + r) / (len(chosen) + 1))
            if a > best_auc:
                best, best_auc = c, a
        chosen.append(best)
        cur = (cur * (len(chosen) - 1) + R[best]) / len(chosen)
    counts = pd.Series(chosen).value_counts()
    return (counts / counts.sum()).to_dict()


def greedy(O, T, y, n_iter: int = 30, seed: int = 0):
    """Returns (honest nested-CV AUROC, oof_blend on all rows, test_blend, weights)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    nested = np.zeros(len(y))
    for tr, va in skf.split(O, y):
        w = greedy_weights(O.iloc[tr], y[tr], n_iter)
        nested[va] = sum(wi * rank01(O[c].to_numpy())[va] for c, wi in w.items())
    w = greedy_weights(O, y, n_iter)
    oof, test = rank_mean(O, T, list(w), list(w.values()))
    return auroc(y, nested), oof, test, w


def stack_lr(O, T, y, C: float = 1.0, seed: int = 0):
    """Logistic-regression stacker on logit(probabilities). Returns
    (CV AUROC, oof_stack, test_stack, coefficients)."""
    X, Xt = logit(O.to_numpy()), logit(T[O.columns].to_numpy())
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    m = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    return auroc(y, oof), oof, m.predict_proba(Xt)[:, 1], dict(zip(O.columns, m.coef_.ravel()))
