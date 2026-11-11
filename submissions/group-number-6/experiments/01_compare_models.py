"""Q1: model comparison and hyperparameter selection.

Protocol
  1. Hyperparameters are chosen by 5-fold stratified CV *inside* the 8,000-row training
     partition (vectorisers are re-fit on each fold's training part). The fixed
     2,000-row validation set is never used for tuning.
  2. Each model is refit with its chosen setting on all 8,000 rows and scored once on the
     validation set. We record train AUROC, validation AUROC and wall-clock fit time.

Usage: python experiments/01_compare_models.py [--quick]
Outputs: results/q1_cv_grid.json, results/q1_comparison.json, results/val_preds_q1.npz
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, get_split  # noqa: E402
from src.features import StyleFeatures  # noqa: E402
from src.models import SEED, SVMProba, build, char_tfidf, proba, word_tfidf  # noqa: E402

RES = ROOT / "results"


def cv_linear(texts, y, folds, make_vec, make_clf, Cs, jobs=2):
    """For one vectoriser setting, return mean CV AUROC for each C (vectoriser fit per fold)."""
    def one_fold(tr, va):
        vec = make_vec()
        Xtr = vec.fit_transform(texts[tr])
        Xva = vec.transform(texts[va])
        out = []
        for C in Cs:
            clf = make_clf(C).fit(Xtr, y[tr])
            s = clf.predict_proba(Xva)[:, 1]
            out.append(roc_auc_score(y[va], s))
        return out
    scores = Parallel(n_jobs=jobs)(delayed(one_fold)(tr, va) for tr, va in folds)
    return np.mean(scores, axis=0), np.std(scores, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grids for a smoke test")
    args = ap.parse_args()

    tr, va = get_split()
    X, y = tr.text.to_numpy(), tr.label.to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    grid = {}

    # ---- Model A: word TF-IDF + LR -------------------------------------------------
    Cs_A = [1, 4, 16, 32, 64, 128, 256]
    grid["A_word_lr"] = []
    for ng in [(1, 1), (1, 2), (1, 3)]:
        t = time.time()
        m, s = cv_linear(X, y, folds, lambda ng=ng: word_tfidf(ng),
                         lambda C: LogisticRegression(C=C, max_iter=3000, solver="liblinear"), Cs_A)
        for C, mm, ss in zip(Cs_A, m, s):
            grid["A_word_lr"].append({"ngram": list(ng), "C": C, "cv_auc": mm, "cv_std": ss})
        print(f"A ngram={ng} best C={Cs_A[int(np.argmax(m))]} cv={m.max():.4f} ({time.time()-t:.0f}s)", flush=True)

    # ---- Model B: char TF-IDF + LR -------------------------------------------------
    Cs_B = [4, 16, 64, 256]
    grid["B_char_lr"] = []
    for an, ng in [("char_wb", (2, 4)), ("char_wb", (3, 5)), ("char", (2, 5))]:
        t = time.time()
        m, s = cv_linear(X, y, folds, lambda an=an, ng=ng: char_tfidf(ng, analyzer=an),
                         lambda C: LogisticRegression(C=C, max_iter=3000, solver="liblinear"), Cs_B)
        for C, mm, ss in zip(Cs_B, m, s):
            grid["B_char_lr"].append({"analyzer": an, "ngram": list(ng), "C": C, "cv_auc": mm, "cv_std": ss})
        print(f"B {an}{ng} best C={Cs_B[int(np.argmax(m))]} cv={m.max():.4f} ({time.time()-t:.0f}s)", flush=True)

    # ---- Model C: stylometric + HistGB (features are stateless -> compute once) -----
    t = time.time()
    S = StyleFeatures().transform(X)
    print(f"style features {S.shape} in {time.time()-t:.0f}s", flush=True)
    grid["C_style_gbm"] = []
    for lr, leaves, iters in itertools.product([0.03, 0.1], [15, 31, 63], [200, 500]):
        sc = []
        for a, b in folds:
            clf = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters,
                                                 l2_regularization=1.0, random_state=SEED).fit(S[a], y[a])
            sc.append(roc_auc_score(y[b], clf.predict_proba(S[b])[:, 1]))
        grid["C_style_gbm"].append({"lr": lr, "max_leaf_nodes": leaves, "max_iter": iters,
                                    "cv_auc": float(np.mean(sc)), "cv_std": float(np.std(sc))})
    best = max(grid["C_style_gbm"], key=lambda r: r["cv_auc"])
    print(f"C best {best}", flush=True)

    # ---- Model D: word counts + NB ---------------------------------------------------
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    grid["D_word_nb"] = []
    for ng in [(1, 1), (1, 2)]:
        m, s = cv_linear(X, y, folds,
                         lambda ng=ng: CountVectorizer(ngram_range=ng, min_df=2, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
                         lambda a: MultinomialNB(alpha=a), alphas)
        for a, mm, ss in zip(alphas, m, s):
            grid["D_word_nb"].append({"ngram": list(ng), "alpha": a, "cv_auc": mm, "cv_std": ss})
    print("D best", max(grid["D_word_nb"], key=lambda r: r["cv_auc"]), flush=True)

    # ---- Model E: word+char TF-IDF + linear SVM --------------------------------------
    from sklearn.pipeline import FeatureUnion
    Cs_E = [0.05, 0.2, 0.5, 2]
    m, s = cv_linear(X, y, folds, lambda: FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())]),
                     lambda C: SVMProba(C=C, max_iter=5000, random_state=SEED), Cs_E)
    grid["E_wordchar_svm"] = [{"C": C, "cv_auc": mm, "cv_std": ss} for C, mm, ss in zip(Cs_E, m, s)]
    print("E best", max(grid["E_wordchar_svm"], key=lambda r: r["cv_auc"]), flush=True)

    RES.mkdir(exist_ok=True)
    (RES / "q1_cv_grid.json").write_text(json.dumps(grid, indent=1, default=float))

    # ---- Refit chosen settings on 8,000 rows, score on fixed validation -----------
    chosen = {}
    for name, rows in grid.items():
        b = dict(max(rows, key=lambda r: r["cv_auc"]))
        b.pop("cv_auc"), b.pop("cv_std")
        if "ngram" in b:
            b["ngram"] = tuple(b["ngram"])
        if name == "C_style_gbm":
            b = {"lr": b["lr"], "max_leaf_nodes": b["max_leaf_nodes"], "max_iter": b["max_iter"]}
        chosen[name] = b

    comparison, val_preds = {}, {}
    for name, params in chosen.items():
        model = build(name, **params)
        t = time.time()
        model.fit(list(X), y)
        fit_s = time.time() - t
        p_tr = proba(model, X)
        p_va = proba(model, va.text)
        val_preds[name] = p_va
        comparison[name] = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                            "train_auc": roc_auc_score(y, p_tr), "val_auc": roc_auc_score(va.label, p_va),
                            "fit_seconds": fit_s}
        print(name, json.dumps(comparison[name]), flush=True)

    (RES / "q1_comparison.json").write_text(json.dumps(comparison, indent=1, default=float))
    np.savez_compressed(RES / "val_preds_q1.npz", ids=va.id.to_numpy(), y=va.label.to_numpy(), **val_preds)


if __name__ == "__main__":
    main()
