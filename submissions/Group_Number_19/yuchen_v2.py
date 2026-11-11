"""Yuchen v2: shift-aware CatBoost (126 features + TF-IDF LR score, optional test pseudo-labels).

Why v2: random-fold AUROC is ~0.999 but the public leaderboard was 0.975, because
test is ~36% poetry/fiction (topic cluster 4) while train has 1.5%. v1 was tuned on a
random holdout that cannot see this shift. v2 selects models with leave-one-topic-
cluster-out CV (LCO) weighted to the test cluster mix -- the team's shift proxy.

Stages (run from the repository root):
  python submissions/Group_Number_19/yuchen_v2.py --stage features   # cache features (~1 min)
  python submissions/Group_Number_19/yuchen_v2.py --stage lr         # cache LR scores (~8 min)
  python submissions/Group_Number_19/yuchen_v2.py --stage cv         # LCO comparison (~1 h on 2 cores)
  python submissions/Group_Number_19/yuchen_v2.py --stage predict    # write submissions (~10 min)

`--stage predict` writes two files to --out:
  submission_v2_cat.csv         inductive: trained on train.csv only
  submission_v2_cat_pseudo.csv  + one round of self-training on confident test predictions
                                (thr 0.95, weight 1). Uses test TEXT only, never labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline_tfidf import build_features, make_classifier  # noqa: E402
from clusters import make_clusters  # noqa: E402
from folds import SEED  # noqa: E402
from yuchen_features import features as v1_features  # noqa: E402
from yuchen_features_v2 import v2_features  # noqa: E402

ROOT = HERE.parents[1]
SEEDS = [SEED, SEED + 1, SEED + 2]
CAT = dict(iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3, eval_metric="AUC",
           od_type="Iter", od_wait=150, use_best_model=True, verbose=False)
PSEUDO_THR, PSEUDO_W = 0.95, 1.0


# ---------------------------------------------------------------- features
def feature_table(texts) -> pd.DataFrame:
    _, ext = v1_features(texts)                       # 53 team + 37 v1 features
    return pd.concat([ext.reset_index(drop=True), v2_features(texts)], axis=1)  # + 36 v2


def stage_features(data: Path, out: Path) -> None:
    for name in ("train", "test"):
        t = time.time()
        d = pd.read_csv(data / f"{name}.csv")
        F = feature_table(d["text"]); F.insert(0, "id", d["id"].to_numpy())
        F.to_pickle(out / f"feat_{name}.pkl")
        print(f"features {name}: {F.shape[1] - 1} columns, {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- LR meta-feature
def lr_crossfit(texts_tr, y_tr, others):
    """Strict cross-fit: each inner fold refits TF-IDF vocabulary/idf and the LR."""
    oof = np.zeros(len(texts_tr))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(texts_tr, y_tr):
        vec = build_features(0)
        clf = make_classifier("logreg", 30).fit(vec.fit_transform(texts_tr.iloc[a]), y_tr[a])
        oof[b] = clf.predict_proba(vec.transform(texts_tr.iloc[b]))[:, 1]
    vec = build_features(0)
    clf = make_classifier("logreg", 30).fit(vec.fit_transform(texts_tr), y_tr)
    return oof, [clf.predict_proba(vec.transform(o))[:, 1] for o in others]


def stage_lr(data: Path, out: Path) -> None:
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = tr["label"].to_numpy()
    t = time.time()
    oof, (pt,) = lr_crossfit(tr["text"], y, [te["text"]])
    np.savez(out / "lr_full.npz", oof=oof, pt=pt)
    print(f"LR full: {time.time() - t:.0f}s", flush=True)
    fold, _ = make_clusters(tr, te)
    for k in np.unique(fold):
        t = time.time()
        tri, vai = np.where(fold != k)[0], np.where(fold == k)[0]
        oof, (pv,) = lr_crossfit(tr["text"].iloc[tri].reset_index(drop=True), y[tri],
                                 [tr["text"].iloc[vai]])
        np.savez(out / f"lr_lco{k}.npz", tri=tri, vai=vai, oof=oof, pv=pv)
        print(f"LR leave-cluster-{k}-out: {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- model
def fit_cat(X, y, X_pred, seed, w=None):
    i_fit, i_stop = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=SEED)
    m = CatBoostClassifier(**CAT, random_seed=seed)
    m.fit(X.iloc[i_fit], y[i_fit], sample_weight=None if w is None else w[i_fit],
          eval_set=(X.iloc[i_stop], y[i_stop]))
    return m.predict_proba(X_pred)[:, 1], m.get_best_iteration()


def fit_predict(X, y, X_pred, seeds, pseudo: bool):
    """Seed-averaged CatBoost; with pseudo=True, one self-training round on X_pred."""
    p = np.mean([fit_cat(X, y, X_pred, s)[0] for s in seeds], axis=0)
    if not pseudo:
        return p, None
    sel = (p > PSEUDO_THR) | (p < 1 - PSEUDO_THR)
    Xs = pd.concat([X, X_pred[sel]], ignore_index=True)
    ys = np.r_[y, (p[sel] > 0.5).astype(int)]
    w = np.r_[np.ones(len(X)), np.full(sel.sum(), PSEUDO_W)]
    p2 = np.mean([fit_cat(Xs, ys, X_pred, s, w)[0] for s in seeds], axis=0)
    return p2, dict(selected=int(sel.sum()), pseudo_ai=int((p[sel] > 0.5).sum()), teacher=p)


# ---------------------------------------------------------------- CV
def stage_cv(data: Path, out: Path, seeds) -> None:
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = tr["label"].to_numpy()
    F = pd.read_pickle(out / "feat_train.pkl").drop(columns="id")
    fold, _ = make_clusters(tr, te)
    cl = pd.read_csv(HERE / "clusters.csv")
    shares = pd.crosstab(cl["cluster"], cl["split"], normalize="columns")
    w_test = pd.Series(fold).map(shares["test"] / shares["train"]).to_numpy()
    blank = tr["text"].str.contains(r"\n[ \t]*\n").to_numpy()
    cols = {"base53": list(F.columns[:53]), "v1_ext90": list(F.columns[:90]), "v2_all126": list(F.columns)}
    runs = [("v1_ext90+lr", "v1_ext90", False), ("v2_all126+lr", "v2_all126", False),
            ("v2_all126+lr+pseudo", "v2_all126", True)]
    results = []
    for name, cset, pseudo in runs:
        t = time.time(); oof = np.zeros(len(y))
        for k in np.unique(fold):
            z = np.load(out / f"lr_lco{k}.npz"); tri, vai = z["tri"], z["vai"]
            Xtr = F.iloc[tri][cols[cset]].reset_index(drop=True).assign(lr_score=z["oof"])
            Xva = F.iloc[vai][cols[cset]].reset_index(drop=True).assign(lr_score=z["pv"])
            oof[vai], _ = fit_predict(Xtr, y[tri], Xva, seeds, pseudo)
        c4 = fold == 4
        r = dict(model=name, lco_test_weighted_auc=roc_auc_score(y, oof, sample_weight=w_test),
                 lco_nonblank_weighted_auc=roc_auc_score(y[~blank], oof[~blank], sample_weight=w_test[~blank]),
                 lco_cluster4_auc=roc_auc_score(y[c4], oof[c4]), lco_pooled_auc=roc_auc_score(y, oof),
                 seconds=round(time.time() - t))
        print(json.dumps(r), flush=True); results.append(r)
        pd.DataFrame({"id": tr["id"], "cluster": fold, "label": y, "pred": oof}).to_csv(out / f"lco_{name}.csv", index=False)
    pd.DataFrame(results).to_csv(out / "lco_results.csv", index=False)


# ---------------------------------------------------------------- predict
def stage_predict(data: Path, out: Path, seeds) -> None:
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    y = tr["label"].to_numpy()
    F = pd.read_pickle(out / "feat_train.pkl"); T = pd.read_pickle(out / "feat_test.pkl")
    assert (F["id"].to_numpy() == tr["id"].to_numpy()).all() and (T["id"].to_numpy() == te["id"].to_numpy()).all()
    z = np.load(out / "lr_full.npz")
    X = F.drop(columns="id").assign(lr_score=z["oof"]); Xt = T.drop(columns="id").assign(lr_score=z["pt"])
    info = {}
    for name, pseudo in (("v2_cat", False), ("v2_cat_pseudo", True)):
        t = time.time()
        p, meta = fit_predict(X, y, Xt, seeds, pseudo)
        assert np.isfinite(p).all() and len(p) == len(te)
        pd.DataFrame({"id": te["id"], "label": p}).to_csv(out / f"submission_{name}.csv", index=False)
        info[name] = dict(seconds=round(time.time() - t), pred_ai_share=float((p > 0.5).mean()),
                          **({k: v for k, v in meta.items() if k != "teacher"} if meta else {}))
        print(name, json.dumps(info[name]), flush=True)
    json.dump(dict(seeds=seeds, catboost=CAT, pseudo_thr=PSEUDO_THR, pseudo_weight=PSEUDO_W,
                   n_features=X.shape[1], runs=info), open(out / "predict_info.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("features", "lr", "cv", "predict", "all"), default="all")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "v2_shift")
    ap.add_argument("--cv-seeds", type=int, default=1, help="seeds per CV fit (1 keeps CV ~1 h)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stages = ["features", "lr", "cv", "predict"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "features": stage_features(a.data, a.out)
        if s == "lr": stage_lr(a.data, a.out)
        if s == "cv": stage_cv(a.data, a.out, SEEDS[:a.cv_seeds])
        if s == "predict": stage_predict(a.data, a.out, SEEDS)


if __name__ == "__main__":
    main()
