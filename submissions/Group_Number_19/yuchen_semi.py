"""Yuchen: learning from unlabeled data + effect of training-set size (on the v3 feature set).

Feature set = v3: 126 style features + TF-IDF/LR score + GPT-2 small/medium LM features
(optionally + Qwen3.5 Binoculars with --bino). Model = 3-seed CatBoost (yuchen_v2.CAT).
Validation = leave-one-topic-cluster-out (LCO), AUC weighted to the test cluster mix.

Stages (from the repository root; needs the v2 caches and GPT-2 features for unlabeled):
  python submissions/Group_Number_19/yuchen_semi.py --stage prep     # unlabeled features + LR scores (~3 min)
  python submissions/Group_Number_19/yuchen_semi.py --stage semi     # self-training pool comparison
  python submissions/Group_Number_19/yuchen_semi.py --stage curve    # learning curve (~15 min)
  python submissions/Group_Number_19/yuchen_semi.py --stage predict  # submissions for each pool

Self-training (one round): fit a teacher on labeled data, pseudo-label the confident
(P>0.95 or P<0.05) rows of a POOL of unlabeled texts, refit on labeled + pseudo-labeled.
Pools compared:
  none      supervised only
  unl       unlabeled.csv (3,000 texts provided for semi-supervised learning)
  target    the texts being predicted (held-out cluster in CV / test.csv at predict time;
            transductive -- this is what v2/v3 did)
  unl+target both
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
from yuchen_v2 import CAT, PSEUDO_THR, PSEUDO_W, SEEDS, feature_table  # noqa: E402
from yuchen_v3 import bino_table, lm_table  # noqa: E402

ROOT = HERE.parents[1]
POOLS = ("none", "unl", "target", "unl+target")


# ---------------------------------------------------------------- data
def lr_fit_predict(texts_tr, y_tr, others):
    vec = build_features(0)
    clf = make_classifier("logreg", 30).fit(vec.fit_transform(texts_tr), y_tr)
    return [clf.predict_proba(vec.transform(o))[:, 1] for o in others]


def lr_crossfit(texts_tr, y_tr, others, seed=SEED):
    oof = np.zeros(len(texts_tr))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(texts_tr, y_tr):
        oof[b] = lr_fit_predict(texts_tr.iloc[a], y_tr[a], [texts_tr.iloc[b]])[0]
    return oof, lr_fit_predict(texts_tr, y_tr, others)


def stage_prep(data: Path, v2: Path) -> None:
    tr, te, un = (pd.read_csv(data / f"{s}.csv") for s in ("train", "test", "unlabeled"))
    t = time.time()
    U = feature_table(un["text"]); U.insert(0, "id", un["id"].to_numpy())
    U.to_pickle(v2 / "feat_unlabeled.pkl")
    print(f"features unlabeled: {U.shape[1] - 1} columns, {time.time() - t:.0f}s", flush=True)
    y = tr["label"].to_numpy()
    t = time.time()
    (pu,) = lr_fit_predict(tr["text"], y, [un["text"]])     # same final model as lr_full.npz
    np.save(v2 / "lr_full_unl.npy", pu)
    fold, _ = make_clusters(tr, te)
    for k in np.unique(fold):
        tri = np.where(fold != k)[0]
        (pu,) = lr_fit_predict(tr["text"].iloc[tri], y[tri], [un["text"]])
        np.save(v2 / f"lr_lco{k}_unl.npy", pu)
    print(f"LR scores for unlabeled (full + 8 folds): {time.time() - t:.0f}s", flush=True)


class Data:
    def __init__(self, data: Path, v2: Path, lm: Path, bino: Path | None):
        self.tr, self.te, self.un = (pd.read_csv(data / f"{s}.csv") for s in ("train", "test", "unlabeled"))
        self.y = self.tr["label"].to_numpy()
        blocks = {}
        for s, d in (("train", self.tr), ("test", self.te), ("unlabeled", self.un)):
            F = pd.read_pickle(v2 / f"feat_{s}.pkl")
            assert (F["id"].to_numpy() == d["id"].to_numpy()).all(), s
            parts = [F.drop(columns="id"), lm_table(lm, s, d["id"].to_numpy())]
            self.style_cols, self.lm_cols = list(parts[0].columns), list(parts[1].columns)
            if bino is not None:
                parts.append(bino_table(bino, s, d["id"].to_numpy()))
            blocks[s] = pd.concat(parts, axis=1)
        self.X, self.Xte, self.Xun = blocks["train"], blocks["test"], blocks["unlabeled"]
        self.fold, _ = make_clusters(self.tr, self.te)
        cl = pd.read_csv(HERE / "clusters.csv")
        sh = pd.crosstab(cl["cluster"], cl["split"], normalize="columns")
        self.w = pd.Series(self.fold).map(sh["test"] / sh["train"]).to_numpy()
        self.blank = self.tr["text"].str.contains(r"\n[ \t]*\n").to_numpy()
        self.v2 = v2

    def metrics(self, oof, idx=None):
        idx = np.arange(len(self.y)) if idx is None else idx
        y, w, b, p = self.y[idx], self.w[idx], self.blank[idx], oof[idx]
        return dict(lco_test_weighted_auc=roc_auc_score(y, p, sample_weight=w),
                    lco_nonblank_weighted_auc=roc_auc_score(y[~b], p[~b], sample_weight=w[~b]),
                    lco_pooled_auc=roc_auc_score(y, p))


# ---------------------------------------------------------------- model
def fit_models(X, y, seeds, w=None):
    i_fit, i_stop = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=SEED)
    ms = []
    for s in seeds:
        m = CatBoostClassifier(**CAT, random_seed=s)
        m.fit(X.iloc[i_fit], y[i_fit], sample_weight=None if w is None else w[i_fit],
              eval_set=(X.iloc[i_stop], y[i_stop]))
        ms.append(m)
    return ms


def predict(ms, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)


def self_train(X, y, targets: list[pd.DataFrame], pool: pd.DataFrame | None, seeds):
    """Return predictions for each frame in `targets` (+ selection stats)."""
    teacher = fit_models(X, y, seeds)
    if pool is None or len(pool) == 0:
        return [predict(teacher, t) for t in targets], {}
    p = predict(teacher, pool)
    sel = (p > PSEUDO_THR) | (p < 1 - PSEUDO_THR)
    Xs = pd.concat([X, pool[sel]], ignore_index=True)
    ys = np.r_[y, (p[sel] > 0.5).astype(int)]
    w = np.r_[np.ones(len(X)), np.full(sel.sum(), PSEUDO_W)]
    student = fit_models(Xs, ys, seeds, w)
    return [predict(student, t) for t in targets], dict(pool=len(pool), selected=int(sel.sum()),
                                                         pseudo_ai_share=float((p[sel] > 0.5).mean()))


def fold_frames(D: Data, k: int, tri: np.ndarray | None = None):
    z = np.load(D.v2 / f"lr_lco{k}.npz")
    tri_full, vai = z["tri"], z["vai"]
    Xtr = D.X.iloc[tri_full].reset_index(drop=True).assign(lr_score=z["oof"])
    Xva = D.X.iloc[vai].reset_index(drop=True).assign(lr_score=z["pv"])
    Xun = D.Xun.reset_index(drop=True).assign(lr_score=np.load(D.v2 / f"lr_lco{k}_unl.npy"))
    return tri_full, vai, Xtr, Xva, Xun


# ---------------------------------------------------------------- stages
def stage_semi(D: Data, out: Path, seeds) -> None:
    rows = []
    for pool_name in POOLS:
        t = time.time(); oof = np.zeros(len(D.y)); stats = []
        for k in np.unique(D.fold):
            tri, vai, Xtr, Xva, Xun = fold_frames(D, k)
            pool = {"none": None, "unl": Xun, "target": Xva,
                    "unl+target": pd.concat([Xun, Xva], ignore_index=True)}[pool_name]
            (oof[vai],), st = self_train(Xtr, D.y[tri], [Xva], pool, seeds)
            stats.append(st.get("selected", 0))
        r = dict(pool=pool_name, **D.metrics(oof), mean_pseudo_selected=float(np.mean(stats)),
                 seconds=round(time.time() - t))
        print(json.dumps(r), flush=True); rows.append(r)
        pd.DataFrame({"id": D.tr["id"], "cluster": D.fold, "label": D.y, "pred": oof}).to_csv(
            out / f"lco_semi_{pool_name}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "semi_results.csv", index=False)


def stage_curve(D: Data, out: Path, seeds, fracs, repeats) -> None:
    """Subsample each LCO training fold (stratified); refit TF-IDF/LR on the subsample."""
    texts = D.tr["text"]
    rows = []
    for frac in fracs:
        for rep in range(repeats if frac < 1 else 1):
            t = time.time()
            oof = {m: np.zeros(len(D.y)) for m in ("tfidf_lr", "style_cat", "lm_cat", "v3_cat")}
            for k in np.unique(D.fold):
                tri, vai = np.where(D.fold != k)[0], np.where(D.fold == k)[0]
                if frac < 1:
                    sub, _ = train_test_split(tri, train_size=frac, stratify=D.y[tri], random_state=SEED + 100 * rep + k)
                    sub = np.sort(sub)
                else:
                    sub = tri
                y = D.y[sub]
                lr_oof, (lr_va,) = lr_crossfit(texts.iloc[sub].reset_index(drop=True), y, [texts.iloc[vai]])
                oof["tfidf_lr"][vai] = lr_va
                Xs, Xv = D.X.iloc[sub].reset_index(drop=True), D.X.iloc[vai].reset_index(drop=True)
                style, lm = D.style_cols, D.lm_cols
                oof["style_cat"][vai] = predict(fit_models(Xs[style], y, seeds), Xv[style])
                oof["lm_cat"][vai] = predict(fit_models(Xs[lm], y, seeds), Xv[lm])
                oof["v3_cat"][vai] = predict(fit_models(Xs.assign(lr_score=lr_oof), y, seeds),
                                             Xv.assign(lr_score=lr_va))
            for m, p in oof.items():
                r = dict(frac=frac, n_train_approx=int(round(frac * len(D.y) * 7 / 8)), repeat=rep, model=m,
                         **D.metrics(p))
                print(json.dumps(r), flush=True); rows.append(r)
            print(f"frac {frac} rep {rep}: {time.time() - t:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "curve_results.csv", index=False)


def stage_predict(D: Data, out: Path, seeds, tag: str) -> None:
    z = np.load(D.v2 / "lr_full.npz")
    X = D.X.assign(lr_score=z["oof"])
    Xte = D.Xte.assign(lr_score=z["pt"])
    Xun = D.Xun.assign(lr_score=np.load(D.v2 / "lr_full_unl.npy"))
    info = {}
    for pool_name in POOLS:
        pool = {"none": None, "unl": Xun, "target": Xte, "unl+target": pd.concat([Xun, Xte], ignore_index=True)}[pool_name]
        t = time.time()
        (p,), st = self_train(X, D.y, [Xte], pool, seeds)
        assert np.isfinite(p).all() and len(p) == len(D.te)
        name = f"submission_{tag}_{pool_name.replace('+', '_')}.csv"
        pd.DataFrame({"id": D.te["id"], "label": p}).to_csv(out / name, index=False)
        info[name] = dict(seconds=round(time.time() - t), pred_ai_share=float((p > 0.5).mean()), **st)
        print(name, json.dumps(info[name]), flush=True)
    json.dump(dict(seeds=seeds, catboost=CAT, pseudo_thr=PSEUDO_THR, n_features=X.shape[1], runs=info),
              open(out / f"predict_info_{tag}.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("prep", "semi", "curve", "predict"), required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--v2", type=Path, default=ROOT / "results" / "yuchen" / "v2_shift")
    ap.add_argument("--lm", type=Path, default=ROOT / "results" / "yuchen" / "lm_features")
    ap.add_argument("--bino", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "semi")
    ap.add_argument("--cv-seeds", type=int, default=1)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "prep":
        return stage_prep(a.data, a.v2)
    D = Data(a.data, a.v2, a.lm, a.bino)
    if a.stage == "semi": stage_semi(D, a.out, SEEDS[:a.cv_seeds])
    if a.stage == "curve": stage_curve(D, a.out, SEEDS[:a.cv_seeds], a.fracs, a.repeats)
    if a.stage == "predict": stage_predict(D, a.out, SEEDS, a.tag)


if __name__ == "__main__":
    main()
