"""Classifier comparison on the fixed v4 feature set, same LCO protocol.

Features (identical for every model): 126 style + TF-IDF/LR score + 35 GPT-2 + 31 Qwen3.5
Binoculars features. Only the classifier changes. No self-training here, so the comparison
is purely about the learner; then rank-average blends of the best learners are scored.

  python submissions/Group_Number_19/yuchen_models.py --stage cv
  python submissions/Group_Number_19/yuchen_models.py --stage predict --models catboost lightgbm
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from folds import SEED  # noqa: E402
from yuchen_semi import Data, fit_models, fold_frames, predict  # noqa: E402

ROOT = HERE.parents[1]


class Imputer:
    """Median-impute and clip so sklearn models get finite inputs."""
    def fit(self, X):
        Z = X.replace([np.inf, -np.inf], np.nan)
        self.med = Z.median()
        self.lo, self.hi = Z.quantile(0.001), Z.quantile(0.999)
        return self

    def transform(self, X):
        Z = X.replace([np.inf, -np.inf], np.nan).fillna(self.med)
        return Z.clip(self.lo, self.hi, axis=1).to_numpy(np.float32)


def run_model(name, Xtr, ytr, Xva, seeds):
    if name == "catboost":
        return predict(fit_models(Xtr, ytr, seeds), Xva)
    if name == "lightgbm":
        i_fit, i_stop = train_test_split(np.arange(len(Xtr)), test_size=0.1, stratify=ytr, random_state=SEED)
        ps = []
        for s in seeds:
            m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                   random_state=s, verbose=-1, n_jobs=4)
            m.fit(Xtr.iloc[i_fit], ytr[i_fit], eval_set=[(Xtr.iloc[i_stop], ytr[i_stop])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            ps.append(m.predict_proba(Xva)[:, 1])
        return np.mean(ps, axis=0)
    imp = Imputer().fit(Xtr)
    A, B = imp.transform(Xtr), imp.transform(Xva)
    if name == "logreg":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    elif name == "mlp":
        ps = []
        for s in seeds:
            m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=s),
                              MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=128,
                                            learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=15, random_state=s))
            ps.append(m.fit(A, ytr).predict_proba(B)[:, 1])
        return np.mean(ps, axis=0)
    elif name == "random_forest":
        m = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=SEED)
    elif name == "extra_trees":
        m = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=SEED)
    else:
        raise ValueError(name)
    return m.fit(A, ytr).predict_proba(B)[:, 1]


def rank(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def stage_cv(D: Data, out: Path, models, seeds) -> None:
    oofs, rows = {}, []
    for name in models:
        t = time.time(); oof = np.zeros(len(D.y))
        for k in np.unique(D.fold):
            tri, vai, Xtr, Xva, _ = fold_frames(D, k)
            oof[vai] = run_model(name, Xtr, D.y[tri], Xva, seeds)
        oofs[name] = oof
        r = dict(model=name, **D.metrics(oof), seconds=round(time.time() - t))
        print(json.dumps(r), flush=True); rows.append(r)
    pd.DataFrame({"id": D.tr["id"], "cluster": D.fold, "label": D.y, **oofs}).to_csv(out / "lco_models_oof.csv", index=False)
    # Ranks are taken within each held-out fold's model before pooling would be ideal, but every
    # model here is a probability model, so global ranks are comparable across folds.
    R = {m: rank(p) for m, p in oofs.items()}
    good = [m for m in models if m != "logreg"]
    for r_ in (2, 3):
        for combo in itertools.combinations(good, r_):
            p = np.mean([R[m] for m in combo], axis=0)
            row = dict(model="blend:" + "+".join(combo), **D.metrics(p))
            print(json.dumps(row), flush=True); rows.append(row)
    pd.DataFrame(rows).to_csv(out / "models_results.csv", index=False)


def stage_predict(D: Data, out: Path, models, seeds) -> None:
    from sklearn.metrics import roc_auc_score  # noqa: F401
    z = np.load(D.v2 / "lr_full.npz")
    X, Xte = D.X.assign(lr_score=z["oof"]), D.Xte.assign(lr_score=z["pt"])
    preds = {}
    for name in models:
        preds[name] = run_model(name, X, D.y, Xte, seeds)
        pd.DataFrame({"id": D.te["id"], "label": preds[name]}).to_csv(out / f"test_{name}.csv", index=False)
    p = np.mean([rank(v) for v in preds.values()], axis=0)
    name = "submission_blend_" + "_".join(models) + ".csv"
    pd.DataFrame({"id": D.te["id"], "label": p}).to_csv(out / name, index=False)
    print("wrote", name, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cv", "predict"), required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--v2", type=Path, default=ROOT / "results" / "yuchen" / "v2_shift")
    ap.add_argument("--lm", type=Path, default=ROOT / "results" / "yuchen" / "lm_features")
    ap.add_argument("--bino", type=Path, default=ROOT / "results" / "yuchen" / "lm_features" / "qwen35_2b")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "models")
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "mlp", "random_forest", "extra_trees", "lightgbm", "catboost"])
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    D = Data(a.data, a.v2, a.lm, a.bino)
    seeds = [SEED + i for i in range(a.seeds)]
    if a.stage == "cv": stage_cv(D, a.out, a.models, seeds)
    if a.stage == "predict": stage_predict(D, a.out, a.models, seeds)


if __name__ == "__main__":
    main()
