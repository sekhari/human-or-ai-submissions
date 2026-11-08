"""Try ways to narrow the gap between validation and the shifted test set (report Q5).

Every method starts from the gpt2_word_masked stack and keeps its boosting settings (chosen by LODO in
experiments.stacking), so no method gets its own tuning. Methods are compared by leave-one-domain-out
on the 8,000-row training partition; the fixed validation set is only scored. Because scores with
blank-line documents sit near 1, LODO on documents without blank lines is reported as a harder view.

Unlabeled data: unlabeled.csv text is used for drift estimates, density weights and pseudo-labels.
test.csv is not used for any of these, and no validation text enters a fit.

Methods:
    base        gpt2_word_masked
    curv        add probability-curvature features (src.neural.curvature_features)
    drop_drift  drop features whose KS distance from unlabeled.csv exceeds 0.1
    bag         mean of 10 bootstrap-resampled boosters
    weighted    weight training rows by how much they resemble unlabeled.csv
    pseudo      add unlabeled rows the booster scores above 0.99 or below 0.01, then refit

Usage:
    python -m experiments.improve                   # compare methods
    python -m experiments.improve --submit curv     # write submissions/improve_curv.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from experiments.make_submission import check_submission
from experiments.model_comparison import bootstrap_ci
from experiments.robustness import load_domains
from experiments.stacking import build_feature_set, text_scores
from src.data import load_split
from src.models import SEED

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "improve"
PARAMS = {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 0.0}
TEXT_MODELS = ["word_lr", "masked_lr"]
METHODS = {
    "base": set(),
    "curv": {"curv"},
    "drop_drift": {"drop_drift"},
    "bag": {"bag"},
    "weighted": {"weighted"},
    "pseudo": {"pseudo"},
    "curv_bag": {"curv", "bag"},
    "curv_pseudo": {"curv", "pseudo"},
}
DRIFT_LIMIT = 0.1
PSEUDO_CONFIDENCE = 0.99
N_BAGS = 10


def all_features(df):
    curvature = pd.read_csv(ROOT / "cache" / "gpt2_curvature.csv").set_index("id").loc[df.id]
    feats = build_feature_set("v2_gpt2", df).assign(**{c: curvature[c].values for c in curvature})
    return feats.set_index(df.id.values)


def fold_design(feats, fit, ev, unlabeled):
    """Stack matrices for one split: out-of-fold text scores to fit, full-fit scores elsewhere."""
    eval_all = pd.concat([ev, unlabeled], ignore_index=True)
    text = {m: text_scores(m, fit, eval_all) for m in TEXT_MODELS}
    n = len(ev)
    Xf = feats.loc[fit.id].assign(**{f"text_{m}": text[m][0] for m in TEXT_MODELS})
    Xe = feats.loc[ev.id].assign(**{f"text_{m}": text[m][1][:n] for m in TEXT_MODELS})
    Xu = feats.loc[unlabeled.id].assign(**{f"text_{m}": text[m][1][n:] for m in TEXT_MODELS})
    return Xf, Xe, Xu


def booster(Xf, y, Xe, weight=None):
    model = HistGradientBoostingClassifier(random_state=SEED, **PARAMS).fit(Xf, y, sample_weight=weight)
    return model.predict_proba(Xe)[:, 1]


def style_columns(X):
    return [c for c in X if not c.startswith("text_")]


def stable_columns(Xf, Xu):
    drifted = {c for c in style_columns(Xf) if ks_2samp(Xf[c], Xu[c]).statistic > DRIFT_LIMIT}
    return [c for c in Xf if c not in drifted]


def density_weights(Xf, Xu):
    """p(unlabeled-like | x) / p(train-like | x), clipped. Text scores are excluded because they are
    out-of-fold for one group and full-fit for the other, which alone would separate the groups."""
    cols = style_columns(Xf)
    X = pd.concat([Xf[cols], Xu[cols]])
    domain = np.r_[np.zeros(len(Xf)), np.ones(len(Xu))]
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=SEED)
    p = cross_val_predict(model, X, domain, cv=5, method="predict_proba")[: len(Xf), 1]
    weight = np.clip(p / (1 - p) * len(Xf) / len(Xu), 0.2, 5)
    return weight / weight.mean()


def fit_predict(method, Xf, y, Xe, Xu):
    spec = METHODS[method]
    curvature = [c for c in Xf if c.startswith("curv_")]
    cols = [c for c in Xf if "curv" in spec or c not in curvature]
    Xf, Xe, Xu = Xf[cols], Xe[cols], Xu[cols]
    if "drop_drift" in spec:
        cols = stable_columns(Xf, Xu)
        Xf, Xe, Xu = Xf[cols], Xe[cols], Xu[cols]
    weight = density_weights(Xf, Xu) if "weighted" in spec else None
    if "pseudo" in spec:
        p = booster(Xf, y, Xu)
        keep = (p >= PSEUDO_CONFIDENCE) | (p <= 1 - PSEUDO_CONFIDENCE)
        Xf, y = pd.concat([Xf, Xu[keep]]), np.r_[y, (p[keep] >= 0.5).astype(int)]
    if "bag" in spec:
        rng = np.random.default_rng(SEED)
        draws = (rng.integers(0, len(Xf), len(Xf)) for _ in range(N_BAGS))
        return np.mean([booster(Xf.iloc[i], y[i], Xe) for i in draws], axis=0)
    return booster(Xf, y, Xe, weight)


def no_blank_auroc(feats, ev, scores):
    keep = (feats.loc[ev.id, "blank_lines"] == 0).values
    y = ev.label.astype(int).values[keep]
    return roc_auc_score(y, scores[keep]) if 0 < y.mean() < 1 else np.nan


def evaluate_split(key, feats, fit, ev, unlabeled):
    Xf, Xe, Xu = fold_design(feats, fit, ev, unlabeled)
    y = fit.label.astype(int).values
    return key, {m: fit_predict(m, Xf, y, Xe, Xu) for m in METHODS}


def compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, val, labeled = load_domains()
    labeled = labeled.reset_index(drop=True)
    unlabeled = load_split("unlabeled")
    in_val = labeled.id.isin(val.id)
    train, val = labeled[~in_val], labeled[in_val]
    frame = pd.concat([labeled[["id", "text"]], unlabeled[["id", "text"]]], ignore_index=True)
    feats = all_features(frame)

    splits = {d: (train[train.domain != d], train[train.domain == d]) for d in sorted(train.domain.unique())}
    splits["val"] = (train, val)
    results = dict(Parallel(n_jobs=-1)(
        delayed(evaluate_split)(k, feats, fit, ev, unlabeled) for k, (fit, ev) in splits.items()))

    domains = [k for k in splits if k != "val"]
    per_domain, rows = [], []
    for method in METHODS:
        for d in domains:
            ev = splits[d][1]
            s = results[d][method]
            per_domain.append({"method": method, "domain": d, "auroc": roc_auc_score(ev.label.astype(int), s),
                               "auroc_no_blank": no_blank_auroc(feats, ev, s)})
    by_domain = pd.DataFrame(per_domain)
    by_domain.to_csv(OUT_DIR / "lodo_by_domain.csv", index=False)
    base = by_domain[by_domain.method == "base"].set_index("domain").auroc_no_blank

    y_val = val.label.astype(int).values
    for method in METHODS:
        d = by_domain[by_domain.method == method].set_index("domain")
        s = results["val"][method]
        low, high = bootstrap_ci(y_val, s)
        rows.append({
            "method": method, "lodo_mean": d.auroc.mean(), "lodo_min": d.auroc.min(),
            "rare_domain": d.auroc["rare_domain"], "lodo_no_blank": d.auroc_no_blank.mean(),
            "lodo_no_blank_min": d.auroc_no_blank.min(),
            "domains_better": int((d.auroc_no_blank > base + 1e-9).sum()),
            "domains_worse": int((d.auroc_no_blank < base - 1e-9).sum()),
            "val": roc_auc_score(y_val, s), "val_ci_low": low, "val_ci_high": high,
            "val_no_blank": no_blank_auroc(feats, val, s),
        })
    table = pd.DataFrame(rows).sort_values("lodo_no_blank", ascending=False)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    (OUT_DIR / "comparison.md").write_text(table.round(4).to_markdown(index=False) + "\n")
    print(f"{len(domains)} domains")
    print(table.round(4).to_string(index=False))


def submit(method):
    train, test, unlabeled = load_split("train"), load_split("test"), load_split("unlabeled")
    frame = pd.concat([d[["id", "text"]] for d in (train, test, unlabeled)], ignore_index=True)
    Xf, Xe, Xu = fold_design(all_features(frame), train, test, unlabeled)
    labels = fit_predict(method, Xf, train.label.astype(int).values, Xe, Xu)
    sub = pd.DataFrame({"id": test.id, "label": labels})
    check_submission(sub, test)
    out = ROOT / "submissions" / f"improve_{method}.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", choices=list(METHODS))
    args = ap.parse_args()
    if args.submit:
        submit(args.submit)
    else:
        compare()


if __name__ == "__main__":
    main()
