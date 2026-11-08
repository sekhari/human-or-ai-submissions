"""Second round of improvements: more language models, and ensembles less tied to any one feature.

Language-model features were the largest gain so far, so this round adds different reference models:
OPT-125M, GPT-2 medium and a word n-gram model fit on unlabeled.csv, plus each model's difference from
GPT-2 (trees cannot easily form differences themselves). Protocol matches experiments.improve: fixed
boosting settings, LODO on the 8,000-row training partition, fixed validation set only scored.
test.csv is not used to fit anything.

Usage:
    python -m experiments.improve_lm                # compare methods
    python -m experiments.improve_lm --submit NAME  # write submissions/improve_NAME.csv

Outputs (results/improve_lm/): comparison.csv, comparison.md, lodo_by_domain.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from experiments.improve import PARAMS, fold_design, no_blank_auroc
from experiments.improve import all_features as base_features
from experiments.make_submission import check_submission
from experiments.model_comparison import bootstrap_ci
from experiments.robustness import load_domains
from src.data import load_split
from src.models import SEED
from src.ngram import ngram_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT_DIR = ROOT / "results" / "improve_lm"
N_BAGS = 10
COLSAMPLE = 0.3
DIFFERENCES = ["nll_mean", "nll_std", "rare", "diff_std"]
BASELINE = "curv_bag"
GROUPS = {"opt": "opt_", "medium": "medium_", "ngram": "ngram_"}
METHODS = {  # name -> (extra feature groups, ensemble)
    "curv_bag": ((), "bag"),
    "opt_bag": (("opt",), "bag"),
    "medium_bag": (("medium",), "bag"),
    "ngram_bag": (("ngram",), "bag"),
    "lm3_bag": (("opt", "medium"), "bag"),
    "all_bag": (("opt", "medium", "ngram"), "bag"),
    "curv_colsample": ((), "colsample"),
    "all_colsample": (("opt", "medium", "ngram"), "colsample"),
    "all_blend": (("opt", "medium", "ngram"), "blend"),
}


def all_features(df, unlabeled_texts):
    feats = base_features(df)
    for prefix in ("opt", "medium"):
        lm = pd.read_csv(CACHE / f"{prefix}_lm.csv").set_index("id").loc[df.id]
        feats = feats.assign(**{c: lm[c].values for c in lm})
        feats = feats.assign(**{f"{prefix}_minus_gpt2_{k}": feats[f"{prefix}_{k}"] - feats[f"gpt2_{k}"] for k in DIFFERENCES})
    ngram = ngram_features(unlabeled_texts, df.text.tolist())
    feats = feats.assign(**{c: ngram[c].values for c in ngram})
    return feats.assign(ngram_minus_gpt2=feats.ngram_bi_nll - feats.gpt2_nll_mean)


def bagged_booster(Xf, y, Xe, **extra):
    rng = np.random.default_rng(SEED)
    scores = []
    for k in range(N_BAGS):
        i = rng.integers(0, len(Xf), len(Xf))
        model = HistGradientBoostingClassifier(random_state=SEED + k, **PARAMS, **extra).fit(Xf.iloc[i], y[i])
        scores.append(model.predict_proba(Xe)[:, 1])
    return np.mean(scores, axis=0)


def extra_trees(Xf, y, Xe):
    fill = Xf.median()
    model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=SEED)
    return model.fit(Xf.fillna(fill), y).predict_proba(Xe.fillna(fill))[:, 1]


def fit_predict(method, Xf, y, Xe):
    groups, ensemble = METHODS[method]
    dropped = tuple(p for g, p in GROUPS.items() if g not in groups)
    cols = [c for c in Xf if not c.startswith(dropped)]
    Xf, Xe = Xf[cols], Xe[cols]
    if ensemble == "bag":
        return bagged_booster(Xf, y, Xe)
    colsampled = bagged_booster(Xf, y, Xe, max_features=COLSAMPLE)
    if ensemble == "colsample":
        return colsampled
    return (rankdata(colsampled) + rankdata(extra_trees(Xf, y, Xe))) / (2 * len(Xe))


def evaluate_split(key, feats, fit, ev, unlabeled):
    Xf, Xe, _ = fold_design(feats, fit, ev, unlabeled)
    y = fit.label.astype(int).values
    return key, {m: fit_predict(m, Xf, y, Xe) for m in METHODS}


def compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, val, labeled = load_domains()
    labeled = labeled.reset_index(drop=True)
    unlabeled = load_split("unlabeled")
    in_val = labeled.id.isin(val.id)
    train, val = labeled[~in_val], labeled[in_val]
    frame = pd.concat([labeled[["id", "text"]], unlabeled[["id", "text"]]], ignore_index=True)
    feats = all_features(frame, unlabeled.text.tolist())

    splits = {d: (train[train.domain != d], train[train.domain == d]) for d in sorted(train.domain.unique())}
    splits["val"] = (train, val)
    results = dict(Parallel(n_jobs=-1)(
        delayed(evaluate_split)(k, feats, fit, ev, unlabeled) for k, (fit, ev) in splits.items()))

    domains = [k for k in splits if k != "val"]
    by_domain = pd.DataFrame([
        {"method": m, "domain": d, "auroc": roc_auc_score(splits[d][1].label.astype(int), results[d][m]),
         "auroc_no_blank": no_blank_auroc(feats, splits[d][1], results[d][m])}
        for m in METHODS for d in domains])
    by_domain.to_csv(OUT_DIR / "lodo_by_domain.csv", index=False)
    base = by_domain[by_domain.method == BASELINE].set_index("domain").auroc_no_blank

    y_val = val.label.astype(int).values
    rows = []
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
    print(table.round(4).to_string(index=False))


def submit(method, out=None):
    train, test, unlabeled = load_split("train"), load_split("test"), load_split("unlabeled")
    frame = pd.concat([d[["id", "text"]] for d in (train, test, unlabeled)], ignore_index=True)
    feats = all_features(frame, unlabeled.text.tolist())
    Xf, Xe, _ = fold_design(feats, train, test, unlabeled)
    sub = pd.DataFrame({"id": test.id, "label": fit_predict(method, Xf, train.label.astype(int).values, Xe)})
    check_submission(sub, test)
    out = out or ROOT / "submissions" / f"improve_{method}.csv"
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
