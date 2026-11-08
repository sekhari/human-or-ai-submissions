"""Stack text-model scores with stylometric features in gradient boosting.

Unlike rank averaging, boosting can weigh the text models differently depending on layout and style.
Text scores used for fitting are out-of-fold, so the booster never sees a text model's in-sample fit.

Protocol: boosting settings are chosen by leave-one-domain-out (LODO) on the 8,000-row training
partition, so the fixed validation set stays untouched until the final score. LODO is the selection
metric because it ranked our first four Kaggle submissions in the same order as the public scores,
though four points are weak evidence.

Usage:
    python -m experiments.stacking                  # compare variants
    python -m experiments.stacking --submit NAME    # refit NAME on 10,000 rows, write submissions/stack_NAME.csv

Outputs (results/stacking/): comparison.csv, comparison.md, lodo_by_domain.csv
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments.make_submission import check_submission
from experiments.model_comparison import bootstrap_ci
from experiments.robustness import load_domains
from src.data import load_split
from src.features import build_extended_features, build_features
from src.models import SEED, build_tuned

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "stacking"
BEST = json.loads((ROOT / "results" / "model_comparison" / "best_params.json").read_text())
CACHE = ROOT / "cache"  # filled by experiments.extract_neural
TEXT_MODELS = ["word_lr", "masked_lr", "char_lr", "minilm_lr"]
VARIANTS = {  # name -> (feature set, text models)
    "style_v1": ("v1", []),
    "style_v2": ("v2", []),
    "v1_word_masked": ("v1", ["word_lr", "masked_lr"]),
    "v2_masked": ("v2", ["masked_lr"]),
    "v2_word_masked": ("v2", ["word_lr", "masked_lr"]),
    "v2_word_masked_char": ("v2", ["word_lr", "masked_lr", "char_lr"]),
    "gpt2_style": ("v2_gpt2", []),
    "gpt2_word_masked": ("v2_gpt2", ["word_lr", "masked_lr"]),
    "v2_word_masked_minilm": ("v2", ["word_lr", "masked_lr", "minilm_lr"]),
    "gpt2_word_masked_minilm": ("v2_gpt2", ["word_lr", "masked_lr", "minilm_lr"]),
}
GRID = [{"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves, "l2_regularization": l2}
        for (lr, it), leaves, l2 in itertools.product([(0.03, 400), (0.1, 200), (0.2, 100)], [7, 15, 31, 63], [0.0, 10.0])]


def minilm_embeddings(df):
    cached = np.load(CACHE / "minilm.npz")
    return pd.DataFrame(cached["emb"], index=cached["id"]).loc[df.id].values


def text_scores(name, fit_df, eval_df):
    """Return out-of-fold log-odds for fit_df and full-fit log-odds for eval_df."""
    y = fit_df.label.astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    if name == "minilm_lr":  # C fixed in advance; the booster decides how much to trust this score
        model, X, X_eval = LogisticRegression(C=10, max_iter=2000), minilm_embeddings(fit_df), minilm_embeddings(eval_df)
    else:
        model, X, X_eval = build_tuned(name, BEST[name]), fit_df.text.tolist(), eval_df.text.tolist()
    oof = cross_val_predict(model, X, y, cv=cv, method="decision_function")
    return oof, clone(model).fit(X, y).decision_function(X_eval)


def build_feature_set(kind, df):
    if kind == "v1":
        return build_features(df)
    feats = build_extended_features(df)
    if kind == "v2_gpt2":
        gpt2 = pd.read_csv(CACHE / "gpt2_surprisal.csv").set_index("id").loc[df.id]
        feats = feats.assign(**{c: gpt2[c].values for c in gpt2})
    return feats


def stack_matrix(feats, text, models):
    return feats.assign(**{f"text_{m}": text[m] for m in models}) if models else feats


def fit_booster(params, feats, y, eval_feats):
    model = HistGradientBoostingClassifier(random_state=SEED, **params).fit(feats, y)
    return model.predict_proba(eval_feats)[:, 1]


def build_splits(train, val):
    """Outer splits: LODO folds inside the training partition, then the fixed validation set."""
    folds = {d: (train[train.domain != d], train[train.domain == d]) for d in sorted(train.domain.unique())}
    return folds, (train, val)


def cached_text(splits):
    tasks = [delayed(text_scores)(m, fit, ev) for fit, ev in splits.values() for m in TEXT_MODELS]
    out = iter(Parallel(n_jobs=-1)(tasks))
    return {key: {m: next(out) for m in TEXT_MODELS} for key in splits}


def evaluate_variant(name, splits, text, feature_sets):
    kind, models = VARIANTS[name]
    feats = feature_sets[kind]

    def run(params, key):
        fit, ev = splits[key]
        Xf = stack_matrix(feats.loc[fit.index], {m: text[key][m][0] for m in models}, models)
        Xe = stack_matrix(feats.loc[ev.index], {m: text[key][m][1] for m in models}, models)
        return fit_booster(params, Xf, fit.label.astype(int).values, Xe)

    domains = [k for k in splits if k != "val"]
    scored = []
    for params in GRID:
        per_domain = {d: roc_auc_score(splits[d][1].label, run(params, d)) for d in domains}
        scored.append((np.mean(list(per_domain.values())), params, per_domain))
    lodo_mean, params, per_domain = max(scored, key=lambda r: r[0])
    y_val = splits["val"][1].label.astype(int).values
    val_scores = run(params, "val")
    low, high = bootstrap_ci(y_val, val_scores)
    blank = (feature_sets["v1"].loc[splits["val"][1].index, "blank_lines"] == 0).values
    return {
        "variant": name, "features": kind, "text_models": "+".join(models) or "-",
        "lodo_mean": lodo_mean, "lodo_min": min(per_domain.values()), "rare_domain": per_domain["rare_domain"],
        "val": roc_auc_score(y_val, val_scores), "val_ci_low": low, "val_ci_high": high,
        "val_no_blank_lines": roc_auc_score(y_val[blank], val_scores[blank]),
        "params": json.dumps(params),
    }, per_domain


def compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, labeled = load_domains()
    labeled = labeled.reset_index(drop=True)
    in_val = labeled.id.isin(val.id)
    train, val = labeled[~in_val], labeled[in_val]
    feature_sets = {kind: build_feature_set(kind, labeled) for kind in ("v1", "v2", "v2_gpt2")}
    folds, val_split = build_splits(train, val)
    splits = {**folds, "val": val_split}
    text = cached_text(splits)
    results = Parallel(n_jobs=-1)(delayed(evaluate_variant)(v, splits, text, feature_sets) for v in VARIANTS)
    table = pd.DataFrame([r[0] for r in results]).sort_values("lodo_mean", ascending=False)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    (OUT_DIR / "comparison.md").write_text(table.round(4).to_markdown(index=False) + "\n")
    pd.DataFrame({r[0]["variant"]: r[1] for r in results}).T.to_csv(OUT_DIR / "lodo_by_domain.csv")
    print(table.drop(columns="params").round(4).to_string(index=False))
    print(table[["variant", "params"]].to_string(index=False))


def submit(name):
    kind, models = VARIANTS[name]
    row = pd.read_csv(OUT_DIR / "comparison.csv").set_index("variant").loc[name]
    params = json.loads(row.params)
    train, test = load_split("train"), load_split("test")
    text = dict(zip(models, Parallel(n_jobs=-1)(delayed(text_scores)(m, train, test) for m in models)))
    Xf = stack_matrix(build_feature_set(kind, train), {m: text[m][0] for m in models}, models)
    Xe = stack_matrix(build_feature_set(kind, test), {m: text[m][1] for m in models}, models)
    sub = pd.DataFrame({"id": test.id, "label": fit_booster(params, Xf, train.label.astype(int).values, Xe)})
    check_submission(sub, test)
    out = ROOT / "submissions" / f"stack_{name}.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out} params={params}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", choices=list(VARIANTS))
    args = ap.parse_args()
    if args.submit:
        submit(args.submit)
    else:
        compare()


if __name__ == "__main__":
    main()
