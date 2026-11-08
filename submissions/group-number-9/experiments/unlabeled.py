"""Phase 4: does unlabeled.csv help? (report Q3)

Methods, each compared with its supervised baseline:
    feature_learning   fit the TF-IDF vocabulary and IDF on train + unlabeled text
    self_training      fit, pseudo-label confident unlabeled rows, refit on labeled + pseudo-labeled

Primary setting, fixed before looking at validation: threshold 0.95, one round. The other settings
are a sensitivity check and are not used to pick a winner.

Leakage control: validation text and labels are never used to fit, to build the vocabulary, or to
choose pseudo-labels. test.csv is not used at all. Pseudo-label accuracy is estimated by holding out
20% of the training partition as mock unlabeled data, so it does not touch validation either.

Scenarios: val (fixed set), val_attacked (see experiments.robustness), rare_domain (fit without the
fiction/poetry clusters; unlabeled.csv does contain some of those passages).

Usage:
    python -m experiments.unlabeled

Outputs (results/unlabeled/): comparison.csv, comparison.md, pseudo_label_quality.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.robustness import load_domains
from src.data import load_split
from src.models import SEED, build_tuned
from src.perturb import attack_ai_docs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "unlabeled"
BASE_MODELS = ["masked_lr", "word_lr"]
PRIMARY = "self_t0.95_r1"
SETTINGS = {  # name -> (threshold, rounds)
    "self_t0.99_r1": (0.99, 1),
    "self_t0.95_r1": (0.95, 1),
    "self_t0.90_r1": (0.90, 1),
    "self_t0.95_r3": (0.95, 3),
}


def select_confident(proba, threshold):
    """Boolean mask of confident rows and their hard pseudo-labels."""
    return (proba >= threshold) | (proba <= 1 - threshold), (proba >= 0.5).astype(int)


def fit_supervised(name, params, texts, y):
    return build_tuned(name, params).fit(texts, y), {}


def fit_feature_learning(name, params, texts, y, unlabeled):
    pipe = build_tuned(name, params)
    vect = pipe.named_steps["tfidf"].fit(texts + unlabeled)
    pipe.named_steps["clf"].fit(vect.transform(texts), y)
    return pipe, {}


def fit_self_training(name, params, texts, y, unlabeled, threshold, rounds):
    model = build_tuned(name, params).fit(texts, y)
    info = {}
    for _ in range(rounds):
        mask, pseudo = select_confident(model.predict_proba(unlabeled)[:, 1], threshold)
        extra = [t for t, keep in zip(unlabeled, mask) if keep]
        model = build_tuned(name, params).fit(texts + extra, np.r_[y, pseudo[mask]])
        info = {"n_pseudo": int(mask.sum()), "pseudo_ai_rate": float(pseudo[mask].mean()) if mask.any() else np.nan}
    return model, info


def fit_method(method, name, params, texts, y, unlabeled):
    if method == "supervised":
        return fit_supervised(name, params, texts, y)
    if method == "feature_learning":
        return fit_feature_learning(name, params, texts, y, unlabeled)
    threshold, rounds = SETTINGS[method]
    return fit_self_training(name, params, texts, y, unlabeled, threshold, rounds)


def proba(model, df):
    """Call steps directly: feature_learning fits them outside Pipeline.fit."""
    steps = model.named_steps
    return steps["clf"].predict_proba(steps["tfidf"].transform(df.text.tolist()))[:, 1]


def run_method(method, name, params, train, val, attacked, source, rare, unlabeled):
    model, info = fit_method(method, name, params, train.text.tolist(), train.label.astype(int).values, unlabeled)
    shift_model, _ = fit_method(method, name, params, source.text.tolist(), source.label.astype(int).values, unlabeled)
    return {
        "model": name, "method": method, **info,
        "val_scores": proba(model, val), "attacked_scores": proba(model, attacked),
        "rare_scores": proba(shift_model, rare),
    }


def paired_delta(y, base, new, n=2000, seed=SEED):
    """Bootstrap 95% interval for AUROC(new) - AUROC(base), resampling the same rows for both."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        deltas.append(roc_auc_score(y[i], new[i]) - roc_auc_score(y[i], base[i]))
    return np.percentile(deltas, [2.5, 97.5])


def pseudo_label_quality(name, params, train):
    """Pseudo-label accuracy on a mock unlabeled set carved from the training partition."""
    fit_df, mock = train_test_split(train, test_size=0.2, stratify=train.label, random_state=SEED)
    model = build_tuned(name, params).fit(fit_df.text.tolist(), fit_df.label.astype(int).values)
    p = model.predict_proba(mock.text.tolist())[:, 1]
    y = mock.label.astype(int).values
    rows = []
    for t in sorted({v[0] for v in SETTINGS.values()}, reverse=True):
        mask, pseudo = select_confident(p, t)
        rows.append({"model": name, "threshold": t, "coverage": mask.mean(),
                     "accuracy": (pseudo[mask] == y[mask]).mean() if mask.any() else np.nan})
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, labeled = load_domains()
    best = json.loads((ROOT / "results" / "model_comparison" / "best_params.json").read_text())
    unlabeled = load_split("unlabeled").text.tolist()
    attacked = attack_ai_docs(val)
    source, rare = labeled[labeled.domain != "rare_domain"], labeled[labeled.domain == "rare_domain"]
    assert not set(val.text) & set(unlabeled), "validation text must not appear in the unlabeled pool"

    methods = ["supervised", "feature_learning", *SETTINGS]
    results = Parallel(n_jobs=-1)(
        delayed(run_method)(meth, m, best[m], train, val, attacked, source, rare, unlabeled)
        for m in BASE_MODELS for meth in methods)
    baseline = {r["model"]: r for r in results if r["method"] == "supervised"}
    y_val = val.label.astype(int).values
    y_rare = rare.label.astype(int).values

    rows = []
    for r in results:
        base = baseline[r["model"]]
        low, high = paired_delta(y_val, base["val_scores"], r["val_scores"])
        rows.append({
            "model": r["model"], "method": r["method"], "primary": r["method"] == PRIMARY,
            "n_pseudo": r.get("n_pseudo"), "pseudo_ai_rate": r.get("pseudo_ai_rate"),
            "val_auroc": roc_auc_score(y_val, r["val_scores"]),
            "val_delta": roc_auc_score(y_val, r["val_scores"]) - roc_auc_score(y_val, base["val_scores"]),
            "val_delta_ci_low": low, "val_delta_ci_high": high,
            "val_attacked": roc_auc_score(y_val, r["attacked_scores"]),
            "rare_domain": roc_auc_score(y_rare, r["rare_scores"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)
    (OUT_DIR / "comparison.md").write_text(table.round(4).to_markdown(index=False) + "\n")
    quality = pd.DataFrame([row for m in BASE_MODELS for row in pseudo_label_quality(m, best[m], train)])
    quality.to_csv(OUT_DIR / "pseudo_label_quality.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(quality.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
