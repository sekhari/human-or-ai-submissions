"""Stress-test models under conditions closer to the test set than the fixed validation set.

Scenarios (hyperparameters stay fixed, so no labels are used for tuning here):
    val            fit on 8,000 rows, score the fixed validation set (i.i.d., optimistic)
    val_attacked   same model, but validation AI passages get line breaks removed, typos, homoglyphs
    lodo           leave-one-domain-out over EDA topic clusters: fit on other domains, score the rest
    rare_domain    the lodo fold made of clusters over-represented in test (test share >= 5x train)

Usage:
    python -m experiments.robustness

Outputs (results/robustness/): summary.csv, summary.md, lodo_by_domain.csv
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from experiments.model_comparison import bootstrap_ci
from src.models import MODELS, build_tuned, scores
from src.perturb import attack_ai_docs
from src.split import load_train_val

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "robustness"
_BEST = json.loads((ROOT / "results" / "model_comparison" / "best_params.json").read_text())

# variant -> (base model, params). Stronger regularization is tested because the i.i.d. CV that
# picked the tuned settings cannot see domain shift.
VARIANTS = {
    "word_lr_tuned": ("word_lr", _BEST["word_lr"]),
    "word_lr_C10": ("word_lr", {**_BEST["word_lr"], "clf__C": 10}),
    "word_lr_C3": ("word_lr", {**_BEST["word_lr"], "clf__C": 3}),
    "char_lr_tuned": ("char_lr", _BEST["char_lr"]),
    "char_lr_C3": ("char_lr", {**_BEST["char_lr"], "clf__C": 3}),
    "masked_lr_tuned": ("masked_lr", _BEST["masked_lr"]),
    "style_tuned": ("style_hgb", _BEST["style_hgb"]),
    "style_nolayout_tuned": ("style_hgb_nolayout", _BEST["style_hgb_nolayout"]),
    "style_nolayout_reg": ("style_hgb_nolayout",
                           {"learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 10.0}),
}
ENSEMBLE_POOL = ["word_lr_C10", "char_lr_tuned", "masked_lr_tuned", "style_tuned", "style_nolayout_reg"]


def rank_average(score_lists):
    """Average of per-model ranks scaled to (0, 1]; models need not share a score scale."""
    return np.mean([rankdata(s) / len(s) for s in score_lists], axis=0)


def fit_and_score(variant, train, eval_sets):
    base, params = VARIANTS[variant]
    spec = MODELS[base]
    model = build_tuned(base, params).fit(spec.inputs(train), train.label.astype(int).values)
    return variant, {name: scores(model, spec.inputs(df)) for name, df in eval_sets.items()}


def load_domains():
    """Labeled rows with a domain column; rare test-heavy clusters are merged into one fold."""
    train, val = load_train_val()
    labeled = pd.concat([train, val], ignore_index=True)
    assign = pd.read_csv(ROOT / "results" / "eda" / "cluster_assignments.csv")
    clusters = pd.read_csv(ROOT / "results" / "eda" / "clusters.csv", index_col=0)
    rare = set(clusters.index[clusters.test_over_train >= 5])
    labeled = labeled.merge(assign[["id", "cluster"]], on="id")
    labeled["domain"] = [("rare_domain" if c in rare else f"cluster_{c}") for c in labeled.cluster]
    return train, val, labeled


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, labeled = load_domains()
    y_val = val.label.astype(int).values
    domains = sorted(labeled.domain.unique())
    tasks = [delayed(fit_and_score)(v, train, {"val": val, "val_attacked": attack_ai_docs(val)}) for v in VARIANTS]
    tasks += [delayed(fit_and_score)(v, labeled[labeled.domain != d], {d: labeled[labeled.domain == d]})
              for v, d in itertools.product(VARIANTS, domains)]
    collected = {v: {} for v in VARIANTS}
    for variant, result in Parallel(n_jobs=-1)(tasks):
        collected[variant].update(result)

    for size in (2, 3, 4, 5):
        for combo in itertools.combinations(ENSEMBLE_POOL, size):
            collected["+".join(combo)] = {
                name: rank_average([collected[m][name] for m in combo]) for name in collected[combo[0]]}

    rows, by_domain = [], {}
    for name, res in collected.items():
        lodo = {d: roc_auc_score(labeled.label[labeled.domain == d], res[d]) for d in domains}
        by_domain[name] = lodo
        y_rare = labeled.label[labeled.domain == "rare_domain"].astype(int).values
        low, high = bootstrap_ci(y_rare, res["rare_domain"])
        rows.append({
            "model": name, "val": roc_auc_score(y_val, res["val"]),
            "val_attacked": roc_auc_score(y_val, res["val_attacked"]),
            "lodo_mean": np.mean(list(lodo.values())), "lodo_min": min(lodo.values()),
            "rare_domain": lodo["rare_domain"], "rare_ci_low": low, "rare_ci_high": high,
        })
    table = pd.DataFrame(rows).sort_values("lodo_mean", ascending=False)
    table.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "summary.md").write_text(table.round(4).to_markdown(index=False) + "\n")
    pd.DataFrame(by_domain).T.to_csv(OUT_DIR / "lodo_by_domain.csv")
    n = labeled.domain.value_counts().to_dict()
    print("domain sizes:", n, "rare AI rate:", round(labeled.label[labeled.domain == "rare_domain"].mean(), 3))
    single = table[~table.model.str.contains(r"\+")]
    print(single.round(4).to_string(index=False))
    print("\nTop ensembles by lodo_mean")
    print(table[table.model.str.contains(r"\+")].head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
