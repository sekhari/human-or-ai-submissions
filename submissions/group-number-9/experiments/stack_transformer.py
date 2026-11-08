"""Add fine-tuned transformer scores to the stacking models and write candidate submissions.

Run experiments.finetune (all jobs) and experiments.stacking first.

Retraining the transformer for every leave-one-domain-out fold is too slow, so these variants have no
LODO score. Boosting settings are not re-tuned: each variant reuses the settings LODO chose for its
base variant without the transformer, and the fixed validation set is only scored.

Usage:
    python -m experiments.stack_transformer

Outputs: results/stacking/transformer_comparison.{csv,md}, submissions/<name>.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from experiments.make_submission import check_submission
from experiments.model_comparison import bootstrap_ci
from experiments.stacking import VARIANTS, build_feature_set, fit_booster, stack_matrix, text_scores
from src.data import load_split
from src.models import to_probability
from src.split import load_train_val

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "stacking"
TRANSFORMER = ROOT / "cache" / "transformer" / "distilroberta-base"
BASES = ["gpt2_word_masked", "gpt2_style", "v2_word_masked"]
COLUMN = "text_distilroberta"


def oof_scores(prefix, df):
    cached = np.load(TRANSFORMER / f"{prefix}.npz")
    return pd.Series(cached["score"], index=cached["id"]).loc[df.id].values


def stack(base, params, fit, ev, fit_transformer, eval_transformer):
    """Return the base stack's scores and the same stack with the transformer score added."""
    kind, models = VARIANTS[base]
    text = dict(zip(models, (text_scores(m, fit, ev) for m in models)))
    Xf = stack_matrix(build_feature_set(kind, fit), {m: text[m][0] for m in models}, models)
    Xe = stack_matrix(build_feature_set(kind, ev), {m: text[m][1] for m in models}, models)
    y = fit.label.astype(int).values
    plain = fit_booster(params, Xf, y, Xe)
    with_transformer = fit_booster(params, Xf.assign(**{COLUMN: fit_transformer}), y,
                                   Xe.assign(**{COLUMN: eval_transformer}))
    return plain, with_transformer


def rank_mean(a, b):
    return (rankdata(a) + rankdata(b)) / (2 * len(a))


def candidates(params, fit, ev, fit_transformer, eval_transformer):
    out = {"distilroberta": to_probability(eval_transformer)}
    for base in BASES:
        plain, added = stack(base, params[base], fit, ev, fit_transformer, eval_transformer)
        out[f"{base}+distil_stack"] = added
        out[f"{base}+distil_rank"] = rank_mean(plain, eval_transformer)
    return out


def main():
    table = pd.read_csv(OUT_DIR / "comparison.csv").set_index("variant")
    params = {b: json.loads(table.loc[b, "params"]) for b in BASES}

    train, val = load_train_val()
    y_val = val.label.astype(int).values
    blank = (build_feature_set("v1", val).blank_lines == 0).values
    rows = []
    for name, s in candidates(params, train, val, oof_scores("oof8k", train), np.load(TRANSFORMER / "val.npy")).items():
        low, high = bootstrap_ci(y_val, s)
        rows.append({"candidate": name, "val": roc_auc_score(y_val, s), "val_ci_low": low, "val_ci_high": high,
                     "val_no_blank_lines": roc_auc_score(y_val[blank], s[blank])})
    result = pd.DataFrame(rows).sort_values("val", ascending=False)
    result.to_csv(OUT_DIR / "transformer_comparison.csv", index=False)
    (OUT_DIR / "transformer_comparison.md").write_text(result.round(4).to_markdown(index=False) + "\n")
    print(result.round(4).to_string(index=False))

    full, test = load_split("train"), load_split("test")
    subs = candidates(params, full, test, oof_scores("oof10k", full), np.load(TRANSFORMER / "test.npy"))
    for name, labels in subs.items():
        sub = pd.DataFrame({"id": test.id, "label": labels})
        check_submission(sub, test)
        sub.to_csv(ROOT / "submissions" / f"{name.replace('+', '_')}.csv", index=False)
        print("wrote", f"{name.replace('+', '_')}.csv")


if __name__ == "__main__":
    main()
