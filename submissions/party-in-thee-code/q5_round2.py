"""Q5 round 2: let the char model see line breaks.

Round 1 (q5_improvement.py) found no clean-validation gain from adding word
features, but the whitespace stress test cut every char model's AUROC by about
0.04. Looking into why: scikit-learn's char analyzer collapses every run of
2+ whitespace characters to one space, so a blank line ("\\n\\n") is invisible
to it. In train.csv, 0% of human texts contain a blank line versus 66% of AI
texts, and 31.9% of test.csv texts contain one (33.0% of train.csv), so the
signal is present in the test data too. It is a formatting artifact of how the
data were collected, not evidence about authorship, so we treat it with care.

Variants (same split, seed and validation views as round 1, run once):
  baseline              char 3-5, 50k features, LinearSVC C=1 (Q1 selection)
  v4_newline            same, but each newline becomes a visible "¶" first,
                        so line structure survives as character n-grams
  v5_newline_rank_avg   v4 rank-averaged with word 1-2 + LR (C=10), equal weight
  v6_newline_200k       v4 with the 200k vocabulary cap from round 1's v3

    python q5_round2.py --train data/train.csv
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from baseline import ROOT, load_data, load_split
from q1 import get_scores, make_q1_model
from q5_improvement import RankAverage, char_vectorizer, perturb_typos, perturb_whitespace, svm


def mark_newlines(text):
    # Lowercasing must happen here because a custom preprocessor replaces the default one.
    return text.lower().replace("\r\n", "\n").replace("\n", "¶")


def make_newline_char(max_features=50000):
    vec = char_vectorizer(max_features)
    vec.set_params(preprocessor=mark_newlines)
    return Pipeline([("tfidf", vec), ("svm", svm())])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "q5"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full = load_data(args.train)
    train, val = load_split(full)
    blank = full.assign(has_blank=full.text.str.contains("\n\n")).groupby("label").has_blank.mean()
    views = {"clean": val.text, "whitespace": perturb_whitespace(val.text), "typos": perturb_typos(val.text)}
    variants = {
        "baseline": lambda: make_q1_model("char_svm", 1.0),
        "v4_newline": make_newline_char,
        "v5_newline_rank_avg": lambda: RankAverage([make_newline_char(), make_q1_model("word_lr", 10.0)]),
        "v6_newline_200k": lambda: make_newline_char(200000),
    }

    rows = []
    with threadpool_limits(1):
        for name, build in variants.items():
            model = build()
            start = time.perf_counter()
            model.fit(train.text, train.label)
            fit_s = time.perf_counter() - start
            scorer = model.score if isinstance(model, RankAverage) else (lambda t, m=model: get_scores(m, t))
            row = {"variant": name, "fit_seconds": round(fit_s, 3),
                   "train_auroc": round(float(roc_auc_score(train.label, scorer(train.text))), 6)}
            for view, texts in views.items():
                row[f"val_auroc_{view}"] = round(float(roc_auc_score(val.label, scorer(texts))), 6)
            rows.append(row)
            print(row, flush=True)

    table = pd.DataFrame(rows)
    base = table.set_index("variant").loc["baseline"]
    for view in views:
        table[f"delta_{view}"] = (table[f"val_auroc_{view}"] - base[f"val_auroc_{view}"]).round(6)
    record = {
        "blank_line_rate_by_label_train_csv": {str(k): round(float(v), 4) for k, v in blank.items()},
        "rows": table.to_dict(orient="records"),
        "notes": "Round 2, chosen after round 1. All variants share the round-1 validation set, so "
                 "round-1 and round-2 winners are selection results on that set.",
    }
    (out / "round2_results.json").write_text(json.dumps(record, indent=2))
    md = ["| variant | fit (s) | train AUROC | val clean | val whitespace | val typos |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['fit_seconds']:.1f} | {r['train_auroc']:.6f} | "
                  f"{r['val_auroc_clean']:.6f} | {r['val_auroc_whitespace']:.6f} | {r['val_auroc_typos']:.6f} |")
    (out / "round2_results.md").write_text("\n".join(md) + "\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
