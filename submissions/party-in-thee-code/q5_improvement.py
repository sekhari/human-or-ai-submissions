"""Q5: does combining word-level and character-level evidence help?

Motivation (from q4_error_analysis.py): the selected char SVM and the word LR
share only 112 of the SVM's 172 validation errors, and their scores correlate
at 0.87, so the two views are partly complementary.

Every variant is fit on the same 8,000 training rows and scored on the same
2,000 validation rows used in Q1/Q2. Each variant is run once, in the order
below, and all results are kept (including ones that do not help).

  baseline     char TF-IDF 3-5 + LinearSVC C=1 (the Q1 selection)
  v1_rank_avg  equal-weight average of rank-normalised scores from the
               baseline and word TF-IDF 1-2 + LR (C=10, the Q1 LR choice)
  v2_union     one LinearSVC (C=1) on word 1-2 and char 3-5 TF-IDF features
               stacked side by side (FeatureUnion)
  v3_char200k  baseline with the char vocabulary cap raised from 50k to 200k

The ensemble weight (0.5) and all settings were fixed before running.

A stress test then perturbs only the validation texts, to probe the
"unfamiliar / adversarially modified" part of the Kaggle test set:
  whitespace   every run of whitespace collapsed to one space
  typos        3% of words get one adjacent-character swap (seed 42)

    python q5_improvement.py --train data/train.csv
"""

import argparse
import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from baseline import ROOT, SEED, load_data, load_split
from q1 import get_scores, make_q1_model


def char_vectorizer(max_features=50000):
    return TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=2, max_df=1.0,
        max_features=max_features, sublinear_tf=True, use_idf=True, smooth_idf=True,
        norm="l2", dtype=np.float64,
    )


def svm():
    return LinearSVC(C=1.0, dual="auto", max_iter=5000, tol=1e-4, random_state=SEED)


def make_union():
    word = make_q1_model("word_lr", 10.0).named_steps["tfidf"]
    return Pipeline([
        ("features", FeatureUnion([("word", word), ("char", char_vectorizer())])),
        ("svm", svm()),
    ])


def make_char200k():
    return Pipeline([("tfidf", char_vectorizer(200000)), ("svm", svm())])


def rank01(x):
    return (rankdata(x) - 1) / (len(x) - 1)


class RankAverage:
    """Equal-weight average of rank-normalised scores from several fitted models."""

    def __init__(self, models):
        self.models = models

    def fit(self, texts, labels):
        for m in self.models:
            m.fit(texts, labels)
        return self

    def score(self, texts):
        return np.mean([rank01(get_scores(m, texts)) for m in self.models], axis=0)


def perturb_whitespace(texts):
    return texts.map(lambda t: re.sub(r"\s+", " ", t).strip())


def perturb_typos(texts, rate=0.03, seed=SEED):
    rng = random.Random(seed)

    def swap(word):
        if len(word) < 4:
            return word
        i = rng.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]

    def one(text):
        parts = re.split(r"(\s+)", text)
        return "".join(swap(p) if not p.isspace() and rng.random() < rate else p for p in parts)

    return texts.map(one)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "q5"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = load_split(load_data(args.train))
    views = {
        "clean": val.text,
        "whitespace": perturb_whitespace(val.text),
        "typos": perturb_typos(val.text),
    }
    variants = {
        "baseline": lambda: make_q1_model("char_svm", 1.0),
        "v1_rank_avg": lambda: RankAverage([make_q1_model("char_svm", 1.0), make_q1_model("word_lr", 10.0)]),
        "v2_union": make_union,
        "v3_char200k": make_char200k,
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
        "rows": table.to_dict(orient="records"),
        "notes": "One run per variant, same split and seed. Fit time = full pipeline fit, one BLAS thread. "
                 "All four variants were scored on the same validation set, so the best one is a "
                 "selection result, not an unbiased estimate.",
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "platform": platform.platform()},
    }
    (out / "q5_results.json").write_text(json.dumps(record, indent=2))
    md = ["| variant | fit (s) | train AUROC | val clean | val whitespace | val typos |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['fit_seconds']:.1f} | {r['train_auroc']:.6f} | "
                  f"{r['val_auroc_clean']:.6f} | {r['val_auroc_whitespace']:.6f} | {r['val_auroc_typos']:.6f} |")
    (out / "q5_results.md").write_text("\n".join(md) + "\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
