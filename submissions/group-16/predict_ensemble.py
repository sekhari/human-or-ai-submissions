"""Write the Kaggle submission with a 4-model rank-average ensemble.

Why: Model A alone scored 0.966 on the random validation split but 0.868 on Kaggle public,
because test passages come from new topics (4.0% unseen words vs 1.3% within train).
Under leave-topic-out CV (shift_eval.py: 12 topic clusters, whole clusters held out),
Model A scored 0.926 while an equal-weight average of four different views scored 0.955,
with the worst held-out topic improving from 0.805 to 0.905. Weights were NOT tuned.

The four members, all trained on whitespace-normalized text (removes the paragraph-break
artifact) and refit on all 10,000 labeled rows:
  1. word 1-2-gram TF-IDF + logistic regression (C=30, min_df=5)       -- Model A
  2. char_wb 1-5-gram TF-IDF + logistic regression (C=10)              -- spelling/morphology
  3. function-word + punctuation 1-3-grams + LR (C=1)                  -- topic-agnostic
  4. 25 stylometric features + gradient boosting                       -- rhythm/richness
Scores are converted to ranks before averaging, so members on different scales count equally.

Usage: python predict_ensemble.py --train data/train.csv --test data/test.csv --out submission_ensemble.csv
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import SEED, normalize_ws
from q1_model_comparison import StyleFeatures

WORD = dict(lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
FUNC_VOCAB = set(ENGLISH_STOP_WORDS) | set(".,;:!?'\"()-—–…")
_TOK = re.compile(r"\w+|[^\w\s]")


def func_view(t):
    """Keep function words and punctuation; replace every content word with 'X'."""
    return " ".join(w if w in FUNC_VOCAB else "X" for w in _TOK.findall(t.lower()))


def tfidf_lr(vec, C):
    return vec, LogisticRegression(C=C, max_iter=3000, solver="liblinear")


def members():
    return {
        "word": tfidf_lr(TfidfVectorizer(ngram_range=(1, 2), min_df=5, **WORD), 30),
        "char": tfidf_lr(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), lowercase=False,
                                         sublinear_tf=True, min_df=3, max_features=300_000), 10),
        "func": tfidf_lr(TfidfVectorizer(ngram_range=(1, 3), min_df=5, sublinear_tf=True, lowercase=True,
                                         token_pattern=r"(?u)\b\w+\b|[^\w\s]", preprocessor=func_view), 1),
        "style": (StyleFeatures(), HistGradientBoostingClassifier(learning_rate=0.1, max_iter=500,
                                                                 random_state=SEED, early_stopping=False)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--out", default="submission_ensemble.csv")
    args = ap.parse_args()
    np.random.seed(SEED)

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    Xtr, ytr = normalize_ws(train.text.values), train.label.values
    Xte = normalize_ws(test.text.values)

    ranks = []
    for name, (feat, clf) in members().items():
        Ftr = feat.fit_transform(Xtr)
        clf.fit(Ftr, ytr)
        p = clf.predict_proba(feat.transform(Xte))[:, 1]
        ranks.append(rankdata(p) / len(p))          # in (0, 1], higher = more AI-like
        print(f"{name:5s} done, mean P(AI)={p.mean():.3f}")

    score = np.mean(ranks, axis=0)
    pd.DataFrame({"id": test.id, "label": score}).to_csv(args.out, index=False)
    print(f"wrote {len(test)} rows to {args.out}")


if __name__ == "__main__":
    main()
