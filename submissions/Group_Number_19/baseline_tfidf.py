"""Baseline: word (1-3) + char (3-5) TF-IDF -> logistic regression, scored with 5-fold CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer,
)

from folds import N_FOLDS, SEED, make_folds
from layout import normalize_texts


TF_SCHEMES = ("log", "log1p", "binary")
# smooth: ln((1 + n) / (1 + df)) + 1 (sklearn default); none: no idf weighting;
# textbook: ln(n / df), so n-grams present in every document get weight 0.
IDF_SCHEMES = ("smooth", "none", "textbook")


class TextbookIdfTransformer(TfidfTransformer):
    """TfidfTransformer with idf = ln(n / df), dropping sklearn's "+1"."""

    def __init__(self, *, sublinear_tf=False):
        super().__init__(smooth_idf=False, sublinear_tf=sublinear_tf)

    def fit(self, X, y=None):
        super().fit(X, y)  # with smooth_idf=False this sets idf_ = ln(n / df) + 1
        self.idf_ = self.idf_ - 1
        return self


def _log1p(X):
    return X.log1p()  # sparse-safe: zeros stay zero


def make_vectorizer(tf: str, idf: str = "smooth", **kwargs):
    """TF-IDF vectorizer with term frequency
    log: 1 + ln(count), log1p: ln(1 + count), binary: 1 if present.
    The vocabulary is the same for all schemes; only the weighting differs.
    """
    if idf not in IDF_SCHEMES:
        raise ValueError(f"unknown idf scheme {idf!r}; expected one of {IDF_SCHEMES}")
    if tf not in TF_SCHEMES:
        raise ValueError(f"unknown tf scheme {tf!r}; expected one of {TF_SCHEMES}")
    if idf == "textbook":
        steps = [("count", CountVectorizer(binary=tf == "binary", **kwargs))]
        if tf == "log1p":
            steps.append(("log1p", FunctionTransformer(_log1p, accept_sparse=True)))
        steps.append(("idf", TextbookIdfTransformer(sublinear_tf=tf == "log")))
        return Pipeline(steps)
    use_idf = idf == "smooth"
    if tf == "log":
        return TfidfVectorizer(sublinear_tf=True, use_idf=use_idf, **kwargs)
    if tf == "binary":
        return TfidfVectorizer(binary=True, use_idf=use_idf, **kwargs)
    if tf == "log1p":
        return Pipeline([
            ("count", CountVectorizer(**kwargs)),
            ("log1p", FunctionTransformer(_log1p, accept_sparse=True)),
            ("idf", TfidfTransformer(use_idf=use_idf)),
        ])
    raise ValueError(f"unknown tf scheme {tf!r}; expected one of {TF_SCHEMES}")


def build_features(
    svd_dim: int,
    tf: str = "binary",
    idf: str = "smooth",
    word_ngrams: tuple[int, int] = (1, 3),
    char_ngrams: tuple[int, int] = (3, 5),
    word_max_features: int | None = None,
    char_max_features: int | None = 300_000,
    layout: bool = True,
) -> Pipeline:
    """TF-IDF (optionally followed by SVD); everything except the classifier.

    A max_features of None keeps every n-gram that passes min_df. With
    layout=False, whitespace and markdown markers are normalized away first.
    """
    features = FeatureUnion([
        ("word", make_vectorizer(
            tf, idf, analyzer="word", ngram_range=tuple(word_ngrams), min_df=2,
            max_features=word_max_features, lowercase=True,
        )),
        # Keep case and whitespace (unless layout=False): formatting is a strong
        # signal in train, though it does not all carry over to test.
        ("char", make_vectorizer(
            tf, idf, analyzer="char", ngram_range=tuple(char_ngrams), min_df=3,
            max_features=char_max_features, lowercase=False,
        )),
    ])
    steps = [] if layout else [("strip_layout", FunctionTransformer(normalize_texts))]
    steps.append(("tfidf", features))
    if svd_dim > 0:
        steps += [
            ("svd", TruncatedSVD(n_components=svd_dim, random_state=SEED)),
            ("norm", Normalizer()),
        ]
    return Pipeline(steps)


MODELS = ("logreg", "svm")


def make_classifier(model: str, C: float):
    if model == "logreg":
        return LogisticRegression(C=C, max_iter=2000)
    if model == "svm":
        # LinearSVC only gives decision scores; Platt-scale them into probabilities
        # using inner 5-fold CV on the training rows, then refit on all of them.
        return CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=10_000, random_state=SEED),
            method="sigmoid", cv=5, ensemble=False,
        )
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


def build_pipeline(svd_dim: int, C: float = 30.0, tf: str = "binary",
                   idf: str = "smooth", model: str = "logreg",
                   **feature_kwargs) -> Pipeline:
    return Pipeline([
        ("features", build_features(svd_dim, tf, idf, **feature_kwargs)),
        ("clf", make_classifier(model, C)),
    ])


def add_ngram_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--word-ngrams", type=int, nargs=2, default=[1, 3],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--char-ngrams", type=int, nargs=2, default=[3, 5],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--word-max-features", type=int, default=0,
                        help="0 = no cap")
    parser.add_argument("--char-max-features", type=int, default=300_000,
                        help="0 = no cap")


def ngram_kwargs(args: argparse.Namespace) -> dict:
    return {
        "word_ngrams": tuple(args.word_ngrams),
        "char_ngrams": tuple(args.char_ngrams),
        "word_max_features": args.word_max_features or None,
        "char_max_features": args.char_max_features or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("../../../data/train.csv"))
    parser.add_argument("--svd", type=int, default=0, help="TruncatedSVD dim (0 = off)")
    parser.add_argument("--model", choices=MODELS, default="logreg")
    parser.add_argument("--C", type=float, default=30.0, help="inverse L2 strength")
    parser.add_argument("--tf", choices=TF_SCHEMES, default="binary")
    parser.add_argument("--idf", choices=IDF_SCHEMES, default="smooth")
    add_ngram_args(parser)
    parser.add_argument("--oof-out", type=Path, default=Path("oof_tfidf.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    fold = make_folds(train)
    y = train["label"].to_numpy()
    oof = np.zeros(len(train))

    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        # Vectorizers are refit inside each fold so validation text never leaks.
        model = build_pipeline(args.svd, args.C, args.tf, args.idf, args.model,
                               **ngram_kwargs(args))
        model.fit(train.loc[tr, "text"], y[tr])
        oof[va] = model.predict_proba(train.loc[va, "text"])[:, 1]
        print(f"fold {k}: auc={roc_auc_score(y[va], oof[va]):.4f} "
              f"logloss={log_loss(y[va], oof[va]):.4f} "
              f"acc={accuracy_score(y[va], oof[va] > 0.5):.4f}")

    print(f"OOF   : auc={roc_auc_score(y, oof):.4f} "
          f"logloss={log_loss(y, oof):.4f} "
          f"acc={accuracy_score(y, oof > 0.5):.4f}")
    pd.DataFrame({"id": train["id"], "fold": fold, "label": y, "pred": oof}).to_csv(
        args.oof_out, index=False)


if __name__ == "__main__":
    main()
