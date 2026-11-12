"""Fold-local word models for the training-only generalization search.

All learned vocabulary, IDF and optional supervised NB ratios belong to the
pipeline, so cross-validation fits them using the current fit rows only.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from preprocess import clean_text


def shared_clean_lower(text):
    """Match shared cleaning followed by vectorizer lowercasing."""
    return clean_text(str(text)).lower()


class NBLogCountRatio(TransformerMixin, BaseEstimator):
    """Multiply sparse features by a smoothed, training-only log-count ratio."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("NB ratios require both binary classes 0 and 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        positive = np.asarray(X[y == 1].sum(axis=0)).ravel() + self.alpha
        negative = np.asarray(X[y == 0].sum(axis=0)).ravel() + self.alpha
        self.ratio_ = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "ratio_")
        X = sparse.csr_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature width differs from the fitted NB ratio")
        return X.multiply(self.ratio_).tocsr()


def build_pipeline(config):
    """Return an unfitted, cloneable and pickle-safe text classification model."""
    token_patterns = {
        "default": r"(?u)\b\w\w+\b",
        "single": r"(?u)\b\w+\b",
        "punctuation": r"(?u)\b\w+\b|[^\w\s]",
    }
    vectorizer = TfidfVectorizer(
        preprocessor=shared_clean_lower if config.get("clean", "raw") == "shared" else None,
        lowercase=True,
        strip_accents=None,
        ngram_range=tuple(config.get("ngram_range", (1, 2))),
        min_df=config.get("min_df", 2),
        max_df=config.get("max_df", 1.0),
        max_features=config.get("max_features", 100000),
        sublinear_tf=config.get("sublinear_tf", True),
        token_pattern=token_patterns[config.get("tokens", "default")],
        dtype=np.float64,
    )
    steps = [("tfidf", vectorizer)]
    if config.get("nb_ratio", False):
        steps.append(("nb_ratio", NBLogCountRatio(alpha=config.get("nb_alpha", 1.0))))
    model = config.get("model", "logistic")
    if model == "logistic":
        classifier = LogisticRegression(
            C=config.get("C", 4.0),
            solver="liblinear",
            max_iter=1000,
            random_state=12,
        )
    elif model == "linear_svc":
        classifier = LinearSVC(
            C=config.get("C", 1.0), dual="auto", max_iter=5000, random_state=12,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def candidate_configs():
    """Small predeclared search across model, cleaning and word representation."""
    base = dict(clean="raw", model="logistic", ngram_range=[1, 2], min_df=2,
                max_df=1.0, max_features=100000, sublinear_tf=True,
                tokens="default", nb_ratio=False)
    candidates = []
    for C in [1.0, 4.0, 16.0]:
        candidates.append(dict(base, name=f"raw_word12_lr_C{C:g}", C=C))
    candidates.extend([
        dict(base, name="clean_word12_lr_C4", clean="shared", C=4.0),
        dict(base, name="raw_word13_lr_C4", ngram_range=[1, 3], max_features=150000, C=4.0),
        dict(base, name="raw_single_word12_lr_C4", tokens="single", C=4.0),
        dict(base, name="raw_punct_word12_lr_C4", tokens="punctuation", C=4.0),
    ])
    for C in [0.1, 1.0, 4.0]:
        candidates.append(dict(base, name=f"raw_word12_svc_C{C:g}", model="linear_svc", C=C))
    candidates.append(dict(base, name="raw_word12_nblr_C4", nb_ratio=True, C=4.0))
    return candidates


if __name__ == "__main__":
    from models.text_generalization.common import run_candidates

    run_candidates("word", candidate_configs(), build_pipeline)
