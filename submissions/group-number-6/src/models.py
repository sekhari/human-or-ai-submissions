"""Model factories. Every model is a scikit-learn Pipeline mapping raw texts -> P(AI).

Model A  word TF-IDF (1-2 grams) + logistic regression        (reference-style baseline)
Model B  char TF-IDF (char_wb n-grams) + logistic regression
Model C  39 stylometric features + histogram gradient boosting
Model D  word counts + multinomial naive Bayes
Model E  word + char TF-IDF + linear SVM (sigmoid-scaled scores)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .features import Normalizer, StyleFeatures

SEED = 20261102


def word_tfidf(ngram=(1, 2), min_df=2, max_features=200_000):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features,
                           sublinear_tf=True, lowercase=True, strip_accents=None,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]")


def char_tfidf(ngram=(2, 5), min_df=3, max_features=300_000, analyzer="char_wb"):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df,
                           max_features=max_features, sublinear_tf=True, lowercase=False)


class SVMProba(LinearSVC):
    """LinearSVC with a logistic squashing of the margin so it exposes predict_proba.
    AUROC only depends on the ranking, so the squashing does not change scores."""

    def predict_proba(self, X):
        d = self.decision_function(X)
        p = 1 / (1 + np.exp(-2 * d))
        return np.column_stack([1 - p, p])


def build(name: str, normalize: bool = False, **p) -> Pipeline:
    steps = [("norm", Normalizer())] if normalize else []
    if name == "A_word_lr":
        steps += [("vec", word_tfidf(p.get("ngram", (1, 2)), p.get("min_df", 2))),
                  ("clf", LogisticRegression(C=p.get("C", 4.0), max_iter=3000, solver="liblinear"))]
    elif name == "B_char_lr":
        steps += [("vec", char_tfidf(p.get("ngram", (2, 5)), p.get("min_df", 3),
                                     analyzer=p.get("analyzer", "char_wb"))),
                  ("clf", LogisticRegression(C=p.get("C", 8.0), max_iter=3000, solver="liblinear"))]
    elif name == "C_style_gbm":
        steps += [("vec", StyleFeatures()),
                  ("clf", HistGradientBoostingClassifier(
                      learning_rate=p.get("lr", 0.05), max_iter=p.get("max_iter", 400),
                      max_leaf_nodes=p.get("max_leaf_nodes", 31), l2_regularization=1.0,
                      random_state=SEED))]
    elif name == "D_word_nb":
        steps += [("vec", CountVectorizer(ngram_range=p.get("ngram", (1, 2)), min_df=2,
                                          token_pattern=r"(?u)\b\w+\b|[^\w\s]")),
                  ("clf", MultinomialNB(alpha=p.get("alpha", 0.1)))]
    elif name == "E_wordchar_svm":
        steps += [("vec", FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])),
                  ("clf", SVMProba(C=p.get("C", 0.5), max_iter=5000, random_state=SEED))]
    else:
        raise ValueError(name)
    return Pipeline(steps)


def proba(model, texts) -> np.ndarray:
    return model.predict_proba(list(texts))[:, 1]
