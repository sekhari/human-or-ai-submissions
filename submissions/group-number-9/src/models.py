"""Model definitions and hyperparameter grids for the model comparison (Q1.2, Q1.3)."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import LAYOUT_FEATURES, build_features, mask_content, strip_layout

SEED = 42


@dataclass
class ModelSpec:
    name: str
    representation: str
    build: Callable
    grid: dict = field(default_factory=dict)
    drop_features: tuple = None  # None: model reads raw text; tuple: style-feature columns to drop

    def inputs(self, df, layout_stripped=False):
        """Turn a DataFrame with a `text` column into this model's X."""
        texts = df.text.map(strip_layout) if layout_stripped else df.text
        if self.drop_features is None:
            return texts.tolist()
        return build_features(texts.to_frame("text")).drop(columns=list(self.drop_features))


def scores(model, X):
    """Continuous ranking score: all AUROC needs. LinearSVC has no predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _word_tfidf(**kw):
    return TfidfVectorizer(sublinear_tf=True, **kw)


def _char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", sublinear_tf=True, min_df=2, max_features=300_000)


def _lr(C=1.0):
    return LogisticRegression(C=C, solver="liblinear", max_iter=2000)


def _hgb():
    return HistGradientBoostingClassifier(random_state=SEED)


_HGB_GRID = {
    "learning_rate": [0.03, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 1.0, 10.0],
}
_WORD_NGRAMS = [(1, 1), (1, 2)]
_CHAR_NGRAMS = [(1, 3), (2, 4), (3, 5)]

MODELS = {m.name: m for m in [
    # Reference point from the report template; deliberately not tuned.
    ModelSpec("word_lr_default", "word 1-gram TF-IDF",
              lambda: Pipeline([("tfidf", TfidfVectorizer()), ("clf", _lr())])),
    ModelSpec("word_nb", "word n-gram counts",
              lambda: Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())]),
              {"vect__ngram_range": _WORD_NGRAMS, "vect__min_df": [2, 5], "clf__alpha": [0.03, 0.1, 0.3, 1.0]}),
    ModelSpec("word_lr", "word n-gram TF-IDF (sublinear)",
              lambda: Pipeline([("tfidf", _word_tfidf()), ("clf", _lr())]),
              {"tfidf__ngram_range": _WORD_NGRAMS, "tfidf__min_df": [2, 5], "clf__C": [0.3, 1, 3, 10, 30, 100, 300]}),
    ModelSpec("char_lr", "char_wb n-gram TF-IDF",
              lambda: Pipeline([("tfidf", _char_tfidf()), ("clf", _lr())]),
              {"tfidf__ngram_range": _CHAR_NGRAMS, "clf__C": [1, 3, 10, 30, 100]}),
    ModelSpec("char_svm", "char_wb n-gram TF-IDF",
              lambda: Pipeline([("tfidf", _char_tfidf()), ("clf", LinearSVC(dual="auto", max_iter=5000))]),
              {"tfidf__ngram_range": _CHAR_NGRAMS, "clf__C": [0.03, 0.1, 0.3, 1]}),
    ModelSpec("masked_lr", "function words + punctuation, content words masked",
              lambda: Pipeline([("tfidf", TfidfVectorizer(preprocessor=mask_content, token_pattern=r"\S+",
                                                          lowercase=False, sublinear_tf=True, min_df=2)),
                                ("clf", _lr())]),
              {"tfidf__ngram_range": [(1, 2), (1, 3), (1, 4)], "clf__C": [0.3, 1, 3, 10, 30, 100]}),
    ModelSpec("style_hgb", "all stylometric features (with line layout)",
              _hgb, _HGB_GRID, drop_features=()),
    ModelSpec("style_hgb_nolayout", "stylometric features minus line layout",
              _hgb, _HGB_GRID, drop_features=tuple(LAYOUT_FEATURES)),
]}


def build_tuned(name, params):
    """Instantiate a model with params loaded from JSON (which turns tuples into lists)."""
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return MODELS[name].build().set_params(**params)


def to_probability(values):
    """Map scores into [0, 1] for Kaggle. Monotonic, so AUROC does not change."""
    values = np.asarray(values, dtype=float)
    if values.min() >= 0 and values.max() <= 1:
        return values
    return 1 / (1 + np.exp(-values))
