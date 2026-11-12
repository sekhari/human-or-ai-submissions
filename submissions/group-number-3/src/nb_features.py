"""NB-weighted binary n-grams with the shared logistic-regression pipeline.

Log-count weighting: Wang and Manning (2012), https://aclanthology.org/P12-2018/.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from models import tfidf_logreg


class NBLogRatio(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, binary_output=True, renormalize=True):
        self.alpha = alpha
        self.binary_output = binary_output
        self.renormalize = renormalize

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y)
        if self.alpha <= 0 or len(y) != X.shape[0] or set(np.unique(y)) != {0, 1}:
            raise ValueError("Require positive alpha and aligned binary training labels.")
        if not np.isfinite(X.data).all() or np.any(X.data < 0):
            raise ValueError("Require finite, nonnegative input features.")
        binary = X.copy()
        binary.eliminate_zeros()
        binary.data = np.ones_like(binary.data)
        positive = np.asarray(binary[y == 1].sum(axis=0)).ravel() + self.alpha
        negative = np.asarray(binary[y == 0].sum(axis=0)).ravel() + self.alpha
        self.ratio_ = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "ratio_")
        X = sparse.csr_matrix(X, copy=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training.")
        if self.binary_output:
            X.eliminate_zeros()
            X.data = np.ones_like(X.data)
        X = X.multiply(self.ratio_).tocsr()
        return normalize(X, norm="l2", copy=False) if self.renormalize else X


def nb_weighted_logreg():
    baseline = tfidf_logreg()
    blocks = [
        (name, Pipeline([
            ("tfidf", clone(vectorizer)),
            ("nb", NBLogRatio(alpha=1, binary_output=True, renormalize=True)),
        ]))
        for name, vectorizer in baseline.named_steps["features"].transformer_list
    ]
    return Pipeline([
        ("features", FeatureUnion(blocks)),
        ("clf", clone(baseline.named_steps["clf"])),
    ])
