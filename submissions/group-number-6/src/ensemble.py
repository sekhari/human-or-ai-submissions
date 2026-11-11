"""Equal-weight logit-average ensemble of independently trained pipelines.

No weights are tuned, so the ensemble adds no hyperparameters that could overfit the
validation set.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LogitAverage(BaseEstimator, ClassifierMixin):
    def __init__(self, members):
        self.members = members  # list of (name, unfitted pipeline)

    def fit(self, X, y):
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def member_scores(self, X):
        return {n: m.predict_proba(X)[:, 1] for n, m in self.fitted_}

    def predict_proba(self, X):
        z = np.mean([_logit(s) for s in self.member_scores(X).values()], axis=0)
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
