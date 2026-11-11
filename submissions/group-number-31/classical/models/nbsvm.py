"""Extra row: NB-SVM (Wang & Manning, 2012) - binarised word 1-2 + char_wb 2-5 n-gram features scaled
by the naive-Bayes log-count ratio r, then logistic regression."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class NBSVM(BaseModel):
    name = "nbsvm"

    def __init__(self, C: float = 4.0, alpha: float = 1.0, max_features: int = 300000, seed: int = 0):
        self.C, self.alpha, self.seed = C, alpha, seed
        kw = dict(binary=True, use_idf=False, norm=None, lowercase=False, max_features=max_features)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, **kw)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, **kw)

    def _X(self, texts, fit=False):
        f = (lambda v: v.fit_transform(texts)) if fit else (lambda v: v.transform(texts))
        return sp.hstack([f(self.word), f(self.char)]).tocsr()

    def fit(self, texts, y, unlabeled_texts=None):
        X = self._X(texts, fit=True)
        y = np.asarray(y)
        p = self.alpha + np.asarray(X[y == 1].sum(0)).ravel()
        q = self.alpha + np.asarray(X[y == 0].sum(0)).ravel()
        self.r = np.log((p / p.sum()) / (q / q.sum()))
        self.clf = LogisticRegression(C=self.C, solver="liblinear", random_state=self.seed).fit(X.multiply(self.r).tocsr(), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts).multiply(self.r).tocsr())[:, 1].astype(np.float64)
