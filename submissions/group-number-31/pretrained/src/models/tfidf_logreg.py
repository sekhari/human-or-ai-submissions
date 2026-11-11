"""TF-IDF (word + char n-grams) -> L2-regularised logistic regression."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..features import TfidfFeaturizer
from .base import BaseModel


class TfidfLogReg(BaseModel):
    name = "tfidf_logreg"

    def __init__(self, tfidf: dict, C: float = 4.0, max_iter: int = 2000,
                 class_weight=None, fit_vocab_on_unlabeled: bool = False):
        super().__init__(tfidf=tfidf, C=C)
        self.featurizer = TfidfFeaturizer(**tfidf)
        self.fit_vocab_on_unlabeled = fit_vocab_on_unlabeled
        self.clf = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight,
                                      solver="liblinear")

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        vocab_texts = list(texts)
        if self.fit_vocab_on_unlabeled and unlabeled_texts is not None:
            vocab_texts += list(unlabeled_texts)
        self.featurizer.fit(vocab_texts)
        X = self.featurizer.transform(texts)
        print(f"  features: {X.shape[1]:,}  |  fitting logistic regression ...")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.featurizer.transform(texts))[:, 1]

    def top_features(self, k: int = 30):
        """Most AI-indicative (positive) and human-indicative (negative) n-grams."""
        names = np.concatenate([v.get_feature_names_out() for v in self.featurizer.vectorizers.values()])
        coef = self.clf.coef_.ravel()
        order = np.argsort(coef)
        return ([(names[i], coef[i]) for i in order[::-1][:k]],
                [(names[i], coef[i]) for i in order[:k]])
