"""Extra row: the C1 TF-IDF features (word 1-2 + char_wb 2-5) -> linear SVM (hinge loss).

AUROC only needs a ranking, so the SVM margin is squashed with a logistic function instead of being
calibrated."""
from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC

from .tfidf_lr import TfidfLR


class TfidfLinearSVM(TfidfLR):
    name = "linear_svm"

    def fit(self, texts, y, unlabeled_texts=None):
        X = self._X(texts, fit=True)
        self.clf = LinearSVC(C=self.C, random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, texts):
        return 1 / (1 + np.exp(-self.clf.decision_function(self._X(texts))))
