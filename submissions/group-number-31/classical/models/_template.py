"""Template for a new model: copy to models/<name>.py, rename the class, register it in
models/__init__.py, and add configs/<name>.yaml. Run `python train.py --model template` to see the
pipeline work end to end (it is a deliberately weak bag-of-characters baseline)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class TemplateModel(BaseModel):
    name = "template"

    def __init__(self, C: float = 1.0, seed: int = 0):
        # every argument comes from `params:` in the config (seed is filled in by train.py)
        self.C, self.seed = C, seed

    def fit(self, texts, y, unlabeled_texts=None):
        self.vec = CountVectorizer(analyzer="char", ngram_range=(1, 1), lowercase=False)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(C=self.C, solver="liblinear", random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1].astype(np.float64)
