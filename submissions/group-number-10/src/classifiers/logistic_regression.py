"""TF-IDF + Logistic Regression (scikit-learn). Owner: Diamond.

Suggested shape: a sklearn Pipeline(TfidfVectorizer, LogisticRegression) built
in __init__, fitted in ``fit`` on training texts only, and
``predict_proba`` returning ``pipeline.predict_proba(texts)[:, 1]``.
Consider ``lowercase=False`` and char n-grams — casing/punctuation may be signal.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class TfidfLogisticRegression:
    def __init__(self, seed: int = 42, **hyperparameters: Any) -> None:
        self.seed = seed
        self.hyperparameters = hyperparameters

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLogisticRegression":
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {"seed": self.seed, **self.hyperparameters}
