"""Extra row: C2's 101 stylometric features -> random forest (bagged deep trees instead of boosting)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from features.stylometry import feature_names, stylometric_features

from .base import BaseModel


class RFStylo(BaseModel):
    name = "rf_stylo"

    def __init__(self, n_estimators: int = 500, max_features: str | float = "sqrt", min_samples_leaf: int = 1,
                 seed: int = 0):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                          min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed)

    @staticmethod
    def reset_cache():
        from features import stylometry
        stylometry._CACHE.clear()

    def fit(self, texts, y, unlabeled_texts=None):
        self.clf.fit(stylometric_features(texts), y)
        self.names_ = feature_names()
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(stylometric_features(texts))[:, 1].astype(np.float64)
