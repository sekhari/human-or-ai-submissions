"""C2: hand-crafted stylometric features -> gradient-boosted trees (LightGBM).

Features (features/stylometry.py): length, sentence-length mean/std and burstiness, type-token and
hapax ratios, repetition, punctuation rates, function-word rates and closed-class POS proxies,
readability, adjacent-sentence coherence, blank lines / line structure, and character classes
including zero-width characters and Cyrillic/Greek homoglyphs. 101 numbers per passage.
"""
from __future__ import annotations

import numpy as np

from features.stylometry import feature_names, stylometric_features

from .base import BaseModel


class StyloGBM(BaseModel):
    name = "stylo_gbm"

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.03, num_leaves: int = 31,
                 min_child_samples: int = 20, colsample_bytree: float = 0.7, subsample: float = 0.8,
                 reg_lambda: float = 1.0, extended: bool = True, added: bool = True, drop: list | None = None,
                 seed: int = 0):
        self.lgbm = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                         min_child_samples=min_child_samples, colsample_bytree=colsample_bytree,
                         subsample=subsample, subsample_freq=1, reg_lambda=reg_lambda, random_state=seed,
                         verbose=-1, n_jobs=4)
        self.extended, self.added, self.drop = extended, added, list(drop or [])

    def _X(self, texts):
        X = stylometric_features(texts, self.extended, self.added)
        names = feature_names(self.extended, self.added)
        keep = [i for i, n in enumerate(names) if n not in self.drop]  # ablations: drop named features
        self.names_ = [names[i] for i in keep]
        return X[:, keep]

    @staticmethod
    def reset_cache():
        from features import stylometry
        stylometry._CACHE.clear()

    def fit(self, texts, y, unlabeled_texts=None):
        import lightgbm as lgb
        self.clf = lgb.LGBMClassifier(**self.lgbm).fit(self._X(texts), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts))[:, 1].astype(np.float64)

    def importances(self):
        return sorted(zip(self.names_, self.clf.booster_.feature_importance("gain")), key=lambda x: -x[1])
