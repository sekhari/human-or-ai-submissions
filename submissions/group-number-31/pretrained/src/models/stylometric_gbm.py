"""Hand-crafted stylometric features (sentence-length variation, vocabulary
richness, punctuation, formatting, repetition, ...) -> tree ensemble.

estimator: lightgbm | random_forest | hist_gb
extended_features: adds function-word, pronoun/conjunction/adverb, readability and
sentence-coherence features (motivated by Ma et al., 2023).
Optionally appends a low-dimensional SVD of char TF-IDF (svd_components > 0)
so the trees also see some content signal.
"""
from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from ..features import TfidfFeaturizer, stylometric_features, stylometric_names
from .base import BaseModel


class _TqdmCallback:
    """LightGBM callback that advances a tqdm bar once per boosting round."""

    def __init__(self, total):
        self.bar = tqdm(total=total, desc="boosting", leave=False)

    def __call__(self, env):
        self.bar.update(1)
        if env.iteration + 1 >= env.end_iteration:
            self.bar.close()


class StylometricGBM(BaseModel):
    name = "stylometric_gbm"

    def __init__(self, estimator: str = "lightgbm", svd_components: int = 0, extended_features: bool = False,
                 lgbm: dict | None = None, random_forest: dict | None = None,
                 hist_gb: dict | None = None, seed: int = 42):
        super().__init__(estimator=estimator, svd_components=svd_components,
                         lgbm=lgbm, random_forest=random_forest, hist_gb=hist_gb)
        self.estimator, self.svd_components, self.seed = estimator, svd_components, seed
        self.extended_features = extended_features
        self.lgbm_params = lgbm or {}
        self.rf_params = random_forest or {}
        self.hgb_params = hist_gb or {}

    def _features(self, texts, fit=False):
        F = stylometric_features(texts, extended=getattr(self, "extended_features", False))
        if self.svd_components:
            from sklearn.decomposition import TruncatedSVD
            if fit:
                self.tfidf = TfidfFeaturizer(char={"ngram_range": [2, 4], "max_features": 100000, "min_df": 3}).fit(texts)
                self.svd = TruncatedSVD(self.svd_components, random_state=self.seed).fit(self.tfidf.transform(texts))
            F = np.hstack([F, self.svd.transform(self.tfidf.transform(texts))])
        return F

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        X = self._features(texts, fit=True)
        if self.estimator == "lightgbm":
            try:
                import lightgbm as lgb
            except (ImportError, OSError):  # e.g. missing libomp on macOS
                print("  lightgbm not importable -> falling back to sklearn hist_gb")
                self.estimator = "hist_gb"
                return self.fit(texts, y, val_texts, val_y, unlabeled_texts)
            p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=self.seed)
            p.update(self.lgbm_params)
            self.clf = lgb.LGBMClassifier(**p)
            self.clf.fit(X, y, callbacks=[_TqdmCallback(p["n_estimators"])])
        elif self.estimator == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            p = dict(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=self.seed)
            p.update(self.rf_params)
            self.clf = RandomForestClassifier(**p).fit(X, y)
        elif self.estimator == "hist_gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            p = dict(max_iter=500, learning_rate=0.05, random_state=self.seed)
            p.update(self.hgb_params)
            self.clf = HistGradientBoostingClassifier(**p).fit(X, y)
        else:
            raise ValueError(f"Unknown estimator {self.estimator}")
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._features(texts))[:, 1]

    def feature_importance(self):
        names = stylometric_names(getattr(self, "extended_features", False)) + [f"svd_{i}" for i in range(self.svd_components)]
        imp = getattr(self.clf, "feature_importances_", None)
        return sorted(zip(names, imp), key=lambda x: -x[1]) if imp is not None else None
