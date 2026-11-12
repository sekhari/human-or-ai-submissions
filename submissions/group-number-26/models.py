"""Model construction shared by the Group 26 experiments.

Currently holds the distance-weighted KNN baseline and the feature
standardization step both it and the later models depend on. The locally
weighted logistic regression is added in a later milestone.

Standardization is always fitted on the training passages only. The helper
below is the single place that happens, so no experiment can accidentally fit
a scaler on validation data.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import config


def fit_scaler(train_features: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training features only."""
    return StandardScaler().fit(train_features)


def build_knn(k: int) -> KNeighborsClassifier:
    """Distance-weighted KNN over standardized features.

    Euclidean distance (scikit-learn's default metric, `minkowski` with p=2)
    and `weights="distance"`, matching the original implementation.
    """
    return KNeighborsClassifier(n_neighbors=k, weights="distance")


def knn_scores(model: KNeighborsClassifier, X: np.ndarray) -> np.ndarray:
    """P(AI) for new passages."""
    return model.predict_proba(X)[:, 1]


def knn_self_excluded_train_scores(model: KNeighborsClassifier) -> np.ndarray:
    """P(AI) for the training rows, each row excluded from its own vote.

    Passing `X=None` to a fitted scikit-learn neighbours estimator queries the
    training set while dropping each point from its own neighbour list. Without
    it a distance-weighted KNN would score every training row against itself at
    distance zero and report a meaningless, near-perfect training AUROC.
    """
    return model.predict_proba(None)[:, 1]


def assert_no_leakage(train_ids, val_ids) -> None:
    """Guard against the train/validation sets sharing any passage."""
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise AssertionError(
            f"{len(overlap)} IDs appear in both train and validation, "
            f"e.g. {sorted(overlap)[:10]}"
        )


__all__ = [
    "config",
    "fit_scaler",
    "build_knn",
    "knn_scores",
    "knn_self_excluded_train_scores",
    "assert_no_leakage",
]
