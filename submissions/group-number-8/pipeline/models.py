"""Classifiers and the rank-average ensembles.

ENS_top2 averages sgd_bpe and gbm_format. ENS_top3 adds lr_bpe with equal
weight. Scores are percentile ranks inside the set being scored, because
AUROC depends only on order and the three models are not on one scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from pipeline.config import (
    GBM_L2,
    GBM_LEARNING_RATE,
    GBM_MAX_ITER,
    GBM_MAX_LEAF_NODES,
    GBM_MIN_SAMPLES_LEAF,
    LR_C,
    MODEL_SEED,
    SGD_ALPHA,
)


def make_sgd(alpha: float = SGD_ALPHA, seed: int = MODEL_SEED) -> SGDClassifier:
    return SGDClassifier(
        loss="modified_huber",
        alpha=alpha,
        max_iter=5000,
        tol=1e-4,
        random_state=seed,
    )


def make_lr(C: float = LR_C) -> LogisticRegression:
    return LogisticRegression(C=C, solver="liblinear", max_iter=2000)


def make_gbm(
    learning_rate: float = GBM_LEARNING_RATE,
    max_leaf_nodes: int = GBM_MAX_LEAF_NODES,
    l2_regularization: float = GBM_L2,
    seed: int = MODEL_SEED,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=GBM_MAX_ITER,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
        l2_regularization=l2_regularization,
        random_state=seed,
    )


def rank01(scores: np.ndarray) -> np.ndarray:
    return (rankdata(np.asarray(scores, dtype=np.float64)) - 1) / max(len(scores) - 1, 1)


def rank_average(score_columns: list[np.ndarray]) -> np.ndarray:
    return np.mean([rank01(column) for column in score_columns], axis=0)


def submission_frame(ids: pd.Series, labels: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids.astype(int).to_numpy(), "label": labels})
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate ids in submission")
    if frame["label"].isna().any() or not np.isfinite(frame["label"]).all():
        raise ValueError("Submission labels must be finite")
    if not ((frame["label"] >= 0) & (frame["label"] <= 1)).all():
        raise ValueError("Submission labels must lie in [0, 1]")
    return frame
