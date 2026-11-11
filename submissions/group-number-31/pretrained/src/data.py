"""Loading the competition CSVs, external data, and building validation splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .config import resolve


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(resolve(path))
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_competition_data(cfg: dict) -> dict[str, pd.DataFrame]:
    d = cfg["data"]
    return {
        "train": load_csv(d["train_file"]),
        "test": load_csv(d["test_file"]),
        "unlabeled": load_csv(d["unlabeled_file"]),
    }


def load_external(cfg: dict) -> pd.DataFrame:
    """Concatenate every CSV in data.external_files (needs text + label columns).

    Competition rule 4(b) forbids externally sourced labelled data for training, validation or choosing
    predictions, so any non-empty list is refused. No logged run used one (experiments/results.csv)."""
    if cfg["data"].get("external_files"):
        raise ValueError("data.external_files must be empty: competition rule 4(b) forbids externally sourced "
                         "labelled data")
    frames = []
    for path in cfg["data"].get("external_files") or []:
        df = load_csv(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        df["source"] = df.get("source", str(path))
        frames.append(df[["text", "label", "source"]])
    if not frames:
        return pd.DataFrame(columns=["text", "label", "source"])
    return pd.concat(frames, ignore_index=True)


def topic_clusters(texts: list[str], n_clusters: int, seed: int) -> np.ndarray:
    """Rough topic/domain clusters (recipes, abstracts, poems, ...) from word TF-IDF."""
    X = TfidfVectorizer(max_features=20000, min_df=3, stop_words="english",
                        sublinear_tf=True).fit_transform(texts)
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=5).fit_predict(X)


def make_splits(texts: list[str], y: np.ndarray, cv_cfg: dict, seed: int, ids=None):
    """Return a list of (train_idx, val_idx) pairs."""
    scheme = cv_cfg.get("scheme", "stratified")
    idx = np.arange(len(y))
    if scheme in ("fixed", "inner"):
        # fixed: the team's 8,000 / 2,000 split (common/val_ids.csv), the same for every model.
        # inner: k-fold CV inside the 8,000 training rows only (hyperparameter search); the 2,000
        #        validation rows are in no fold, so they are never trained on or scored.
        if ids is None:
            raise ValueError(f"cv.scheme={scheme} needs the train ids")
        val_ids = pd.read_csv(resolve(cv_cfg["val_ids_file"]))
        is_val = np.isin(np.asarray(ids), val_ids["id"].to_numpy())
        if is_val.sum() != len(val_ids):
            raise ValueError(f"{is_val.sum()} of {len(val_ids)} validation ids found in the training file")
        if scheme == "fixed":
            return [(idx[~is_val], idx[is_val])]
        inner = idx[~is_val]
        skf = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=seed)
        return [(inner[tr], inner[va]) for tr, va in skf.split(inner, y[inner])]
    if scheme == "holdout":
        tr, va = train_test_split(idx, test_size=cv_cfg.get("val_size", 0.2),
                                  stratify=y, random_state=seed)
        return [(tr, va)]
    if scheme == "stratified":
        skf = StratifiedKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=seed)
        return list(skf.split(idx, y))
    if scheme == "cluster":
        groups = topic_clusters(texts, cv_cfg.get("n_clusters", 8), seed)
        sgkf = StratifiedGroupKFold(n_splits=cv_cfg["n_folds"], shuffle=True, random_state=seed)
        return list(sgkf.split(idx, y, groups))
    raise ValueError(f"Unknown cv.scheme: {scheme}")


def subsample_n(idx: np.ndarray, y: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Nested stratified subsample of n training indices (learning curves): each class is shuffled
    once with `seed` and its first n * class-share indices are kept, so a smaller n is always a
    subset of a larger one."""
    if n is None or n >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[idx]):
        members = rng.permutation(idx[y[idx] == c])
        keep.append(members[:int(round(n * len(members) / len(idx)))])
    return np.sort(np.concatenate(keep))


def subsample(idx: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Stratified subsample of training indices (for learning curves)."""
    if frac >= 1.0:
        return idx
    keep, _ = train_test_split(idx, train_size=frac, stratify=y[idx], random_state=seed)
    return np.sort(keep)
