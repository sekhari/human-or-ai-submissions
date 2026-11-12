"""Competition CSV loading, split provenance, and submission alignment."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_texts(path, id_column, text_column):
    frame = pd.read_csv(path, dtype={id_column: str, text_column: str}, keep_default_na=False)
    missing = {id_column, text_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    check_ids(frame[id_column], path)
    return frame


def check_ids(ids, path):
    if ids.empty or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path} needs nonempty, unique IDs")


def labels_from(frame, column):
    if column is None:
        candidates = [name for name in ("label", "generated", "target", "is_ai") if name in frame]
        if len(candidates) != 1:
            raise ValueError(f"Set --label-col explicitly. CSV columns: {list(frame.columns)}")
        column = candidates[0]
    if column not in frame:
        raise ValueError(f"Label column {column!r} is missing")
    labels = pd.to_numeric(frame[column], errors="raise")
    if set(labels) != {0, 1}:
        raise ValueError("Expected both binary labels: human=0, AI=1")
    return labels.to_numpy(dtype=int), column


def split_rows(frame, labels, text_column, seed):
    if len(frame) != 10_000:
        raise ValueError(f"Use the full train.csv with 10,000 rows, found {len(frame)}")
    train, valid = train_test_split(
        np.arange(len(frame)), test_size=2000, stratify=labels, random_state=seed
    )
    # Keep the team's validation IDs fixed. Quarantine training copies of
    # validation text, including case and whitespace variants.
    normalized = frame[text_column].map(lambda text: " ".join(text.casefold().split()))
    overlapping = normalized.iloc[train].isin(set(normalized.iloc[valid])).to_numpy()
    fit = train[~overlapping]
    if set(labels[fit]) != {0, 1}:
        raise ValueError("Both labels must remain after duplicate quarantine")
    return train, valid, fit


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def submission(test_ids, scores, sample_path, id_column):
    template = pd.read_csv(sample_path, dtype={id_column: str}, keep_default_na=False)
    if id_column not in template or len(template.columns) != 2:
        raise ValueError("Expected exactly an ID column and a score column in the sample submission")
    check_ids(template[id_column], sample_path)
    if set(template[id_column]) != set(test_ids):
        raise ValueError("Test and sample submission IDs differ")
    scores = np.asarray(scores)
    if scores.shape != (len(test_ids),) or not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Expected one finite AI probability in [0, 1] per test row")
    score_column = next(name for name in template.columns if name != id_column)
    template[score_column] = template[id_column].map(dict(zip(test_ids, scores)))
    return template
