"""Competition input and submission contracts; IDs are preserved as strings."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path, *, labeled=False, expected_rows=None):
    frame = pd.read_csv(path, dtype={"id": str, "text": str}, keep_default_na=False)
    expected = {"id", "text", "label"} if labeled else {"id", "text"}
    if set(frame.columns) != expected:
        raise ValueError(f"{path}: expected columns {sorted(expected)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(frame)}")
    if frame.empty or frame.id.str.strip().eq("").any() or frame.id.duplicated().any():
        raise ValueError(f"{path}: IDs must be nonempty and unique")
    if frame.text.str.strip().eq("").any():
        raise ValueError(f"{path}: blank text requires an explicit data-cleaning decision")
    if labeled and (not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2):
        raise ValueError(f"{path}: labels must contain both 0 and 1 only")
    return frame


def split_manifest(frame):
    # Use original CSV row order to match the team plan. Share this file, not
    # merely a seed: reordering the input changes train_test_split membership.
    train, validation = train_test_split(
        np.arange(len(frame)), test_size=0.2, stratify=frame.label, random_state=42)
    manifest = frame[["id"]].copy()
    manifest["split"] = "train"
    manifest.loc[validation, "split"] = "validation"
    return manifest


def partition(frame, manifest):
    if (list(manifest.columns) != ["id", "split"] or manifest.id.duplicated().any()
            or set(manifest.id) != set(frame.id)
            or set(manifest.split) != {"train", "validation"}):
        raise ValueError("Split manifest must assign every training ID exactly once")
    assignments = manifest.set_index("id")["split"]
    mask = frame.id.map(assignments).eq("train")
    train, validation = frame[mask], frame[~mask]
    if train.label.nunique() != 2 or validation.label.nunique() != 2:
        raise ValueError("Both partitions must contain both classes")
    return train, validation


def make_submission(test, probabilities, sample, *, expected_rows=3000):
    p = np.asarray(probabilities, dtype=float)
    if (len(test) != expected_rows or test.id.duplicated().any()
            or p.shape != (len(test),) or not np.isfinite(p).all()
            or not ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Submission needs unique test IDs and finite probabilities in [0, 1]")
    if (list(sample.columns) != ["id", "label"] or sample.id.duplicated().any()
            or set(sample.id) != set(test.id)):
        raise ValueError("Sample submission IDs/columns do not match test data")
    return pd.DataFrame({"id": test.id.to_numpy(), "label": p})
