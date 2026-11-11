"""Fixed 5-fold split of train.csv, shared by every experiment.

The split is built once from SEED with stratified shuffling and cached to
folds.csv (id -> fold). Later runs read the cache, so every model is scored
on exactly the same five 20% blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 20261109
N_FOLDS = 5
DEFAULT_FOLDS_PATH = Path(__file__).resolve().parent / "folds.csv"


def make_folds(train: pd.DataFrame, path: Path = DEFAULT_FOLDS_PATH) -> np.ndarray:
    """Return a fold index (0..N_FOLDS-1) for each row of `train`."""
    if path.exists():
        folds = pd.read_csv(path)
        if set(folds["id"]) != set(train["id"]):
            raise ValueError(f"{path} does not match the ids in train; delete it to rebuild")
        return train["id"].map(folds.set_index("id")["fold"]).to_numpy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.empty(len(train), dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(train, train["label"])):
        fold[val_idx] = k
    pd.DataFrame({"id": train["id"], "fold": fold}).to_csv(path, index=False)
    return fold
