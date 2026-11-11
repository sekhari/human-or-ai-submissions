"""Interface every classical model implements (see CONTRIBUTING.md)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np


class BaseModel:
    """A model trained from scratch: no pretrained weights, embeddings or tokenizers.

    __init__ takes the keyword arguments in configs/<name>.yaml `params:` (plus `seed`, and `device`
    if the signature has it). train.py calls fit() on the training rows and predict_proba() on any
    texts; it never shows a model validation labels.
    """

    name = "base"
    device_used = "cpu"  # torch models set this to the device they trained on (reported as hardware)

    def fit(self, texts: Sequence[str], y: np.ndarray, unlabeled_texts: Sequence[str] | None = None) -> "BaseModel":
        """texts / y: labeled training rows. unlabeled_texts: unlabeled.csv passages, usable for
        unsupervised steps (vocabulary, tokenizer, pseudo-labels). Never contains validation or test text."""
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """P(AI-written) for each text, shape (n,), values in [0, 1]."""
        raise NotImplementedError

    # Optional: a model that caches features across fits (e.g. for the search grid) can define
    # reset_cache(); train.py calls it before the timed final fit so fit_time_s is honest.

    # Default persistence pickles the whole object; override for torch models.
    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / "model.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return joblib.load(Path(path) / "model.joblib")
