"""Common interface every model implements.

A model receives *already preprocessed* raw strings and returns the
probability (or any monotone score in [0, 1]) that each text is AI-generated.
AUROC only depends on ranking, so calibrated probabilities are not required.

To add a model: create src/models/<name>.py with a subclass of BaseModel,
register it in src/models/__init__.py, and add configs/models/<name>.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np


class BaseModel:
    name = "base"

    def __init__(self, **params):
        self.params = params

    def fit(self, texts: Sequence[str], y: np.ndarray,
            val_texts: Sequence[str] | None = None, val_y: np.ndarray | None = None,
            unlabeled_texts: Sequence[str] | None = None) -> "BaseModel":
        """unlabeled_texts: optional extra text (unlabeled + test) that a model
        may use for unsupervised steps such as fitting a vocabulary. Labels of
        the validation set must never be used for fitting, only for logging."""
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D array: P(AI-generated) for each text."""
        raise NotImplementedError

    # Default persistence: pickle the whole object. Torch models override this.
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.joblib")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "BaseModel":
        """device: optional torch device override for models that use one."""
        model = joblib.load(Path(path) / "model.joblib")
        if device is not None and hasattr(model, "set_device"):
            model.set_device(device)
        return model
