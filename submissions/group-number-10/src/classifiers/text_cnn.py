"""Text CNN trained from scratch (PyTorch). Owner: Joshua.

Notes for the implementer:
* Build the vocabulary from training texts only (inside ``fit``).
* Use ``src.utils.set_seed`` and ``src.utils.get_device`` (CUDA -> MPS -> CPU).
* If you need early stopping, carve it out of the 8,000 training examples —
  do not tune on the 2,000 validation examples during training.
* A custom loop is fine; finish with ``src.evaluate.record_experiment`` or keep
  this fit/predict_proba shape and use ``evaluate_classifier``.
* Save checkpoints under ``checkpoints/`` (git-ignored).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class TextCNNClassifier:
    def __init__(self, seed: int = 42, device: str | None = None, **hyperparameters: Any) -> None:
        self.seed = seed
        self.device = device
        self.hyperparameters = hyperparameters

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TextCNNClassifier":
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {"seed": self.seed, **self.hyperparameters}
