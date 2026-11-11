"""The minimal contract every model must satisfy.

This is a structural ``Protocol``: no inheritance needed. Any object with these
methods works with ``src.evaluate.evaluate_classifier``. Neural models may
instead use their own training loop and call ``src.evaluate.record_experiment``
directly — the only hard requirement is the output format below.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class TextClassifier(Protocol):
    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TextClassifier":
        """Train on the training partition ONLY. Fit any vectorizer/tokenizer vocab here."""
        ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D float array of P(AI) in [0, 1], one per input, in input order.

        Note: unlike scikit-learn's predict_proba, this is 1-D (the AI column only).
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """JSON-serializable hyperparameters and seed, saved with every experiment."""
        ...
