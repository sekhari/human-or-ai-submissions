"""Shared evaluation: metrics, timing, experiment records, prediction files.

Two ways to use it:

* scikit-learn-style models (fit / predict_proba): call ``evaluate_classifier``.
* custom training loops (PyTorch): time training with ``Timer``, compute
  probabilities yourself, then call ``record_experiment``.

Both produce the same JSON record in results/metrics/ and the same validation
prediction CSV in results/predictions/.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import RESULTS_DIR, SplitData

METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def validate_probabilities(probs: Any, n_expected: int | None = None) -> np.ndarray:
    """Check model output: 1-D, finite, within [0, 1], one value per example."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"Expected a 1-D array of P(AI); got shape {p.shape}. "
                         "For sklearn estimators use predict_proba(X)[:, 1].")
    if n_expected is not None and len(p) != n_expected:
        raise ValueError(f"Expected {n_expected} probabilities, got {len(p)}")
    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities contain NaN or inf")
    if p.min() < 0 or p.max() > 1:
        raise ValueError(f"Probabilities must be in [0, 1]; range is [{p.min()}, {p.max()}]")
    return p


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true)
    p = validate_probabilities(y_prob, n_expected=len(y))
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


# --------------------------------------------------------------------------- #
# Experiment records
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    model: str
    seed: int
    train_auc: float | None = None
    val_auc: float | None = None
    train_time_seconds: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)  # per-epoch losses/AUCs (neural models)
    run_name: str = ""
    timestamp: str = ""
    git_commit: str | None = None
    notes: str = ""

    def save(self, directory: Path | str = METRICS_DIR) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_name}.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def load_result(path: Path | str) -> ExperimentResult:
    return ExperimentResult(**json.loads(Path(path).read_text()))


def load_all_results(directory: Path | str = METRICS_DIR) -> pd.DataFrame:
    rows = [asdict(load_result(p)) for p in sorted(Path(directory).glob("*.json"))]
    cols = ["run_name", "model", "seed", "train_auc", "val_auc", "train_time_seconds", "hyperparameters", "timestamp"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def save_predictions(path: Path | str, ids: Sequence, y_prob: Sequence[float], y_true: Sequence[int] | None = None) -> Path:
    """Write id, (label), prob_ai — one row per example, in input order."""
    p = validate_probabilities(y_prob, n_expected=len(ids))
    df = pd.DataFrame({"id": list(ids)})
    if y_true is not None:
        if len(y_true) != len(ids):
            raise ValueError("ids and y_true have different lengths")
        df["label"] = np.asarray(y_true, dtype=int)
    df["prob_ai"] = p
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


# --------------------------------------------------------------------------- #
# End-to-end helpers
# --------------------------------------------------------------------------- #
def record_experiment(
    model_name: str,
    split: SplitData,
    train_probs: Sequence[float],
    val_probs: Sequence[float],
    train_time_seconds: float,
    seed: int,
    hyperparameters: dict[str, Any] | None = None,
    notes: str = "",
    history: list[dict[str, Any]] | None = None,
    run_name: str | None = None,
    save: bool = True,
    metrics_dir: Path | str = METRICS_DIR,
    predictions_dir: Path | str = PREDICTIONS_DIR,
) -> ExperimentResult:
    """Build (and by default save) the standard record from precomputed probabilities."""
    now = datetime.now(timezone.utc)
    train_m = compute_metrics(split.train_labels, train_probs)
    val_m = compute_metrics(split.val_labels, val_probs)
    result = ExperimentResult(
        model=model_name,
        seed=seed,
        train_auc=train_m["auc"],
        val_auc=val_m["auc"],
        train_time_seconds=round(float(train_time_seconds), 3),
        hyperparameters=hyperparameters or {},
        train_metrics=train_m,
        val_metrics=val_m,
        history=history or [],
        run_name=run_name or f"{model_name}_{now.strftime('%Y%m%dT%H%M%SZ')}",
        timestamp=now.isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        notes=notes,
    )
    if save:
        result.save(metrics_dir)
        save_predictions(Path(predictions_dir) / f"{result.run_name}_val.csv",
                         split.val_ids, val_probs, split.val_labels)
    return result


def evaluate_classifier(
    model: Any,
    split: SplitData,
    model_name: str,
    seed: int,
    hyperparameters: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ExperimentResult:
    """Fit on the training partition only, then score train and validation."""
    with Timer() as t:
        model.fit(split.train_texts, split.train_labels)
    train_probs = model.predict_proba(split.train_texts)
    val_probs = model.predict_proba(split.val_texts)
    if hyperparameters is None and hasattr(model, "get_config"):
        hyperparameters = model.get_config()
    return record_experiment(model_name, split, train_probs, val_probs, t.seconds, seed, hyperparameters, **kwargs)
