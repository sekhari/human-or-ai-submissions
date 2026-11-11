"""Predicting with saved runs and checking submission files (used by predict.py,
make_submission.py and scripts/blend.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import read_yaml, resolve
from .data import load_csv
from .models import get_model_class
from .preprocessing import preprocess


def find_run(run: str | Path, runs_dir: str = "outputs/runs") -> Path:
    """Accept a run directory, or just a run name under outputs/runs/."""
    p = resolve(run)
    if p.is_dir():
        return p
    p = resolve(runs_dir) / str(run)
    if p.is_dir():
        return p
    raise FileNotFoundError(f"No run directory '{run}' (looked in {resolve(runs_dir)})")


def model_dirs(run: Path) -> list[Path]:
    """Saved model(s) of a run: the refit model, or one per CV fold for fold-averaged runs."""
    if (run / "model.joblib").exists() or (run / "transformer_hparams.json").exists():
        return [run]
    folds = sorted(p for p in run.glob("fold_*") if p.is_dir())
    if folds:
        return folds
    raise FileNotFoundError(
        f"No saved model in {run}. Fold-averaged runs only keep fold models when trained with "
        f"output.save_fold_models=true (on by default for fine-tunes since 2026-10-30); their test "
        f"predictions are in {run / 'test_pred.csv'}.")


def has_model(run: Path) -> bool:
    try:
        model_dirs(run)
        return True
    except FileNotFoundError:
        return False


def predict_run(run: Path, texts_raw, device: str | None = None) -> np.ndarray:
    """Predict raw texts with a saved run: same preprocessing as in training, mean over its models."""
    cfg = read_yaml(run / "config.yaml")
    cls = get_model_class(cfg["model"]["name"])
    texts = preprocess(pd.Series(list(texts_raw)), cfg["preprocessing"], "preprocess")
    preds = []
    for d in model_dirs(run):
        model = cls.load(d, device=device)
        preds.append(model.predict_proba(texts))
        if hasattr(model, "release"):
            model.release()
    return np.mean(preds, 0)


def validate_submission(sub: pd.DataFrame, sample_file: str = "data/sample_submission.csv") -> list[str]:
    """Problems with a submission (empty list = OK): columns, row count, ids vs the sample
    (Kaggle matches rows by id, so order doesn't matter), duplicates, range, NaN."""
    problems = []
    sample = pd.read_csv(resolve(sample_file))
    if list(sub.columns) != ["id", "label"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['id', 'label']")
        return problems
    ids, want = sub["id"].astype(str), set(sample["id"].astype(str))
    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, expected {len(sample)}")
    if ids.duplicated().any():
        problems.append(f"{int(ids.duplicated().sum())} duplicate ids")
    if set(ids) != want:
        problems.append(f"ids differ from sample_submission ({len(set(ids) - want)} unexpected, "
                        f"{len(want - set(ids))} missing)")
    lab = pd.to_numeric(sub["label"], errors="coerce")
    if lab.isna().any():
        problems.append(f"{int(lab.isna().sum())} missing / non-numeric labels")
    if ((lab < 0) | (lab > 1)).any():
        problems.append("labels outside [0, 1]")
    if lab.nunique() < 2:
        problems.append("all labels identical (AUROC would be 0.5)")
    return problems
