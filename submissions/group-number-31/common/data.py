"""Load the Kaggle CSVs (kept outside git) and apply the team's fixed 8,000 / 2,000 split.

Both tracks (pretrained/ and classical/) import this module, so every number in both
reports is measured on the same 2,000 validation rows listed in common/val_ids.csv.

Where the data lives (first match wins):
  1. --data-dir on the command line (passed in as `data_dir`)
  2. the HOA_DATA_DIR environment variable
  3. submissions/<team>/data/          (git-ignored)
  4. ../AI-vs-Human-Text-Cornell-Tech-Competition-/data next to the course repo (the research repo)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

COMMON_DIR = Path(__file__).resolve().parent
TEAM_DIR = COMMON_DIR.parent
VAL_IDS_FILE = COMMON_DIR / "val_ids.csv"
SEED = 20261023
N_VAL = 2000

_CANDIDATES = [
    TEAM_DIR / "data",
    TEAM_DIR.parents[1].parent / "AI-vs-Human-Text-Cornell-Tech-Competition-" / "data",
]


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Return the folder holding train.csv / test.csv / unlabeled.csv / sample_submission.csv."""
    tried = []
    for cand in [data_dir, os.environ.get("HOA_DATA_DIR"), *_CANDIDATES]:
        if not cand:
            continue
        p = Path(cand).expanduser().resolve()
        tried.append(str(p))
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError(
        "train.csv not found. Pass --data-dir or set HOA_DATA_DIR. Tried:\n  " + "\n  ".join(tried))


def load_csv(name: str, data_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """name: train | test | unlabeled | sample_submission | train_starter_2000."""
    df = pd.read_csv(resolve_data_dir(data_dir) / f"{name}.csv")
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)
    return df


def load_val_ids(path: str | os.PathLike = VAL_IDS_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing - run `python common/make_split.py` first")
    return pd.read_csv(path)


def train_val_split(data_dir: str | os.PathLike | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fixed split: (8,000 training rows, 2,000 validation rows) of train.csv, both in file order."""
    train = load_csv("train", data_dir)
    val_ids = load_val_ids()
    is_val = train["id"].isin(val_ids["id"])
    if is_val.sum() != len(val_ids):
        raise ValueError(f"only {is_val.sum()} of {len(val_ids)} validation ids found in train.csv")
    labels = train.loc[is_val].set_index("id")["label"]
    if not labels.reindex(val_ids["id"]).eq(val_ids.set_index("id")["label"]).all():
        raise ValueError("labels in train.csv disagree with common/val_ids.csv")
    return train.loc[~is_val].reset_index(drop=True), train.loc[is_val].reset_index(drop=True)
