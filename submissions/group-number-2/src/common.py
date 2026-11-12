"""Shared data loading, validation split, features and metrics.

Every experiment imports from here so all models use the *same* fixed
stratified 2,000-row validation split (seed 42).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TEAM_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", TEAM_DIR / "data"))
RESULTS_DIR = Path(os.environ.get("AML_RESULTS_DIR", TEAM_DIR / "results"))
PRED_DIR = RESULTS_DIR / "preds"
SEED = 42
VAL_SIZE = 2000

TEXT_CANDS = ["text", "passage", "generation", "content"]
LABEL_CANDS = ["label", "generated", "is_ai", "target", "y"]
ID_CANDS = ["id", "ID", "Id"]


def find_col(df: pd.DataFrame, cands) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_csv(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    p = Path(data_dir) / name
    return pd.read_csv(p) if p.exists() else None


class Data:
    """Labeled train/val split plus unlabeled and test texts."""

    def __init__(self, data_dir: Path = DATA_DIR, seed: int = SEED):
        df = load_csv("train.csv", data_dir)
        if df is None:
            raise FileNotFoundError(f"train.csv not found in {data_dir}")
        self.text = find_col(df, TEXT_CANDS)
        self.label = find_col(df, LABEL_CANDS)
        self.id = find_col(df, ID_CANDS)
        assert self.text and self.label, f"Unknown columns: {df.columns.tolist()}"
        if self.id is None:
            df["row_id"] = np.arange(len(df))
            self.id = "row_id"
        df[self.text] = df[self.text].fillna("").astype(str)
        df[self.label] = df[self.label].astype(int)
        self.full = df

        tr_idx, va_idx = train_test_split(
            np.arange(len(df)), test_size=VAL_SIZE, stratify=df[self.label], random_state=seed
        )
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        # leakage guard: drop train rows whose normalized text also appears in val
        va_norm = set(va[self.text].map(_norm))
        dup = tr[self.text].map(_norm).isin(va_norm)
        self.n_dups_removed = int(dup.sum())
        self.tr = tr[~dup].reset_index(drop=True)
        self.va = va

        self.unl_df = load_csv("unlabeled.csv", data_dir)
        self.test_df = load_csv("test.csv", data_dir)
        self.sample_sub = load_csv("sample_submission.csv", data_dir)

    # convenience accessors
    @property
    def X_tr(self):
        return self.tr[self.text].values

    @property
    def y_tr(self):
        return self.tr[self.label].values

    @property
    def X_va(self):
        return self.va[self.text].values

    @property
    def y_va(self):
        return self.va[self.label].values

    def texts_of(self, df):
        if df is None:
            return None
        c = self.text if self.text in df.columns else find_col(df, TEXT_CANDS)
        return df[c].fillna("").astype(str).values

    @property
    def X_unl(self):
        return self.texts_of(self.unl_df)

    @property
    def X_test(self):
        return self.texts_of(self.test_df)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ---------------- "weird character" features ----------------
ZW = re.compile("[​-‏⁠﻿]")
WEIRD_NAMES = [
    "non_ascii_frac", "zero_width", "odd_whitespace", "multi_space", "greek_cyrillic_frac",
    "n_words", "mean_word_len", "upper_frac", "digit_frac", "newlines",
]


def weird_feats(texts) -> np.ndarray:
    rows = []
    for t in texts:
        n = max(len(t), 1)
        w = t.split()
        gc_count = sum(
            1 for c in t
            if ord(c) > 127 and unicodedata.name(c, "").split(" ")[0] in ("CYRILLIC", "GREEK")
        )
        rows.append([
            sum(ord(c) > 127 for c in t) / n,
            len(ZW.findall(t)),
            len(re.findall(r"[^\S\n ]", t)),
            len(re.findall(r" {2,}", t)),
            gc_count / n,
            len(w),
            np.mean([len(x) for x in w]) if w else 0.0,
            sum(c.isupper() for c in t) / n,
            sum(c.isdigit() for c in t) / n,
            t.count("\n"),
        ])
    return np.array(rows, dtype=float)


# ---------------- metrics / io ----------------
def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def rank01(p) -> np.ndarray:
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def save_preds(name: str, **arrays) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{name}.npz"
    np.savez(path, **arrays)
    return path


def load_preds(name: str) -> dict | None:
    path = PRED_DIR / f"{name}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def update_results(key: str, value) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = value
    path.write_text(json.dumps(res, indent=2, default=float))


def read_results() -> dict:
    path = RESULTS_DIR / "results.json"
    return json.loads(path.read_text()) if path.exists() else {}
