"""load competition csvs and the fixed validation split used by every experiment."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parents[1] / "results"
SEED = 3737
VAL_SIZE = 2000


def load(name):
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)


def load_all():
    return load("train"), load("unlabeled"), load("test")


def fixed_split(train):
    """positional indices of the 8000/2000 stratified split required by the report."""
    idx = np.arange(len(train))
    return train_test_split(idx, test_size=VAL_SIZE, stratify=train.label, random_state=SEED)
