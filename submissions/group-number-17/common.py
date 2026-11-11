"""
Loads train.csv and splits it with our shared val_ids.csv.
Used by nb_model.py, pseudo_label.py and ensemble.py.
"""
import pandas as pd

SEED = 42


def load_split():
    train = pd.read_csv("data/train.csv")
    val_ids = set(pd.read_csv("val_ids.csv")["id"])
    is_val = train["id"].isin(val_ids)
    tr = train[~is_val].reset_index(drop=True)
    va = train[is_val].reset_index(drop=True)
    assert len(va) == 2000, f"expected 2000 val rows, got {len(va)}"
    return tr, va
