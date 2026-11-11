"""common.py: shared helpers imported by every experiment script (Q1-Q5).

Why this file exists
--------------------
Every report question has to be evaluated on the SAME held-out rows. Otherwise
"model X beat model Y" could just mean "X got an easier validation set". So the
data loading, the train/validation split, and the random seed live here, in one
place, and every script imports them instead of redefining them.

What it provides
----------------
SEED          : the global random seed (42), used for the split, CV folds and models.
load_csv()    : reads a competition CSV from the data folder.
normalize_ws(): collapses all whitespace to single spaces (removes the formatting leak).
get_split()   : returns the fixed 8,000-row train / 2,000-row validation DataFrames.

The split is created once and its ids are saved to splits/val_ids.csv. That file
is committed to git, so a teammate (or a grader) re-running any script gets the
identical validation rows even with a different pandas/sklearn version.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# One seed for everything stochastic. Recorded in the README, as the PR rubric requires.
SEED = 42

# The data folder can be overridden with an environment variable so that nobody has
# to edit code to point at their own copy of the Kaggle files:
#   HOA_DATA=/path/to/kaggle/csvs python q1_model_comparison.py
# The default is a data/ folder next to this file, which is git-ignored (the course
# forbids committing train.csv / test.csv / unlabeled.csv).
DATA_DIR = os.environ.get("HOA_DATA", os.path.join(os.path.dirname(__file__), "data"))

# Where the validation ids are cached. This is the one data-derived file we DO commit:
# it holds only ids, not text or labels, so it leaks nothing.
SPLIT_PATH = os.path.join(os.path.dirname(__file__), "splits", "val_ids.csv")


def load_csv(name):
    """Read one of the competition files (e.g. "train.csv") from DATA_DIR."""
    return pd.read_csv(os.path.join(DATA_DIR, name))


# Compiled once at import time; matches any run of spaces, tabs, \n, \r, etc.
_WS = re.compile(r"\s+")


def normalize_ws(texts):
    """Collapse every whitespace run to a single space and strip the ends.

    Motivation: in train.csv, blank-line paragraph breaks ("\\n\\n") occur in 65% of
    AI passages and 0% of human ones. A model can score very well by detecting
    formatting alone, and that shortcut is unlikely to survive on adversarially
    edited or unseen-domain test text. Running a model on normalized text tells us
    how much of its accuracy comes from the actual wording.

    Returns an object-dtype numpy array so it can be passed straight into sklearn
    vectorizers, just like df["text"].values.
    """
    return np.array([_WS.sub(" ", t).strip() for t in texts], dtype=object)


def get_split():
    """Return (train_df, val_df): 8,000 / 2,000 rows from train.csv, stratified on label.

    First call: makes a stratified random split (so validation keeps the 50/50 class
    balance of the full data) and saves the validation ids to SPLIT_PATH.
    Later calls: read those saved ids back, so the split never changes between
    runs, scripts, or machines.

    Splitting on ids (rather than row positions) means the split still holds if
    train.csv is ever re-saved with its rows in a different order.
    """
    df = load_csv("train.csv")

    if os.path.exists(SPLIT_PATH):
        # Reuse the frozen split.
        val_ids = set(pd.read_csv(SPLIT_PATH)["id"])
    else:
        # Create it once. Only the validation half is needed: everything not in
        # validation is training data.
        _, val = train_test_split(df, test_size=2000, stratify=df["label"], random_state=SEED)
        os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
        val[["id"]].to_csv(SPLIT_PATH, index=False)
        val_ids = set(val["id"])

    # Boolean mask: True for validation rows. reset_index gives both frames a clean
    # 0..n-1 index, so positional lookups like .iloc stay simple later on.
    is_val = df["id"].isin(val_ids)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
