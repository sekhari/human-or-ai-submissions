# data loaded and fixed val split 

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 20261107
VAL_SIZE = 2000

# our folder (submissions/group-number-34) relative to this file
TEAM_FOLDER = Path(__file__).resolve().parent.parent
SPLIT_FILE = TEAM_FOLDER / "splits" / "val_ids.json"


def load_data(csv_path, has_labels=True):
  # load data and keep only the id, text, and label columns (if found)
  df = pd.read_csv(csv_path)
  df["text"] = df["text"].fillna("").astype(str)

  if has_labels:
    df["label"] = df["label"].astype(int)
    return df[["id", "text", "label"]]
  return df[["id", "text"]]


def create_val_split(train_df, seed=SEED):
  # select 2,000 validation rows 
  _, val_df = train_test_split(
    train_df,
    test_size=VAL_SIZE,
    stratify=train_df["label"],
    random_state=seed,
  )

  val_ids = sorted(int(i) for i in val_df["id"])

  SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
  with open(SPLIT_FILE, "w") as f:
    json.dump({"seed": seed, "val_ids": val_ids}, f)

  return val_ids


def get_train_val(full_df):
  # split labeled data into (train_df, val_df) using the saved validation IDs 
  if not SPLIT_FILE.exists():
    raise FileNotFoundError("splits/val_ids.json is missing. Run make_split.py first.")

  with open(SPLIT_FILE) as f:
    val_ids = set(json.load(f)["val_ids"])

  in_val = full_df["id"].isin(val_ids)
  train_df = full_df[~in_val].reset_index(drop=True)
  val_df = full_df[in_val].reset_index(drop=True)

  if len(val_df) != VAL_SIZE:
    raise ValueError(f"Expected {VAL_SIZE} validation rows but got {len(val_df)}")

  return train_df, val_df