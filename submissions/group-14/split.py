"""Fixed labeled validation partition shared by every model.

Create once (already committed as val_ids.txt):
    python split.py --train data/train.csv

Use in a model script:
    from split import load_split
    train_df, val_df = load_split(pd.read_csv("data/train.csv"))
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 20261023
VAL_SIZE = 2000
VAL_IDS_PATH = Path(__file__).with_name("val_ids.txt")


def load_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_ids = set(int(x) for x in VAL_IDS_PATH.read_text().split())
    is_val = train["id"].isin(val_ids)
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the fixed validation ids")
    parser.add_argument("--train", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    _, val = train_test_split(train, test_size=VAL_SIZE, stratify=train["label"], random_state=SEED)
    VAL_IDS_PATH.write_text("\n".join(str(i) for i in sorted(val["id"])) + "\n")
    print(f"wrote {len(val)} validation ids ({val['label'].mean():.1%} AI) to {VAL_IDS_PATH}")


if __name__ == "__main__":
    main()
