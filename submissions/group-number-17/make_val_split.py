"""
Step 1: make the shared validation split.
Holds out 2,000 rows of train.csv (half human, half AI) with a fixed seed,
so the whole team tests on the exact same rows.

Run (from your team folder):  python make_val_split.py
Needs: data/train.csv (from Kaggle, not committed)
Out:   val_ids.csv  (the ids of the 2,000 validation rows)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

train = pd.read_csv("data/train.csv")
print("Rows in train.csv:", len(train))
print("Label counts:\n", train["label"].value_counts())

# stratify = keep the same human/AI ratio in the validation set
_, val = train_test_split(train, test_size=2000, stratify=train["label"], random_state=SEED)

val[["id"]].to_csv("val_ids.csv", index=False)
print("\nValidation rows:", len(val))
print("Validation label counts:\n", val["label"].value_counts())
print("Saved val_ids.csv")