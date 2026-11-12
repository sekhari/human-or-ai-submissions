"""
Shared code for the whole team. Everyone imports from here so that
we all read the data and preprocess text in exactly the same way.
"""
import re
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

DATA_DIR = "data"


def load_data(data_dir=DATA_DIR):
    """Read the three competition files."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    unlabeled = pd.read_csv(f"{data_dir}/unlabeled.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, unlabeled, test


def clean_text(text, mode="raw"):
    """
    mode="raw":     keep the text exactly as it is (default)
    mode="flatten": turn every run of whitespace (spaces, newlines, tabs)
                    into one space, so blank lines can't be used as a clue
    """
    if mode == "raw":
        return text
    if mode == "flatten":
        return re.sub(r"\s+", " ", text).strip()
    raise ValueError(f"unknown mode: {mode}")


SEED = 20261023
SPLIT_FILE = "splits/val_ids.txt"


def make_split(train):
    """Draw the fixed 2,000-row validation set once and save its ids."""
    rest, val = train_test_split(
        train,
        test_size=2000,
        stratify=train["label"],
        random_state=SEED,
    )
    os.makedirs("splits", exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        for i in sorted(val["id"]):
            f.write(f"{i}\n")


def get_split(train):
    """Split train into (training rows, validation rows) using the saved ids."""
    with open(SPLIT_FILE) as f:
        val_ids = [int(line) for line in f]
    is_val = train["id"].isin(val_ids)
    return train[~is_val], train[is_val]


def get_scores(model, texts):
    """P(AI) if the model has probabilities, otherwise the SVM-style score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def evaluate(model, train_part, val_part, mode="raw"):
    """Fit on the training rows, report train/val AUROC and fit time."""
    X_train = train_part["text"].map(lambda t: clean_text(t, mode))
    X_val = val_part["text"].map(lambda t: clean_text(t, mode))

    start = time.time()
    model.fit(X_train, train_part["label"])
    fit_seconds = time.time() - start

    train_auc = roc_auc_score(train_part["label"], get_scores(model, X_train))
    val_pred = get_scores(model, X_val)
    val_auc = roc_auc_score(val_part["label"], val_pred)

    print(f"train AUROC {train_auc:.4f} | val AUROC {val_auc:.4f} | "
          f"gap {train_auc - val_auc:.4f} | fit {fit_seconds:.1f}s")
    return {"train_auc": train_auc, "val_auc": val_auc,
            "fit_seconds": fit_seconds, "val_pred": val_pred}

if __name__ == "__main__":
    # quick sanity check: python common.py
    train, unlabeled, test = load_data()
    print("train", train.shape, "unlabeled", unlabeled.shape, "test", test.shape)

    example = "Ingredients:\n\n1 cup sugar\n\n2 eggs"
    print("raw:    ", repr(clean_text(example, "raw")))
    print("flatten:", repr(clean_text(example, "flatten")))

    flat = train["text"].map(lambda t: clean_text(t, "flatten"))
    print("passages with a blank line after flatten:", flat.str.contains("\n\n").sum())
