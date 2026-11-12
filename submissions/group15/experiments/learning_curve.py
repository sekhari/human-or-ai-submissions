"""Q2: effect of training data size, using Model A (fixed hyperparameters)."""

import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common import load_data, get_split, evaluate, SEED
from model_a import build_model

SIZES = [500, 1000, 2000, 5000, 8000]


def run(train_part, val_part):
    results = []
    for n in SIZES:
        if n < len(train_part):
            subset, _ = train_test_split(train_part, train_size=n,
                                         stratify=train_part["label"],
                                         random_state=SEED)
        else:
            subset = train_part
        print(f"--- {n} training rows ---")
        r = evaluate(build_model(), subset, val_part)
        results.append({"train_rows": n,
                        "train_auc": round(r["train_auc"], 4),
                        "val_auc": round(r["val_auc"], 4)})
    return pd.DataFrame(results)


def plot(table):
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(table["train_rows"], table["train_auc"], marker="o", label="Training AUROC")
    plt.plot(table["train_rows"], table["val_auc"], marker="o", label="Validation AUROC")
    plt.xscale("log")
    plt.xticks(SIZES, [str(n) for n in SIZES])
    plt.minorticks_off()
    plt.xlabel("Number of training rows (log scale)")
    plt.ylabel("AUROC")
    plt.title("Learning curve: Model A (word TF-IDF + LR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/learning_curve.png", dpi=200)


if __name__ == "__main__":
    train, unlabeled, test = load_data()
    train_part, val_part = get_split(train)
    table = run(train_part, val_part)
    print(table.to_string(index=False))
    with open("experiments/learning_curve.txt", "w") as f:
        f.write(table.to_string(index=False))
    plot(table)
    print("saved experiments/learning_curve.txt and figures/learning_curve.png")