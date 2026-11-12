"""Robustness to adversarial edits (the test set contains modified AI passages).

Both main models are trained once on the 8,000-row training fold, then scored on
perturbed copies of the fixed 2,000-row validation set:

  typos_5 / typos_15   random char swap/delete/insert/replace in 5% / 15% of words (AI passages only)
  homoglyphs           ~10% of a/e/o/c/p/x letters swapped for Cyrillic look-alikes (AI only)
  word_drop_10         10% of words deleted (AI only)
  no_newlines          every newline replaced by a space (ALL passages) -> breaks the formatting shortcut

    python src/robustness.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from models import style_gbm, tfidf_logreg

SEED = 42
LETTERS = "abcdefghijklmnopqrstuvwxyz"
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х"}


def typo_word(w, rng):
    if len(w) < 3:
        return w
    i = rng.integers(1, len(w) - 1)
    op = rng.integers(4)
    if op == 0:   # swap adjacent
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 1:   # delete
        return w[:i] + w[i + 1:]
    if op == 2:   # insert
        return w[:i] + LETTERS[rng.integers(26)] + w[i:]
    return w[:i] + LETTERS[rng.integers(26)] + w[i + 1:]  # replace


def typos(text, rate, rng):
    # split on spaces only, so newlines/paragraph structure are kept
    return " ".join(typo_word(w, rng) if rng.random() < rate else w for w in text.split(" "))


def homoglyphs(text, rate, rng):
    return "".join(HOMOGLYPHS[ch] if ch in HOMOGLYPHS and rng.random() < rate else ch for ch in text)


def word_drop(text, rate, rng):
    return " ".join(w for w in text.split(" ") if rng.random() >= rate)


def no_newlines(text, rng=None):
    return re.sub(r"\s*\n\s*", " ", text)


ATTACKS = {
    "clean": (None, False),
    "typos_5": (lambda t, r: typos(t, 0.05, r), True),
    "typos_15": (lambda t, r: typos(t, 0.15, r), True),
    "homoglyphs": (lambda t, r: homoglyphs(t, 0.10, r), True),
    "word_drop_10": (lambda t, r: word_drop(t, 0.10, r), True),
    "no_newlines": (no_newlines, False),  # applied to every passage
}


def perturb(va, fn, ai_only):
    rng = np.random.default_rng(SEED)
    texts = va["text"].tolist()
    labels = va["label"].tolist()
    return [fn(t, rng) if (not ai_only or y == 1) else t for t, y in zip(texts, labels)]


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")

    models = {"tfidf_logreg": tfidf_logreg(), "style_gbm": style_gbm()}
    for m in models.values():
        m.fit(tr["text"], tr["label"])

    rows = []
    for attack, (fn, ai_only) in ATTACKS.items():
        texts = va["text"].tolist() if fn is None else perturb(va, fn, ai_only)
        for name, m in models.items():
            auc = roc_auc_score(va["label"], m.predict_proba(texts)[:, 1])
            rows.append({"attack": attack, "model": name, "val_auroc": round(auc, 4)})

    df = pd.DataFrame(rows).pivot(index="attack", columns="model", values="val_auroc")
    df = df.loc[list(ATTACKS)]
    print("Validation AUROC under each perturbation:\n")
    print(df.to_string())
    df.to_csv(ROOT / "results" / "robustness.csv")

    ax = df.plot.bar(figsize=(7, 4), rot=20)
    ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Robustness to edits (AI passages perturbed; no_newlines = all)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "robustness.png", dpi=200)
    print("\nSaved results/robustness.csv and results/robustness.png")


if __name__ == "__main__":
    main()
