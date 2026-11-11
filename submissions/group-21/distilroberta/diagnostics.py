"""Inspect the final blend on the 2,000 original held-out passages."""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def read_rows(path, originals_only=False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            if originals_only and row["variant"] != "original":
                continue
            passage_id = int(row["id"])
            if passage_id in rows:
                raise ValueError(f"Duplicate ID {passage_id} in {path.name}")
            rows[passage_id] = row
        return rows


def scaled(values):
    values = np.array(values, dtype=float)
    if not np.isfinite(values).all() or values.std() == 0:
        raise ValueError("Scores must be finite and nonconstant.")
    return (values - values.mean()) / values.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--v2-probes", type=Path, required=True)
    parser.add_argument("--transformer-probes", type=Path, required=True)
    args = parser.parse_args()
    source = read_rows(args.train)
    v2 = read_rows(args.v2_probes, originals_only=True)
    transformer = read_rows(args.transformer_probes, originals_only=True)
    ids = sorted(v2)
    if len(ids) != 2000 or set(ids) != set(transformer) or not set(ids) <= source.keys():
        raise ValueError("Expected the same 2,000 validation IDs from train.csv.")

    for i in ids:
        for field in ("label", "text"):
            if not source[i][field] == v2[i][field] == transformer[i][field]:
                raise ValueError(f"{field} mismatch on ID {i}")

    labels = np.array([int(source[i]["label"]) for i in ids])
    if not np.array_equal(np.bincount(labels), [1000, 1000]):
        raise ValueError("Expected 1,000 examples of each class.")
    scores = 0.75 * scaled([v2[i]["v2_score"] for i in ids])
    scores += 0.25 * scaled([transformer[i]["margin"] for i in ids])
    probs = expit(scores)
    predictions = probs >= 0.5
    words = np.array([len(source[i]["text"].split()) for i in ids])
    cuts = np.quantile(words, [1 / 3, 2 / 3])
    bins = np.searchsorted(cuts, words, side="left")
    print("group words n human ai auc false_positive false_negative")
    for name, mask in [("all", np.ones(len(ids), dtype=bool))] + [
        (name, bins == j) for j, name in enumerate(["short", "medium", "long"])
    ]:
        y, p = labels[mask], predictions[mask]
        auc = roc_auc_score(y, scores[mask])
        print(name, f"{words[mask].min()}-{words[mask].max()}", len(y),
              sum(y == 0), sum(y == 1), f"{auc:.6f}",
              sum((y == 0) & p), sum((y == 1) & ~p))
    print("error id words blend_probability")
    for name, mask in [("FP", (labels == 0) & predictions),
                       ("FN", (labels == 1) & ~predictions)]:
        cases = np.flatnonzero(mask)
        order = np.argsort(scores[cases] if name == "FN" else -scores[cases])
        for j in cases[order[:5]]:
            print(name, ids[j], words[j], f"{probs[j]:.6f}")


if __name__ == "__main__":
    main()
