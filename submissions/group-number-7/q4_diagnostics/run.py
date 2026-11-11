"""Q4: reproduce length-bin AUROC, overfit gap, and selected FP/FN IDs.

Only local Model 2 validation predictions (including true validation labels)
are read. CSV output is local-only and MUST NOT be committed to GitHub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BINS = [(-np.inf, 100, "<100 words"), (100, 200, "100-199 words"),
        (200, 300, "200-299 words"), (300, np.inf, ">=300 words")]


def load(path: Path, *, need_word_count: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    needed = {"id", "label", "probability"}
    if need_word_count:
        needed.add("word_count")
    if not needed.issubset(df.columns) or df.id.duplicated().any():
        raise ValueError(f"Missing columns or duplicate IDs in {path}")
    if not df.label.isin([0, 1]).all() or not df.probability.between(0, 1).all():
        raise ValueError(f"Invalid labels or probabilities in {path}")
    if need_word_count and (df.word_count.isna().any() or (df.word_count < 0).any()):
        raise ValueError("Invalid word counts")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train", type=Path, help="Optional training predictions for gap")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"Use a new output directory: {args.outdir}")
    val = load(args.validation, need_word_count=True)
    if len(val) != 2000 or val.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Expected fixed 2,000-row balanced validation split")
    args.outdir.mkdir(parents=True)
    overall = float(roc_auc_score(val.label, val.probability))
    summary = {"validation_auroc": overall, "validation_rows": len(val),
               "classification_threshold": 0.5}
    if args.train is not None:
        train = load(args.train, need_word_count=False)
        if len(train) != 8000 or set(train.id) & set(val.id):
            raise ValueError("Expected nonoverlapping 8,000 training and 2,000 validation IDs")
        train_auc = float(roc_auc_score(train.label, train.probability))
        summary.update(train_auroc=train_auc, train_validation_gap=train_auc - overall)
    rows = []
    for lo, hi, name in BINS:
        part = val[(val.word_count >= lo) & (val.word_count < hi)]
        if len(part) == 0:
            raise ValueError(f"Empty text-length bin: {name}")
        n_human = int((part.label == 0).sum())
        n_ai = int((part.label == 1).sum())
        auc = float(roc_auc_score(part.label, part.probability)) if n_human and n_ai else None
        fp = int(((part.label == 0) & (part.probability >= 0.5)).sum())
        fn = int(((part.label == 1) & (part.probability < 0.5)).sum())
        rows.append({"bin": name, "count": len(part), "human_rows": n_human,
                     "ai_rows": n_ai, "auroc": auc, "false_positive_rate": fp / n_human if n_human else None,
                     "false_negative_rate": fn / n_ai if n_ai else None})
    pd.DataFrame(rows).to_csv(args.outdir / "length_bins.csv", index=False)
    fp = val[(val.label == 0) & (val.probability >= 0.5)].nlargest(5, "probability")
    fn = val[(val.label == 1) & (val.probability < 0.5)].nsmallest(5, "probability")
    if len(fp) < 5 or len(fn) < 5:
        raise ValueError("Fewer than 5 FP or FN; cannot reproduce the report examples")
    errors = pd.concat([fp.assign(error_type="FP"), fn.assign(error_type="FN")])
    errors[["id", "error_type", "probability", "word_count"]].to_csv(
        args.outdir / "error_cases_LOCAL_ONLY.csv", index=False)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(rows).round(6).to_string(index=False))
    print("\nTop confidence errors:\n", errors[["id", "error_type", "probability"]].to_string(index=False))
    print("\nSaved local aggregate and row-level diagnostics to", args.outdir)


if __name__ == "__main__":
    main()
