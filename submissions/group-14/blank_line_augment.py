"""Blank-line augmentation: does the GBDT still work when the layout cue is removed? (report Q5)

Baseline     the train_gbdt.py model: 8,000 training rows, results_gbdt.json hyperparameters.
Improvement  the same model, features and hyperparameters, trained on the same 8,000 rows plus one
             extra copy of every training passage that contains a blank line, with its blank lines
             collapsed to single line breaks (label unchanged).

Both are scored on three views of the fixed 2,000-row validation set:
  standard   the validation passages as given
  stripped   every validation passage with its blank lines collapsed (a synthetic stress test)
  hard slice AI passages that have no blank line vs. all human passages (unmodified text)

Example:
    python blank_line_augment.py --train data/train.csv --unlabeled data/unlabeled.csv \
        --test data/test.csv --submission submission_gbdt_aug.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from split import SEED, load_split
from train_gbdt import featurize, fit_lsa, make_gbdt

PARAMS = json.loads(Path(__file__).with_name("results_gbdt.json").read_text())["params"]
_BLANK_RUN_RE = re.compile(r"\n(?:[^\S\n]*\n)+")


def strip_blank_lines(text: str) -> str:
    """Collapse each run of blank lines to a single line break; nothing else changes."""
    return _BLANK_RUN_RE.sub("\n", text)


def augment(texts: pd.Series, labels: np.ndarray, lsa) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and labels for the stripped copies of the passages that contain a blank line."""
    stripped = texts.map(strip_blank_lines)
    changed = (stripped != texts).to_numpy()
    return featurize(stripped[changed], lsa), labels[changed]


def main() -> None:
    parser = argparse.ArgumentParser(description="GBDT with and without blank-line augmentation")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--unlabeled", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--submission", type=Path, default=Path("submission_gbdt_aug.csv"))
    parser.add_argument("--repeats", type=int, default=5, help="seeds to evaluate; 0 skips evaluation")
    parser.add_argument("--results", type=Path, default=Path("results_blank_line_augment.json"))
    args = parser.parse_args()

    train_all = pd.read_csv(args.train)
    unlabeled_texts = pd.read_csv(args.unlabeled)["text"].tolist() if args.unlabeled else []
    if args.repeats:
        evaluate(train_all, unlabeled_texts, args)
    if args.test:
        write_submission(train_all, unlabeled_texts, args)


def evaluate(train_all: pd.DataFrame, unlabeled_texts: list[str], args: argparse.Namespace) -> None:
    tr, va = load_split(train_all)
    y_tr, y_va = tr["label"].to_numpy(), va["label"].to_numpy()

    lsa = fit_lsa(tr["text"].tolist() + unlabeled_texts)
    F_tr, F_va = featurize(tr["text"], lsa), featurize(va["text"], lsa)
    F_va_stripped = featurize(va["text"].map(strip_blank_lines), lsa)
    F_aug, y_aug = augment(tr["text"], y_tr, lsa)
    print(f"features built; {len(F_aug):,} stripped copies added ({int(y_aug.sum()):,} AI)", flush=True)

    hard = (y_va == 0) | (F_va["n_blank_lines"].to_numpy() == 0)  # all human + AI without a blank line
    train_sets = {
        "baseline": (F_tr, y_tr),
        "augmented": (pd.concat([F_tr, F_aug], ignore_index=True), np.concatenate([y_tr, y_aug])),
    }

    runs = {name: [] for name in train_sets}
    for seed in range(SEED, SEED + args.repeats):
        for name, (F, y) in train_sets.items():
            start = time.time()
            model = make_gbdt(**PARAMS).set_params(random_state=seed).fit(F, y)
            fit_s = time.time() - start
            p, p_stripped = model.predict_proba(F_va)[:, 1], model.predict_proba(F_va_stripped)[:, 1]
            runs[name].append({
                "val_auroc": roc_auc_score(y_va, p),
                "val_auroc_stripped": roc_auc_score(y_va, p_stripped),
                "val_auroc_hard_slice": roc_auc_score(y_va[hard], p[hard]),
                "false_negatives_at_0.5": int(((p < 0.5) & (y_va == 1)).sum()),
                "false_positives_at_0.5": int(((p >= 0.5) & (y_va == 0)).sum()),
                "n_trees": int(model.n_iter_),
                "fit_s": fit_s,
            })
            r = runs[name][-1]
            print(f"seed {seed} {name:>9}: val {r['val_auroc']:.4f} | stripped {r['val_auroc_stripped']:.4f} | "
                  f"hard slice {r['val_auroc_hard_slice']:.4f} | FN {r['false_negatives_at_0.5']} "
                  f"FP {r['false_positives_at_0.5']} | {r['n_trees']} trees, {fit_s:.1f}s", flush=True)

    summary = {name: {k: {"mean": round(float(np.mean([r[k] for r in rs])), 6),
                          "std": round(float(np.std([r[k] for r in rs])), 6)} for k in rs[0]}
               for name, rs in runs.items()}
    results = {"params": PARAMS, "repeats": args.repeats, "train_rows": {n: len(y) for n, (_, y) in train_sets.items()},
               "hard_slice_rows": int(hard.sum()), "hard_slice_ai_rows": int((hard & (y_va == 1)).sum()),
               "summary": summary}
    args.results.write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=2))


def write_submission(train_all: pd.DataFrame, unlabeled_texts: list[str], args: argparse.Namespace) -> None:
    test = pd.read_csv(args.test)
    y_all = train_all["label"].to_numpy()
    lsa = fit_lsa(train_all["text"].tolist() + unlabeled_texts)
    F_extra, y_extra = augment(train_all["text"], y_all, lsa)
    F_all = pd.concat([featurize(train_all["text"], lsa), F_extra], ignore_index=True)
    final = make_gbdt(**PARAMS).fit(F_all, np.concatenate([y_all, y_extra]))
    scores = final.predict_proba(featurize(test["text"], lsa))[:, 1]
    pd.DataFrame({"id": test["id"], "label": scores}).to_csv(args.submission, index=False)
    print(f"wrote {args.submission} (refit on all {len(train_all):,} labeled rows + {len(y_extra):,} stripped copies)")


if __name__ == "__main__":
    main()
