"""Write a Kaggle submission (id,label = P(AI)) from a saved DistilBERT checkpoint.

Examples (run from submissions/group-number-10/):
    python predict.py                                                  # default checkpoint -> submission.csv
    python predict.py --checkpoint checkpoints/distilbert/<run> --out results/submissions/<run>.csv

No retraining: the checkpoint is loaded as saved by experiments/distilbert_train.py.
Rows are written in test.csv order and checked against data/sample_submission.csv when it exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifiers.distilbert import DistilBertClassifier
from src.data import DATA_DIR, TEAM_ROOT, load_unlabeled, read_csv

DEFAULT_CHECKPOINT = TEAM_ROOT / "checkpoints" / "distilbert" / "distilbert_nl_lr3e-05_ep4_bs16_len512_seed42"


def check_against_sample(sub: pd.DataFrame, sample_path: Path) -> None:
    """Fail loudly if IDs or columns differ from the sample submission."""
    sample = read_csv(sample_path)
    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
    if len(sub) != len(sample) or set(sub.iloc[:, 0]) != set(sample.iloc[:, 0]):
        raise ValueError("submission IDs do not match sample_submission.csv")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--test", type=Path, default=DATA_DIR / "test.csv")
    p.add_argument("--sample", type=Path, default=DATA_DIR / "sample_submission.csv")
    p.add_argument("--out", type=Path, default=TEAM_ROOT / "submission.csv")
    p.add_argument("--device", default=None, help="force cuda / mps / cpu (default: auto)")
    a = p.parse_args(argv)

    ids, texts = load_unlabeled(a.test)
    clf = DistilBertClassifier.load(a.checkpoint, device=a.device)
    print(f"checkpoint={a.checkpoint.name} best_epoch={clf.best_epoch} device={clf.device} n_test={len(texts)}",
          flush=True)
    probs = clf.predict_proba(texts)
    if not np.all(np.isfinite(probs)) or probs.min() < 0 or probs.max() > 1:
        raise ValueError("predicted probabilities are not all finite values in [0, 1]")

    sub = pd.DataFrame({"id": ids, "label": probs})
    if a.sample.exists():
        check_against_sample(sub, a.sample)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows); P(AI) mean={probs.mean():.3f}, "
          f"share > 0.5 = {(probs > 0.5).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
