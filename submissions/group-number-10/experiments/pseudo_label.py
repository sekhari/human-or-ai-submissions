"""Pseudo-label unlabeled.csv with saved checkpoints for self-training.

Each checkpoint scores all unlabeled passages; probabilities are averaged and only confident
passages (mean P(AI) <= threshold or >= 1 - threshold) are kept with their hard label.
Use split-trained checkpoints for validation experiments (they never trained on validation
passages) and full-data checkpoints for final models. test.csv is never used.

    python -m experiments.pseudo_label --name large8k \
        --checkpoints checkpoints/deberta/deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42

Outputs:
    results/pseudo/<name>_scores.csv   id, prob per checkpoint, mean prob (all unlabeled passages)
    results/pseudo/<name>.csv          id, text, label for the confident passages
                                       (pass to --extra-train of distilbert_train / distilbert_full)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.classifiers.distilbert import predict_proba_from_checkpoint
from src.data import DATA_DIR, RESULTS_DIR, load_unlabeled

PSEUDO_DIR = RESULTS_DIR / "pseudo"


def select_confident(probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean mask of confident passages and their hard labels (1 = AI)."""
    keep = (probs <= threshold) | (probs >= 1 - threshold)
    return keep, (probs >= 0.5).astype(int)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--name", required=True, help="output file stem, e.g. large8k")
    p.add_argument("--threshold", type=float, default=0.02, help="keep mean P(AI) <= t or >= 1 - t")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N unlabeled passages")
    a = p.parse_args(argv)

    ids, texts = load_unlabeled(DATA_DIR / "unlabeled.csv")
    ids, texts = ids[: a.limit], texts[: a.limit]
    scores = pd.DataFrame({"id": ids})
    for ckpt in a.checkpoints:
        print(f"scoring {len(texts)} unlabeled passages with {ckpt}", flush=True)
        scores[ckpt.name] = predict_proba_from_checkpoint(texts, ckpt, device=a.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probs = scores[[c.name for c in a.checkpoints]].mean(axis=1).to_numpy()
    scores["prob"] = probs

    keep, labels = select_confident(probs, a.threshold)
    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(PSEUDO_DIR / f"{a.name}_scores.csv", index=False)
    pseudo = pd.DataFrame({"id": ids[keep], "text": np.asarray(texts, dtype=object)[keep], "label": labels[keep]})
    pseudo.to_csv(PSEUDO_DIR / f"{a.name}.csv", index=False)
    summary = {"name": a.name, "checkpoints": [str(c) for c in a.checkpoints], "threshold": a.threshold,
               "n_unlabeled": len(texts), "n_kept": int(keep.sum()), "n_ai": int(labels[keep].sum()),
               "n_human": int((1 - labels[keep]).sum())}
    (PSEUDO_DIR / f"{a.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
