"""Line-break stress test: how much does a trained model rely on the layout of a passage?

Each validation passage is re-laid-out without changing its words, then rescored:
    none          original text
    no_newlines   every line break replaced by a space (one paragraph)
    unwrap        single line breaks joined, blank-line paragraph breaks kept (undoes hard wrapping)
    hard_wrap     every paragraph re-wrapped at 70 characters (typical hard-wrapped human text)
    short_lines   all words re-broken into lines of 4-9 words (poem-like layout)
If AUROC drops or one class moves toward the other under a layout change, the model is using layout as a
shortcut, and passages laid out unlike the training data (e.g. poems in test.csv) are at risk.

    python -m experiments.format_stress --checkpoints checkpoints/deberta/<run> [<run> ...]

Output: results/format_stress/<first checkpoint>.json, plus a table on stdout (checkpoint probabilities averaged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.classifiers.distilbert import DistilBertClassifier
from src.data import RESULTS_DIR, load_split
from src.layout import hard_wrap, no_newlines, short_lines, unwrap

STRESS_DIR = RESULTS_DIR / "format_stress"


def variants(texts: list[str], seed: int = 42) -> dict[str, list[str]]:
    return {
        "none": list(texts),
        "no_newlines": [no_newlines(t) for t in texts],
        "unwrap": [unwrap(t) for t in texts],
        "hard_wrap": [hard_wrap(t) for t in texts],
        "short_lines": [short_lines(t, np.random.default_rng([seed, i])) for i, t in enumerate(texts)],
    }


def summarize(y: np.ndarray, p: np.ndarray, changed: np.ndarray) -> dict:
    return {"auroc": round(float(roc_auc_score(y, p)), 5), "errors": int(((p > 0.5) != (y == 1)).sum()),
            "human_called_ai": int(((p > 0.5) & (y == 0)).sum()), "ai_called_human": int(((p <= 0.5) & (y == 1)).sum()),
            "mean_p_human": round(float(p[y == 0].mean()), 4), "mean_p_ai": round(float(p[y == 1].mean()), 4),
            "n_changed": int(changed.sum())}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N validation passages")
    a = p.parse_args(argv)

    split = load_split()
    texts, y = split.val_texts[: a.limit], split.val_labels[: a.limit]
    nl = np.array([t.count("\n") for t in texts])
    for c, name in ((0, "human"), (1, "AI")):
        print(f"{name}: {np.mean(nl[y == c] > 0):.1%} of passages have line breaks, median {np.median(nl[y == c]):.0f}")
    var = variants(texts)
    probs = {k: np.zeros(len(texts)) for k in var}
    for ckpt in a.checkpoints:
        clf = DistilBertClassifier.load(ckpt, device=a.device)
        for k, v in var.items():
            print(f"scoring {k} with {ckpt.name}", flush=True)
            probs[k] += clf.predict_proba(v) / len(a.checkpoints)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = {k: summarize(y, probs[k], np.array([s != t for s, t in zip(var[k], texts)])) for k in var}
    print(pd.DataFrame(rows).T.to_string())
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    if a.limit is None:
        out = STRESS_DIR / f"{a.checkpoints[0].name}.json"
        out.write_text(json.dumps({"checkpoints": [str(c) for c in a.checkpoints], "results": rows}, indent=2) + "\n")
        pd.DataFrame({"id": split.val_ids, "label": y, **{f"p_{k}": v for k, v in probs.items()}}).to_csv(
            out.with_suffix(".csv"), index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
