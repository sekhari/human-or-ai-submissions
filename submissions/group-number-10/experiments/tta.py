"""Test-time augmentation over line layouts: score each test passage as-is and re-laid-out, then average.

The words never change, only the line breaks (src.layout). Prediction only: nothing is trained on test.csv.
Models trained with --layout-aug should give similar scores under every layout; averaging smooths out
passages where one layout still pushes the score the wrong way.

Validation check (aug 0.5 model, seed 42, results/format_stress/*.csv): none 0.99942, none + unwrap 0.99945,
adding hard_wrap / no_newlines / short_lines lowers AUROC. So the default is none + unwrap.

    python -m experiments.tta --checkpoints checkpoints/full/<run_seed42> ... --name <output name>

Output:
    results/tta/<name>_parts.csv        id, then one probability column per checkpoint x layout
    results/submissions/<name>.csv      id,label: mean over layouts of the mean over checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.classifiers.distilbert import DistilBertClassifier
from src.data import DATA_DIR, RESULTS_DIR, load_unlabeled
from src.layout import hard_wrap, no_newlines, unwrap

LAYOUTS = {"none": lambda t: t, "unwrap": unwrap, "hard_wrap": hard_wrap, "no_newlines": no_newlines}
TTA_DIR = RESULTS_DIR / "tta"


def apply_layout(texts: list[str], layout: str) -> list[str]:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
    return [LAYOUTS[layout](t) for t in texts]


def tta_average(parts: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    """parts[(checkpoint, layout)] -> probabilities. Mean over checkpoints per layout, then over layouts."""
    layouts = sorted({lay for _, lay in parts})
    return np.mean([np.mean([p for (_, lay), p in parts.items() if lay == layout], axis=0) for layout in layouts], axis=0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--layouts", nargs="+", default=["none", "unwrap"], choices=sorted(LAYOUTS))
    p.add_argument("--name", required=True, help="output file name (no extension)")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N test passages, nothing written")
    a = p.parse_args(argv)

    ids, texts = load_unlabeled(DATA_DIR / "test.csv")
    ids, texts = ids[: a.limit], list(texts[: a.limit])
    variants = {lay: apply_layout(texts, lay) for lay in a.layouts}
    for lay, v in variants.items():
        print(f"{lay}: {sum(s != t for s, t in zip(v, texts))} of {len(texts)} passages changed", flush=True)

    parts: dict[tuple[str, str], np.ndarray] = {}
    for ckpt in a.checkpoints:
        clf = DistilBertClassifier.load(ckpt, device=a.device)
        for lay, v in variants.items():
            print(f"scoring {lay} with {ckpt.name}", flush=True)
            parts[(ckpt.name, lay)] = clf.predict_proba(v)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final = tta_average(parts)
    if a.limit is not None:
        print(f"smoke test ok: {len(final)} predictions, range {final.min():.4f}-{final.max():.4f}")
        return
    TTA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, **{f"{c}|{lay}": v for (c, lay), v in parts.items()}}).to_csv(
        TTA_DIR / f"{a.name}_parts.csv", index=False)
    out = RESULTS_DIR / "submissions" / f"{a.name}.csv"
    pd.DataFrame({"id": ids, "label": final}).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
