"""Combine several submission CSVs (id,label) into one submission.

Seeds of one config are already probability-averaged by experiments.distilbert_full. Across
different models the probabilities are calibrated differently, so the default here averages
rank percentiles instead (AUROC depends only on the ranking).

    python -m experiments.ensemble --out results/submissions/ens_deberta_distil.csv \
        results/submissions/<config A>_full_seeds42-43.csv results/submissions/<config B>_full_seeds42-43-44.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import check_against_sample
from src.data import DATA_DIR


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs="+", default=None)
    p.add_argument("--method", choices=("rank", "mean"), default="rank")
    a = p.parse_args(argv)

    weights = np.asarray(a.weights if a.weights else [1.0] * len(a.files), dtype=float)
    if len(weights) != len(a.files):
        raise SystemExit("need one weight per file")
    subs = [pd.read_csv(f).sort_values("id").reset_index(drop=True) for f in a.files]
    ids = subs[0]["id"]
    if any(not s["id"].equals(ids) for s in subs[1:]):
        raise SystemExit("files do not share the same ids")

    cols = [s["label"].rank(pct=True) if a.method == "rank" else s["label"] for s in subs]
    mat = np.column_stack(cols)
    combined = mat @ (weights / weights.sum())

    names = [f.stem[:40] for f in a.files]
    print(pd.DataFrame(mat, columns=names).corr(method="spearman").round(3).to_string())
    out = pd.DataFrame({"id": ids, "label": combined})
    check_against_sample(out, DATA_DIR / "sample_submission.csv")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows, method={a.method}, weights={weights.tolist()})")


if __name__ == "__main__":
    main()
