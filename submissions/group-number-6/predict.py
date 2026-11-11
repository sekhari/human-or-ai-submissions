"""Write the Kaggle submission: one P(AI) in [0, 1] per test id.

Example
  python predict.py --test data/test.csv --model models/final.joblib --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="Write Kaggle submission.csv")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.csv")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "final.joblib")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    args = ap.parse_args()

    test = pd.read_csv(args.test)
    model = joblib.load(args.model)
    p = model.predict_proba(test["text"].astype(str).tolist())[:, 1]
    assert np.all((p >= 0) & (p <= 1)) and len(p) == len(test)
    pd.DataFrame({"id": test["id"], "label": p}).to_csv(args.out, index=False)
    print(f"wrote {len(p)} predictions to {args.out} (mean P(AI)={p.mean():.3f})")


if __name__ == "__main__":
    main()
