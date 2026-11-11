"""write the kaggle submission from a trained blend."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
import pandas as pd

from src.surprisal_cache import get_surprisal


def main() -> None:
    p = argparse.ArgumentParser(description="write kaggle submission.csv")
    p.add_argument("--test", type=Path, required=True, help="path to test.csv")
    p.add_argument("--model", type=Path, default=Path("model.joblib"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    a = p.parse_args()
    test = pd.read_csv(a.test, keep_default_na=False)
    m = joblib.load(a.model)
    texts = list(test.text.values)
    score = m.score(texts, get_surprisal(texts))
    sub = pd.DataFrame({"id": test.id, "label": score})
    assert len(sub) == 3000 and sub.id.is_unique and np.isfinite(sub.label).all() and sub.label.between(0, 1).all()
    sub.to_csv(a.out, index=False)
    print("wrote", a.out, len(sub))


if __name__ == "__main__":
    main()
