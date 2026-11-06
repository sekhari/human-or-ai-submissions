"""train the final blend. --validate fits on the fixed 8000 rows and reports the 2000 validation rows instead."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.data import fixed_split
from src.pipeline import Blend
from src.surprisal_cache import get_surprisal


def main() -> None:
    p = argparse.ArgumentParser(description="train human or ai detector")
    p.add_argument("--train", type=Path, required=True, help="path to train.csv")
    p.add_argument("--seed", type=int, default=3737)
    p.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    train = pd.read_csv(a.train, keep_default_na=False)
    texts, y = train.text.values, train.label.values
    S = get_surprisal(texts)
    if a.validate:
        fit, val = fixed_split(train)
        m = Blend(seed=a.seed).fit(list(texts[fit]), y[fit], S[fit])
        print("train auroc", round(auc(y[fit], m.score(list(texts[fit]), S[fit])), 4))
        print("validation auroc", round(auc(y[val], m.score(list(texts[val]), S[val])), 4))
        parts = m.parts(list(texts[val]), S[val])
        print({k: round(auc(y[val], v), 4) for k, v in parts.items()})
    m = Blend(seed=a.seed).fit(list(texts), y, S)
    joblib.dump(m, a.model_out)
    print("wrote", a.model_out)


if __name__ == "__main__":
    main()
