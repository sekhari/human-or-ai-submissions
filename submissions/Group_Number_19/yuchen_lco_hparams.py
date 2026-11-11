"""CatBoost depth / learning-rate check under leave-one-topic-cluster-out (LCO) CV, v4 features.

The fixed random split (yuchen_report.py, Q1.3) is flat across the grid and favoured
depth 4 / lr 0.1 by <0.0008 inner AUROC. This re-checks the candidates under the
protocol that tracks the test genre shift, with and without self-training
(pool = unlabeled.csv + held-out cluster texts, as in the final recipe). Labels of
train only; 1 seed per fit.

  python submissions/Group_Number_19/yuchen_lco_hparams.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yuchen_v2  # noqa: E402
from folds import SEED  # noqa: E402
from yuchen_semi import Data, fold_frames, self_train  # noqa: E402

ROOT = HERE.parents[1]
GRID = [(6, 0.05), (4, 0.1), (4, 0.05), (8, 0.05), (6, 0.1)]


def main() -> None:
    D = Data(ROOT / "data", ROOT / "results" / "yuchen" / "v2_shift", ROOT / "results" / "yuchen" / "lm_features",
             ROOT / "results" / "yuchen" / "lm_features" / "qwen35_2b")
    out = ROOT / "results" / "yuchen" / "lco_hparams"; out.mkdir(parents=True, exist_ok=True)
    base = dict(yuchen_v2.CAT)
    rows = []
    for pseudo in (False, True):
        for depth, lr in (GRID if not pseudo else GRID[:2]):
            yuchen_v2.CAT.update(base, depth=depth, learning_rate=lr)   # shared dict used by fit_models
            t = time.time(); oof = np.zeros(len(D.y))
            for k in np.unique(D.fold):
                tri, vai, Xtr, Xva, Xun = fold_frames(D, k)
                pool = pd.concat([Xun, Xva], ignore_index=True) if pseudo else None
                (oof[vai],), _ = self_train(Xtr, D.y[tri], [Xva], pool, [SEED])
            r = dict(depth=depth, learning_rate=lr, self_training=pseudo, **D.metrics(oof), seconds=round(time.time() - t))
            print(json.dumps(r), flush=True); rows.append(r)
    yuchen_v2.CAT.update(base)
    pd.DataFrame(rows).to_csv(out / "lco_hparams.csv", index=False)


if __name__ == "__main__":
    main()
