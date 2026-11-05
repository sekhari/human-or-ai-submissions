"""single feature auroc for each surprisal feature, overall, per domain, and how it looks on test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from src.harness import Ctx, REPORTS
from src.surprisal import NAMES

c = Ctx()
S = c.surprisal()
print("shape", S.shape, "nan", np.isnan(S).sum(), "all zero rows", int((S == 0).all(1).sum()))
y, n = c.y, c.n
rows = []
for j, name in enumerate(NAMES):
    a = auc(y, S[:n, j])
    per = [auc(y[c.dom[:n] == d], S[:n][c.dom[:n] == d, j]) for d in range(10)]
    rows.append(dict(feature=name, auc=a, best_dir=max(a, 1 - a), dom_min=min(max(p, 1 - p) for p in per),
                     big_domain=per[c.big], flips=int(len({p > .5 for p in per}) > 1)))
df = pd.DataFrame(rows).round(4).sort_values("best_dir", ascending=False)
df.to_csv(REPORTS / "11_surprisal_signal.csv", index=False)
print(df.to_string(index=False))
k = NAMES.index("discrepancy")
print("discrepancy by source: human", S[:n][y == 0, k].mean().round(2), "ai", S[:n][y == 1, k].mean().round(2),
      "test", S[c.te_i, k].mean().round(2), "test big domain", S[c.te_i][c.dom[c.te_i] == c.big, k].mean().round(2))
print("test big domain deciles", np.quantile(S[c.te_i][c.dom[c.te_i] == c.big, k], np.linspace(.1, .9, 9)).round(1).tolist())
print("train human deciles    ", np.quantile(S[:n][y == 0, k], np.linspace(.1, .9, 9)).round(1).tolist())
print("train ai deciles       ", np.quantile(S[:n][y == 1, k], np.linspace(.1, .9, 9)).round(1).tolist())
