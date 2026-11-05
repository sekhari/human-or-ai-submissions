"""grid search with 3 fold cv inside the 8000 training rows. the 2000 validation rows stay out of selection."""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

from src.baseline import fit_style
from src.harness import Ctx, REPORTS
from src.models import fit_clean_lexical

c = Ctx()
f, y, texts = c.fit_i, c.y, c.texts
folds = list(StratifiedKFold(3, shuffle=True, random_state=3737).split(f, y[f]))


def cv(fit):
    return float(np.mean([auc(y[f[b]], fit(list(texts[f[a]]), y[f[a]])(list(texts[f[b]]))) for a, b in folds]))


rows = []
for C, wmax in product((.25, 1, 4, 16, 64), (40000, 120000)):
    s = cv(lambda t, l: fit_clean_lexical(t, l, c=C, word_max=wmax, char_max=wmax * 2))
    rows.append(dict(model="clean word+char lr", params=f"C={C} word_max={wmax}", cv_auc=s))
    print(rows[-1], flush=True)
for it, depth in product((300, 850), (4, 6, 8)):
    s = cv(lambda t, l: fit_style(t, l, iterations=it, depth=depth))
    rows.append(dict(model="reference style catboost", params=f"iterations={it} depth={depth}", cv_auc=s))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows).round(5)
df.to_csv(REPORTS / "07_hyperparams.csv", index=False)
print(df.sort_values("cv_auc", ascending=False).groupby("model").head(1))
