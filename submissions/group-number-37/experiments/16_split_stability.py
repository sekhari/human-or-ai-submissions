"""is the blend gain stable? five other stratified 8000/2000 splits and three catboost seeds on the fixed one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import train_test_split

from src.data import fixed_split, load
from src.harness import REPORTS
from src.pipeline import Blend
from src.surprisal_cache import get_surprisal

train = load("train")
t, y = train.text.values, train.label.values
S = get_surprisal(t)
r = lambda x: rankdata(x) / len(x)
runs = [("split", s, *train_test_split(np.arange(len(t)), test_size=2000, stratify=y, random_state=s)) for s in (1, 2, 3, 4, 5)]
f, v = fixed_split(train)
runs += [("model seed", s, f, v) for s in (11, 22, 33)]
rows = []
for kind, s, a, b in runs:
    p = Blend(seed=3737 if kind == "split" else s).fit(list(t[a]), y[a], S[a]).parts(list(t[b]), S[b])
    row = dict(kind=kind, seed=s, reference=auc(y[b], p["reference"]), surprisal=auc(y[b], p["surprisal"]))
    for w in (.1, .2, .3, .4):
        row[f"blend_{w}"] = auc(y[b], (1 - w) * r(p["reference"]) + w * r(p["surprisal"]))
    rows.append(row)
    print({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in row.items()}, flush=True)
df = pd.DataFrame(rows).round(5)
df.to_csv(REPORTS / "16_split_stability.csv", index=False)
g = df.drop(columns=["kind", "seed"])
print(pd.DataFrame({"mean": g.mean(), "std": g.std(), "wins vs reference": [(g[c] > g.reference).sum() for c in g]}).round(5))
