"""is the surprisal blend gain real? picks the weight without the scored domain, then paired bootstrap inside each domain."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.data import OUT, fixed_split, load
from src.harness import REPORTS

y = load("train").label.values
st = dict(np.load(OUT / "lodo_scores.npz"))
domains = sorted(int(k[4:]) for k in st if k.startswith("idx_"))
W = (0, .05, .1, .15, .2, .3, .4, .5, .7)
r = lambda x: rankdata(x) / len(x)
blend = lambda d, w: (1 - w) * r(st[f"ref_{d}"]) + w * r(st[f"sur_{d}"])
score = {(d, w): auc(y[st[f"idx_{d}"]], blend(d, w)) for d in domains for w in W}

rows = []
for d in domains:
    pick = max(W, key=lambda w: np.mean([score[(o, w)] for o in domains if o != d]))
    rows.append(dict(domain=d, n=len(st[f"idx_{d}"]), weight_from_other_domains=pick, reference=score[(d, 0)], blend=score[(d, pick)]))
t = pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
print("nested mean", round(t.blend.mean(), 4), "worst", round(t.blend.min(), 4), "reference mean", round(t.reference.mean(), 4), "worst", round(t.reference.min(), 4))

rng = np.random.default_rng(3737)
diffs = {w: [] for w in (.2, .3, .4)}
for _ in range(500):
    gain = {w: [] for w in diffs}
    for d in domains:
        lab = y[st[f"idx_{d}"]]
        pick = rng.integers(0, len(lab), len(lab))
        if lab[pick].min() == lab[pick].max():
            continue
        base = auc(lab[pick], r(st[f"ref_{d}"])[pick])
        for w in diffs:
            gain[w].append(auc(lab[pick], blend(d, w)[pick]) - base)
    for w in diffs:
        diffs[w].append(np.mean(gain[w]))
ci = pd.DataFrame([dict(weight=w, mean_gain=np.mean(v), low=np.quantile(v, .025), high=np.quantile(v, .975)) for w, v in diffs.items()]).round(5)
print(ci.to_string(index=False))
t.round(4).to_csv(REPORTS / "15_nested_weight.csv", index=False)
ci.to_csv(REPORTS / "15_bootstrap_gain.csv", index=False)
