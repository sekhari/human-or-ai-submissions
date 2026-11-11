"""pick three way weights honestly. each domain is scored with weights chosen on the other nine.

also scores every weight with domains reweighted to the test mix, reads the fixed validation set for the cost on
train-like text, and bootstraps the gain of the leading blends against the reference and the two way blend.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.data import OUT
from src.harness import Ctx, REPORTS

c = Ctx()
y = c.y
ls = dict(np.load(OUT / "lodo_scores.npz"))
en = dict(np.load(OUT / "lodo_encoder.npz"))
domains = sorted(int(k[4:]) for k in en)
r = lambda x: rankdata(x) / len(x)
parts = {d: (r(ls[f"ref_{d}"]), r(ls[f"sur_{d}"]), r(en[f"enc_{d}"])) for d in domains}
labels = {d: y[ls[f"idx_{d}"]] for d in domains}
blend = lambda d, w: w[0] * parts[d][0] + w[1] * parts[d][1] + w[2] * parts[d][2]

step = 20
grid = [(a / step, b / step, (step - a - b) / step) for a in range(step + 1) for b in range(step + 1 - a)]
score = {(d, w): auc(labels[d], blend(d, w)) for d in domains for w in grid}
mean = {w: np.mean([score[(d, w)] for d in domains]) for w in grid}
REF, TWO = (1., 0., 0.), (.7, .3, 0.)

test_share = np.bincount(c.dom[c.te_i], minlength=len(domains)) / len(c.te_i)
train_share = np.bincount(c.dom[:c.n], minlength=len(domains)) / c.n
print("test share", dict(enumerate(test_share.round(3))))
print("train share", dict(enumerate(train_share.round(3))))
print("largest test domain", c.big)
mixed = {w: sum(test_share[d] * score[(d, w)] for d in domains) for w in grid}

rows = []
for d in domains:
    others = [o for o in domains if o != d]
    pick_mean = max(grid, key=lambda w: np.mean([score[(o, w)] for o in others]))
    pick_worst = max(grid, key=lambda w: min(score[(o, w)] for o in others))
    rows.append(dict(domain=d, n=len(labels[d]), test_share=test_share[d], pick_mean="/".join(map(str, pick_mean)), nested_mean_rule=score[(d, pick_mean)],
                     pick_worst="/".join(map(str, pick_worst)), nested_worst_rule=score[(d, pick_worst)], reference=score[(d, REF)], two_way=score[(d, TWO)]))
t = pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
for col in ("nested_mean_rule", "nested_worst_rule", "reference", "two_way"):
    print(col, "mean", round(t[col].mean(), 4), "worst", round(t[col].min(), 4), "test mix", round((t[col] * t.test_share).sum(), 4))
t.round(4).to_csv(REPORTS / "21_nested_three_way.csv", index=False)

enc_f = np.load(OUT / "enc_distilroberta-base_fixed_aug0.0_s3737_ep2_drop0.0.npy")
ref_f = .35 * np.load(OUT / "base_lex_raw.npy") + .65 * np.load(OUT / "base_style_raw.npy")
sur_f = np.load(OUT / "base_surprisal.npy")
v = c.val_i
val = lambda w: auc(y[v], w[0] * r(ref_f[v]) + w[1] * r(sur_f[v]) + w[2] * r(enc_f[v]))
top = sorted(grid, key=lambda w: -mean[w])[:8]
named = [REF, TWO, (.8, .1, .1), (.6, .2, .2), (.5, .3, .2), (.5, .25, .25), (.4, .3, .3)]
show = list(dict.fromkeys(named + top))
g = pd.DataFrame([dict(reference=w[0], surprisal=w[1], encoder=w[2], lodo_mean=mean[w], lodo_worst=min(score[(d, w)] for d in domains), test_mix=mixed[w], fixed_val=val(w)) for w in show])
g = g.sort_values("test_mix", ascending=False).round(5)
print(g.to_string(index=False))
g.to_csv(REPORTS / "21_weight_grid.csv", index=False)

rng = np.random.default_rng(3737)
cands = [(.6, .2, .2), (.5, .3, .2), (.5, .25, .25)]
res = {(w, b): [] for w in cands for b in (REF, TWO)}
for _ in range(500):
    gain = {k: [] for k in res}
    for d in domains:
        pick = rng.integers(0, len(labels[d]), len(labels[d]))
        lab = labels[d][pick]
        if lab.min() == lab.max():
            continue
        base = {b: auc(lab, blend(d, b)[pick]) for b in (REF, TWO)}
        for w in cands:
            s = auc(lab, blend(d, w)[pick])
            for b in (REF, TWO):
                gain[(w, b)].append(s - base[b])
    for k, v_ in gain.items():
        res[k].append(np.mean(v_))
ci = pd.DataFrame([dict(blend="/".join(map(str, w)), against="reference" if b == REF else "two way", mean_gain=np.mean(x), low=np.quantile(x, .025), high=np.quantile(x, .975)) for (w, b), x in res.items()]).round(5)
print(ci.to_string(index=False))
ci.to_csv(REPORTS / "21_bootstrap_gain.csv", index=False)
