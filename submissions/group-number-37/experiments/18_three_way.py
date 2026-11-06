"""three way rank blend on the fixed validation set and the held out book and poetry domain.

the book and poetry column uses models that never saw that domain. reference and surprisal scores for it come from the
lodo cache, the encoder from the big holdout run. only 143 rows, 41 human, so read it loosely.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

from src.data import OUT, fixed_split, load
from src.harness import REPORTS

tr = load("train")
y = tr.label.values
_, v = fixed_split(tr)
ls = dict(np.load(OUT / "lodo_scores.npz"))
big = 5
b = ls[f"idx_{big}"]
enc_f = np.load(OUT / "enc_distilroberta-base_fixed_aug0.0_s3737_ep2_drop0.0.npy")
enc_b = np.load(OUT / "enc_distilroberta-base_big_aug0.0_s3737_ep2_drop0.0.npy")
ref = .35 * np.load(OUT / "base_lex_raw.npy") + .65 * np.load(OUT / "base_style_raw.npy")
sur = np.load(OUT / "base_surprisal.npy")
r = lambda x: rankdata(x) / len(x)
rows = []
for ws in ((1, 0, 0), (.7, .3, 0), (.8, .2, 0), (.9, .1, 0), (.9, 0, .1), (.8, 0, .2), (.7, 0, .3), (.5, 0, .5), (.8, .1, .1), (.7, .2, .1), (.7, .1, .2),
           (.6, .2, .2), (.5, .2, .3), (.6, .3, .1), (.5, .3, .2), (.4, .3, .3), (.3, .3, .4), (0, .5, .5), (0, 0, 1)):
    val = ws[0] * r(ref[v]) + ws[1] * r(sur[v]) + ws[2] * r(enc_f[v])
    held = ws[0] * r(ls[f"ref_{big}"]) + ws[1] * r(ls[f"sur_{big}"]) + ws[2] * r(enc_b[b])
    rows.append(dict(reference=ws[0], surprisal=ws[1], encoder=ws[2], val=auc(y[v], val), held_book_poetry=auc(y[b], held)))
df = pd.DataFrame(rows).round(5)
df.to_csv(REPORTS / "18_three_way.csv", index=False)
print(df.to_string(index=False))
