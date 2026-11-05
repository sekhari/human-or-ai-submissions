"""report q2: nested 500 to 8000 row subsets, fixed validation set and hyperparameters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline import fit_style
from src.harness import Ctx, REPORTS
from src.models import cleaned, fit_clean_lexical

c = Ctx()
f, y, texts = c.fit_i, c.y, c.texts
order = {}
rest = f
for size in (5000, 2000, 1000, 500):
    rest, _ = train_test_split(rest, train_size=size, stratify=y[rest], random_state=3737)
    order[size] = rest
order[8000] = f
rows = []
models = {"clean word+char lr": lambda t, l: fit_clean_lexical(t, l), "reference style catboost": lambda t, l: fit_style(t, l)}
held_big = np.flatnonzero(c.dom[:c.n] == c.big)
for name, fit in models.items():
    for size in (500, 1000, 2000, 5000, 8000):
        a = order[size]
        s = fit(list(texts[a]), y[a])
        rows.append(dict(model=name, rows=size, train_auc=auc(y[a], s(list(texts[a]))), val_auc=auc(y[c.val_i], s(list(texts[c.val_i])))))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows).round(4)
df.to_csv(REPORTS / "08_learning_curve.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 3.6))
for (name, g), col in zip(df.groupby("model"), ("#1f77b4", "#d95f02")):
    ax.plot(g.rows, g.train_auc, "--", color=col, label=f"{name}, train")
    ax.plot(g.rows, g.val_auc, "-o", color=col, label=f"{name}, validation")
ax.set_xscale("log"); ax.set_xlabel("training rows"); ax.set_ylabel("AUROC"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(REPORTS / "08_learning_curve.png", dpi=160)
