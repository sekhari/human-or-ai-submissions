"""how much does each text edit hurt each model on the fixed validation set?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.baseline import fit_lexical, fit_style
from src.harness import Ctx, REPORTS
from src.models import fit_clean_lexical
from src.validate import attack_table

c = Ctx()
t, y = list(c.texts[c.fit_i]), c.y[c.fit_i]
vt, vy = c.texts[c.val_i], c.y[c.val_i]
lx, st, cl = fit_lexical(t, y), fit_style(t, y), fit_clean_lexical(t, y)
models = {"reference blend raw": lambda x: .35 * lx(x) + .65 * st(x), "reference style raw": st, "clean word+char lr": cl}
out = None
for name, f in models.items():
    a = attack_table(f, vt, vy).rename(columns={"auroc": name})
    out = a if out is None else out.merge(a, on="attack")
    print(name, "done", flush=True)
out.round(4).to_csv(REPORTS / "10_attacks.csv", index=False)
print(out.round(4).to_string(index=False))
