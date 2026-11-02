"""Check whether whitespace formatting separates the classes, and whether that pattern appears in test."""
import re, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from common import load_train, split
d = load_train("data/train.csv"); u = pd.read_csv("data/unlabeled.csv"); e = pd.read_csv("data/test.csv")
F = {"has_dbl_nl": lambda t: "\n\n" in t, "has_single_nl": lambda t: bool(re.search(r"(?<!\n)\n(?!\n)", t)), "has_any_nl": lambda t: "\n" in t,
     "has_tab": lambda t: "\t" in t, "trail_ws": lambda t: t != t.rstrip(), "lead_ws": lambda t: t != t.lstrip(), "dbl_space": lambda t: "  " in t,
     "curly": lambda t: bool(re.search("[‘’“”]", t)), "straight_apos": lambda t: "'" in t, "crlf": lambda t: "\r" in t}
rows = {}
for k, f in F.items():
    a = d.text.map(f).astype(int)
    rows[k] = dict(human=a[d.label == 0].mean(), ai=a[d.label == 1].mean(), unl=u.text.map(f).mean(), test=e.text.map(f).mean())
print(pd.DataFrame(rows).T.round(3))
c = d.assign(dbl=d.text.map(F["has_dbl_nl"]), single=d.text.map(F["has_single_nl"]))
print(c.groupby(["dbl", "single"]).label.agg(["mean", "size"]).round(3))
for n, x in (("unl", u), ("test", e)):
    print(n, pd.crosstab(x.text.map(F["has_dbl_nl"]), x.text.map(F["has_single_nl"])).to_dict())
R = lambda t: t.count("\n\n") * 2 / max(t.count("\n"), 1) if "\n" in t else -1
print("dbl share of newlines auc", auc(d.label, d.text.map(R)))
