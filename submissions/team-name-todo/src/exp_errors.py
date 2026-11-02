"""Q4: subgroup AUROC and the most confident validation errors at threshold 0.5."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from common import load_train
r = json.load(open("results/final.json")); df = load_train("data/train.csv").set_index("id").loc[r["ids"]].reset_index()
df["p"] = r["p"]; df["w"] = df.text.str.split().str.len(); df["blank"] = df.text.str.contains("\n\n")
out = dict(train=r["train_auroc"], val=r["val_auroc"], len_bins=[], fmt=[])
for a, b in ((0, 150), (150, 200), (200, 250), (250, 400)):
    x = df[(df.w >= a) & (df.w < b)]; out["len_bins"].append(dict(bin=f"{a}-{b}", n=len(x), ai=float(x.label.mean()), auc=auc(x.label, x.p)))
for n, m in (("no line breaks", ~df.text.str.contains("\n")), ("single breaks only", df.text.str.contains("\n") & ~df.blank), ("has blank line", df.blank)):
    x = df[m]; out["fmt"].append(dict(group=n, n=len(x), ai=float(x.label.mean()), auc=auc(x.label, x.p) if x.label.nunique() > 1 else None))
yh = df.p >= .5; out["acc"] = float((yh == df.label).mean()); out["fp"] = int((yh & (df.label == 0)).sum()); out["fn"] = int((~yh & (df.label == 1)).sum())
fp = df[yh & (df.label == 0)].sort_values("p", ascending=False).head(8); fn = df[~yh & (df.label == 1)].sort_values("p").head(8)
out["errors"] = [dict(id=int(a.id), t="FP" if a.label == 0 else "FN", p=a.p, words=int(a.w)) for _, a in pd.concat([fp, fn]).iterrows()]
json.dump(out, open("results/q4_errors.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "errors"}, indent=0))
for _, a in pd.concat([fp, fn]).iterrows(): print(a.id, "FP" if a.label == 0 else "FN", a.p, a.w, repr(a.text[:230]))
