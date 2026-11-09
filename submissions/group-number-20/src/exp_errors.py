"""Q4: subgroup AUROC and the most confident validation errors at threshold 0.5.
Default reads results/final.json -> results/q4_errors.json (same as before).
Final 2: python src/exp_errors.py --results results/final_v2.json --out results/q4_errors_v2.json"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from common import load_train
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/final.json"); ap.add_argument("--out", default="results/q4_errors.json"); ap.add_argument("--train", default="data/train.csv")
args = ap.parse_args()
r = json.load(open(args.results)); df = load_train(args.train).set_index("id").loc[r["ids"]].reset_index()
df["p"] = r["p"]; df["w"] = df.text.str.split().str.len(); df["blank"] = df.text.str.contains("\n\n")
out = dict(train=r["train_auroc"], val=r["val_auroc"], len_bins=[], fmt=[])
for a, b in ((0, 150), (150, 200), (200, 250), (250, 400)):
    x = df[(df.w >= a) & (df.w < b)]; out["len_bins"].append(dict(bin=f"{a}-{b}", n=len(x), ai=float(x.label.mean()), auc=auc(x.label, x.p)))
for n, m in (("no line breaks", ~df.text.str.contains("\n")), ("single breaks only", df.text.str.contains("\n") & ~df.blank), ("has blank line", df.blank)):
    x = df[m]; out["fmt"].append(dict(group=n, n=len(x), ai=float(x.label.mean()), auc=auc(x.label, x.p) if x.label.nunique() > 1 else None))
yh = df.p >= .5; out["acc"] = float((yh == df.label).mean()); out["fp"] = int((yh & (df.label == 0)).sum()); out["fn"] = int((~yh & (df.label == 1)).sum())
fp = df[yh & (df.label == 0)].sort_values("p", ascending=False).head(8); fn = df[~yh & (df.label == 1)].sort_values("p").head(8)
out["errors"] = [dict(id=int(a.id), t="FP" if a.label == 0 else "FN", p=a.p, words=int(a.w)) for _, a in pd.concat([fp, fn]).iterrows()]
json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "errors"}, indent=0))
for _, a in pd.concat([fp, fn]).iterrows(): print(a.id, "FP" if a.label == 0 else "FN", a.p, a.w, ascii(a.text[:230]))
