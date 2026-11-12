"""Summarize the recorded predictions after running rebuild.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

out = Path(__file__).resolve().parent / "results"
train = pd.read_csv(out / "train.csv")
val = pd.read_csv(out / "validation.csv")
unlabeled = pd.read_csv(out / "unlabeled_validation.csv")
assert set(val.id) == set(unlabeled.id) and set(train.id).isdisjoint(val.id)
labels = unlabeled.label.to_numpy()
old = unlabeled.supervised.to_numpy()
new = unlabeled.with_unlabeled.to_numpy()
groups = [np.flatnonzero(labels == k) for k in (0, 1)]
rng = np.random.default_rng(42)
deltas = []
for _ in range(1000):
    rows = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
    deltas.append(roc_auc_score(labels[rows], new[rows]) - roc_auc_score(labels[rows], old[rows]))
summary = {
    "reconstructed_train_auc": roc_auc_score(train.label, train.prob),
    "reconstructed_validation_auc": roc_auc_score(val.label, val.prob),
    "unlabeled_auc_difference": roc_auc_score(labels, new) - roc_auc_score(labels, old),
    "unlabeled_stratified_paired_bootstrap_95": np.quantile(deltas, [.025,.975]).tolist(),
}
(out / "diagnostics.json").write_text(json.dumps(summary, indent=2))
val["length_bin"] = pd.cut(val.words, [0,150,250,np.inf], labels=["up to 150","151 to 250","251 or more"])
metrics = []
for group, rows in val.groupby("length_bin", observed=True):
    metrics.append({"length_bin":group,"count":len(rows),"human":int((rows.label==0).sum()),"ai":int(rows.label.sum()),
                    "auc":roc_auc_score(rows.label,rows.prob)})
pd.DataFrame(metrics).to_csv(out / "length_bins.csv",index=False)
fp=val[(val.label==0)&(val.prob>=.5)].nlargest(5,"prob").assign(error="FP")
fn=val[(val.label==1)&(val.prob<.5)].nsmallest(5,"prob").assign(error="FN")
pd.concat([fp,fn]).to_csv(out / "error_examples.csv",index=False)
print(summary,metrics)
