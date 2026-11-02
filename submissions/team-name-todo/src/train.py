"""Evaluate the baseline on the fixed validation set, then refit on all labeled rows."""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import SEED, auroc_ci, baseline, load_train, split

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.csv"); ap.add_argument("--model-out", default="model.joblib")
ap.add_argument("--metrics-out", default="results/baseline.json"); a = ap.parse_args()
df = load_train(a.train); tr, va = split(df)
t0 = time.time(); m = baseline().fit(tr.text, tr.label); fit = time.time() - t0
ptr = m.predict_proba(tr.text)[:, 1]; pva = m.predict_proba(va.text)[:, 1]
cv = []
for i, j in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, tr.label):
    f = baseline().fit(tr.text.iloc[i], tr.label.iloc[i])
    cv.append(roc_auc_score(tr.label.iloc[j], f.predict_proba(tr.text.iloc[j])[:, 1]))
r = dict(seed=SEED, n_train=len(tr), n_val=len(va), train_auroc=roc_auc_score(tr.label, ptr),
         val_auroc=roc_auc_score(va.label, pva), val_ci95=auroc_ci(va.label, pva).tolist(),
         cv5_mean=float(np.mean(cv)), cv5_sd=float(np.std(cv)), fit_seconds=fit)
Path(a.metrics_out).parent.mkdir(exist_ok=True); Path(a.metrics_out).write_text(json.dumps(r, indent=2)); print(json.dumps(r, indent=2))
joblib.dump(baseline().fit(df.text, df.label), a.model_out)
