"""Evaluate the final detector on the fixed validation set, then refit on all labeled rows."""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score as auc
from common import SEED, auroc_ci, load_train, split
from features import perturb_format
from pipeline import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.csv"); ap.add_argument("--no-format", action="store_true")
ap.add_argument("--model-out", default="model.joblib"); ap.add_argument("--metrics-out", default="results/final.json"); a = ap.parse_args()
df = load_train(a.train); tr, va = split(df); t0 = time.time(); m = Detector(not a.no_format).fit(tr.text.tolist(), tr.label.values); fit = time.time() - t0
pv = m.predict(va.text.tolist()); px = m.predict(va.text.map(perturb_format).tolist())
r = dict(seed=SEED, use_format=not a.no_format, train_auroc=auc(tr.label, m.predict(tr.text.tolist())), val_auroc=auc(va.label, pv),
         val_ci95=auroc_ci(va.label, pv).tolist(), val_perturbed=auc(va.label, px), fit_seconds=fit, ids=va.id.tolist(), p=[round(float(x), 6) for x in pv], y=va.label.tolist())
Path(a.metrics_out).write_text(json.dumps(r)); print({k: v for k, v in r.items() if k not in ("ids", "p", "y")})
joblib.dump(Detector(not a.no_format).fit(df.text.tolist(), df.label.values), a.model_out)
