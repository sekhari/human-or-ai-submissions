"""Compare a re-run against reference results (used for results/reproduction_simin.md).

  python src/check_repro.py --metrics results/final.json data/repro_final.json
  python src/check_repro.py --sub ../../../submission.csv submission.csv

AUROC depends only on the order of the probabilities, so an identical ranking gives an
identical AUROC on any labels, even when floating-point values differ slightly.
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--metrics", nargs=2, metavar=("REFERENCE", "RERUN"), help="two metrics JSON files written by train.py")
ap.add_argument("--sub", nargs=2, metavar=("REFERENCE", "RERUN"), help="two submission CSV files")
a = ap.parse_args()

def compare(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); d = np.abs(p - q)
    same_rank = bool((np.argsort(p, kind="stable") == np.argsort(q, kind="stable")).all())
    return f"rows={len(d)}  exactly equal={int((d == 0).sum())}  max|diff|={d.max():.2g}  same ranking={same_rank}"

if a.metrics:
    r, n = (json.load(open(f)) for f in a.metrics)
    for k in ("train_auroc", "val_auroc", "val_ci95", "val_perturbed", "fit_seconds"):
        print(f"{k:14s} reference={r[k]}  rerun={n[k]}  equal={r[k] == n[k]}")
    assert r["ids"] == n["ids"] and r["y"] == n["y"], "validation ids or labels differ"
    print("validation p (stored rounded to 6 decimals):", compare(r["p"], n["p"]))
if a.sub:
    r, n = (pd.read_csv(f) for f in a.sub)
    assert list(r.columns) == list(n.columns) == ["id", "label"] and list(r.id) == list(n.id), "columns or id order differ"
    print("submission:", compare(r.label, n.label))
