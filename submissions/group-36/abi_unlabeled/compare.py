"""Compare paired validation rankings for the same two thousand labeled passages."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, default=here.parent / "group36_evidence/results/validation.csv")
args = parser.parse_args()
scores = pd.read_csv(args.baseline)[["id", "label", "prob"]].merge(
    pd.read_csv(here / "results/validation.csv")[["id", "label", "prob"]],
    on=["id", "label"], suffixes=("_supervised", "_unlabeled"), validate="one_to_one"
)
assert len(scores) == 2000 and scores.label.value_counts().to_dict() == {0: 1000, 1: 1000}
labels = scores.label.to_numpy()
supervised = scores.prob_supervised.to_numpy()
augmented = scores.prob_unlabeled.to_numpy()
old_auc = roc_auc_score(labels, supervised)
new_auc = roc_auc_score(labels, augmented)
rng = np.random.default_rng(42)
groups = [np.flatnonzero(labels == label) for label in (0, 1)]
differences = []
for _ in range(1000):
    rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
    differences.append(roc_auc_score(labels[rows], augmented[rows]) - roc_auc_score(labels[rows], supervised[rows]))
summary = dict(validation_rows=len(scores), supervised_auc=old_auc, unlabeled_auc=new_auc,
               delta_auc=new_auc-old_auc, paired_95_interval=np.quantile(differences,[.025,.975]).tolist(),
               bootstrap_replicates=1000, seed=42)
(here / "results/comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
