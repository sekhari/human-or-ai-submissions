"""Validation AUROC of the 10% RoBERTa rank blend (all_blend_roberta_rank10_10000).

Both components are scored on the fixed 2,000-row validation set (splits/val_ids.csv) after fitting on
the other 8,000 labeled rows only:

- improve_all_blend is refit here exactly as in experiments.improve_lm.
- RoBERTa-base (seed 42, epoch 1) validation probabilities come from the earlier teammate run,
  results/curvature_roberta/single_validation_predictions.csv (column `neural`).

The blend formula and its 10% weight match the submitted CSV (src/rank_blend.py); the weight was fixed
before this check and is not tuned here. Needs the language-model feature cache from
experiments.extract_neural.

Usage:
    python -m experiments.rank10_validation   # writes results/all_blend_roberta_rank/validation.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from experiments import improve_lm
from experiments.improve import fold_design, no_blank_auroc
from experiments.model_comparison import bootstrap_ci
from experiments.robustness import load_domains
from src.data import load_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results" / "all_blend_roberta_rank"
NEURAL_VAL = ROOT / "results" / "curvature_roberta" / "single_validation_predictions.csv"


def main():
    config = json.loads((ARTIFACTS / "config.json").read_text())
    weight = config["neural_weight"]
    _, val, labeled = load_domains()
    labeled = labeled.reset_index(drop=True)
    unlabeled = load_split("unlabeled")
    in_val = labeled.id.isin(val.id)
    train, val = labeled[~in_val], labeled[in_val]
    frame = pd.concat([labeled[["id", "text"]], unlabeled[["id", "text"]]], ignore_index=True)
    feats = improve_lm.all_features(frame, unlabeled.text.tolist())
    Xf, Xe, _ = fold_design(feats, train, val, unlabeled)
    y = val.label.astype(int).values
    baseline = improve_lm.fit_predict("all_blend", Xf, train.label.astype(int).values, Xe)

    neural = pd.read_csv(NEURAL_VAL).set_index("id")
    if set(neural.index) != set(val.id):
        raise ValueError("RoBERTa validation predictions do not cover exactly the validation IDs")
    neural = neural.loc[val.id]
    if (neural.label.values != y).any():
        raise ValueError("Label mismatch between RoBERTa predictions and the validation split")
    neural = neural.neural.values
    blend = (1 - weight) * baseline + weight * rankdata(neural, method="average") / len(neural)

    report = {"validation_rows": len(y), "neural_weight": weight}
    for name, scores in [("improve_all_blend", baseline), ("roberta_alone", neural), ("rank10_blend", blend)]:
        low, high = bootstrap_ci(y, scores)
        report[name] = {"auroc": round(roc_auc_score(y, scores), 7), "ci95": [round(low, 5), round(high, 5)],
                        "auroc_no_blank": round(no_blank_auroc(feats, val, scores), 7)}
    report["note"] = ("Validation was reused across many experiments, so these scores are optimistic. "
                      "No LODO score: RoBERTa per-domain held-out predictions for this recipe do not exist.")
    (ARTIFACTS / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
