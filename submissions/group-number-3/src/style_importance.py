"""Which stylometric features drive the style_gbm model?

1. Single-feature AUROC: how well each feature ALONE separates human vs AI
   on the fixed validation set (0.5 = useless; far from 0.5 either way = strong).
2. Permutation importance: how much the full GBM's val AUROC drops when one
   feature is shuffled.
3. Per-class means of the top features, so you can see the direction.

    python src/style_importance.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from features import FEATURE_NAMES, StyleFeatures


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")

    sf = StyleFeatures()
    Xtr = pd.DataFrame(sf.transform(tr["text"]), columns=FEATURE_NAMES)
    Xva = pd.DataFrame(sf.transform(va["text"]), columns=FEATURE_NAMES)
    ytr, yva = tr["label"].values, va["label"].values

    single = {f: roc_auc_score(yva, Xva[f]) for f in FEATURE_NAMES}

    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    gbm.fit(Xtr, ytr)
    full_auc = roc_auc_score(yva, gbm.predict_proba(Xva)[:, 1])
    perm = permutation_importance(gbm, Xva, yva, scoring="roc_auc",
                                  n_repeats=5, random_state=42)

    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "single_auroc": [single[f] for f in FEATURE_NAMES],
        "perm_drop": perm.importances_mean,
        "human_mean": [Xva.loc[yva == 0, f].mean() for f in FEATURE_NAMES],
        "ai_mean": [Xva.loc[yva == 1, f].mean() for f in FEATURE_NAMES],
    })
    df["single_strength"] = (df["single_auroc"] - 0.5).abs() + 0.5
    df = df.sort_values("perm_drop", ascending=False)

    pd.set_option("display.width", 120)
    print(f"Full style_gbm val AUROC: {full_auc:.4f}\n")
    print(df.drop(columns="single_strength").round(4).to_string(index=False))

    out = ROOT / "results" / "style_importance.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
