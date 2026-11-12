"""Report Q1.3 (second model): grid search for the style-feature gradient boosting model.

Grid: learning_rate x max_iter x max_leaf_nodes, scored on the fixed 2,000-row
validation set. Style features are computed once, so this runs quickly.

    python src/tune_gbm.py
"""
import itertools

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train
from features import StyleFeatures

GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 300, 600],
    "max_leaf_nodes": [15, 31, 63],
}


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    tr, va = get_split(train, splits, "random_fold")
    sf = StyleFeatures()
    Xtr, Xva = sf.transform(tr["text"]), sf.transform(va["text"])

    rows = []
    for lr, it, leaves in itertools.product(*GRID.values()):
        gbm = HistGradientBoostingClassifier(learning_rate=lr, max_iter=it,
                                             max_leaf_nodes=leaves, random_state=42)
        gbm.fit(Xtr, tr["label"])
        tr_auc = roc_auc_score(tr["label"], gbm.predict_proba(Xtr)[:, 1])
        va_auc = roc_auc_score(va["label"], gbm.predict_proba(Xva)[:, 1])
        rows.append({"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves,
                     "train_auroc": round(tr_auc, 4), "val_auroc": round(va_auc, 4)})

    df = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results" / "tune_gbm.csv", index=False)
    best = df.iloc[0]
    print(f"\nBest: learning_rate={best.learning_rate}, max_iter={int(best.max_iter)}, "
          f"max_leaf_nodes={int(best.max_leaf_nodes)} -> val AUROC {best.val_auroc}")
    print("Current default (0.05, 300, 31):",
          df[(df.learning_rate == 0.05) & (df.max_iter == 300) & (df.max_leaf_nodes == 31)]
          ["val_auroc"].iloc[0])
    print("Saved results/tune_gbm.csv")


if __name__ == "__main__":
    main()
