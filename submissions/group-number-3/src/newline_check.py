"""Is the newline signal a formatting artifact?

1. Distribution of newline counts per class (train.csv) and in unlabeled/test,
   to see whether the test set even has the same formatting.
2. style_gbm with vs. without newline features, on the fixed validation set.

    python src/newline_check.py
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_test, load_train, load_unlabeled
from features import FEATURE_NAMES, StyleFeatures


def bucket(n):
    return str(n) if n <= 5 else ("6-10" if n <= 10 else "11+")


def main():
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    train, splits = load_train(), load_splits()
    order = [str(i) for i in range(6)] + ["6-10", "11+"]

    # 1. Newline count distributions
    nl = train["text"].str.count("\n").map(bucket)
    dist = pd.crosstab(nl, train["label"].map({0: "human", 1: "AI"}), normalize="columns")
    for name, df in [("unlabeled", load_unlabeled()), ("test", load_test())]:
        dist[name] = df["text"].str.count("\n").map(bucket).value_counts(normalize=True)
    dist = dist.reindex(order).fillna(0)
    print("Share of passages by number of newlines (each column sums to 1):")
    print((dist * 100).round(1).to_string(), "\n")

    print("Double newlines (paragraph breaks) present:")
    for name, df in [("train human", train[train.label == 0]), ("train AI", train[train.label == 1]),
                     ("unlabeled", load_unlabeled()), ("test", load_test())]:
        print(f"  {name:<12} {df['text'].str.contains(chr(10) * 2).mean() * 100:5.1f}%")
    print()

    # 2. style_gbm with vs without newline information
    tr, va = get_split(train, splits, "random_fold")
    sf = StyleFeatures()
    Xtr = pd.DataFrame(sf.transform(tr["text"]), columns=FEATURE_NAMES)
    Xva = pd.DataFrame(sf.transform(va["text"]), columns=FEATURE_NAMES)

    rows = []
    for label, cols in [("all features", FEATURE_NAMES),
                        ("without newline_ratio", [f for f in FEATURE_NAMES if f != "newline_ratio"])]:
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
        gbm.fit(Xtr[cols], tr["label"])
        auc = roc_auc_score(va["label"], gbm.predict_proba(Xva[cols])[:, 1])
        rows.append({"variant": label, "val_auroc": round(auc, 4)})
        print(f"style_gbm {label:<24} val AUROC = {auc:.4f}")

    # Same check with newlines stripped from the raw text (also shows TF-IDF is unaffected)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "newline_check.csv", index=False)
    dist.to_csv(ROOT / "results" / "newline_distribution.csv")
    print("\nSaved results/newline_check.csv and results/newline_distribution.csv")


if __name__ == "__main__":
    main()
