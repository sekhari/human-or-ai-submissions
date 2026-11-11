"""
Model C: word counts + Multinomial Naive Bayes.
Tries every combo of alpha and max_features on the shared val split.

Run (from team folder): python nb_model.py
Out: nb_results.csv
"""
import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from common import load_split

tr, va = load_split()
print(f"Train rows: {len(tr)}, Val rows: {len(va)}")

# First grid was alpha 0.01-1 and max_features 10k/50k/all. Best ended up on the
# edge (10k, 0.01), so I widened both ends to check it wasn't still improving.
ALPHAS = [0.001, 0.01, 0.1, 0.5, 1.0]
MAX_FEATURES = [2_000, 5_000, 10_000, 50_000, None]  # None = keep everything (min_df=2)

results = []
for mf in MAX_FEATURES:
    for alpha in ALPHAS:
        # single words + word pairs, drop anything that shows up in only 1 passage
        model = make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=mf),
            MultinomialNB(alpha=alpha),
        )
        start = time.time()
        model.fit(tr["text"], tr["label"])
        secs = time.time() - start

        train_auc = roc_auc_score(tr["label"], model.predict_proba(tr["text"])[:, 1])
        val_auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])
        results.append({"max_features": mf or "all", "alpha": alpha,
                        "train_auroc": round(train_auc, 4), "val_auroc": round(val_auc, 4),
                        "train_sec": round(secs, 1)})
        print(f"max_features={str(mf or 'all'):<6} alpha={alpha:<5} "
              f"train AUROC={train_auc:.4f}  val AUROC={val_auc:.4f}  ({secs:.1f}s)")

df = pd.DataFrame(results)
df.to_csv("nb_results.csv", index=False)
best = df.loc[df["val_auroc"].idxmax()]
print(f"\nBest: max_features={best.max_features}, alpha={best.alpha}, "
      f"val AUROC={best.val_auroc}, train AUROC={best.train_auroc}")
