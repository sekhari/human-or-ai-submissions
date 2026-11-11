"""
Q3: does unlabeled.csv help? Pseudo-labeling with Naive Bayes.

Steps:
1. Train NB (best setting from nb_results.csv) on the 8,000 training rows.
2. Predict the 3,000 unlabeled passages.
3. Take the k most confident AI and k most confident human ones and label them 1/0.
   Same k for both classes so we don't mess up the 50/50 balance.
4. Retrain on real rows + pseudo rows, score on the same 2,000 val rows.

No leakage: val rows never go into training and aren't used to pick pseudo-labels.
Val labels are only used for the final score and the noise estimate.

Run (from team folder): python nb_model.py   (first)
                        python pseudo_label.py
Out: pseudo_label_results.csv
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from common import load_split

tr, va = load_split()
unl = pd.read_csv("data/unlabeled.csv")
print(f"Train {len(tr)}, Val {len(va)}, Unlabeled {len(unl)}")

best = pd.read_csv("nb_results.csv").sort_values("val_auroc", ascending=False).iloc[0]
mf = None if str(best["max_features"]) == "all" else int(best["max_features"])
alpha = float(best["alpha"])
print(f"Using NB setting: max_features={best['max_features']}, alpha={alpha}")


def make_nb():
    return make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=mf),
                         MultinomialNB(alpha=alpha))


def n_words(texts):
    return texts.str.split().str.len()


# baseline: real labels only
base = make_nb().fit(tr["text"], tr["label"])
base_val = roc_auc_score(va["label"], base.predict_proba(va["text"])[:, 1])
print(f"\nSupervised only: val AUROC={base_val:.4f}")

p_unl = base.predict_proba(unl["text"])[:, 1]
order = np.argsort(p_unl)  # low P(AI) first
print(f"Unlabeled: {np.mean(p_unl > 0.5):.1%} predicted AI, "
      f"{np.mean((p_unl > 0.99) | (p_unl < 0.01)):.1%} have P below 0.01 or above 0.99")

# Rough noise check: how often is the model right on its most confident val rows?
# Same fraction as the pseudo-labels we keep.
p_val = base.predict_proba(va["text"])[:, 1]
conf_val = np.abs(p_val - 0.5)

rows = [{"method": "supervised only", "k_per_class": 0, "pseudo_rows": 0,
         "val_auroc": round(base_val, 4), "est_pseudo_label_acc": None,
         "median_words_pseudo": None}]
for k in [250, 500, 1000, 1500]:
    pseudo = pd.concat([unl.iloc[order[:k]].assign(label=0),
                        unl.iloc[order[-k:]].assign(label=1)])
    aug = pd.concat([tr[["text", "label"]], pseudo[["text", "label"]]])
    model = make_nb().fit(aug["text"], aug["label"])
    auc = roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])

    top = conf_val >= np.quantile(conf_val, 1 - 2 * k / len(unl))
    est_acc = np.mean((p_val[top] > 0.5) == va["label"].values[top])
    med_len = n_words(pseudo["text"]).median()

    rows.append({"method": "with pseudo-labels", "k_per_class": k, "pseudo_rows": 2 * k,
                 "val_auroc": round(auc, 4), "est_pseudo_label_acc": round(est_acc, 3),
                 "median_words_pseudo": med_len})
    print(f"k={k:<5} ({2*k} pseudo rows): val AUROC={auc:.4f} ({auc - base_val:+.4f}), "
          f"est. label accuracy={est_acc:.1%}, median length={med_len:.0f} words")

print(f"\nFor comparison, median length in the 8k training rows: {n_words(tr['text']).median():.0f} words")
pd.DataFrame(rows).to_csv("pseudo_label_results.csv", index=False)
