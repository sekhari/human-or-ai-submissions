# error_analysis.py
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from train_gbdt import make_gbdt, fit_lsa, featurize
from split import load_split
import json

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--unlabeled", required=True)
a = p.parse_args()

train_all = pd.read_csv(a.train)
unl = pd.read_csv(a.unlabeled)["text"].tolist()
tr, va = load_split(train_all)
va = va.copy()

best = json.load(open("results_gbdt.json"))["params"]
lsa = fit_lsa(tr["text"].tolist() + unl)
F_tr, F_va = featurize(tr["text"], lsa), featurize(va["text"], lsa)
model = make_gbdt(**best).fit(F_tr, tr["label"])
va["p_ai"] = model.predict_proba(F_va)[:, 1]
va["words"] = va["text"].str.split().str.len()
va["blank_lines"] = F_va["n_blank_lines"].values
va.drop(columns="text").to_csv("val_predictions.csv", index=False)

# Q4.2: word-count terciles
va["bin"] = pd.qcut(va["words"], 3, labels=["short", "medium", "long"])
for b, g in va.groupby("bin", observed=True):
    print(b, len(g), f"words {g.words.min()}-{g.words.max()}",
          round(roc_auc_score(g.label, g.p_ai), 4), "class counts:", g.label.value_counts().to_dict())

# Q4.3: errors at 0.5, plus most confident mistakes as fallback
fp = va[(va.label == 0) & (va.p_ai >= 0.5)]
fn = va[(va.label == 1) & (va.p_ai < 0.5)]
print(f"\nAt 0.5: {len(fp)} FPs, {len(fn)} FNs")
cols = [c for c in ["id", "label", "p_ai", "words", "blank_lines"] if c in va.columns]
print("Top FP candidates:\n", va[va.label == 0].nlargest(5, "p_ai")[cols])
print("Top FN candidates:\n", va[va.label == 1].nsmallest(5, "p_ai")[cols])