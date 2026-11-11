"""
Final Kaggle submission: ensemble of Model B (word TF-IDF + LinearSVC) and
Model D (char 3-5 TF-IDF + LinearSVC). Best val AUROC on our split: 0.9497.

After picking B + D on the val split, both models are retrained on all 10,000
labeled rows (allowed by the rubric), then scored on test.csv. Each model's test
scores are turned into ranks and averaged, same as ensemble.py. The output is in
[0, 1] and higher means more likely AI. AUROC only uses order, so ranks are fine.

Run (from team folder): python make_submission.py
Needs: data/train.csv, data/test.csv
Out:   submission.csv  (id,label for all 3,000 test rows)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(f"Training on all {len(train)} labeled rows, predicting {len(test)} test rows")

model_b = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LinearSVC(C=1))
model_d = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                        sublinear_tf=True),
                        LinearSVC(C=1))

ranks = []
for name, model in [("B", model_b), ("D", model_d)]:
    model.fit(train["text"], train["label"])
    scores = model.decision_function(test["text"])
    ranks.append(rankdata(scores) / len(scores))
    print(f"Model {name} done")

sub = pd.DataFrame({"id": test["id"], "label": np.mean(ranks, axis=0)})
assert len(sub) == 3000 and sub["label"].between(0, 1).all() and sub["id"].is_unique
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(sub.head())