import joblib
from sklearn.metrics import roc_auc_score
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

with open("submissions/group-number-5/embed_svm/embeddings.json") as f:
    data = json.load(f)

starter_df = pd.read_csv("submissions/group-number-5/embed_svm/train_starter_2000.csv")
full_train_df = pd.read_csv("submissions/group-number-5/embed_svm/train.csv")

X = [data[str(i)]["embedding"] for i in full_train_df["id"]]
y = full_train_df["label"].tolist()

RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)



model = joblib.load(
    "submissions/group-number-5/embed_svm/SVM_train"
)

#weights = model.coef_

predictions = model.predict_proba(X_test)[:, 1]

print("Dev AUROC:", roc_auc_score(y_test, predictions))

predictions = model.predict_proba(X_train)[:, 1]

print("Train AUROC:", roc_auc_score(y_train, predictions))