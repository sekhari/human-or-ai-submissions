import json
import pandas as pd
import numpy as np
import time
import ollama
from tqdm import tqdm
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PATH = "submissions/group-number-5/embed_svm/"

# Load original embeddings and training data
with open(PATH + "embeddings.json") as f:
    data = json.load(f)

train_df = pd.read_csv(PATH + "train.csv")

X = [data[str(i)]["embedding"] for i in train_df["id"]]
y = train_df["label"].tolist()

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

# Load existing SVM
model = joblib.load(PATH + "SVM_train")

# Embed unlabelled data
unlabelled_df = pd.read_csv(PATH + "unlabeled.csv")

X_unlabelled = np.array([
    ollama.embed(
        model="embeddinggemma-2",
        input=text
    )["embeddings"][0]
    for text in tqdm(unlabelled_df["text"])
])

with open("submissions/group-number-5/embed_svm/unlabelled_embeddings.json","w") as writer:
    json.dump({str(i):{"embedding":list(v)} for i,v in enumerate(list(X_unlabelled))},writer)

# Generate pseudo-labels
pseudo_labels = model.predict(X_unlabelled)

unlabelled_df["label"] = pseudo_labels

unlabelled_df.to_csv(PATH + "pseudo_labelled.csv", index=False)

# Retrain on original training data + pseudo-labelled data
X_combined = np.vstack([X_train, X_unlabelled])
y_combined = np.concatenate([y_train, pseudo_labels])

new_model = SVC(
    kernel="poly",
    C=10.0,
    probability=True
)

start = time.time()
new_model.fit(X_combined, y_combined)
elapsed = time.time() - start

# Evaluate on original held-out dev set
predictions = new_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, predictions)

print(f"Training time: {elapsed:.2f}s")
print(f"Validation AUROC: {auroc:.4f}")

joblib.dump(new_model, PATH + "SVM_pseudo_labelled")