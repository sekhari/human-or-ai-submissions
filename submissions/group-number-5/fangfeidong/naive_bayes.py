
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("data/train_starter_2000.csv")
X = np.load("data/starter_embeddings.npy")
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = GaussianNB()

start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start


train_prob = model.predict_proba(X_train)[:, 1]
val_prob = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)
train_auc = roc_auc_score(y_train, train_prob)
val_auc = roc_auc_score(y_val, val_prob)
val_accuracy = accuracy_score(y_val, y_pred)

print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Training AUROC:", round(train_auc, 4))
print("Validation AUROC:", round(val_auc, 4))
print("Validation Accuracy:", round(val_accuracy, 4))
print("Training Time:", round(train_time, 4), "seconds")
