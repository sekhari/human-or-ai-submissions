import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


df = pd.read_csv("data/train.csv")


with open("embed_svm/embeddings.json") as f:
    embeddings = json.load(f)

X = np.array([embeddings[str(i)]["embedding"] for i in df["id"]])
y = df["label"].to_numpy()


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# part4
sizes = [500, 1000, 2000, 5000, 8000]

np.random.seed(42)
indices = np.random.permutation(len(X_train))

train_aucs = []
val_aucs = []

for size in sizes:
    selected = indices[:size]
    X_subset = X_train[selected]
    y_subset = y_train[selected]

    model = SVC(kernel="poly", C=10.0, probability=True)
    model.fit(X_subset, y_subset)

    train_probs = model.predict_proba(X_subset)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_subset, train_probs)
    val_auc = roc_auc_score(y_val, val_probs)

    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    print(size, "Train AUROC:", round(train_auc, 4),
          "Validation AUROC:", round(val_auc, 4))

# plot
plt.plot(sizes, train_aucs, marker="o", label="Train AUROC")
plt.plot(sizes, val_aucs, marker="o", label="Validation AUROC")
plt.xlabel("Training Data Size")
plt.ylabel("AUROC")
plt.title("Effect of Training Data Size")
plt.legend()
plt.grid(True)
plt.savefig("fangfeidong/part4_learning_curve.png")
plt.show()
