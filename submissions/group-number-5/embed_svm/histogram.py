import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

PATH = "submissions/group-number-5/embed_svm/"

# Load embeddings and labelled data
with open(PATH + "embeddings.json") as f:
    data = json.load(f)

df = pd.read_csv(PATH + "train_starter_2000.csv")

X = np.array([data[str(i)]["embedding"] for i in df["id"]])
y = df["label"].to_numpy()

# Recreate original split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load models
model1 = joblib.load(PATH + "SVM_train")
model2 = joblib.load(PATH + "SVM_pseudo_labelled")

# Get prediction probabilities
probs1 = model1.predict_proba(X_test)[:, 1]
probs2 = model2.predict_proba(X_test)[:, 1]

def extremity_rate(probs, threshold=0.1):
    return np.mean((probs <= threshold) | (probs >= 1 - threshold))

print(f"SVM_train extremity rate: {extremity_rate(probs1):.2%}")
print(f"SVM_unlabelled_train extremity rate: {extremity_rate(probs2):.2%}")

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(probs1, bins=20, range=(0, 1), edgecolor="black")
axes[0].set_title("SVM_train")
axes[0].set_xlabel("Prediction probability (class 1)")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0,10)

axes[1].hist(probs2, bins=20, range=(0, 1), edgecolor="black")
axes[1].set_title("SVM_pseudo_labelled")
axes[1].set_xlabel("Prediction probability (class 1)")
axes[1].set_ylim(0,10)

plt.tight_layout()
plt.savefig(PATH + "prediction_histograms.png", dpi=300)
plt.show()