import json
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PATH = "submissions/group-number-5/embed_svm/"

with open(PATH + "smaller_embeddings.json") as f:
    data = json.load(f)

starter_df = pd.read_csv(PATH + "train_starter_2000.csv")

X = [data[str(i)]["embedding"] for i in starter_df["id"]]
y = starter_df["label"].tolist()

RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

kernels = ["linear", "rbf", "poly", "sigmoid"]
C_values = [0.01, 0.1, 1, 10, 100]

results = []

for kernel in kernels:
    for C in C_values:
        model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=RANDOM_SEED
        )

        model.fit(X_train, y_train)

        predictions = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, predictions)

        results.append({
            "kernel": kernel,
            "C": C,
            "AUROC": auroc
        })

        print(f"kernel={kernel}, C={C}: AUROC={auroc:.4f}")

results_df = pd.DataFrame(results).sort_values(
    "AUROC", ascending=False
)

print("\nBest configurations:")
print(results_df.to_string(index=False))

results_df.to_csv(PATH + "smaller_grid_search_results.csv", index=False)