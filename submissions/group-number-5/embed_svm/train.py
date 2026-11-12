import json
import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib

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

if False:
    # Load pseudo-labelled data
    pseudo_df = pd.read_csv(PATH + "pseudo_labelled.csv")

    with open(PATH + "unlabelled_embeddings.json") as f:
        pseudo_data = json.load(f)

    X_pseudo = [pseudo_data[str(i)]["embedding"] for i in pseudo_df["id"]]
    y_pseudo = pseudo_df["label"].tolist()

    # Combine datasets
    X_train = np.vstack([X_train, X_pseudo])
    y_train = np.concatenate([y_train, y_pseudo])


def train_model(X_train, y_train, model_path, **hyperparameters):
    model = SVC(**hyperparameters)

    start = time.time()
    model.fit(X_train, y_train)
    print(f"{model_path}: {time.time() - start:.2f}s")

    joblib.dump(model, model_path)

train_model(
    X_train,
    y_train,
    "submissions/group-number-5/embed_svm/SVM_train",
    kernel="poly",
    C=10.0,
    probability=True
)