"""Model A: word TF-IDF + Logistic Regression."""

import os
import sys
import itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_data, get_split, evaluate, get_scores, SEED

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import load_data, get_split, evaluate, SEED

WORDS_ONLY = r"(?u)\b\w\w+\b"
WORDS_AND_PUNCT = r"(?u)\b\w+\b|[^\w\s]"


def build_model(lowercase=False, keep_punct=True, ngram_range=(1, 2), C=10.0):
    return make_pipeline(
        TfidfVectorizer(
            lowercase=lowercase,
            token_pattern=WORDS_AND_PUNCT if keep_punct else WORDS_ONLY,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        ),
        LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    )


def tune(train_part, val_part):
    grid = {
        "lowercase": [True, False],
        "keep_punct": [False, True],
        "ngram_range": [(1, 1), (1, 2)],
        "C": [0.1, 1, 10, 100],
    }
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        print(params)
        r = evaluate(build_model(**params), train_part, val_part)
        results.append({**params, "train_auc": round(r["train_auc"], 4),
                        "val_auc": round(r["val_auc"], 4),
                        "fit_seconds": round(r["fit_seconds"], 1)})
    table = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    with open("experiments/model_a_tuning.txt", "w") as f:
        f.write(table.to_string(index=False))
    print(table.head(10).to_string(index=False))
    return table

def save_predictions(train, train_part, val_part, test, name="model_a", **params):
    os.makedirs("preds", exist_ok=True)

    # 1) validation predictions (model trained on the 8,000 rows) -> for ensemble later
    result = evaluate(build_model(**params), train_part, val_part)
    np.save(f"preds/{name}_val.npy", result["val_pred"])

    # 2) retrain on all 10,000 labeled rows, then predict the test set
    final_model = build_model(**params)
    final_model.fit(train["text"], train["label"])
    test_pred = get_scores(final_model, test["text"])
    np.save(f"preds/{name}_test.npy", test_pred)

    # 3) Kaggle file: id + P(AI)
    pd.DataFrame({"id": test["id"], "label": test_pred}).to_csv(
        f"submission_{name}.csv", index=False)
    print(f"saved preds/{name}_val.npy, preds/{name}_test.npy, submission_{name}.csv")

if __name__ == "__main__":
    train, unlabeled, test = load_data()
    train_part, val_part = get_split(train)
    if "--tune" in sys.argv:
        tune(train_part, val_part)
    elif "--submit" in sys.argv:
        save_predictions(train, train_part, val_part, test, name="model_a")
        save_predictions(train, train_part, val_part, test, name="model_a_plain",
                         lowercase=True, keep_punct=False)
    else:
        evaluate(build_model(), train_part, val_part)