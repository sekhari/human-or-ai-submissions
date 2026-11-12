"""Model B. Run from group15 with python -m experiments.model_b_char_svm."""

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import SEED, clean_text, evaluate, get_scores, get_split, load_data


TEXT_MODE = "raw"


def build_model():
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_features=120_000,
            sublinear_tf=True,
        ),
        LinearSVC(C=1.0, dual="auto", random_state=SEED),
    )


def main():
    train, _, test = load_data()
    train_part, val_part = get_split(train)
    model = build_model()
    print("Model B: raw text, char TF-IDF (3–5), min_df=3, LinearSVC C=1")
    result = evaluate(model, train_part, val_part, mode=TEXT_MODE)

    # Fit on the full labeled set for test predictions after validation.
    model.fit(train["text"].map(lambda text: clean_text(text, TEXT_MODE)), train["label"])
    test_scores = get_scores(model, test["text"].map(lambda text: clean_text(text, TEXT_MODE)))

    preds = Path("preds")
    preds.mkdir(exist_ok=True)
    # Scores follow val_part and test.csv row order and are SVM margins.
    np.save(preds / "model_b_val.npy", result["val_pred"])
    np.save(preds / "model_b_test.npy", test_scores)
    print("Saved model_b_val.npy and model_b_test.npy under preds/.")


if __name__ == "__main__":
    main()
