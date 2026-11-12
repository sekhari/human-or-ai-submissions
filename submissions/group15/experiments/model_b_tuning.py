"""Small hyperparameter grid for Model B (char TF-IDF + LinearSVC).

Run from submissions/group15/:  python experiments/model_b_tuning.py
Uses the same fixed validation split as every other model.
Everything except ngram_range and C is kept the same as model_b_char_svm.py.
Results are written to experiments/model_b_tuning.txt.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import SEED, evaluate, get_split, load_data


def build_model(ngram_range, C):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=3,
            max_features=120_000,
            sublinear_tf=True,
        ),
        LinearSVC(C=C, dual="auto", random_state=SEED),
    )


def main():
    train, _, _ = load_data()
    train_part, val_part = get_split(train)

    rows = []
    for ngram_range in [(2, 4), (3, 5)]:
        for C in [0.1, 1.0, 10.0]:
            print(f"char ngram {ngram_range}, C={C}")
            result = evaluate(build_model(ngram_range, C), train_part, val_part, mode="raw")
            rows.append({
                "ngram_range": str(ngram_range),
                "C": C,
                "train_auc": round(result["train_auc"], 4),
                "val_auc": round(result["val_auc"], 4),
                "gap": round(result["train_auc"] - result["val_auc"], 4),
            })

    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    print(table.to_string(index=False))
    (HERE / "model_b_tuning.txt").write_text(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
