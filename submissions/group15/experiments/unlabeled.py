"""Q3. Run from group15 with python -m experiments.unlabeled."""

import pandas as pd

from common import clean_text, evaluate, get_scores, get_split, load_data
from experiments.model_b_char_svm import TEXT_MODE, build_model


MAX_PER_CLASS = 300


def select_pseudo_labels(unlabeled, margins):
    candidates = unlabeled.assign(margin=margins)
    human = candidates[candidates["margin"] < 0].nsmallest(MAX_PER_CLASS, "margin")
    ai = candidates[candidates["margin"] > 0].nlargest(MAX_PER_CLASS, "margin")

    count = min(len(human), len(ai))
    return pd.concat([
        human.head(count).assign(label=0),
        ai.head(count).assign(label=1),
    ], ignore_index=True)


def main():
    train, unlabeled, _ = load_data()
    train_part, val_part = get_split(train)

    print("Supervised baseline")
    teacher = build_model()
    baseline = evaluate(teacher, train_part, val_part, mode=TEXT_MODE)

    texts = unlabeled["text"].map(lambda text: clean_text(text, TEXT_MODE))
    margins = get_scores(teacher, texts)
    pseudo = select_pseudo_labels(unlabeled, margins)
    if pseudo.empty:
        raise ValueError("Need unlabeled predictions on both sides of the SVM boundary.")

    for label, rows in pseudo.groupby("label"):
        print(f"Pseudo-label {label}: {len(rows)} selected, min |margin| {rows['margin'].abs().min():.4f}")

    augmented = pd.concat([train_part, pseudo.drop(columns="margin")], ignore_index=True)
    print(f"Self-training: {len(train_part)} labeled + {len(pseudo)} pseudo-labeled")
    result = evaluate(build_model(), augmented, val_part, mode=TEXT_MODE)
    print(f"Change in validation AUROC: {result['val_auc'] - baseline['val_auc']:+.6f}")


if __name__ == "__main__":
    main()
