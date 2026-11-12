"""Evaluate a style-only model and blend with Model B's stored predictions."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from common import SEED, get_split, load_data


def style_features(texts):
    rows = []
    for text in texts:
        words = re.findall(r"\b[\w']+\b", text)
        word_lengths = [len(word) for word in words] or [0]
        sentences = [s.split() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_lengths = [len(sentence) for sentence in sentences] or [0]
        n_chars = max(1, len(text))
        n_words = max(1, len(words))
        rows.append([
            len(text),
            len(words),
            len(set(word.lower() for word in words)) / n_words,
            np.mean(word_lengths),
            np.std(word_lengths),
            np.mean(sentence_lengths),
            np.std(sentence_lengths),
            sum(word.isupper() for word in words) / n_words,
            sum(char.isdigit() for char in text) / n_chars,
            *[text.count(char) / n_chars for char in ",:;!?'()-—/"],
        ])
    return np.array(rows)


def blend(char_scores, style_scores):
    return (0.75 * rankdata(char_scores) + 0.25 * rankdata(style_scores)) / len(char_scores)


def main():
    train, _, test = load_data()
    training, validation = get_split(train)
    features_train = style_features(train["text"])
    features_val = style_features(validation["text"])

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
    model.fit(features_train[train.index.isin(training.index)], training["label"])
    style_val = model.predict_proba(features_val)[:, 1]
    char_val = np.load("preds/model_b_val.npy")
    val_blend = blend(char_val, style_val)

    labels = validation["label"]
    char_auc = roc_auc_score(labels, char_val)
    blend_auc = roc_auc_score(labels, val_blend)
    print(f"Style validation: {roc_auc_score(labels, style_val):.6f}")
    print(f"75% character / 25% style rank blend validation: {blend_auc:.6f}")
    print(f"Blend delta: {blend_auc - char_auc:+.6f}")
    no_blank_line = ~validation["text"].str.contains(r"\n\s*\n")
    print(
        "No-blank-line validation subset: "
        f"char={roc_auc_score(labels[no_blank_line], char_val[no_blank_line]):.6f}, "
        f"blend={roc_auc_score(labels[no_blank_line], val_blend[no_blank_line]):.6f}"
    )

    if blend_auc > char_auc:
        model.fit(features_train, train["label"])
        style_test = model.predict_proba(style_features(test["text"]))[:, 1]
        char_test = np.load("preds/model_b_test.npy")
        scores = blend(char_test, style_test)
        sample = pd.read_csv("data/sample_submission.csv")
        assert len(scores) == len(test) == 3000 and np.isfinite(scores).all()
        assert test["id"].is_unique and set(test["id"]) == set(sample["id"])
        assert ((scores >= 0) & (scores <= 1)).all()
        output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("submission_char_style_blend.csv")
        test[["id"]].assign(label=scores).to_csv(output, index=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
