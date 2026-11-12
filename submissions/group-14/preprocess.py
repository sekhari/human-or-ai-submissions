"""Turn the Kaggle CSVs into TF-IDF feature matrices.

Example:
    python preprocess.py --train data/train.csv --test data/test.csv \
        --unlabeled data/unlabeled.csv --out-dir features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from features import build_vectorizer

TEXT_CANDIDATES = ("text", "generation", "passage", "content")
LABEL_CANDIDATES = ("label", "is_ai", "generated", "target")
ID_CANDIDATES = ("id", "ID", "row_id")


def pick_column(df: pd.DataFrame, override: str | None, candidates: tuple[str, ...], what: str) -> str | None:
    if override:
        if override not in df.columns:
            raise SystemExit(f"--{what}-col '{override}' not in columns {list(df.columns)}")
        return override
    return next((c for c in candidates if c in df.columns), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TF-IDF features")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--unlabeled", type=Path, help="Extra text used only to fit the vocabulary/IDF")
    parser.add_argument("--out-dir", type=Path, default=Path("features"))
    parser.add_argument("--text-col")
    parser.add_argument("--label-col")
    parser.add_argument("--id-col")
    parser.add_argument("--word-max-features", type=int, default=100_000)
    parser.add_argument("--char-max-features", type=int, default=200_000)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    text_col = pick_column(train, args.text_col, TEXT_CANDIDATES, "text")
    label_col = pick_column(train, args.label_col, LABEL_CANDIDATES, "label")
    if text_col is None or label_col is None:
        raise SystemExit(f"Could not find text/label columns in {list(train.columns)}; pass --text-col/--label-col")
    print(f"train: {len(train):,} rows, text='{text_col}', label='{label_col}'")
    print(f"label distribution:\n{train[label_col].value_counts().to_string()}")

    fit_texts = train[text_col].tolist()
    if args.unlabeled:
        unlabeled = pd.read_csv(args.unlabeled)
        fit_texts += unlabeled[pick_column(unlabeled, args.text_col, TEXT_CANDIDATES, "text")].tolist()
        print(f"unlabeled: {len(unlabeled):,} rows added to the vocabulary/IDF fit")

    vectorizer = build_vectorizer(args.word_max_features, args.char_max_features, args.min_df)
    vectorizer.fit(fit_texts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X_train = vectorizer.transform(train[text_col]).tocsr()
    sp.save_npz(args.out_dir / "X_train.npz", X_train)
    np.save(args.out_dir / "y_train.npy", train[label_col].to_numpy())

    n_word = len(vectorizer.transformer_list[0][1].vocabulary_)
    n_char = len(vectorizer.transformer_list[1][1].vocabulary_)
    meta = {"text_col": text_col, "label_col": label_col, "n_word_features": n_word,
            "n_char_features": n_char, "X_train_shape": list(X_train.shape)}

    if args.test:
        test = pd.read_csv(args.test)
        X_test = vectorizer.transform(test[pick_column(test, args.text_col, TEXT_CANDIDATES, "text")]).tocsr()
        sp.save_npz(args.out_dir / "X_test.npz", X_test)
        id_col = pick_column(test, args.id_col, ID_CANDIDATES, "id")
        ids = test[id_col] if id_col else pd.Series(range(len(test)), name="id")
        ids.to_frame(name=id_col or "id").to_csv(args.out_dir / "test_ids.csv", index=False)
        meta["X_test_shape"] = list(X_test.shape)

    joblib.dump(vectorizer, args.out_dir / "vectorizer.joblib")
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"word features: {n_word:,} | char features: {n_char:,}")
    print(f"X_train: {X_train.shape}, nnz/row = {X_train.nnz / X_train.shape[0]:.0f}")
    if "X_test_shape" in meta:
        print(f"X_test:  {tuple(meta['X_test_shape'])}")
    print(f"saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
