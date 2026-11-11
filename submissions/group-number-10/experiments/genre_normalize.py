"""Per-genre rank normalization: put every genre's scores on the same scale before ranking globally.

AUROC ranks all passages together. If a model shifts a whole genre toward "AI" (on test.csv it calls
59% of poems AI vs 48% of prose), human passages of that genre outrank AI passages of other genres.
Replacing each score with its percentile within its genre removes such shifts and keeps the order
inside each genre. Assumption: every genre is roughly class-balanced, as in train.csv (49-53% AI
for recipes, abstracts, and other prose).

    # labeled predictions (e.g. validation): reports AUROC before and after, overall and per genre
    python -m experiments.genre_normalize results/predictions/<run>_val.csv --texts data/train.csv
    # submission: writes a normalized copy
    python -m experiments.genre_normalize results/submissions/<file>.csv --texts data/test.csv --out <out>.csv
    # only separate one genre from the rest (e.g. poems vs everything else)
    python -m experiments.genre_normalize ... --only poetry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from predict import check_against_sample
from src.data import DATA_DIR
from src.genres import GENRES, genre


def genre_rank_normalize(scores: np.ndarray, genres: np.ndarray) -> np.ndarray:
    """Percentile rank of each score within its genre (ties averaged), in (0, 1]."""
    return pd.Series(scores).groupby(np.asarray(genres)).rank(pct=True).to_numpy()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scores", type=Path, help="id,label submission or id,label,prob_ai labeled predictions")
    p.add_argument("--texts", type=Path, required=True, help="CSV with id,text for the scored passages")
    p.add_argument("--out", type=Path, default=None, help="write the normalized id,label submission here")
    p.add_argument("--only", choices=GENRES, default=None, help="normalize this genre against all others pooled")
    a = p.parse_args(argv)

    df = pd.read_csv(a.scores).merge(pd.read_csv(a.texts)[["id", "text"]], on="id", how="left", validate="1:1")
    if df["text"].isna().any():
        raise SystemExit(f"{int(df['text'].isna().sum())} ids have no text in {a.texts}")
    labeled = "prob_ai" in df
    score = df["prob_ai" if labeled else "label"].to_numpy()
    df["genre"] = df["text"].map(genre)
    groups = df["genre"].to_numpy()
    if a.only:
        groups = np.where(groups == a.only, a.only, "rest")
    df["norm"] = genre_rank_normalize(score, groups)

    summary = df.assign(pred_ai=score > 0.5).groupby("genre").agg(n=("id", "size"), pred_ai=("pred_ai", "mean"))
    if labeled:
        y = df["label"].to_numpy()
        print(f"AUROC raw {roc_auc_score(y, score):.5f} -> genre-normalized {roc_auc_score(y, df['norm']):.5f}")
        summary["ai_rate"] = df.groupby("genre")["label"].mean()
        summary["auc"] = [roc_auc_score(g["label"], g["prob_ai"]) if g["label"].nunique() > 1 else np.nan
                          for _, g in df.groupby("genre")]
    print(summary.round(4).to_string())
    print(f"Spearman raw vs normalized: {pd.Series(score).corr(df['norm'], method='spearman'):.4f}")

    if a.out:
        out = pd.DataFrame({"id": df["id"], "label": df["norm"]})
        check_against_sample(out, DATA_DIR / "sample_submission.csv")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
