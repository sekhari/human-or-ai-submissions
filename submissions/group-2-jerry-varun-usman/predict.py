"""Write the Kaggle submission.csv.

Uses the cached DeBERTa test predictions from `train_deberta.py --mode final`
and, if the chosen final config is an ensemble, refits the character TF-IDF
(and weird-char) model on all 10,000 labeled rows with its CV-selected
hyperparameters, then rank-averages.

Usage:  python predict.py --out submission.csv [--config deberta+char]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from baselines import char_pipe, weird_pipe
from src.common import Data, find_col, load_preds, rank01, read_results, weird_feats


def refit(pipe, best_params, X, y, Xt):
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best_params.items()}
    pipe.set_params(**params); pipe.fit(X, y)
    return pipe.predict_proba(Xt)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--config", default=None, help="override results.json final_config")
    args = ap.parse_args()
    d = Data(); res = read_results()
    config = args.config or res.get("final_config", "deberta")
    deb = load_preds("deberta_final_test")
    if deb is None:
        raise SystemExit("Run `python train_deberta.py --mode final` first.")
    X, y, Xt = d.full[d.text].values, d.full[d.label].values, d.X_test
    p = rank01(deb["p_test"]) if config != "deberta" else deb["p_test"]
    parts = [p]
    if config in ("deberta+char", "deberta+char+weird"):
        parts.append(rank01(refit(char_pipe(), res["model:char_tfidf_lr"]["best_params"], X, y, Xt)))
    if config == "deberta+char+weird":
        parts.append(rank01(refit(weird_pipe(), res["model:weird_char_lr"]["best_params"],
                                  weird_feats(X), y, weird_feats(Xt))))
    p_final = np.clip(np.mean(parts, axis=0), 0, 1)
    test_id = find_col(d.test_df, [d.id, "id", "ID", "Id"])
    id_col = d.sample_sub.columns[0] if d.sample_sub is not None else test_id
    pred_col = d.sample_sub.columns[1] if d.sample_sub is not None else "label"
    pd.DataFrame({id_col: d.test_df[test_id].values, pred_col: p_final}).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(p_final)} rows) using config={config}")


if __name__ == "__main__":
    main()
