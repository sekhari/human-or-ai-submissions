"""Unseen-domain experiment: how much does AUROC drop on a genre absent from training?

The test set contains domains the labeled data lacks (e.g. ~19% poems in test.csv vs ~1-2% in train.csv),
which cannot be measured without test labels. This simulates it inside train.csv with recipes, a
large, reliably detectable genre (~19% of train, balanced labels):

    eval sets (held out of both conditions): 1,000 recipes and 1,000 non-recipes
    --condition holdout   train on N non-recipes                      (recipes never seen)
    --condition control   train on N rows from all remaining passages (recipes at their natural share)

Both conditions train on the same number of rows with the same config; the last epoch is kept
(no early stopping, so no evaluation passage influences training). The gap between conditions on the
recipe eval set is the cost of an unseen domain; the non-recipe eval set checks that it is genre-specific.

    python -m experiments.genre_holdout --condition holdout --model-name models/deberta-v3-large \
        --lr 1e-5 --epochs 3 --warmup-ratio 0.1 --newline-marker "[NL]"

Outputs: results/genre_holdout/<run>.json (AUROC and errors per eval set) and <run>_preds.csv.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.distilbert_train import run_name_for
from src.classifiers.distilbert import DistilBertClassifier, DistilBertConfig
from src.data import DATA_DIR, RESULTS_DIR, load_labeled
from src.evaluate import Timer
from src.genres import is_recipe

N_EVAL = 1000
SPLIT_SEED = 42


def holdout_split(is_genre: np.ndarray, condition: str, train_size: int,
                  n_eval: int = N_EVAL, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Row indices for the genre eval set, the other-genre eval set, and the training rows of a condition.

    Eval sets are drawn first and are identical for both conditions.
    """
    rng = np.random.default_rng(seed)
    genre, other = rng.permutation(np.flatnonzero(is_genre)), rng.permutation(np.flatnonzero(~is_genre))
    eval_genre, eval_other = genre[:n_eval], other[:n_eval]
    pool = other[n_eval:] if condition == "holdout" else rng.permutation(np.concatenate([genre[n_eval:], other[n_eval:]]))
    if len(pool) < train_size:
        raise ValueError(f"only {len(pool)} training rows available for {condition}, need {train_size}")
    return {"eval_genre": eval_genre, "eval_other": eval_other, "train": np.sort(pool[:train_size])}


def main(argv=None) -> None:
    d = DistilBertConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=("holdout", "control"), required=True)
    p.add_argument("--train-size", type=int, default=7000)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--newline-marker", default=None)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for forward passes; weights stay fp32")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test: N eval rows per set and N training rows")
    a = p.parse_args(argv)

    data = load_labeled(DATA_DIR / "train.csv")
    texts, labels = np.asarray(data.texts, dtype=object), data.labels
    recipe = np.array([is_recipe(t) for t in texts])
    n_eval, train_size = (a.limit, a.limit) if a.limit else (N_EVAL, a.train_size)
    idx = holdout_split(recipe, a.condition, train_size, n_eval=n_eval)

    cfg = DistilBertConfig(model_name=a.model_name, max_length=a.max_length, learning_rate=a.lr,
                           batch_size=a.batch_size, epochs=a.epochs, seed=a.seed,
                           newline_marker=a.newline_marker, warmup_ratio=a.warmup_ratio, bf16=a.bf16)
    run = f"{run_name_for(cfg)}_recipe-{a.condition}_n{train_size}" + (f"_smoke{a.limit}" if a.limit else "")
    tr = idx["train"]
    print(f"[{run}] train={len(tr)} (recipes {int(recipe[tr].sum())}) eval recipe={len(idx['eval_genre'])} "
          f"eval other={len(idx['eval_other'])} | recipes in train.csv: {int(recipe.sum())}", flush=True)

    clf = DistilBertClassifier(cfg, device=a.device)
    with Timer() as t:
        clf.fit(list(texts[tr]), labels[tr], log=lambda s: print(s, flush=True))
    if clf.device.type == "cuda":
        torch.cuda.empty_cache()

    out_dir = RESULTS_DIR / "genre_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, preds = {"run": run, "condition": a.condition, "config": asdict(cfg), "n_train": len(tr),
                     "n_train_recipes": int(recipe[tr].sum()), "train_time_seconds": round(t.seconds, 1),
                     "history": clf.history}, []
    for name in ("eval_genre", "eval_other"):
        rows = idx[name]
        probs = clf.predict_proba(list(texts[rows]))
        y = labels[rows]
        key = "recipe" if name == "eval_genre" else "non_recipe"
        result[f"{key}_auc"] = float(roc_auc_score(y, probs))
        result[f"{key}_errors"] = int(((probs > 0.5) != (y == 1)).sum())
        result[f"{key}_n"] = len(rows)
        preds.append(pd.DataFrame({"id": data.ids[rows], "eval_set": key, "label": y, "prob_ai": probs}))
    (out_dir / f"{run}.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.concat(preds).to_csv(out_dir / f"{run}_preds.csv", index=False)
    print(f"[{run}] recipe_auc={result['recipe_auc']:.5f} ({result['recipe_errors']} errors) "
          f"non_recipe_auc={result['non_recipe_auc']:.5f} ({result['non_recipe_errors']} errors) "
          f"time={t.seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
