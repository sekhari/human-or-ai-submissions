"""Report experiments for the classical track, on the team's fixed 8,000 / 2,000 split.

  python experiments.py curve   --models stylo_gbm tfidf_lr   # Q2: 500 ... 8,000 nested training rows
  python experiments.py semisup --models tfidf_lr stylo_gbm   # Q3: self-training with unlabeled.csv
  python experiments.py errors  --model stylo_gbm             # Q4: subgroups + 5 FP / 5 FN [--show]

Each model uses the params chosen by its `python train.py --model <m>` run (results/<m>/metrics.json),
so hyperparameters stay fixed across sizes / methods. Results go to results/experiments/*.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from data import SEED, load_csv, train_val_split  # noqa: E402
from metrics import errors, length_bin_auroc, summary, verse_prose_auroc, word_count  # noqa: E402
from train import build, load_config  # noqa: E402

OUT = HERE / "results" / "experiments"


def chosen(model: str) -> tuple[dict, dict]:
    cfg = load_config(model)
    return cfg, json.loads((HERE / "results" / model / "metrics.json").read_text(encoding="utf-8"))["params"]


def nested_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Same rule as the pretrained track (src/data.subsample_n): shuffle each class once with `seed`,
    keep its first n * class-share rows, so smaller sets are subsets of larger ones."""
    idx = np.arange(len(y))
    if n >= len(y):
        return idx
    rng = np.random.default_rng(seed)
    keep = [rng.permutation(idx[y == c])[:int(round(n * (y == c).sum() / len(y)))] for c in np.unique(y)]
    return np.sort(np.concatenate(keep))


def fit_eval(cfg, params, X, y, Xv, yv, unl, device=None):
    m = build(cfg, params, SEED, device)
    if hasattr(m, "reset_cache"):
        m.reset_cache()
    t0 = time.time()
    m.fit(X, y, unlabeled_texts=unl)
    fit_s = time.time() - t0
    return m, {"train_auroc": summary(y, m.predict_proba(X))["auroc"], "val_auroc": summary(yv, m.predict_proba(Xv))["auroc"],
               "fit_time_s": round(fit_s, 1)}


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, lineterminator="\n", float_format="%.6g")
    print(f"wrote results/experiments/{name}")


def cmd_curve(args):
    tr, va = train_val_split(args.data_dir)
    unl = load_csv("unlabeled", args.data_dir)["text"].tolist()
    X, y = tr["text"].to_numpy(), tr["label"].to_numpy()
    rows = []
    for model in args.models:
        cfg, params = chosen(model)
        for n in (500, 1000, 2000, 5000, 8000):
            keep = nested_subset(y, n, SEED)
            _, r = fit_eval(cfg, params, X[keep].tolist(), y[keep], va["text"].tolist(), va["label"].to_numpy(), unl,
                            args.device)
            rows.append({"model": model, "train_rows": len(keep), **r})
            print(rows[-1], flush=True)
    write(pd.DataFrame(rows), "learning_curve.csv")


def cmd_semisup(args):
    """Self-training: fit on the 8,000 rows, label the most confident `frac` of unlabeled.csv (half per
    class, by rank), add them, refit; repeat for `rounds`. The pool is unlabeled.csv only."""
    tr, va = train_val_split(args.data_dir)
    unl = load_csv("unlabeled", args.data_dir)["text"].tolist()
    overlap = int(pd.Series(unl).str.strip().isin(set(va["text"].str.strip())).sum())
    X, y, Xv, yv = tr["text"].tolist(), tr["label"].to_numpy(), va["text"].tolist(), va["label"].to_numpy()
    rows = []
    for model in args.models:
        cfg, params = chosen(model)
        m, r = fit_eval(cfg, params, X, y, Xv, yv, unl, args.device)
        rows.append({"model": model, "method": "supervised", "rounds": 0, "pseudo_labels": 0, **r})
        for rounds in (1, 3):
            mm, Xa, ya, t_total = m, list(X), y.copy(), 0.0
            for _ in range(rounds):
                p = mm.predict_proba(unl)
                k = int(len(unl) * args.frac / 2)
                order = np.argsort(p)
                pos, neg = order[-k:], order[:k]
                Xa = list(X) + [unl[i] for i in pos] + [unl[i] for i in neg]
                ya = np.concatenate([y, np.ones(k, int), np.zeros(k, int)])
                mm, rr = fit_eval(cfg, params, Xa, ya, Xv, yv, unl, args.device)
                t_total += rr["fit_time_s"]
            rows.append({"model": model, "method": f"self-training ({args.frac:.0%} of unlabeled per round)",
                         "rounds": rounds, "pseudo_labels": 2 * k, "train_auroc": summary(y, mm.predict_proba(X))["auroc"],
                         "val_auroc": summary(yv, mm.predict_proba(Xv))["auroc"], "fit_time_s": round(t_total, 1),
                         "pseudo_label_p_min_ai": float(p[pos].min()), "pseudo_label_p_max_human": float(p[neg].max())})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["unlabeled_texts_identical_to_a_val_text"] = overlap
    write(df, "semi_supervised.csv")


def cmd_errors(args):
    tr, va = train_val_split(args.data_dir)
    pred = pd.read_csv(HERE / "results" / args.model / "val_pred.csv").set_index("id").loc[va["id"]]
    texts, yv, p = va["text"].tolist(), va["label"].to_numpy(), pred["pred"].to_numpy()
    lb = length_bin_auroc(texts, yv, p).assign(grouping="word count")
    vp = verse_prose_auroc(texts, yv, p).assign(grouping="line structure")
    write(pd.concat([lb, vp], ignore_index=True).assign(model=args.model), f"subgroups_{args.model}.csv")
    err = errors(va["id"], yv, p, threshold=0.5, k=5)
    t = va.set_index("id")["text"]
    err["words"] = word_count(t.loc[err["id"]])
    err["blank_lines"] = [s.count("\n\n") for s in t.loc[err["id"]]]
    err["lines"] = [len([ln for ln in s.splitlines() if ln.strip()]) for s in t.loc[err["id"]]]
    write(err.assign(model=args.model), f"fp_fn_{args.model}.csv")
    write(pd.DataFrame([{"model": args.model, "threshold": 0.5, "false_positives": int(((p >= 0.5) & (yv == 0)).sum()),
                         "false_negatives": int(((p < 0.5) & (yv == 1)).sum()), "n_val": len(yv)}]),
          f"confusion_{args.model}.csv")
    if args.show:  # printed only, never written to results/
        for _, r in err.iterrows():
            print(f"\n=== {r.error} id={r.id} P(AI)={r.p_ai:.3f} words={r.words}\n{t.loc[r.id][:1200]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["curve", "semisup", "errors"])
    ap.add_argument("--models", nargs="+", default=["stylo_gbm", "tfidf_lr"])
    ap.add_argument("--model", default="stylo_gbm")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
