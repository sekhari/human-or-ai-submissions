"""Compare transformer variants and their rank-average blends (report Q5), write a submission.

Each variant is a cache prefix written by train_deberta.py: `deberta` (the default
DeBERTa-v3-small run) or a `--tag` such as `base512`. A variant needs
results/preds/<prefix>_main.npz (validation preds) to be scored, and
<prefix>_final_test.npz (test preds from the all-10k retrain) to be submitted.

Blends are rank averages with equal weights (or fixed --weights chosen in advance),
so nothing is fit on the validation set.

Usage:
  python blend.py                                   # score all variants found + all blends
  python blend.py --submit base512 deberta --out results/submission_blend.csv
  python blend.py --submit base512 lm --weights 0.7 0.3 --out results/submission_base_lm.csv
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from src.common import PRED_DIR, RESULTS_DIR, Data, auc, find_col, load_preds, rank01, update_results


def variants() -> list[str]:
    return sorted(p.name[: -len("_main.npz")] for p in PRED_DIR.glob("*_main.npz"))


def blend(arrs, weights=None):
    if len(arrs) == 1:
        return arrs[0]
    w = np.ones(len(arrs)) if weights is None else np.asarray(weights, dtype=float)
    return np.average([rank01(a) for a in arrs], axis=0, weights=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", nargs="*", default=None, help="variants to blend into the submission")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="one weight per --submit variant")
    ap.add_argument("--out", default=str(RESULTS_DIR / "submission_blend.csv"))
    args = ap.parse_args()
    d = Data()
    names = variants()
    val = {n: load_preds(f"{n}_main")["p_va"] for n in names}
    rows = []
    for k in range(1, len(names) + 1):
        for combo in combinations(names, k):
            rows.append(dict(config="+".join(combo), val_auc=auc(d.y_va, blend([val[n] for n in combo])),
                             has_test=all(load_preds(f"{n}_final_test") is not None for n in combo)))
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    print(df.to_string(index=False, float_format="%.4f"))
    update_results("blend_val_auc", {r["config"]: r["val_auc"] for r in rows})
    (RESULTS_DIR / "blend_table.md").write_text(
        "## Q5 Transformer variants and blends (equal-weight rank average, fixed validation)\n"
        + df.to_markdown(index=False, floatfmt=".4f") + "\n")

    if args.submit:
        if args.weights is not None and len(args.weights) != len(args.submit):
            raise SystemExit("--weights needs one value per --submit variant")
        cfg = "+".join(args.submit) + (f" w={args.weights}" if args.weights else "")
        v_auc = auc(d.y_va, blend([val[n] for n in args.submit], args.weights))
        print(f"submission config {cfg}: val AUROC {v_auc:.4f}")
        tests = [load_preds(f"{n}_final_test") for n in args.submit]
        missing = [n for n, t in zip(args.submit, tests) if t is None]
        if missing:
            raise SystemExit(f"no test preds for {missing}; run train_deberta.py --mode final for them")
        p = blend([t["p_test"] for t in tests], args.weights)
        test_id = find_col(d.test_df, [d.id, "id", "ID", "Id"])
        cols = d.sample_sub.columns if d.sample_sub is not None else [test_id, "label"]
        pd.DataFrame({cols[0]: d.test_df[test_id].values, cols[1]: p}).to_csv(args.out, index=False)
        update_results("blend_submission", dict(config=cfg, val_auc=v_auc, out=str(args.out)))
        print(f"wrote {args.out} ({len(p)} rows) from {cfg}")


if __name__ == "__main__":
    main()
