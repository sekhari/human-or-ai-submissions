"""Compare transformer variants and their rank-average blends (report Q5), write a submission.

Each variant is a cache prefix written by train_deberta.py: `deberta` (the default
DeBERTa-v3-small run) or a `--tag` such as `base512`. A variant needs
results/preds/<prefix>_main.npz (validation preds) to be scored, and
<prefix>_final_test.npz (test preds from the all-10k retrain) to be submitted.

Blends are equal-weight rank averages, so nothing is fit on the validation set.

Usage:
  python blend.py                                   # score all variants found + all blends
  python blend.py --submit base512 deberta --out results/submission_blend.csv
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from src.common import PRED_DIR, RESULTS_DIR, Data, auc, find_col, load_preds, rank01, update_results


def variants() -> list[str]:
    return sorted(p.name[: -len("_main.npz")] for p in PRED_DIR.glob("*_main.npz"))


def blend(arrs):
    return arrs[0] if len(arrs) == 1 else np.mean([rank01(a) for a in arrs], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", nargs="*", default=None, help="variants to blend into the submission")
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
        tests = [load_preds(f"{n}_final_test") for n in args.submit]
        missing = [n for n, t in zip(args.submit, tests) if t is None]
        if missing:
            raise SystemExit(f"no test preds for {missing}; run train_deberta.py --mode final for them")
        p = blend([t["p_test"] for t in tests])
        test_id = find_col(d.test_df, [d.id, "id", "ID", "Id"])
        cols = d.sample_sub.columns if d.sample_sub is not None else [test_id, "label"]
        pd.DataFrame({cols[0]: d.test_df[test_id].values, cols[1]: p}).to_csv(args.out, index=False)
        update_results("blend_submission", dict(config="+".join(args.submit), out=str(args.out)))
        print(f"wrote {args.out} ({len(p)} rows) from {'+'.join(args.submit)}")


if __name__ == "__main__":
    main()
