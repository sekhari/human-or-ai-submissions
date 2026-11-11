"""Rank-average blend of the v4 CatBoost and the fine-tuned transformer (RoBERTa-base), chosen on LCO OOF.

  python submissions/Group_Number_19/yuchen_blend.py \
      --cat-oof results/yuchen/v4_bino/lco_v2+lm+bino+lr+pseudo.csv \
      --deb-dir results/yuchen/roberta/rob_a results/yuchen/roberta/rob_b \
      --cat-test results/yuchen/v4_bino/submission_v4_unl_target.csv
--deb-dir takes one or more yuchen_deberta.py output folders (the RoBERTa run was split
into two GPU jobs: folds 0-3 and folds 4-7 + full fit); their out-of-fold predictions are
merged. Reports test-weighted / non-blank LCO AUC for the transformer alone and for blend
weights w in [0, 1] (w = transformer share), and writes submission_blend_w{w}.csv for the
best w (the final Kaggle submission is submission_blend_w0.1.csv, public AUROC 0.99162).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]


def rank(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def merge_runs(dirs: list[Path], test_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the per-fold OOF predictions of several yuchen_deberta.py runs; take the full-fit test file."""
    oofs = [pd.read_csv(d / "oof_lco.csv") for d in dirs]
    oof = oofs[0].copy()
    for o in oofs[1:]:
        assert (o["id"].to_numpy() == oof["id"].to_numpy()).all()
        oof["pred"] = oof["pred"].fillna(o["pred"])
    tests = [d / test_name for d in dirs if (d / test_name).exists()]
    assert len(tests) == 1, f"expected exactly one {test_name} among {dirs}"
    return oof, pd.read_csv(tests[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-oof", type=Path, required=True)
    ap.add_argument("--deb-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--cat-test", type=Path, required=True)
    ap.add_argument("--deb-test", default="test_full.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "blend")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    cl = pd.read_csv(HERE / "clusters.csv")
    sh = pd.crosstab(cl["cluster"], cl["split"], normalize="columns")
    c = pd.read_csv(a.cat_oof).set_index("id").loc[tr["id"]]
    oof_all, dt = merge_runs(a.deb_dir, a.deb_test)
    d = oof_all.set_index("id").loc[tr["id"]]
    assert d["pred"].notna().all(), "transformer OOF incomplete"
    y = tr["label"].to_numpy()
    w_test = c["cluster"].map(sh["test"] / sh["train"]).to_numpy()
    blank = tr["text"].str.contains(r"\n[ \t]*\n").to_numpy()
    # rank within each held-out cluster's model is not comparable across folds for a pooled AUC,
    # so blend probabilities' ranks globally (both are calibrated-ish probabilities per fold)
    rc, rd = rank(c["pred"].to_numpy()), rank(d["pred"].to_numpy())
    rows = []
    for w in np.round(np.linspace(0, 1, 11), 2):
        p = (1 - w) * rc + w * rd
        rows.append(dict(w_deberta=w, weighted_auc=roc_auc_score(y, p, sample_weight=w_test),
                         nonblank_auc=roc_auc_score(y[~blank], p[~blank], sample_weight=w_test[~blank]),
                         pooled_auc=roc_auc_score(y, p)))
    R = pd.DataFrame(rows)
    print(R.round(5).to_string(index=False))
    print("Transformer per-cluster AUC:", {int(k): round(roc_auc_score(g["label"], g["pred"]), 5)
                                       for k, g in d.groupby("cluster") if g["label"].nunique() > 1})
    R.to_csv(a.out / "blend_lco.csv", index=False)
    best = R.loc[R["weighted_auc"].idxmax(), "w_deberta"]
    ct = pd.read_csv(a.cat_test)
    assert (ct["id"].to_numpy() == dt["id"].to_numpy()).all()
    for w in sorted({best, 0.3, 0.5}):
        p = (1 - w) * rank(ct["label"].to_numpy()) + w * rank(dt["label"].to_numpy())
        pd.DataFrame({"id": ct["id"], "label": p}).to_csv(a.out / f"submission_blend_w{w:.1f}.csv", index=False)
    print("best w_deberta on LCO:", best)


if __name__ == "__main__":
    main()
