"""Q5: does a low-weight blend help Usman's base512?

Baseline : DeBERTa-v3-base (base512) alone.
Improved : rank-average of base512 and the LM-feature model
           (gradient boosting, depth 3, lr 0.1, 300 trees).
LM weights are fixed in advance (0.1, 0.2, 0.3). Nothing is
tuned on the validation set.

Alignment: Usman's *_main.npz hold p_va in the row order of
train_test_split(np.arange(n), stratify=label, random_state=42).
We rebuild that order and check each file's validation AUROC
against his results.json before blending anything.

Usage (run inside lm_stack, data in ./data, features in
./features):
  python ensemble.py --npz-dir /path/to/folder/with/npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
VAL_SIZE = 2000
WEIGHTS = [0.1, 0.2, 0.3]
GB = dict(max_depth=3, learning_rate=0.1, n_estimators=300)
LM = [
    "base_lp_mean", "base_lp_std",
    "instr_lp_mean", "instr_lp_std",
    "base_logrank_mean", "instr_logrank_mean",
    "base_entropy_mean", "instr_entropy_mean",
    "bino_ppl", "bino_xppl", "binoculars",
]


def rank01(p):
    return (rankdata(p) - 1) / (len(p) - 1)


def lm_matrix(ids, feats, what):
    """LM features in the row order of ids; fail on any gap."""
    out = ids.to_frame("id").merge(
        feats[["id"] + LM], on="id", how="left",
        validate="one_to_one")
    if out[LM].isna().any().any():
        raise ValueError(f"{what}: missing LM features")
    return out[LM].to_numpy(dtype=float)


def fit_gb(X, y):
    sc = StandardScaler().fit(X)
    m = GradientBoostingClassifier(random_state=SEED, **GB)
    m.fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",
                    default=os.path.join(HERE, "data"))
    ap.add_argument("--features-dir",
                    default=os.path.join(HERE, "features"))
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out-dir",
                    default=os.path.join(HERE, "blend"))
    ap.add_argument("--expect-base512", type=float,
                    default=0.9943)
    ap.add_argument("--expect-deberta", type=float,
                    default=0.9927)
    ap.add_argument("--tol", type=float, default=0.0005)
    a = ap.parse_args()

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    y = train["label"].to_numpy()
    tr, va = train_test_split(
        np.arange(len(train)), test_size=VAL_SIZE,
        stratify=train["label"], random_state=SEED)

    def load(name, key):
        path = os.path.join(a.npz_dir, name + ".npz")
        return np.load(path)[key]

    base_va = load("base512_main", "p_va")
    base_te = load("base512_final_test", "p_test")
    deb_va = load("deberta_main", "p_va")
    if len(base_te) != len(test):
        raise SystemExit("test preds and test.csv differ in length")

    # 1. alignment check against Usman's own numbers
    checks = [("base512", base_va, a.expect_base512),
              ("deberta", deb_va, a.expect_deberta)]
    for name, p, want in checks:
        got = roc_auc_score(y[va], p)
        print(f"align {name}: val AUROC {got:.4f} "
              f"(his results.json: {want:.4f})")
        if abs(got - want) > a.tol:
            raise SystemExit(f"{name} does not line up; stop")

    # 2. LM model: 8,000-row fit for validation, all 10,000 for test
    f_tr = pd.read_parquet(
        os.path.join(a.features_dir, "train_qwen0.5B.parquet"))
    f_te = pd.read_parquet(
        os.path.join(a.features_dir, "test_qwen0.5B.parquet"))
    X = lm_matrix(train["id"], f_tr, "train")
    X_te = lm_matrix(test["id"], f_te, "test")
    lm_va = fit_gb(X[tr], y[tr])(X[va])
    lm_te = fit_gb(X, y)(X_te)
    print(f"LM model val AUROC {roc_auc_score(y[va], lm_va):.4f}"
          " (train.py seed-42 run: about 0.9700)")

    # 3. blends, weights fixed in advance
    rows = []
    for w in [0.0] + WEIGHTS:
        p = (1 - w) * rank01(base_va) + w * rank01(lm_va)
        rows.append({"lm_weight": w,
                     "val_auroc": roc_auc_score(y[va], p)})
    table = pd.DataFrame(rows)
    print("\n" + table.round(4).to_string(index=False))

    # 3b. paired bootstrap: blend (LM weight 0.2) minus base512 alone.
    # Same resampled rows for both models; ranks are computed once on the
    # full val set, as in the table above. Print only.
    yv = y[va]
    blend_va = 0.8 * rank01(base_va) + 0.2 * rank01(lm_va)
    brng = np.random.default_rng(SEED)
    diffs = np.empty(1000)
    for i in range(1000):
        idx = brng.integers(0, len(yv), len(yv))
        diffs[i] = (roc_auc_score(yv[idx], blend_va[idx])
                    - roc_auc_score(yv[idx], base_va[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nPaired bootstrap (1000 resamples, seed {SEED}), "
          "LM weight 0.2 minus base512 alone, val AUROC:\n"
          f"  mean {diffs.mean():.4f}, 2.5% {lo:.4f}, 97.5% {hi:.4f}")

    # 3c. save val predictions for LM weight 0.2 (rows in val order)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"id": train["id"].iloc[va].to_numpy(),
                  "label": yv,
                  "base512": base_va,
                  "lm": lm_va,
                  "blend": blend_va}).to_csv(
        os.path.join(a.out_dir, "val_preds_w0.2.csv"), index=False)

    # 3d. top-1000 cutoff: the 1,000 highest-scoring val rows are called AI,
    # the rest human. Print only.
    def top_half_wrong(score):
        pred_ai = np.zeros(len(score), dtype=bool)
        pred_ai[np.argsort(-score, kind="stable")[:len(score) // 2]] = True
        return pred_ai, pred_ai != (yv == 1)

    print("\nTop-1000 cutoff (highest 1,000 val scores called AI):")
    wrong_by = {}
    for name, s in [("base512", base_va), ("LM", lm_va), ("blend w0.2", blend_va)]:
        pred_ai, wrong = top_half_wrong(s)
        wrong_by[name] = wrong
        print(f"  {name:<11} FP {int((pred_ai & (yv == 0)).sum()):>3}  "
              f"FN {int((~pred_ai & (yv == 1)).sum()):>3}  "
              f"errors {int(wrong.sum()):>3}")
    print("  errors shared by base512 and LM: "
          f"{int((wrong_by['base512'] & wrong_by['LM']).sum())}")

    # 4. do the models make different mistakes? (0.5 cutoff)
    wrong_b = (base_va >= 0.5) != (y[va] == 1)
    wrong_l = (lm_va >= 0.5) != (y[va] == 1)
    print(f"\nval errors at 0.5: base512 {wrong_b.sum()}, "
          f"LM {wrong_l.sum()}, both {(wrong_b & wrong_l).sum()}")
    print("Spearman base512 vs LM: val "
          f"{spearmanr(base_va, lm_va)[0]:.3f}, test "
          f"{spearmanr(base_te, lm_te)[0]:.3f}")

    # 5. write one submission file per weight (gitignored *.csv)
    os.makedirs(a.out_dir, exist_ok=True)
    table.to_csv(os.path.join(a.out_dir, "blend_val.csv"),
                 index=False)
    for w in WEIGHTS:
        p = (1 - w) * rank01(base_te) + w * rank01(lm_te)
        sub = pd.DataFrame({"id": test["id"],
                            "label": np.clip(p, 0, 1)})
        assert len(sub) == len(test) and sub["id"].is_unique
        assert sub["label"].between(0, 1).all()
        out = os.path.join(a.out_dir, f"submission_w{w}.csv")
        sub.to_csv(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    main()