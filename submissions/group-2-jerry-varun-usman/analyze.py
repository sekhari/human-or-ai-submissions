"""Turn cached predictions into the report's tables and figures.

Q1.2 model comparison, Q2 learning-curve plot, Q3 pseudo-label table,
Q4 train/val gap + subgroups + FP/FN examples, Q5 ensemble comparison.
Writes results/report_tables.md, results/learning_curve.png, results/errors.csv.

The final configuration is picked among {deberta, deberta+char, deberta+char+weird}
by validation AUROC; ensembles use equal-weight rank averaging (nothing is fit on val).

Usage:  python analyze.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import RESULTS_DIR, Data, auc, load_preds, rank01, read_results, update_results, weird_feats

LENGTH_BINS = [0, 150, 250, 10_000]


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main():
    d = Data(); res = read_results(); out = []
    deb = load_preds("deberta_main")
    char = load_preds("char_tfidf_lr"); weird = load_preds("weird_char_lr")

    # ---- Q1.2 model comparison
    rows = []
    for key, v in res.items():
        if key.startswith("model:"):
            rows.append(dict(model=key[6:], train_auc=v["train_auc"], val_auc=v["val_auc"],
                             train_time_s=v["train_time_s"]))
    if rows:
        out += ["## Q1.2 Model comparison (fixed 2,000-row validation)", md_table(pd.DataFrame(rows)), ""]
    for key, v in res.items():
        if key.startswith("model:") and "cv_table" in v:
            out += [f"### Q1.3 grid: {key[6:]} (3-fold CV AUROC on training partition)",
                    md_table(pd.DataFrame([dict(params=str(r["params"]), cv_auc=r["cv_auc"]) for r in v["cv_table"]])),
                    f"Chosen: `{v['best_params']}`", ""]

    # ---- Q2 learning curve
    if "learning_curve" in res:
        lc = res["learning_curve"]
        lc_df = pd.DataFrame([dict(rows=int(k), **v) for k, v in lc.items()]).sort_values("rows")
        out += ["## Q2 Learning curve (DeBERTa-v3-small)", md_table(lc_df), ""]
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(lc_df.rows, lc_df.train_auc, "o-", label="Train AUROC")
        plt.plot(lc_df.rows, lc_df.val_auc, "s-", label="Validation AUROC")
        plt.xscale("log"); plt.xticks(lc_df.rows, [str(x) for x in lc_df.rows])
        plt.xlabel("Training rows (log scale)"); plt.ylabel("AUROC")
        plt.title("DeBERTa-v3-small learning curve"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=200)

    # ---- Q3 pseudo labels
    if "pseudo_label" in res:
        pl = res["pseudo_label"]
        out += ["## Q3 Unlabeled data (confidence pseudo-labeling)",
                md_table(pd.DataFrame([dict(method="Supervised only", val_auc=pl["val_auc_supervised"]),
                                       dict(method=f"+ {pl['n_pseudo']} pseudo-labels (p>={pl['threshold']})",
                                            val_auc=pl["val_auc_with_unlabeled"])])), ""]

    if deb is None:
        (RESULTS_DIR / "report_tables.md").write_text("\n".join(out)); print("\n".join(out)); return

    # ---- Q5 ensembles
    ens = {"deberta": deb["p_va"]}
    if char is not None:
        ens["deberta+char"] = (rank01(deb["p_va"]) + rank01(char["p_va"])) / 2
        if weird is not None:
            ens["deberta+char+weird"] = (rank01(deb["p_va"]) + rank01(char["p_va"]) + rank01(weird["p_va"])) / 3
    ens_auc = {k: auc(d.y_va, v) for k, v in ens.items()}
    final = max(ens_auc, key=ens_auc.get)
    update_results("ensemble_val_auc", ens_auc); update_results("final_config", final)
    out += ["## Q5 Ensemble (equal-weight rank average)",
            md_table(pd.DataFrame([dict(config=k, val_auc=v) for k, v in ens_auc.items()])),
            f"Final config by validation AUROC: **{final}**", ""]

    # ---- Q4.1 gap
    out += ["## Q4.1 Train vs validation (DeBERTa)",
            f"train AUROC {auc(d.y_tr, deb['p_tr']):.4f} | val AUROC {auc(d.y_va, deb['p_va']):.4f}", ""]

    # ---- Q4.2 subgroups
    nw = np.array([len(t.split()) for t in d.X_va]); W = weird_feats(d.X_va)
    has_weird = (W[:, 1] + W[:, 2] + W[:, 4]) > 0
    groups = {f"{lo + 1}-{hi} words": (nw > lo) & (nw <= hi) for lo, hi in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:])}
    groups["has weird chars"] = has_weird; groups["no weird chars"] = ~has_weird
    sg = pd.DataFrame([dict(group=g, count=int(m.sum()), ai_frac=float(d.y_va[m].mean()) if m.sum() else np.nan,
                            auc_deberta=auc(d.y_va[m], deb["p_va"][m]), auc_final=auc(d.y_va[m], ens[final][m]))
                       for g, m in groups.items()])
    update_results("subgroups", sg.to_dict("records"))
    out += ["## Q4.2 Subgroups", md_table(sg), ""]

    # ---- Q4.3 errors (threshold 0.5, DeBERTa)
    p, y = deb["p_va"], d.y_va
    info = pd.DataFrame({"id": d.va[d.id].values, "label": y, "p_ai": p, "n_words": nw, "weird": has_weird,
                         "snippet": [t[:300].replace("\n", " ") for t in d.X_va]})
    fp = info[y == 0].sort_values("p_ai", ascending=False).head(10).assign(
        error=lambda x: np.where(x.p_ai >= .5, "FP", "near-FP(<0.5)"))
    fn = info[y == 1].sort_values("p_ai").head(10).assign(
        error=lambda x: np.where(x.p_ai < .5, "FN", "near-FN(>=0.5)"))
    errs = pd.concat([fp, fn]); errs.to_csv(RESULTS_DIR / "errors.csv", index=False)
    n_fp, n_fn = int(((p >= .5) & (y == 0)).sum()), int(((p < .5) & (y == 1)).sum())
    update_results("errors_at_0.5", dict(n_fp=n_fp, n_fn=n_fn))
    out += [f"## Q4.3 Errors at threshold 0.5: {n_fp} FP, {n_fn} FN (top 10 each in results/errors.csv)",
            md_table(errs[["id", "error", "p_ai", "n_words", "weird"]]), ""]

    (RESULTS_DIR / "report_tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
