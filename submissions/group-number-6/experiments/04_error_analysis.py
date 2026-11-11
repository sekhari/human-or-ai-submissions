"""Q4: generalisation and error analysis for the selected model.

  * train vs validation AUROC (model fit on the 8,000-row partition)
  * validation AUROC by length bin (<=160, 161-240, >240 words; fixed before looking) and
    by pseudo-domain (KMeans topic clusters fit on the training partition only)
  * the 5 most confident false positives and false negatives at threshold 0.5, plus
    the stylometric features that differ most from the class average, saved for reading

Usage: python experiments/04_error_analysis.py [--config ensemble+norm]
Outputs: results/q4_error_analysis.json, results/q4_errors.md, figures/q4_subgroups.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, get_split  # noqa: E402
from src.features import STYLE_NAMES, style_vector  # noqa: E402
from src.models import SEED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

imp = import_module("05_improvement")

BINS = [(0, 160, "<=160 words"), (161, 240, "161-240 words"), (241, 10_000, ">240 words")]
K = 5


def auc_or_none(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def plot(out):
    groups = [(b["bin"], b["val_auc"], b["n"]) for b in out["length_bins"]] + \
             [(f"Topic C{c['cluster']}: {', '.join(c['top_terms'][:3])}", c["val_auc"], c["n"]) for c in out["clusters"]] + \
             [(g["group"], g["val_auc"], g["n"]) for g in out["structure"] if g["val_auc"] is not None]
    n_len, n_top = len(out["length_bins"]), len(out["clusters"])
    cols = ["#5c7a99"] * n_len + ["#e9a03b"] * n_top + ["#8a9a5b"] * (len(groups) - n_len - n_top)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [f"{g} (n={n})" for g, _, n in groups][::-1]
    vals = [v for _, v, _ in groups][::-1]
    ax.barh(labels, vals, color=cols[::-1])
    lo = min(vals) - 0.004
    for i, v in enumerate(vals):
        ax.text(v - 0.0002, i, f"{v:.4f}", va="center", ha="right", fontsize=8, color="white", fontweight="bold")
    ax.axvline(out["val_auc"], color="k", ls="--", lw=1)
    ax.text(out["val_auc"], len(vals) - 0.35, f" overall {out['val_auc']:.4f}", fontsize=8, ha="right", va="bottom")
    ax.set_xlim(lo, 1.0005)
    ax.set_xlabel("Validation AUROC (final model trained on 8,000 rows)")
    ax.set_title("Validation AUROC by length bin, topic cluster and format", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q4_subgroups.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ensemble+norm")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results")
    args = ap.parse_args()
    if args.plot_only:
        plot(json.loads((ROOT / "results" / "q4_error_analysis.json").read_text()))
        return
    tr, va = get_split()
    X, y = list(tr.text), tr.label.to_numpy()
    yv = va.label.to_numpy()
    m = imp.make(args.config).fit(X, y)
    p_tr = m.predict_proba(X)[:, 1]
    p_va = m.predict_proba(list(va.text))[:, 1]
    out = {"config": args.config, "train_auc": roc_auc_score(y, p_tr), "val_auc": roc_auc_score(yv, p_va),
           "val_accuracy@0.5": float(((p_va >= 0.5) == yv).mean())}
    if hasattr(m, "member_scores"):
        mtr, mva = m.member_scores(X), m.member_scores(list(va.text))
        out["members"] = {n: {"train_auc": roc_auc_score(y, mtr[n]), "val_auc": roc_auc_score(yv, mva[n])} for n in mtr}
    print(json.dumps(out, indent=1), flush=True)

    # length bins
    wc = va.text.str.split().str.len().to_numpy()
    out["length_bins"] = []
    for lo, hi, name in BINS:
        s = (wc >= lo) & (wc <= hi)
        out["length_bins"].append({"bin": name, "n": int(s.sum()), "ai_frac": float(yv[s].mean()),
                                   "val_auc": auc_or_none(yv[s], p_va[s])})
    # pseudo-domains: fit on training partition only, assign validation rows
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True).fit(X)
    svd = TruncatedSVD(100, random_state=SEED).fit(vec.transform(X))

    def emb(t):
        z = svd.transform(vec.transform(t))
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    km = KMeans(K, n_init=10, random_state=SEED).fit(emb(X))
    cv = km.predict(emb(list(va.text)))
    terms = np.array(vec.get_feature_names_out())
    Xt = vec.transform(X)
    out["clusters"] = []
    for c in range(K):
        top = terms[np.argsort(-np.asarray(Xt[km.labels_ == c].mean(axis=0)).ravel())[:8]].tolist()
        s = cv == c
        out["clusters"].append({"cluster": c, "top_terms": top, "n": int(s.sum()), "ai_frac": float(yv[s].mean()),
                                "val_auc": auc_or_none(yv[s], p_va[s])})
    # structural subgroup: passages with line breaks (lists, recipes, poems, code) vs prose
    nl = va.text.str.contains("\n").to_numpy()
    out["structure"] = [{"group": "has line breaks", "n": int(nl.sum()), "ai_frac": float(yv[nl].mean()),
                         "val_auc": auc_or_none(yv[nl], p_va[nl])},
                        {"group": "single block of prose", "n": int((~nl).sum()), "ai_frac": float(yv[~nl].mean()),
                         "val_auc": auc_or_none(yv[~nl], p_va[~nl])}]
    bl = va.text.str.contains("\n\n").to_numpy()
    out["structure"] += [{"group": "has blank line", "n": int(bl.sum()), "ai_frac": float(yv[bl].mean()),
                          "val_auc": auc_or_none(yv[bl], p_va[bl])},
                         {"group": "no blank line", "n": int((~bl).sum()), "ai_frac": float(yv[~bl].mean()),
                          "val_auc": auc_or_none(yv[~bl], p_va[~bl])}]
    if hasattr(m, "member_scores"):
        out["members_no_blank_line"] = {n: roc_auc_score(yv[~bl], s[~bl]) for n, s in mva.items()}
    print(json.dumps({k: out[k] for k in ["length_bins", "clusters", "structure"]}, indent=1), flush=True)

    # errors
    S = np.array([style_vector(t) for t in va.text])
    mu = {c: S[yv == c].mean(axis=0) for c in (0, 1)}
    sd = S.std(axis=0) + 1e-9
    fp = np.where((yv == 0) & (p_va >= 0.5))[0]
    fn = np.where((yv == 1) & (p_va < 0.5))[0]
    out["n_fp"], out["n_fn"] = int(len(fp)), int(len(fn))
    errs = []
    md = [f"# Validation errors ({args.config}, threshold 0.5)\n",
          f"FP (human predicted AI): {len(fp)}; FN (AI predicted human): {len(fn)}\n"]
    for kind, idx, order, true_c in [("FP", fp, -1, 0), ("FN", fn, 1, 1)]:
        idx = idx[np.argsort(order * p_va[idx])][:8]
        for i in idx:
            z = (S[i] - mu[true_c]) / sd
            top = [f"{STYLE_NAMES[j]}={S[i, j]:.3g} (z={z[j]:+.1f})" for j in np.argsort(-np.abs(z))[:4]]
            rec = {"id": int(va.id[i]), "type": kind, "p_ai": float(p_va[i]), "words": int(wc[i]),
                   "cluster": int(cv[i]), "unusual_style": top}
            if hasattr(m, "member_scores"):
                rec["members"] = {n: float(s[i]) for n, s in mva.items()}
            errs.append(rec)
            md.append(f"\n## {kind} id={rec['id']} p={rec['p_ai']:.3f} words={rec['words']} cluster={rec['cluster']}\n")
            if "members" in rec:
                md.append("members: " + ", ".join(f"{n}={v:.2f}" for n, v in rec["members"].items()) + "\n")
            md.append("unusual: " + "; ".join(top) + "\n\n")
            md.append("> " + va.text[i][:700].replace("\n", "\n> ") + "\n")
    out["errors"] = errs
    (ROOT / "results" / "q4_error_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    (ROOT / "results" / "q4_errors.md").write_text("".join(md))

    plot(out)

if __name__ == "__main__":
    main()
