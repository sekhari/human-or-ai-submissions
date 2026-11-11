"""Q5: robustness-oriented improvement.

Observation from Q1: the best single model (C, stylometric GBM, val AUROC 0.997) leans on
formatting cues -- no human training passage contains a blank line, 66% of AI passages
do -- and on character statistics (non-ASCII ratio, zero-width count, spacing) that
adversarial edits manipulate directly. The test set is described as mixing unfamiliar
domains/generators and adversarially edited AI passages, which validation does not contain.

Hypothesis: an equal-weight logit-average ensemble of three different views (word
n-grams, character n-grams, stylometry) on Unicode-normalised text keeps near-baseline
clean accuracy but degrades less when formatting cues are removed, when text is
adversarially edited, and when the topic is unseen.

Configurations (same 8,000 training rows, member hyperparameters fixed from Q1 CV):
  B_char_lr       best n-gram model (reference)
  C_style_gbm     baseline = best Q1 model, raw text
  ensemble        LogitAverage(A word-LR, B char-LR, C style-GBM), raw text
  ensemble+norm   same ensemble, normalised text (NFKC, zero-width removal, homoglyph undo)

Evaluation (nothing here touches test.csv; no setting is tuned on these numbers):
  clean_val       fixed 2,000-row validation set
  format_stripped validation with every blank line collapsed to one newline and runs of
                  spaces collapsed, in BOTH classes (simulates sources whose formatting
                  differs from training)
  attack_ai_only  every AI validation passage gets one random synthetic attack
  attack_both     every passage (human and AI) gets one random attack
  topic_shift     leave-one-cluster-out inside the 8,000-row training partition (KMeans k=5
                  on word TF-IDF+SVD); mean AUROC over held-out clusters

Usage: python experiments/05_improvement.py [--skip-shift]
Outputs: results/q5_improvement.json, figures/q5_robustness.png
"""

import argparse
import json
import re
import sys
import time
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
from src.ensemble import LogitAverage  # noqa: E402
from src.models import SEED, build  # noqa: E402
from src.perturb import perturb_texts  # noqa: E402


def q1():
    return json.loads((ROOT / "results" / "q1_comparison.json").read_text())


def params(name):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in q1()[name]["params"].items()}


def make(cfg, best_single=None):
    norm = cfg.endswith("+norm")
    if not cfg.startswith("ensemble"):
        name = cfg.replace("+norm", "")
        return build(name, normalize=norm, **params(name))
    members = [(n, build(n, normalize=norm, **params(n))) for n in ["A_word_lr", "B_char_lr", "C_style_gbm"]]
    return LogitAverage(members)


def pseudo_domains(texts, k=5):
    vec = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True)
    Z = vec.fit_transform(texts)
    Z = TruncatedSVD(100, random_state=SEED).fit_transform(Z)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    terms = np.array(vec.get_feature_names_out())
    # describe clusters by top mean TF-IDF terms
    Zt = vec.transform(texts)
    desc = {}
    for c in range(k):
        mean = np.asarray(Zt[km.labels_ == c].mean(axis=0)).ravel()
        desc[c] = terms[np.argsort(-mean)[:8]].tolist()
    return km.labels_, desc


LABELS = {"B_char_lr": "B: char TF-IDF LR", "C_style_gbm": "C: style GBM (baseline)",
          "ensemble": "A+B+C ensemble", "ensemble+norm": "A+B+C ensemble + normalisation"}


def plot(out, configs):
    metrics = [("clean_val", "Clean\nvalidation"), ("format_stripped", "Blank lines\nremoved"),
               ("attack_ai_only", "Attacked\nAI only"), ("attack_both", "Attacked\nboth classes")]
    if "topic_shift_mean" in out["configs"][configs[0]]:
        metrics.append(("topic_shift_mean", "Held-out topic\ncluster (mean)"))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.2
    colors = ["#9aa5b1", "#5c7a99", "#e9a03b", "#c4572a"]
    for i, cfg in enumerate(configs):
        vals = [out["configs"][cfg][m] for m, _ in metrics]
        xs = np.arange(len(metrics)) + (i - 1.5) * w
        ax.bar(xs, vals, w, label=LABELS.get(cfg, cfg), color=colors[i])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([l for _, l in metrics], fontsize=9)
    lo = min(out["configs"][c][m] for c in configs for m, _ in metrics)
    ax.set_ylim(max(0.5, lo - 0.02), 1.017)
    ax.set_ylabel("AUROC")
    ax.set_title("Validation AUROC on clean and stress-tested data (higher is better)", pad=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "q5_robustness.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-shift", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results")
    args = ap.parse_args()
    if args.plot_only:
        out = json.loads((ROOT / "results" / "q5_improvement.json").read_text())
        plot(out, list(out["configs"]))
        return
    best_single = None

    tr, va = get_split()
    X, y = list(tr.text), tr.label.to_numpy()
    yv = va.label.to_numpy()
    ai = yv == 1
    att_ai, names_ai = perturb_texts(va.text[ai], seed=SEED)
    v_ai = np.array(va.text, dtype=object)
    v_ai[ai] = att_ai
    att_both, _ = perturb_texts(va.text, seed=SEED + 1)
    attack_names = np.array(names_ai)

    v_fmt = [re.sub(r" {2,}", " ", re.sub(r"\n\s*\n+", "\n", t)) for t in va.text]
    configs = ["B_char_lr", "C_style_gbm", "ensemble", "ensemble+norm"]
    out = {"configs": {}}
    for cfg in configs:
        m = make(cfg, best_single)
        t = time.time()
        m.fit(X, y)
        fit_s = time.time() - t
        t = time.time()
        p_clean = m.predict_proba(list(va.text))[:, 1]
        pred_s = time.time() - t
        p_ai = m.predict_proba(list(v_ai))[:, 1]
        p_both = m.predict_proba(att_both)[:, 1]
        p_fmt = m.predict_proba(v_fmt)[:, 1]
        per_attack = {}
        for k in sorted(set(attack_names)):
            # human passages unchanged + AI passages that received attack k
            mask = ~ai.copy()
            idx = np.where(ai)[0][attack_names == k]
            mask[idx] = True
            per_attack[k] = roc_auc_score(yv[mask], p_ai[mask])
        r = {"fit_seconds": fit_s, "predict_seconds_2000": pred_s,
             "clean_val": roc_auc_score(yv, p_clean), "format_stripped": roc_auc_score(yv, p_fmt),
             "attack_ai_only": roc_auc_score(yv, p_ai),
             "attack_both": roc_auc_score(yv, p_both), "per_attack_ai_only": per_attack,
             "mean_p_ai_on_attacked_ai": float(p_ai[ai].mean()), "mean_p_ai_on_clean_ai": float(p_clean[ai].mean())}
        if hasattr(m, "member_scores"):
            ms = m.member_scores(list(va.text))
            r["member_clean_val"] = {n: roc_auc_score(yv, s) for n, s in ms.items()}
        out["configs"][cfg] = r
        print(cfg, json.dumps({k: v for k, v in r.items() if k != "per_attack_ai_only"}), flush=True)
        print("   per attack:", {k: round(v, 4) for k, v in per_attack.items()}, flush=True)
        if cfg == "ensemble+norm":
            np.savez_compressed(ROOT / "results" / "val_preds_final.npz", ids=va.id.to_numpy(), y=yv, p=p_clean)

    if not args.skip_shift:
        labels, desc = pseudo_domains(X)
        out["clusters"] = {int(c): {"n": int((labels == c).sum()), "ai_frac": float(y[labels == c].mean()),
                                    "top_terms": desc[c]} for c in desc}
        print("clusters", json.dumps(out["clusters"]), flush=True)
        for cfg in configs:
            aucs = {}
            t = time.time()
            for c in sorted(desc):
                trn, tst = labels != c, labels == c
                m = make(cfg, best_single).fit([X[i] for i in np.where(trn)[0]], y[trn])
                aucs[int(c)] = roc_auc_score(y[tst], m.predict_proba([X[i] for i in np.where(tst)[0]])[:, 1])
            out["configs"][cfg]["topic_shift_per_cluster"] = aucs
            out["configs"][cfg]["topic_shift_mean"] = float(np.mean(list(aucs.values())))
            out["configs"][cfg]["topic_shift_seconds"] = time.time() - t
            print(cfg, "topic shift", {k: round(v, 4) for k, v in aucs.items()},
                  "mean", round(np.mean(list(aucs.values())), 4), flush=True)

    (ROOT / "results" / "q5_improvement.json").write_text(json.dumps(out, indent=1, default=float))

    plot(out, configs)

if __name__ == "__main__":
    main()
