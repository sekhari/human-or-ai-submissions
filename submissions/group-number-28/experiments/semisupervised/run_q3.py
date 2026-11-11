import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256

SEED = 20261109
C_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]
PRIMARY_THRESHOLD = 0.95
SWEEP_THRESHOLDS = [0.99, 0.95, 0.90, 0.80, 0.50] 
MAX_ROUNDS = 3
N_BOOT = 1000
RESULTS = ROOT / "results" / "semisupervised"


def vectorizer():
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                           max_features=60000, sublinear_tf=True)


def classifier(c):
    return LogisticRegression(C=c, solver="liblinear", max_iter=5000,
                              random_state=SEED)


def git_state():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "uncommitted_changes": bool(run("status", "--porcelain"))}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "platform": platform.platform()}


def fit_predict(train_texts, train_y, eval_sets, c, weights=None,
                vocab_texts=None):
    """Fit vectorizer (on vocab_texts or train_texts) + LR; score eval sets."""
    start = time.perf_counter()
    vec = vectorizer()
    if vocab_texts is None:
        x_train = vec.fit_transform(train_texts)
    else:
        vec.fit(vocab_texts)
        x_train = vec.transform(train_texts)
    clf = classifier(c).fit(x_train, train_y, sample_weight=weights)
    fit_seconds = time.perf_counter() - start
    p_train = clf.predict_proba(x_train)[:, 1]
    outs = [clf.predict_proba(vec.transform(t))[:, 1] for t in eval_sets]
    return p_train, outs, fit_seconds


def paired_bootstrap(y, p_a, p_b, seed=SEED):
    """Stratified paired bootstrap of AUROC(b) - AUROC(a)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    diffs = np.array(diffs)
    return {"difference": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_positive": float((diffs > 0).mean()), "replicates": N_BOOT}


def save_run(run_id, config, metrics, val_ids, val_y, p_val, extra=None):
    out = RESULTS / run_id
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --prefix (runs are immutable)")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(out / "validation_predictions.csv", "w") as f:
        f.write("id,label,p_ai\n")
        for i, y, p in zip(val_ids, val_y, p_val):
            f.write(f"{i},{int(y)},{p:.10f}\n")
    for name, rows in (extra or {}).items():
        with open(out / name, "w") as f:
            f.write(rows)


def select_pseudo(p_unl, threshold):
    conf = np.maximum(p_unl, 1 - p_unl)
    keep = conf >= threshold
    return keep, (p_unl >= 0.5).astype(int), conf


def noise_estimate(p_val, y_val, threshold):
    """How wrong is the teacher on labeled rows it is this confident about?"""
    keep, pred, _ = select_pseudo(p_val, threshold)
    n = int(keep.sum())
    return {"threshold": threshold, "selected": n,
            "selected_share": n / len(y_val),
            "error_rate": float((pred[keep] != y_val[keep]).mean()) if n else None,
            "selected_ai_share": float(pred[keep].mean()) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()

    train, val = load_partitions()
    unl = read_csv(raw_path("unlabeled.csv"))
    tr_text = [r["text"] for r in train]
    tr_y = np.array([int(r["label"]) for r in train])
    va_ids = [r["id"] for r in val]
    va_text = [r["text"] for r in val]
    va_y = np.array([int(r["label"]) for r in val])
    un_ids = [r["id"] for r in unl]
    un_text = [r["text"] for r in unl]
    common = {"seed": SEED, "split": "data/splits (canonical 8000/2000)",
              "train_rows": len(train), "validation_rows": len(val),
              "unlabeled_rows": len(unl),
              "unlabeled_sha256": sha256(raw_path("unlabeled.csv")),
              "representation": "TfidfVectorizer(analyzer='char', ngram_range=(3,5), "
                                "min_df=2, max_features=60000, sublinear_tf=True)",
              "classifier": "LogisticRegression(solver='liblinear', L2)",
              "git": git_state(), "environment": environment()}

    cv_scores = {c: [] for c in C_GRID}
    folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(folds.split(tr_text, tr_y)):
        vec = vectorizer()
        xa = vec.fit_transform([tr_text[i] for i in a])
        xb = vec.transform([tr_text[i] for i in b])
        for c in C_GRID:
            p = classifier(c).fit(xa, tr_y[a]).predict_proba(xb)[:, 1]
            cv_scores[c].append(roc_auc_score(tr_y[b], p))
        print(f"cv fold {k} done", flush=True)
    cv_mean = {c: float(np.mean(s)) for c, s in cv_scores.items()}
    best_c = max(cv_mean, key=cv_mean.get)
    print("cv", cv_mean, "-> C =", best_c, flush=True)

    p_tr, (p_va, p_un), secs = fit_predict(tr_text, tr_y, [va_text, un_text], best_c)
    base = {"train_auroc": roc_auc_score(tr_y, p_tr),
            "validation_auroc": roc_auc_score(va_y, p_va), "fit_seconds": secs}
    print("baseline", base, flush=True)
    noise = [noise_estimate(p_va, va_y, t) for t in SWEEP_THRESHOLDS]
    save_run(f"{args.prefix}_baseline",
             {**common, "method": "supervised only", "C": best_c,
              "C_grid": C_GRID, "cv_mean_auroc": cv_mean},
             {**base, "evaluation_rows": len(va_y),
              "teacher_noise_on_validation": noise},
             va_ids, va_y, p_va)
    results = [{"run": "baseline", "method": "Supervised only", "threshold": None,
                "round": 0, "pseudo_added": 0, **base}]
    p_by_run = {"baseline": p_va}

    def self_train(threshold, rounds, tag):
        teacher_p_un = p_un
        history = []
        for rnd in range(1, rounds + 1):
            keep, pl, conf = select_pseudo(teacher_p_un, threshold)
            texts = tr_text + [t for t, k in zip(un_text, keep) if k]
            ys = np.concatenate([tr_y, pl[keep]])
            p_tr2, (p_va2, p_un2), secs2 = fit_predict(texts, ys, [va_text, un_text], best_c)
            m = {"round": rnd, "pseudo_added": int(keep.sum()),
                 "pseudo_ai": int(pl[keep].sum()), "pseudo_human": int((1 - pl[keep]).sum()),
                 "train_auroc_labeled_rows": roc_auc_score(tr_y, p_tr2[: len(tr_y)]),
                 "validation_auroc": roc_auc_score(va_y, p_va2), "fit_seconds": secs2}
            history.append(m)
            print(tag, m, flush=True)
            pseudo_csv = "id,pseudo_label,teacher_confidence\n" + "".join(
                f"{i},{l},{c:.6f}\n" for i, l, c, k in zip(un_ids, pl, conf, keep) if k)
            save_run(f"{args.prefix}_{tag}_r{rnd}",
                     {**common, "method": "confidence-based self-training",
                      "C": best_c, "threshold": threshold, "round": rnd,
                      "teacher": "baseline" if rnd == 1 else f"{tag}_r{rnd-1}",
                      "pseudo_label_weight": 1.0,
                      "vectorizer_fit_on": "train + selected pseudo-labeled texts"},
                     {**m, "evaluation_rows": len(va_y)},
                     va_ids, va_y, p_va2, {"pseudo_labels.csv": pseudo_csv})
            results.append({"run": f"{tag}_r{rnd}", "method": "Self-training",
                            "threshold": threshold, **m,
                            "train_auroc": m["train_auroc_labeled_rows"]})
            p_by_run[f"{tag}_r{rnd}"] = p_va2
            teacher_p_un = p_un2
        return history

    self_train(PRIMARY_THRESHOLD, MAX_ROUNDS, "st95")
    for t in SWEEP_THRESHOLDS:
        if t != PRIMARY_THRESHOLD:
            self_train(t, 1, f"st{int(round(t * 100))}")

    p_tr3, (p_va3,), secs3 = fit_predict(tr_text, tr_y, [va_text], best_c,
                                         vocab_texts=tr_text + un_text)
    m3 = {"train_auroc": roc_auc_score(tr_y, p_tr3),
          "validation_auroc": roc_auc_score(va_y, p_va3), "fit_seconds": secs3}
    save_run(f"{args.prefix}_vocab_unl",
             {**common, "method": "TF-IDF vocabulary/IDF fit on train + unlabeled text; "
                                  "classifier on labeled train only", "C": best_c},
             {**m3, "evaluation_rows": len(va_y)}, va_ids, va_y, p_va3)
    results.append({"run": "vocab_unl", "method": "Vocabulary from unlabeled",
                    "threshold": None, "round": 0, "pseudo_added": 0, **m3})
    p_by_run["vocab_unl"] = p_va3
    print("vocab_unl", m3, flush=True)

    study = RESULTS / f"{args.prefix}_study"
    study.mkdir(parents=True, exist_ok=False)
    boot = {k: paired_bootstrap(va_y, p_va, v) for k, v in p_by_run.items() if k != "baseline"}
    cols = ["run", "method", "threshold", "round", "pseudo_added", "pseudo_ai",
            "pseudo_human", "train_auroc", "validation_auroc", "fit_seconds"]
    lines = [",".join(cols + ["diff_vs_baseline", "ci_low", "ci_high"])]
    for r in results:
        b = boot.get(r["run"], {"difference": 0.0, "ci95": [0.0, 0.0]})
        lines.append(",".join(
            "" if r.get(c) is None else (f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r[c]))
            for c in cols) + f",{b['difference']:.6f},{b['ci95'][0]:.6f},{b['ci95'][1]:.6f}")
    (study / "comparison.csv").write_text("\n".join(lines) + "\n")
    (study / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
    (study / "teacher_noise_on_validation.json").write_text(json.dumps(noise, indent=2) + "\n")
    (study / "plan.json").write_text(json.dumps(
        {"primary_comparison": f"st95_r1 vs baseline (threshold {PRIMARY_THRESHOLD}, "
                               "predeclared)",
         "C_grid": C_GRID, "chosen_C": best_c, "sweep_thresholds": SWEEP_THRESHOLDS,
         "max_rounds": MAX_ROUNDS, "bootstrap_replicates": N_BOOT, **common},
        indent=2) + "\n")
    print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
