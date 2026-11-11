"""Session report for run_all.py: results table, sanity checks, ensembles, graphs.

Writes (tracked in git, so teammates / AI assistants can read results after a pull):
  experiments/sessions/<session>/report.md        human-readable report with embedded figures
  experiments/sessions/<session>/summary.csv      one row per job, all metrics
  experiments/sessions/<session>/ensembles.csv    blend / stack results
  experiments/sessions/<session>/sanity.csv       every check with PASS / WARN / FAIL
  experiments/sessions/<session>/figures/*.png
and (git-ignored) outputs/sessions/<session>/submissions/*.csv plus
outputs/submissions/<session>__best.csv (the recommended Kaggle upload).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, data_dir, read_yaml
from .ensemble import greedy, load_runs, rank_mean, stack_lr
from .utils import auroc, binary_metrics, file_lock

warnings.filterwarnings("ignore", category=UserWarning)

FAMILY_COLORS = {"classical": "#4C78A8", "stylometric": "#F58518", "LM features": "#54A24B",
                 "embeddings": "#B279A2", "fine-tuned": "#E45756", "ensemble": "#333333"}


def family(config: str) -> str:
    c = config or ""
    if c.startswith("tfidf"):
        return "classical"
    if c.startswith("stylometric"):
        return "stylometric"
    if c.startswith(("lm_", "binoculars")):
        return "LM features"
    if c.startswith("embed"):
        return "embeddings"
    if c.startswith("ft_"):
        return "fine-tuned"
    return "ensemble" if "blend" in c else "other"


def _ks(a, b) -> float:
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


# ------------------------------------------------------------------ collection
def job_error(run_dir: Path, log_tail: str) -> str:
    """The exception that ended a job: train.py's status.json if it got that far, else the
    last 'SomethingError: ...' line of the log tail (warnings printed after a traceback
    must not hide it)."""
    try:
        with open(run_dir / "status.json", encoding="utf-8") as f:
            err = json.load(f).get("error")
        if err:
            return str(err)[:200]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    lines = log_tail.splitlines()
    exc = [l for l in lines if re.match(r"^\s*[\w.]+(Error|Exception|Interrupt)\b", l)]
    return ((exc or lines or [""])[-1]).strip()[:200]


def collect(session: Path):
    meta = read_yaml(session / "session.yaml")
    with open(session / "jobs.csv", newline="", encoding="utf-8") as f:
        status = {r["name"]: r for r in csv.DictReader(f)}
    rows, runs = [], {}
    for j in meta["jobs"]:
        st = status.get(j["name"], {})
        run_dir = ROOT / "outputs" / "runs" / j["run_name"]
        row = {"job": j["name"], "config": j["config"], "family": family(j["config"]),
               "group": j.get("group", "main"), "status": st.get("status", "?"), "run_name": j["run_name"]}
        mpath = run_dir / "metrics.json"
        if row["status"] == "ok" and mpath.exists():
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            om = m.get("oof_metrics", {})
            row.update({
                "cv_scheme": m.get("cv_scheme"), "n_folds": len(m.get("fold_aucs", [])),
                "cv_auroc": m.get("cv_auc"), "fold_std": m.get("cv_auc_std"),
                "fold_min": min(m.get("fold_aucs", [np.nan])),
                "tpr@1%fpr": om.get("tpr_at_1fpr"), "tpr@5%fpr": om.get("tpr_at_5fpr"),
                "accuracy": om.get("accuracy"), "f1": om.get("f1"), "log_loss": om.get("log_loss"),
                "brier": om.get("brier"), "time_min": round(m.get("total_time_s", np.nan) / 60, 1),
                "device": m.get("gpu_name", m.get("device")), "gpu_mem_gb": m.get("gpu_peak_mem_gb"),
                "test_mean": m.get("test_pred_mean"),
                "train_frac": j.get("set", {}).get("data.train_frac", 1.0),
                **{f"eval:{k}": v.get("auc") for k, v in (m.get("eval") or {}).items()},
            })
            runs[j["name"]] = run_dir
        else:
            row["error"] = job_error(run_dir, st.get("error") or "")
        rows.append(row)
    return meta, pd.DataFrame(rows), runs


# ------------------------------------------------------------------ sanity checks
def sanity_checks(df: pd.DataFrame, runs: dict, oofs: dict, tests: dict) -> pd.DataFrame:
    checks = []

    def add(scope, check, ok, detail="", warn=False):
        checks.append({"scope": scope, "check": check,
                       "result": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": detail})

    train = pd.read_csv(data_dir() / "train.csv")
    test = pd.read_csv(data_dir() / "test.csv")
    sample = pd.read_csv(data_dir() / "sample_submission.csv")
    add("data", "train labels are 0/1 and roughly balanced",
        set(train.label.unique()) <= {0, 1} and 0.4 < train.label.mean() < 0.6, f"AI share {train.label.mean():.3f}")
    add("data", "test ids unique and identical to sample_submission ids",
        test.id.is_unique and set(test.id) == set(sample.id), f"{len(test)} test rows")
    add("data", "no exact text overlap between train and test",
        len(set(train.text) & set(test.text)) == 0, f"{len(set(train.text) & set(test.text))} overlaps")

    for _, r in df.iterrows():
        if r.status != "ok":
            add(r.job, "job finished", False, r.get("error", "") or r.status)
            continue
        if r.job not in tests or r.job not in oofs:
            add(r.job, "OOF + test predictions written", False, "missing oof.csv or test_pred.csv")
            continue
        o, t = oofs[r.job], tests[r.job]
        seen = o.fold >= 0
        add(r.job, "OOF predictions finite", bool(np.isfinite(o.pred[seen]).all()))
        a = r.cv_auroc
        add(r.job, "CV AUROC > 0.5 (labels not flipped)", a > 0.5, f"{a:.4f}")
        add(r.job, "CV AUROC > 0.75 (model learned something)", a > 0.75, f"{a:.4f}", warn=True)
        if r.n_folds > 1:
            add(r.job, "fold AUROC spread small (std < 0.03)", r.fold_std < 0.03, f"std {r.fold_std:.4f}", warn=True)
        add(r.job, "test predictions: 3000 rows, ids match sample, in [0,1], no NaN",
            len(t) == len(sample) and set(t.id) == set(sample.id) and t.label.between(0, 1).all()
            and t.label.notna().all())
        add(r.job, "test predictions not constant", t.label.std() > 1e-6, f"std {t.label.std():.4f}")
        mean = t.label.mean()
        add(r.job, "mean test prediction plausible (0.15-0.85)", 0.15 < mean < 0.85, f"{mean:.3f}", warn=True)
        ks = _ks(t.label, o.pred[seen])
        add(r.job, "test score distribution close to OOF distribution (KS < 0.25)", ks < 0.25,
            f"KS {ks:.3f} - large values mean distribution shift or a preprocessing mismatch", warn=True)

    ok_jobs = [j for j in df.job[df.status == "ok"] if j in tests]
    main_ok = df[(df.status == "ok") & (df.group == "main") & df.job.isin(ok_jobs)]
    if len(ok_jobs) > 1 and len(main_ok):
        best = main_ok.sort_values("cv_auroc", ascending=False).job.iloc[0]
        for j in ok_jobs:
            if j == best:
                continue
            rho = pd.Series(tests[j].label.values).corr(pd.Series(tests[best].label.values), method="spearman")
            add(j, f"test predictions agree with best model ({best}) - Spearman > 0.2", rho > 0.2,
                f"rho {rho:.3f}", warn=rho >= 0)
    return pd.DataFrame(checks)


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    return plt


def fig_auroc_bar(ok: pd.DataFrame, path: Path):
    plt = _plt()
    d = ok.sort_values("cv_auroc")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(d) + 1)))
    ax.barh(d.job, d.cv_auroc, xerr=d.fold_std.fillna(0), color=[FAMILY_COLORS.get(f, "#999") for f in d.family],
            error_kw={"elinewidth": 0.8, "capsize": 2})
    lo = np.floor(max(0.0, (d.cv_auroc - d.fold_std.fillna(0)).min() - 0.02) * 20) / 20
    hi = 1.0 + (1.0 - lo) * 0.12  # room for the value labels
    ax.set_xlim(lo, hi)
    ax.axvline(1.0, color="k", lw=0.6)
    ax.axvline(0.5, color="k", ls=":", lw=0.6) if lo < 0.5 else None
    for y, (v, e) in enumerate(zip(d.cv_auroc, d.fold_std.fillna(0))):
        ax.text(min(v + e, 1.0) + (hi - lo) * 0.01, y, f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlabel("CV AUROC (error bar = std over folds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLORS.items() if f in set(d.family)]
    ax.legend(handles, [f for f in FAMILY_COLORS if f in set(d.family)], loc="lower right", fontsize=7)
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_roc(ok, oofs, extra: dict, path: Path):
    from sklearn.metrics import roc_curve
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    top = ok.sort_values("cv_auroc", ascending=False).head(8)
    curves = [(r.job, oofs[r.job]) for _, r in top.iterrows()]
    for name, (y, s) in extra.items():
        curves.append((name, pd.DataFrame({"label": y, "pred": s, "fold": 0})))
    for name, o in curves:
        o = o[o.fold >= 0]
        fpr, tpr, _ = roc_curve(o.label, o.pred)
        lw = 2.2 if name in extra else 1.1
        for ax in axes:
            ax.plot(fpr, tpr, lw=lw, label=f"{name} ({auroc(o.label, o.pred):.4f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_title("ROC (out-of-fold)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-3, 1)
    axes[1].axvline(0.01, color="k", ls=":", lw=0.8)
    axes[1].set_title("Low-FPR region (log scale; dotted = 1% FPR)")
    for ax in axes:
        ax.set_xlabel("false positive rate (human flagged as AI)")
        ax.set_ylabel("true positive rate")
    axes[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_score_hists(ok, oofs, tests, path: Path):
    plt = _plt()
    d = ok.sort_values("cv_auroc", ascending=False).head(12)
    n = len(d)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    bins = np.linspace(0, 1, 31)
    for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
        o = oofs[r.job]
        o = o[o.fold >= 0]
        ax.hist(o.pred[o.label == 0], bins=bins, alpha=0.55, density=True, label="OOF human", color="#4C78A8")
        ax.hist(o.pred[o.label == 1], bins=bins, alpha=0.55, density=True, label="OOF AI", color="#E45756")
        ax.hist(tests[r.job].label, bins=bins, histtype="step", density=True, color="k", lw=1.1, label="test")
        ax.set_title(r.job, fontsize=8)
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Predicted P(AI): out-of-fold by true class vs. test (sanity: test should look like a mix)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_corr(ok, oofs, tests, path: Path):
    plt = _plt()
    jobs = list(ok.sort_values("cv_auroc", ascending=False).job)
    T = pd.DataFrame({j: tests[j].label.values for j in jobs})
    C = T.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.45 * len(jobs) + 3, 0.45 * len(jobs) + 2.5))
    im = ax.imshow(C.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(jobs)), jobs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(jobs)), jobs, fontsize=7)
    ax.grid(False)
    if len(jobs) <= 20:
        for i in range(len(jobs)):
            for k in range(len(jobs)):
                ax.text(k, i, f"{C.values[i, k]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation of test predictions")
    ax.set_title("Model agreement on test (low correlation + good CV = useful for ensembling)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def topic_auroc(ok, oofs, k: int = 8):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    train = pd.read_csv(data_dir() / "train.csv").sort_values("id").reset_index(drop=True)
    vec = TfidfVectorizer(max_features=20000, min_df=3, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(train.text)
    km = MiniBatchKMeans(k, random_state=0, n_init=5).fit(X)
    terms = np.array(vec.get_feature_names_out())
    names = [", ".join(terms[np.argsort(c)[::-1][:3]]) for c in km.cluster_centers_]
    lab = pd.Series(km.labels_, index=train.id)
    res = {}
    for _, r in ok.iterrows():
        o = oofs[r.job].set_index("id")
        o = o[o.fold >= 0]
        cl = lab.loc[o.index]
        res[r.job] = {f"{c}: {names[c]}": (auroc(o.label[cl == c], o.pred[cl == c])
                                          if o.label[cl == c].nunique() == 2 else np.nan) for c in range(k)}
    sizes = {f"{c}: {names[c]}": int((km.labels_ == c).sum()) for c in range(k)}
    return pd.DataFrame(res).T, sizes


def fig_topic(topic_df: pd.DataFrame, sizes, path: Path):
    plt = _plt()
    d = topic_df.loc[topic_df.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.1 * d.shape[1] + 4, 0.32 * d.shape[0] + 2.5))
    im = ax.imshow(d.values, cmap="viridis", vmin=max(0.5, np.nanmin(d.values)), vmax=1, aspect="auto")
    ax.set_xticks(range(d.shape[1]), [f"{c} (n={sizes[c]})" for c in d.columns], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(d.shape[0]), d.index, fontsize=7)
    ax.grid(False)
    for i in range(d.shape[0]):
        for k in range(d.shape[1]):
            v = d.values[i, k]
            if np.isfinite(v):
                ax.text(k, i, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v < (np.nanmin(d.values) + 1) / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="OOF AUROC within topic cluster")
    ax.set_title("Error analysis: AUROC per topic cluster (k-means on train TF-IDF)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_time(ok, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for f, g in ok.groupby("family"):
        ax.scatter(g.time_min, g.cv_auroc, label=f, color=FAMILY_COLORS.get(f, "#999"), s=30)
        for _, r in g.iterrows():
            ax.annotate(r.job, (r.time_min, r.cv_auroc), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("wall time (min, incl. CV)")
    ax.set_ylabel("CV AUROC")
    ax.legend(fontsize=7)
    ax.set_title("Cost vs. accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_learning_curve(ok, path: Path) -> bool:
    lc = ok[ok.group == "learning_curve"]
    if lc.empty:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cfg_name, g in lc.groupby("config"):
        full = ok[(ok.config == cfg_name) & (ok.group == "main")]
        pts = pd.concat([g, full])
        pts = pts.assign(train_frac=pts.train_frac.astype(float)).sort_values("train_frac")
        ax.errorbar(pts.train_frac * 8000, pts.cv_auroc, yerr=pts.fold_std.fillna(0), marker="o",
                    capsize=3, label=cfg_name)
    ax.set_xscale("log")
    ax.set_xlabel("labeled training examples per fold (approx.)")
    ax.set_ylabel("CV AUROC")
    ax.set_title("Effect of training-set size")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_calibration(ok, oofs, path: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    for _, r in ok.sort_values("cv_auroc", ascending=False).head(6).iterrows():
        o = oofs[r.job]
        o = o[o.fold >= 0]
        b = np.clip((o.pred * 10).astype(int), 0, 9)
        g = o.groupby(b).agg(p=("pred", "mean"), y=("label", "mean"), n=("label", "size"))
        g = g[g.n >= 20]
        ax.plot(g.p, g.y, marker="o", ms=3, lw=1, label=r.job)
    ax.set_xlabel("mean predicted P(AI)")
    ax.set_ylabel("observed AI fraction")
    ax.set_title("Calibration (AUROC ignores it, but it matters for thresholds)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main entry
def _md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else format(v, floatfmt.get(c, ".4f")))
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_report(session: Path) -> Path:
    session = Path(session)
    meta, df, runs = collect(session)
    name = session.name
    out = ROOT / "experiments" / "sessions" / name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    subs = session / "submissions"
    subs.mkdir(parents=True, exist_ok=True)

    oofs = {j: pd.read_csv(r / "oof.csv").sort_values("id").reset_index(drop=True) for j, r in runs.items()}
    tests = {j: pd.read_csv(r / "test_pred.csv").sort_values("id").reset_index(drop=True) for j, r in runs.items()
             if (r / "test_pred.csv").exists()}
    ok = df[(df.status == "ok") & df.job.isin(tests)].copy()
    for c in ["cv_auroc", "fold_std", "time_min", "train_frac"]:
        if c in ok:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # ---------------- ensembles over the main group
    ens_rows, extra_curves, best = [], {}, None
    groups = (meta.get("report") or {}).get("ensemble_groups", ["main"])
    pool = ok[ok.group.isin(groups)].sort_values("cv_auroc", ascending=False)
    if len(pool) >= 2:
        O, T, y, names = load_runs([runs[j] for j in pool.job])
        O.columns = T.columns = list(pool.job)
        n_common = len(y)
        cands = []
        top5 = list(pool.job[:5])
        o, t = rank_mean(O, T, top5)
        cands.append(("rank_mean_top5", auroc(y, o), o, t, " + ".join(top5)))
        o, t = rank_mean(O, T)
        cands.append(("rank_mean_all", auroc(y, o), o, t, f"{len(pool)} models"))
        g_auc, o, t, w = greedy(O, T, y)
        cands.append(("greedy_selection", g_auc, o, t, " + ".join(f"{k}:{v:.2f}" for k, v in w.items())))
        s_auc, o, t, w = stack_lr(O, T, y)
        cands.append(("stack_logreg", s_auc, o, t, " + ".join(f"{k}:{v:+.2f}" for k, v in w.items())))
        single = pool.iloc[0]
        cands.append((f"single_best ({single.job})", auroc(y, O[single.job]), O[single.job].values,
                      T[single.job].values, single.job))
        for nm, a, o, t, desc in cands:
            m = binary_metrics(y, o if nm.startswith("stack") else np.clip(o, 0, 1))
            ens_rows.append({"method": nm, "auroc": a, "tpr@1%fpr": m["tpr_at_1fpr"],
                             "rows_evaluated": n_common, "members": desc})
            fname = nm.split(" ")[0]
            pd.DataFrame({"id": T.index, "label": np.clip(t, 0, 1)}).to_csv(subs / f"{fname}.csv", index=False)
        ens = pd.DataFrame(ens_rows).sort_values("auroc", ascending=False)
        best = ens.iloc[0]
        extra_curves = {c[0]: (y, c[2]) for c in cands if c[0] == best["method"] and not c[0].startswith("single")}
        best_file = subs / f"{best['method'].split(' ')[0]}.csv"
        shutil.copy(best_file, ROOT / "outputs" / "submissions" / f"{name}__best.csv")
        ens.to_csv(out / "ensembles.csv", index=False)
    else:
        ens = pd.DataFrame()

    # ---------------- sanity + tables
    sanity = sanity_checks(df, runs, oofs, tests)
    sanity.to_csv(out / "sanity.csv", index=False)
    cols = ["job", "config", "family", "group", "status", "cv_scheme", "n_folds", "cv_auroc", "fold_std", "fold_min",
            "tpr@1%fpr", "tpr@5%fpr", "accuracy", "f1", "log_loss", "brier", "time_min", "device", "gpu_mem_gb",
            "test_mean", "train_frac", "run_name", "error"]
    eval_cols = sorted(c for c in df.columns if c.startswith("eval:"))  # older sessions only (removed option)
    cols = cols[:8] + eval_cols + cols[8:]
    summary = df.reindex(columns=cols)
    summary["n_folds"] = pd.to_numeric(summary["n_folds"], errors="coerce").astype("Int64")
    summary["_ok"] = summary.status == "ok"
    summary = summary.sort_values(["_ok", "cv_auroc"], ascending=[False, False]).drop(columns="_ok")
    summary.to_csv(out / "summary.csv", index=False)

    # ---------------- figures
    made = []
    if len(ok):
        fig_auroc_bar(ok, figs / "auroc_bar.png"); made.append(("Model comparison", "auroc_bar.png"))
        fig_roc(ok, oofs, extra_curves, figs / "roc.png"); made.append(("ROC curves", "roc.png"))
        fig_score_hists(ok, oofs, tests, figs / "score_hists.png"); made.append(("Score distributions", "score_hists.png"))
        fig_calibration(ok, oofs, figs / "calibration.png"); made.append(("Calibration", "calibration.png"))
        fig_time(ok, figs / "time_vs_auroc.png"); made.append(("Cost vs accuracy", "time_vs_auroc.png"))
        if len(ok) > 1:
            fig_corr(ok, oofs, tests, figs / "test_corr.png"); made.append(("Model agreement", "test_corr.png"))
        try:
            topic_df, sizes = topic_auroc(ok, oofs)
            topic_df.to_csv(out / "topic_auroc.csv")
            fig_topic(topic_df, sizes, figs / "topic_auroc.png"); made.append(("Per-topic AUROC", "topic_auroc.png"))
        except Exception as e:  # never let a figure kill the report
            print(f"topic figure skipped: {e}")
        if fig_learning_curve(ok, figs / "learning_curve.png"):
            made.append(("Learning curve", "learning_curve.png"))

    # ---------------- markdown
    n_fail = int((sanity.result == "FAIL").sum())
    n_warn = int((sanity.result == "WARN").sum())
    md = [f"# Session `{name}`", "",
          f"Profile **{meta['profile']}** — {meta.get('description', '')}", "",
          f"Jobs: {int((df.status == 'ok').sum())} ok, {int((df.status == 'failed').sum())} failed, "
          f"{int((~df.status.isin(['ok', 'failed'])).sum())} not run.  "
          f"Sanity checks: {n_fail} FAIL, {n_warn} WARN (details below).", ""]
    if best is not None:
        md += ["## Recommended submission", "",
               f"**{best['method']}** — OOF AUROC {best['auroc']:.4f} on {int(best['rows_evaluated'])} rows → "
               f"`outputs/submissions/{name}__best.csv`", "",
               "_Greedy and stacking scores are cross-validated (nested), so they are comparable to single models. "
               "Remember the test set contains unseen domains/generators — prefer robust gains over tiny CV gains._", ""]
    md += ["## Results", "", _md_table(summary[summary.status == "ok"][
        ["job", "family", "group", "cv_scheme", "n_folds", "cv_auroc", "fold_std", *eval_cols, "tpr@1%fpr",
         "accuracy", "log_loss", "time_min", "device", "test_mean"]],
        {"time_min": ".1f", "test_mean": ".3f", "fold_std": ".4f", **{c: ".4f" for c in eval_cols}}), ""]
    failed = summary[summary.status != "ok"]
    if len(failed):
        md += ["### Jobs that did not finish", "", _md_table(failed[["job", "config", "status", "error"]]), "",
               "Full logs: `outputs/sessions/" + name + "/logs/<job>.log`", ""]
    if len(ens):
        md += ["## Ensembles", "", _md_table(ens), ""]
    for g, title in [("learning_curve", "Learning curve runs"), ("semi_supervised", "Semi-supervised runs"),
                     ("robustness", "Robustness / generalisation runs")]:
        sub = ok[ok.group == g]
        if len(sub):
            base = ok[(ok.group == "main") & ok.config.isin(sub.config)][["config", "cv_auroc"]].rename(
                columns={"cv_auroc": "baseline_cv_auroc"})
            t = sub.merge(base, on="config", how="left")[["job", "config", "cv_scheme", "train_frac", "cv_auroc",
                                                          "baseline_cv_auroc"]]
            t["delta"] = t.cv_auroc - t.baseline_cv_auroc
            md += [f"## {title}", "", _md_table(t), ""]
    md += ["## Figures", ""]
    for title, f in made:
        md += [f"### {title}", "", f"![{title}](figures/{f})", ""]
    md += ["## Sanity checks", "", f"{n_fail} FAIL, {n_warn} WARN, {int((sanity.result == 'PASS').sum())} PASS.", ""]
    shown = sanity[sanity.result != "PASS"]
    md += [_md_table(shown) if len(shown) else "All checks passed.", "",
           "Full list: `sanity.csv`. Summary of every job and metric: `summary.csv`.", ""]
    report = out / "report.md"
    with file_lock(report):
        report.write_text("\n".join(md), encoding="utf-8")

    print(f"\nReport: {report.relative_to(ROOT)}")
    if best is not None:
        print(f"Best: {best['method']} (OOF AUROC {best['auroc']:.4f}) -> outputs/submissions/{name}__best.csv")
    print(f"Sanity checks: {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        print(sanity[sanity.result == "FAIL"].to_string(index=False))
    return report
