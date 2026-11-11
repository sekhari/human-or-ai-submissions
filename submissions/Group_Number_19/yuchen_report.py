"""Report-protocol experiments on the fixed 2,000-row validation set (template Q1-Q4).

Validation = fold 0 of the team's stratified 5-fold split (folds.py, SEED 20261109):
2,000 rows (1,000 human / 1,000 AI). Training partition = the other 8,000 rows. Every
model below is fit only on (subsets of) the training partition; all hyperparameter and
iteration choices use an inner 20% split of the training partition, never the validation set.
The TF-IDF/LR score fed to the tree/MLP models is out-of-fold (3-fold) on the training rows.

  python submissions/Group_Number_19/yuchen_report.py --stage all
Writes results/yuchen/report/*.csv, learning_curve_fixed_val.png and REPORT_TABLES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from folds import SEED, make_folds  # noqa: E402
from yuchen_models import Imputer  # noqa: E402
from yuchen_semi import Data, lr_crossfit, lr_fit_predict  # noqa: E402
from yuchen_v2 import CAT, PSEUDO_THR, PSEUDO_W  # noqa: E402

ROOT = HERE.parents[1]
SIZES = [500, 1000, 2000, 5000, 8000]


# ---------------------------------------------------------------- setup
class Split:
    def __init__(self, a):
        self.D = Data(a.data, a.v2, a.lm, a.bino)
        D = self.D
        fold = make_folds(D.tr, HERE / "folds.csv")
        self.va = np.where(fold == 0)[0]
        self.tr = np.where(fold != 0)[0]
        self.y = D.y
        self.out = a.out
        cache = a.out / "lr_fixed.npz"
        if cache.exists():
            z = np.load(cache)
        else:
            t = time.time()
            oof, (pv, pu, pt, ptr) = lr_crossfit(D.tr["text"].iloc[self.tr].reset_index(drop=True), self.y[self.tr],
                                                 [D.tr["text"].iloc[self.va], D.un["text"], D.te["text"],
                                                  D.tr["text"].iloc[self.tr]])
            np.savez(cache, oof=oof, pv=pv, pu=pu, pt=pt, ptr=ptr, seconds=time.time() - t)
            z = np.load(cache)
        self.lr = z
        self.style53 = D.style_cols[:53]
        self.style126 = D.style_cols
        self.lm = D.lm_cols
        self.bino = [c for c in D.X.columns if c not in D.style_cols and c not in D.lm_cols]

    def frames(self, cols, idx_tr=None, lr_tr=None):
        D = self.D
        idx_tr = self.tr if idx_tr is None else idx_tr
        Xtr = D.X.iloc[idx_tr][cols].reset_index(drop=True)
        Xva = D.X.iloc[self.va][cols].reset_index(drop=True)
        Xun = D.Xun[cols].reset_index(drop=True)
        Xte = D.Xte[cols].reset_index(drop=True)
        Xtr = Xtr.assign(lr_score=self.lr["oof"] if lr_tr is None else lr_tr)
        return Xtr, Xva.assign(lr_score=self.lr["pv"]), Xun.assign(lr_score=self.lr["pu"]), Xte.assign(lr_score=self.lr["pt"])


def auc(y, p):
    return float(roc_auc_score(y, p))


def cat_fit(X, y, params=None, w=None, seed=SEED):
    p = dict(CAT, **(params or {}))
    i_fit, i_stop = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
    m = CatBoostClassifier(**p, random_seed=seed, thread_count=8)
    m.fit(X.iloc[i_fit], y[i_fit], sample_weight=None if w is None else w[i_fit], eval_set=(X.iloc[i_stop], y[i_stop]))
    return m


def lgb_fit(X, y):
    i_fit, i_stop = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, random_state=SEED, verbose=-1, n_jobs=8)
    m.fit(X.iloc[i_fit], y[i_fit], eval_set=[(X.iloc[i_stop], y[i_stop])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def mlp_fit(X, y, alpha=1e-3, hidden=(256, 128)):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    imp = Imputer().fit(X)
    m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED),
                      MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, batch_size=128, max_iter=200,
                                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                                    random_state=SEED)).fit(imp.transform(X), y)
    return lambda Z: m.predict_proba(imp.transform(Z))[:, 1]


# ---------------------------------------------------------------- Q1
def q1(S: Split) -> pd.DataFrame:
    y, ytr, yva = S.y, S.y[S.tr], S.y[S.va]
    rows, preds = [], {}

    def add(name, rep, ptr, pva, sec):
        rows.append(dict(model=name, representation=rep, train_auroc=auc(ytr, ptr), val_auroc=auc(yva, pva),
                         train_seconds=round(sec, 1)))
        preds[name] = pva
        print(json.dumps(rows[-1]), flush=True)

    add("TF-IDF + logistic regression (C=30)", "word 1-3 + char 3-5 n-gram TF-IDF (binary tf)",
        S.lr["ptr"], S.lr["pv"], float(S.lr["seconds"]) / 4)  # crossfit = 4 LR fits; one fit ~ 1/4
    for name, cols, kind in [
        ("LightGBM, 53 style + LR score (team baseline)", S.style53, "lgb"),
        ("CatBoost, 126 style + LR score (v2)", S.style126, "cat"),
        ("CatBoost, v4 features (style + LR + GPT-2 + Qwen)", S.style126 + S.lm + S.bino, "cat"),
        ("MLP 256-128, v4 features", S.style126 + S.lm + S.bino, "mlp"),
    ]:
        Xtr, Xva, _, _ = S.frames(cols)
        t = time.time()
        if kind == "lgb":
            m = lgb_fit(Xtr, ytr); f = lambda Z: m.predict_proba(Z)[:, 1]  # noqa: E731
        elif kind == "cat":
            m = cat_fit(Xtr, ytr); f = lambda Z: m.predict_proba(Z)[:, 1]  # noqa: E731
        else:
            f = mlp_fit(Xtr, ytr)
        sec = time.time() - t
        rep = {"lgb": "53 style features + LR score", "cat": f"{Xtr.shape[1]} features",
               "mlp": f"{Xtr.shape[1]} features, quantile-normalized"}[kind]
        add(name, rep, f(Xtr), f(Xva), sec)
    R = pd.DataFrame(rows)
    R.to_csv(S.out / "q1_models.csv", index=False)
    pd.DataFrame({"id": S.D.tr["id"].iloc[S.va].to_numpy(), "label": yva, **preds}).to_csv(S.out / "q1_val_preds.csv", index=False)
    return R


def q1_hparams(S: Split) -> pd.DataFrame:
    """Selection on an inner 20% split of the training partition (never the validation set)."""
    ytr = S.y[S.tr]
    cols = S.style126 + S.lm + S.bino
    Xtr, Xva, _, _ = S.frames(cols)
    i_a, i_b = train_test_split(np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED + 7)
    rows = []
    for depth in (4, 6, 8):
        for lr_ in (0.03, 0.05, 0.1):
            m = cat_fit(Xtr.iloc[i_a].reset_index(drop=True), ytr[i_a], dict(depth=depth, learning_rate=lr_))
            rows.append(dict(model="CatBoost", setting=f"depth={depth}, lr={lr_}",
                             inner_auroc=auc(ytr[i_b], m.predict_proba(Xtr.iloc[i_b])[:, 1]),
                             best_iter=m.get_best_iteration(),
                             val_auroc=auc(S.y[S.va], m.predict_proba(Xva)[:, 1])))
            print(json.dumps(rows[-1]), flush=True)
    for hidden in ((128,), (256, 128)):
        for alpha in (1e-4, 1e-3, 1e-2):
            f = mlp_fit(Xtr.iloc[i_a].reset_index(drop=True), ytr[i_a], alpha, hidden)
            rows.append(dict(model="MLP", setting=f"hidden={hidden}, alpha={alpha}",
                             inner_auroc=auc(ytr[i_b], f(Xtr.iloc[i_b])), best_iter=None,
                             val_auroc=auc(S.y[S.va], f(Xva))))
            print(json.dumps(rows[-1]), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(S.out / "q1_hparams.csv", index=False)
    return R


# ---------------------------------------------------------------- Q2
def q2(S: Split) -> pd.DataFrame:
    """CatBoost on v4 features; TF-IDF/LR refit (cross-fitted) on each subsample."""
    texts = S.D.tr["text"]
    cols = S.style126 + S.lm + S.bino
    rows = []
    for n in SIZES:
        t = time.time()
        if n < len(S.tr):
            sub, _ = train_test_split(S.tr, train_size=n, stratify=S.y[S.tr], random_state=SEED + n)
            sub = np.sort(sub)
        else:
            sub = S.tr
        y = S.y[sub]
        oof, (pv,) = lr_crossfit(texts.iloc[sub].reset_index(drop=True), y, [texts.iloc[S.va]])
        Xs = S.D.X.iloc[sub][cols].reset_index(drop=True).assign(lr_score=oof)
        Xv = S.D.X.iloc[S.va][cols].reset_index(drop=True).assign(lr_score=pv)
        m = cat_fit(Xs, y)
        rows.append(dict(training_rows=n, train_auroc=auc(y, m.predict_proba(Xs)[:, 1]),
                         val_auroc=auc(S.y[S.va], m.predict_proba(Xv)[:, 1]),
                         tfidf_lr_val_auroc=auc(S.y[S.va], pv), best_iter=m.get_best_iteration(),
                         seconds=round(time.time() - t)))
        print(json.dumps(rows[-1]), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(S.out / "q2_learning_curve.csv", index=False)
    plot_curve(R, S.out)
    return R


def plot_curve(R: pd.DataFrame, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
    from matplotlib.ticker import NullFormatter, NullLocator
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(R.training_rows, R.train_auroc, "o-", label="CatBoost v4 features: training AUROC")
    ax.plot(R.training_rows, R.val_auroc, "o-", label="CatBoost v4 features: validation AUROC")
    ax.plot(R.training_rows, R.tfidf_lr_val_auroc, "s--", label="TF-IDF + LR: validation AUROC")
    ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES])
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("training rows (from the 8,000-row training partition)"); ax.set_ylabel("AUROC")
    ax.set_title("Effect of training-set size (fixed 2,000-row validation set)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "learning_curve_fixed_val.png", dpi=160)


# ---------------------------------------------------------------- Q3 / Q4
def self_train(S: Split, pool: pd.DataFrame | None, thr=PSEUDO_THR, seeds=(SEED,)):
    cols = S.style126 + S.lm + S.bino
    Xtr, Xva, Xun, Xte = S.frames(cols)
    ytr = S.y[S.tr]
    teachers = [cat_fit(Xtr, ytr, seed=s) for s in seeds]
    P = lambda ms, Z: np.mean([m.predict_proba(Z)[:, 1] for m in ms], axis=0)  # noqa: E731
    info = dict(pool_rows=0, selected=0)
    if pool is None:
        return P(teachers, Xva), P(teachers, Xtr), info, teachers, (Xtr, Xva, Xte)
    p = P(teachers, pool)
    sel = (p > thr) | (p < 1 - thr)
    Xs = pd.concat([Xtr, pool[sel]], ignore_index=True)
    ys = np.r_[ytr, (p[sel] > 0.5).astype(int)]
    w = np.r_[np.ones(len(Xtr)), np.full(sel.sum(), PSEUDO_W)]
    students = [cat_fit(Xs, ys, w=w, seed=s) for s in seeds]
    # proxy for pseudo-label noise: teacher accuracy on validation rows it is equally confident about
    pv = P(teachers, Xva); conf = (pv > thr) | (pv < 1 - thr)
    info = dict(pool_rows=len(pool), selected=int(sel.sum()), pseudo_ai_share=float((p[sel] > 0.5).mean()),
                teacher_val_confident_share=float(conf.mean()),
                teacher_val_confident_accuracy=float(((pv[conf] > 0.5) == S.y[S.va][conf]).mean()))
    return P(students, Xva), P(students, Xtr), info, students, (Xtr, Xva, Xte)


def q3(S: Split) -> pd.DataFrame:
    cols = S.style126 + S.lm + S.bino
    _, _, Xun, Xte = S.frames(cols)
    rows = []
    for name, pool, thr in [("supervised baseline (no unlabeled data)", None, PSEUDO_THR),
                            ("self-training on unlabeled.csv, thr 0.95", Xun, 0.95),
                            ("self-training on unlabeled.csv, thr 0.98", Xun, 0.98),
                            ("self-training on unlabeled.csv, thr 0.90", Xun, 0.90)]:
        pv, ptr, info, _, _ = self_train(S, pool, thr)
        rows.append(dict(method=name, val_auroc=auc(S.y[S.va], pv), train_auroc=auc(S.y[S.tr], ptr), **info))
        print(json.dumps(rows[-1]), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(S.out / "q3_unlabeled.csv", index=False)
    return R


def q4(S: Split) -> dict:
    """Final recipe (v4 features, self-training on unlabeled + test text, 3 seeds) refit on 8,000 rows."""
    cols = S.style126 + S.lm + S.bino
    _, _, Xun, Xte = S.frames(cols)
    pool = pd.concat([Xun, Xte], ignore_index=True)
    pv, ptr, info, _, _ = self_train(S, pool, PSEUDO_THR, seeds=(SEED, SEED + 1, SEED + 2))
    yva = S.y[S.va]
    v = S.D.tr.iloc[S.va][["id", "text"]].reset_index(drop=True).assign(label=yva, pred=pv)
    v["words"] = v.text.str.split().str.len()
    v["blank_line"] = v.text.str.contains(r"\n[ \t]*\n")
    v["length_bin"] = pd.cut(v.words, [0, 150, 250, 10_000], labels=["<150 words", "150-250 words", ">250 words"])
    groups = []
    for col in ("length_bin", "blank_line"):
        for g, d in v.groupby(col, observed=True):
            groups.append(dict(group=f"{col}={g}", n=len(d), ai_share=float(d.label.mean()),
                               auroc=auc(d.label, d.pred) if d.label.nunique() > 1 else None,
                               errors_at_0_5=int(((d.pred > 0.5) != (d.label == 1)).sum())))
    G = pd.DataFrame(groups); G.to_csv(S.out / "q4_subgroups.csv", index=False)
    fp = v[v.label == 0].sort_values("pred", ascending=False).head(8)
    fn = v[v.label == 1].sort_values("pred").head(8)
    E = pd.concat([fp.assign(error_type="FP (human -> AI)"), fn.assign(error_type="FN (AI -> human)")])
    E = E.assign(snippet=E.text.str.replace(r"\s+", " ", regex=True).str[:140],
                 wrong_at_0_5=(E.pred > 0.5) != (E.label == 1))
    E[["id", "error_type", "pred", "wrong_at_0_5", "words", "blank_line", "snippet"]].to_csv(S.out / "q4_errors.csv", index=False)
    v.drop(columns="text").to_csv(S.out / "q4_val_preds.csv", index=False)
    pd.DataFrame({"id": S.D.tr["id"].iloc[S.tr].to_numpy(), "label": S.y[S.tr], "pred": ptr}).to_csv(
        S.out / "q4_train_preds.csv", index=False)
    res = dict(train_auroc=auc(S.y[S.tr], ptr), val_auroc=auc(yva, pv),
               n_fp_at_0_5=int(((v.pred > 0.5) & (v.label == 0)).sum()),
               n_fn_at_0_5=int(((v.pred <= 0.5) & (v.label == 1)).sum()), **info)
    json.dump(res, open(S.out / "q4_final_model.json", "w"), indent=2)
    print(json.dumps(res), flush=True)
    return res


def q4_blend(S: Split, transformer_dir: Path, w: float = 0.1) -> dict:
    """Q4 for the final submission: rank blend (1-w) * CatBoost final recipe + w * RoBERTa, both
    trained on the same 8,000 rows (yuchen_deberta.py --fixed-split). Ranks are taken within each set."""
    from scipy.stats import rankdata
    R = lambda x: (rankdata(x) - 1) / (len(x) - 1)  # noqa: E731
    out = {}
    for part in ("val", "train"):
        c = pd.read_csv(S.out / f"q4_{part}_preds.csv")
        t = pd.read_csv(transformer_dir / f"fixed_{part}.csv").set_index("id").loc[c["id"]]
        assert (t["label"].to_numpy() == c["label"].to_numpy()).all()
        c["cat_pred"], c["roberta_pred"] = c["pred"], t["pred"].to_numpy()
        c["pred"] = (1 - w) * R(c["cat_pred"]) + w * R(c["roberta_pred"])
        out[part] = c
    v = out["val"]
    text = S.D.tr.set_index("id").loc[v["id"], "text"].to_numpy()
    res = dict(blend_weight=w, train_auroc=auc(out["train"].label, out["train"].pred), val_auroc=auc(v.label, v.pred),
               catboost_val_auroc=auc(v.label, v.cat_pred), roberta_val_auroc=auc(v.label, v.roberta_pred),
               roberta_train_auroc=auc(out["train"].label, out["train"].roberta_pred))
    # the blend output is a rank score, not a probability: pick errors by rank (as many FPs/FNs as CatBoost at 0.5)
    thr = float(np.quantile(v.pred, 1 - (v.cat_pred > 0.5).mean()))
    res.update(threshold_rank=thr, n_fp=int(((v.pred > thr) & (v.label == 0)).sum()),
               n_fn=int(((v.pred <= thr) & (v.label == 1)).sum()))
    groups = []
    for col in ("length_bin", "blank_line"):
        for g, d in v.groupby(col, observed=True):
            groups.append(dict(group=f"{col}={g}", n=len(d), auroc=auc(d.label, d.pred) if d.label.nunique() > 1 else None,
                               catboost_auroc=auc(d.label, d.cat_pred) if d.label.nunique() > 1 else None,
                               errors=int(((d.pred > thr) != (d.label == 1)).sum())))
    pd.DataFrame(groups).to_csv(S.out / "q4b_subgroups.csv", index=False)
    v = v.assign(snippet=pd.Series(text).str.replace(r"\s+", " ", regex=True).str[:140].to_numpy())
    fp = v[v.label == 0].sort_values("pred", ascending=False).head(8).assign(error_type="FP")
    fn = v[v.label == 1].sort_values("pred").head(8).assign(error_type="FN")
    pd.concat([fp, fn])[["id", "error_type", "pred", "cat_pred", "roberta_pred", "words", "blank_line", "snippet"]].to_csv(
        S.out / "q4b_errors.csv", index=False)
    json.dump(res, open(S.out / "q4b_final_blend.json", "w"), indent=2)
    print(json.dumps(res), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("q1", "q1hp", "q2", "q3", "q4", "q4blend", "all"), default="all")
    ap.add_argument("--transformer", type=Path, default=ROOT / "results" / "yuchen" / "roberta" / "fixed",
                    help="yuchen_deberta.py --fixed-split output (for --stage q4blend)")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--v2", type=Path, default=ROOT / "results" / "yuchen" / "v2_shift")
    ap.add_argument("--lm", type=Path, default=ROOT / "results" / "yuchen" / "lm_features")
    ap.add_argument("--bino", type=Path, default=ROOT / "results" / "yuchen" / "lm_features" / "qwen35_2b")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "report")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = Split(a)
    print(f"train partition {len(S.tr)} rows, validation {len(S.va)} rows (AI share {S.y[S.va].mean():.2f})", flush=True)
    stages = ["q1", "q1hp", "q2", "q3", "q4"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"=== {s}", flush=True)
        if s == "q4blend":
            q4_blend(S, a.transformer); continue
        {"q1": q1, "q1hp": q1_hparams, "q2": q2, "q3": q3, "q4": q4}[s](S)


if __name__ == "__main__":
    main()
