"""Yuchen v3: v2 shift-aware CatBoost + frozen GPT-2 token-probability features.

Adds the zero-shot detector signals from yuchen_lm_features.py (gpt2 124M and
gpt2-medium 355M, computed on a GPU: see gpt2_feats.sbatch) plus a few cross-model
ratios (Binoculars-style: how much the bigger model is "less surprised" than the small
one). These do not depend on topic, so they should help most where the v2 features are
weak: test poetry/fiction without blank lines.

Prerequisites (from the repository root):
  python submissions/Group_Number_19/yuchen_v2.py --stage features   # -> results/yuchen/v2_shift/feat_*.pkl
  python submissions/Group_Number_19/yuchen_v2.py --stage lr         # -> results/yuchen/v2_shift/lr_*.npz
  results/yuchen/lm_features/{gpt2,gpt2-medium}/lm_{train,test}.csv  # from the GPU job

Stages:
  python submissions/Group_Number_19/yuchen_v3.py --stage cv        # LCO comparison, same proxy as v2
  python submissions/Group_Number_19/yuchen_v3.py --stage predict   # submission_v3_cat[_pseudo].csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clusters import make_clusters  # noqa: E402
from yuchen_v2 import CAT, PSEUDO_THR, PSEUDO_W, SEEDS, fit_predict  # noqa: E402

ROOT = HERE.parents[1]
LM_MODELS = {"gpt2": "s", "gpt2-medium": "m"}


def lm_table(lm_dir: Path, split: str, ids: np.ndarray) -> pd.DataFrame:
    parts = []
    for model, tag in LM_MODELS.items():
        d = pd.read_csv(lm_dir / model / f"lm_{split}.csv")
        assert (d["id"].to_numpy() == ids).all(), f"{model} {split}: id order mismatch"
        d = d.drop(columns="id")
        if tag != "s":
            d = d.drop(columns="lm_ntok")
        parts.append(d.add_prefix(f"{tag}_"))
    L = pd.concat(parts, axis=1)
    # cross-model signals: AI text is predictable for both models, human text mostly for the big one
    L["x_logp_diff"] = L["m_lm_logp_mean"] - L["s_lm_logp_mean"]
    L["x_logp_ratio"] = L["m_lm_logp_mean"] / L["s_lm_logp_mean"].clip(upper=-1e-3)
    L["x_bino_s_on_m"] = (-L["s_lm_logp_mean"]) / L["m_lm_ent_mean"].clip(lower=1e-3)
    L["x_bino_m_on_s"] = (-L["m_lm_logp_mean"]) / L["s_lm_ent_mean"].clip(lower=1e-3)
    L["x_fastdetect_diff"] = L["m_lm_fastdetect"] - L["s_lm_fastdetect"]
    return L.fillna(L.median(numeric_only=True))


def bino_table(bino_dir: Path | None, split: str, ids: np.ndarray) -> pd.DataFrame | None:
    if bino_dir is None:
        return None
    d = pd.read_csv(bino_dir / f"bino_{split}.csv")
    assert (d["id"].to_numpy() == ids).all(), f"bino {split}: id order mismatch"
    d = d.drop(columns="id").replace([np.inf, -np.inf], np.nan)
    return d.fillna(d.median(numeric_only=True))


def load(data: Path, v2: Path, lm_dir: Path, bino_dir: Path | None = None):
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    F = pd.read_pickle(v2 / "feat_train.pkl"); T = pd.read_pickle(v2 / "feat_test.pkl")
    assert (F["id"].to_numpy() == tr["id"].to_numpy()).all() and (T["id"].to_numpy() == te["id"].to_numpy()).all()
    F, T = F.drop(columns="id"), T.drop(columns="id")
    Ltr, Lte = lm_table(lm_dir, "train", tr["id"].to_numpy()), lm_table(lm_dir, "test", te["id"].to_numpy())
    Btr, Bte = bino_table(bino_dir, "train", tr["id"].to_numpy()), bino_table(bino_dir, "test", te["id"].to_numpy())
    return tr, te, F, T, Ltr, Lte, Btr, Bte


def stage_cv(data: Path, v2: Path, lm_dir: Path, out: Path, seeds, bino_dir: Path | None = None) -> None:
    tr, te, F, _, L, _, B, _ = load(data, v2, lm_dir, bino_dir)
    y = tr["label"].to_numpy()
    fold, _ = make_clusters(tr, te)
    cl = pd.read_csv(HERE / "clusters.csv")
    shares = pd.crosstab(cl["cluster"], cl["split"], normalize="columns")
    w_test = pd.Series(fold).map(shares["test"] / shares["train"]).to_numpy()
    blank = tr["text"].str.contains(r"\n[ \t]*\n").to_numpy()
    c4 = fold == 4

    def report(name, oof, t):
        r = dict(model=name, lco_test_weighted_auc=roc_auc_score(y, oof, sample_weight=w_test),
                 lco_nonblank_weighted_auc=roc_auc_score(y[~blank], oof[~blank], sample_weight=w_test[~blank]),
                 lco_cluster4_auc=roc_auc_score(y[c4], oof[c4]), lco_pooled_auc=roc_auc_score(y, oof),
                 seconds=round(time.time() - t))
        print(json.dumps(r), flush=True)
        pd.DataFrame({"id": tr["id"], "cluster": fold, "label": y, "pred": oof}).to_csv(out / f"lco_{name}.csv", index=False)
        return r

    results = []
    # single-feature zero-shot AUCs (no training at all): how strong is the raw signal?
    Z = L if B is None else pd.concat([L, B], axis=1)
    zcols = ["s_lm_logp_std", "m_lm_logp_std", "s_lm_fastdetect", "m_lm_fastdetect", "m_lm_bino"]
    if B is not None:
        zcols += ["q_bino", "q_bino_perf", "q_fastdetect_cross", "q_fastdetect_obs", "qo_logp_mean",
                  "qo_logp_std", "q_logp_gap", "q_tok_bino_std"]
    for c in zcols:
        s = Z[c].to_numpy()
        auc = roc_auc_score(y, s, sample_weight=w_test)
        nb = roc_auc_score(y[~blank], s[~blank], sample_weight=w_test[~blank])
        print(json.dumps(dict(zero_shot=c, weighted_auc=round(max(auc, 1 - auc), 5),
                              nonblank_auc=round(max(nb, 1 - nb), 5), sign="+" if auc >= .5 else "-")), flush=True)

    if B is None:
        runs = [("lm_only", "L", False), ("v2+lr", "F", False), ("v2+lm+lr", "FL", False),
                ("v2+lm+lr+pseudo", "FL", True)]
    else:
        runs = [("bino_only", "B", False), ("lm+bino", "LB", False), ("v2+lm+lr", "FL", False),
                ("v2+lm+bino+lr", "FLB", False), ("v2+lm+lr+pseudo", "FL", True), ("v2+lm+bino+lr+pseudo", "FLB", True)]
    blocks = {"F": F, "L": L, "B": B}
    for name, use, pseudo in runs:
        t = time.time(); oof = np.zeros(len(y))
        for k in np.unique(fold):
            z = np.load(v2 / f"lr_lco{k}.npz"); tri, vai = z["tri"], z["vai"]
            Xtr = pd.concat([blocks[c].iloc[tri].reset_index(drop=True) for c in use], axis=1)
            Xva = pd.concat([blocks[c].iloc[vai].reset_index(drop=True) for c in use], axis=1)
            if "F" in use:
                Xtr, Xva = Xtr.assign(lr_score=z["oof"]), Xva.assign(lr_score=z["pv"])
            oof[vai], _ = fit_predict(Xtr, y[tri], Xva, seeds, pseudo)
        results.append(report(name, oof, t))
    pd.DataFrame(results).to_csv(out / "lco_results.csv", index=False)


def stage_predict(data: Path, v2: Path, lm_dir: Path, out: Path, seeds, bino_dir: Path | None = None) -> None:
    tr, te, F, T, L, Lt, B, Bt = load(data, v2, lm_dir, bino_dir)
    y = tr["label"].to_numpy()
    z = np.load(v2 / "lr_full.npz")
    X = pd.concat([F, L] + ([B] if B is not None else []), axis=1).assign(lr_score=z["oof"])
    Xt = pd.concat([T, Lt] + ([Bt] if Bt is not None else []), axis=1).assign(lr_score=z["pt"])
    info = {}
    tag = "v3_cat" if B is None else "v4_cat"
    for name, pseudo in ((tag, False), (tag + "_pseudo", True)):
        t = time.time()
        p, meta = fit_predict(X, y, Xt, seeds, pseudo)
        assert np.isfinite(p).all() and len(p) == len(te)
        pd.DataFrame({"id": te["id"], "label": p}).to_csv(out / f"submission_{name}.csv", index=False)
        info[name] = dict(seconds=round(time.time() - t), pred_ai_share=float((p > 0.5).mean()),
                          **({k: v for k, v in meta.items() if k != "teacher"} if meta else {}))
        print(name, json.dumps(info[name]), flush=True)
    json.dump(dict(seeds=seeds, catboost=CAT, pseudo_thr=PSEUDO_THR, pseudo_weight=PSEUDO_W,
                   lm_models=list(LM_MODELS), bino=str(bino_dir) if bino_dir else None, n_features=X.shape[1], runs=info),
              open(out / "predict_info.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cv", "predict", "all"), default="all")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--v2", type=Path, default=ROOT / "results" / "yuchen" / "v2_shift")
    ap.add_argument("--lm", type=Path, default=ROOT / "results" / "yuchen" / "lm_features")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "yuchen" / "v3_lm")
    ap.add_argument("--bino", type=Path, default=None, help="dir with bino_{train,test}.csv (Qwen Binoculars)")
    ap.add_argument("--cv-seeds", type=int, default=1)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("cv", "all"): stage_cv(a.data, a.v2, a.lm, a.out, SEEDS[:a.cv_seeds], a.bino)
    if a.stage in ("predict", "all"): stage_predict(a.data, a.v2, a.lm, a.out, SEEDS, a.bino)


if __name__ == "__main__":
    main()
