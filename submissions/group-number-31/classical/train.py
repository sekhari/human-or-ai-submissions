"""Train and validate one classical model on the team's fixed split, or refit it for Kaggle.

  python train.py --model tfidf_lr                    # inner-CV search (if the config has one), fit on the
                                                      # 8,000 rows, score the fixed 2,000 validation rows
  python train.py --model tfidf_lr --final            # refit on all 10,000 rows with the chosen params and
                                                      # write kaggle_submissions/final_E_tfidf_lr.csv: one file
                                                      # per model, lettered after the pretrained track's A-C
                                                      # (LETTERS; a new model passes --letter <next free>)
  python train.py --model tfidf_lr --set C=8 --no-search --tag c8   # a variant, saved as results/tfidf_lr@c8/
  python train.py --model tfidf_lr --cv topic         # inner CV holds out whole topics instead of random rows

Validation run writes results/<name>/metrics.json (chosen params, train/val AUROC, fit time, hardware),
val_pred.csv (id,label,pred), search.csv (inner-CV grid, if any), and upserts the model's row in the
team's testing history (submission_log.csv, "not submitted"). --final also saves the fitted model to
artifacts/<name>/ (git-ignored) so predict.py can re-create the submission file.

Rules this enforces: models see only the 8,000 training rows (+ unlabeled.csv text) while being
validated; hyperparameters are chosen by 5-fold CV inside those rows, never on the 2,000 validation
rows; nothing here uses test labels (there are none) or pretrained weights.

Inner CV schemes (--cv, or `cv:` in the config): `stratified` (default) splits the training rows at random;
`topic` clusters the training rows into TOPIC_K topics (k-means on word TF-IDF of those rows only: no
validation, test or external text) and each fold holds out whole topics, a label-safe estimate of how a
design copes with domain shift (test has topics train lacks). Folds and topics: results/topic_cv/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
sys.path.insert(0, str(TEAM / "common"))
sys.path.insert(0, str(HERE))
from data import SEED, load_csv, train_val_split  # noqa: E402
from metrics import summary  # noqa: E402
from models import get_model_class  # noqa: E402
from validate_submission import problems  # noqa: E402

RESULTS, ARTIFACTS, SUBS = HERE / "results", HERE / "artifacts", HERE / "kaggle_submissions"
LETTERS = {"stylo_gbm": "D", "tfidf_lr": "E", "transformer_scratch": "F"}  # one Kaggle file per model (A-C: pretrained)
TOPIC_K = 10  # topic clusters for --cv topic; with 5 folds each fold holds out 2 whole clusters


# ------------------------------------------------------------------ helpers
def load_config(name: str) -> dict:
    path = HERE / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config {path.relative_to(TEAM)}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("model", name)
    cfg.setdefault("params", {})
    cfg.setdefault("search", {})
    cfg.setdefault("cv_folds", 5)
    cfg.setdefault("cv", "stratified")
    return cfg


def parse_sets(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def build(cfg: dict, params: dict, seed: int, device: str | None):
    cls = get_model_class(cfg["model"])
    accepted = inspect.signature(cls.__init__).parameters
    kw = dict(params)
    if "seed" in accepted:
        kw.setdefault("seed", seed)
    if "device" in accepted and device:
        kw["device"] = device
    return cls(**kw)


def hardware(model) -> str:
    """The GPU for torch models that ran on one, else the CPU."""
    dev = str(getattr(model, "device_used", "cpu"))
    if dev.startswith("cuda"):
        import torch
        return torch.cuda.get_device_name(torch.device(dev))
    return f"{dev.upper()} ({platform.processor() or platform.machine()})" if dev != "cpu" else         f"CPU ({platform.processor() or platform.machine()})"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()


def log_row(**kw) -> None:
    cmd = [sys.executable, str(TEAM / "common" / "update_submission_log.py"), "--track", "classical"]
    for k, v in kw.items():
        if v is not None:
            cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ inner-CV search
def topic_clusters(texts, k: int = TOPIC_K, seed: int = SEED) -> np.ndarray:
    """Topic of each text: k-means on word TF-IDF fit on these texts only (pass the training rows)."""
    X = TfidfVectorizer(max_features=20000, min_df=3, stop_words="english", sublinear_tf=True).fit_transform(texts)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


def cv_splits(scheme: str, texts, y, n_folds: int, seed: int) -> list:
    """(train_idx, held_out_idx) pairs. `topic`: GroupKFold over the topic clusters, so no topic is on both sides."""
    if scheme == "stratified":
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(texts, y))
    if scheme != "topic":
        raise ValueError(f"unknown CV scheme {scheme!r} (stratified | topic)")
    groups = topic_clusters(texts, TOPIC_K, seed)
    splits = list(GroupKFold(n_splits=n_folds).split(texts, y, groups))
    for f, (_, va) in enumerate(splits):
        if len(set(y[va])) < 2:
            raise ValueError(f"topic fold {f} holds out one class only; AUROC is undefined")
    return splits


def write_topic_folds(texts, y, n_folds: int, seed: int, out: Path) -> None:
    """Per-cluster summary of the topic folds (size, AI share, top terms); no text or row ids."""
    groups = topic_clusters(texts, TOPIC_K, seed)
    vec = TfidfVectorizer(max_features=20000, min_df=3, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    fold_of = {}
    for f, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(texts, y, groups)):
        fold_of.update({g: f for g in set(groups[va])})
    rows = []
    for g in range(TOPIC_K):
        m = groups == g
        top = terms[np.argsort(np.asarray(X[m].mean(axis=0)).ravel())[::-1][:6]]
        rows.append({"cluster": g, "fold": fold_of[g], "n": int(m.sum()), "ai_share": float(y[m].mean()),
                     "top_terms": " ".join(top)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["fold", "cluster"]).to_csv(out / "clusters.csv", index=False,
                                                                 lineterminator="\n", float_format="%.4f")


def inner_cv(cfg, params, texts, y, unl, seed, device, splits=None) -> dict:
    if splits is None:
        splits = cv_splits(cfg["cv"], texts, y, cfg["cv_folds"], seed)
    oof, aucs = np.zeros(len(y)), []
    for f, (tr, va) in enumerate(splits):
        m = build(cfg, params, seed + f, device).fit([texts[i] for i in tr], y[tr], unlabeled_texts=unl)
        oof[va] = m.predict_proba([texts[i] for i in va])
        aucs.append(summary(y[va], oof[va])["auroc"])
    return {"cv_auroc_mean": float(np.mean(aucs)), "cv_auroc_std": float(np.std(aucs)),
            "cv_auroc_pooled": summary(y, oof)["auroc"], "fold_aurocs": [float(a) for a in aucs], "oof": oof}


def search(cfg, params, texts, y, unl, seed, device, out_dir: Path) -> dict:
    grid = cfg["search"]
    keys = list(grid)
    rows, best = [], None
    splits = cv_splits(cfg["cv"], texts, y, cfg["cv_folds"], seed)
    for values in itertools.product(*(grid[k] for k in keys)):
        cand = {**params, **dict(zip(keys, values))}
        t0 = time.time()
        r = inner_cv(cfg, cand, texts, y, unl, seed, device, splits)
        rows.append({**{k: json.dumps(v) for k, v in zip(keys, values)}, "cv": cfg["cv"],
                     **{k: v for k, v in r.items() if k not in ("oof", "fold_aurocs")},
                     "time_s": round(time.time() - t0, 1)})
        print(f"  search {dict(zip(keys, values))}: inner-CV AUROC {r['cv_auroc_mean']:.4f} +/- {r['cv_auroc_std']:.4f}",
              flush=True)
        if best is None or r["cv_auroc_mean"] > best[0]:
            best = (r["cv_auroc_mean"], cand)
    df = pd.DataFrame(rows)
    df["chosen"] = df["cv_auroc_mean"] == df["cv_auroc_mean"].max()
    df.to_csv(out_dir / "search.csv", index=False, lineterminator="\n", float_format="%.6g")
    return best[1]


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="config name in configs/ (and registry name)")
    ap.add_argument("--final", action="store_true", help="refit on 10,000 rows and write the Kaggle CSV")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override params (YAML values)")
    ap.add_argument("--no-search", action="store_true", help="skip the inner-CV grid, use params as given")
    ap.add_argument("--cv", choices=["stratified", "topic"], default=None,
                    help="inner-CV scheme for the search (default: the config's `cv:`, else stratified)")
    ap.add_argument("--tag", default=None, help="save as results/<model>@<tag>/ (variants, ablations)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None, help="torch models: cpu | cuda | cuda:<i> | mps (default: auto)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--author", default=None, help="for the testing history (default: git user.name)")
    ap.add_argument("--no-log", action="store_true", help="don't touch submission_log.csv")
    ap.add_argument("--letter", default=None, help="--final: name the file and log row final_<letter>_<model> "
                                                   "(default: the model's letter in LETTERS)")
    args = ap.parse_args(argv)

    cfg = load_config(args.model)
    if args.cv:
        cfg["cv"] = args.cv
    name = args.model + (f"@{args.tag}" if args.tag else "")
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    author = args.author or git("config", "user.name")
    params = {**cfg["params"], **parse_sets(args.set)}
    unl = load_csv("unlabeled", args.data_dir)["text"].tolist()

    if args.final:
        return final(args, cfg, name, out_dir, params, unl, author)

    train, val = train_val_split(args.data_dir)
    X, y = train["text"].tolist(), train["label"].to_numpy()
    if cfg["cv"] == "topic":
        write_topic_folds(X, y, cfg["cv_folds"], args.seed, RESULTS / "topic_cv")
    if cfg["search"] and not args.no_search:
        print(f"{name}: inner {cfg['cv_folds']}-fold {cfg['cv']} CV on {len(y)} training rows over {cfg['search']}")
        params = search(cfg, params, X, y, unl, args.seed, args.device, out_dir)
        print(f"  chosen: {params}")
    model = build(cfg, params, args.seed, args.device)
    if hasattr(model, "reset_cache"):  # time the fit including feature extraction, not from the search's cache
        model.reset_cache()
    t0 = time.time()
    model.fit(X, y, unlabeled_texts=unl)
    fit_s = time.time() - t0
    p_tr = model.predict_proba(X)
    t1 = time.time()
    p_va = model.predict_proba(val["text"].tolist())
    pred_s = time.time() - t1
    tr_m, va_m = summary(y, p_tr), summary(val["label"], p_va)
    metrics = {
        "model": name, "registry_name": cfg["model"], "params": params, "seed": args.seed,
        "train_auroc": tr_m["auroc"], "val_auroc": va_m["auroc"], "val_tpr_at_1fpr": va_m["tpr_at_1fpr"],
        "n_train": len(y), "n_val": len(val), "fit_time_s": round(fit_s, 1), "val_predict_time_s": round(pred_s, 1),
        "hardware": hardware(model), "searched": bool(cfg["search"] and not args.no_search), "cv": cfg["cv"],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "HEAD"),
    }
    if hasattr(model, "weights"):  # blends: the member weights chosen inside fit()
        metrics["blend_weights"] = model.weights
    if hasattr(model, "history"):
        pd.DataFrame(model.history).to_csv(out_dir / "training_curve.csv", index=False, lineterminator="\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame({"id": val["id"], "label": val["label"], "pred": p_va}).to_csv(
        out_dir / "val_pred.csv", index=False, lineterminator="\n", float_format="%.6g")
    print(f"{name}: train AUROC {tr_m['auroc']:.4f} | val AUROC {va_m['auroc']:.4f} "
          f"(TPR@1%FPR {va_m['tpr_at_1fpr']:.3f}) | fit {fit_s:.0f}s on {metrics['hardware']}")
    if not args.no_log:
        log_row(model=name, val=f"{va_m['auroc']:.5f}", author=author)


def final(args, cfg, name, out_dir, params, unl, author):
    mfile = out_dir / "metrics.json"
    val = json.loads(mfile.read_text(encoding="utf-8")) if mfile.exists() else {}
    if val and not args.set:  # the params chosen by the validation run
        params = val["params"]
    train, test = load_csv("train", args.data_dir), load_csv("test", args.data_dir)
    t0 = time.time()
    model = build(cfg, params, args.seed, args.device).fit(train["text"].tolist(), train["label"].to_numpy(),
                                                          unlabeled_texts=unl)
    fit_s = time.time() - t0
    sub = pd.DataFrame({"id": test["id"], "label": np.clip(model.predict_proba(test["text"].tolist()), 0, 1)})
    bad = problems(sub, load_csv("sample_submission", args.data_dir))
    if bad:
        raise SystemExit("invalid submission: " + "; ".join(bad))
    model.save(ARTIFACTS / name)
    (ARTIFACTS / name / "params.json").write_text(json.dumps({"model": cfg["model"], "params": params,
                                                              "seed": args.seed}, default=str), encoding="utf-8")
    SUBS.mkdir(exist_ok=True)
    letter = args.letter or LETTERS.get(name)
    sub_name = f"final_{letter}_{name}" if letter else name
    path = SUBS / f"{sub_name}.csv"  # the build date is in submission_log.csv
    sub.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")
    meta = {"model": name, "params": params, "seed": args.seed, "n_train": len(train), "fit_time_s": round(fit_s, 1),
            "hardware": hardware(model), "file": str(path.relative_to(TEAM)).replace("\\", "/"),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
    (out_dir / "final.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{name}: refit on {len(train)} rows in {fit_s:.0f}s -> {meta['file']} (validated OK)")
    if not args.no_log:
        # a refit replaces the model's file under the same name, so any earlier score no longer applies
        log_row(model=sub_name, file=meta["file"], author=author, public="pending upload",
                val=f"{val['val_auroc']:.5f}" if val and not args.set else None)


if __name__ == "__main__":
    main()
