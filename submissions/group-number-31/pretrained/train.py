"""Train + validate one model, then produce test predictions and a Kaggle submission.

Examples
--------
  python train.py --model tfidf_logreg
  python train.py --model tfidf_logreg --set model.params.C=10 cv.scheme=cluster
  python train.py --model stylometric_gbm --set model.params.estimator=random_forest
  python train.py --model tfidf_logreg --set data.train_frac=0.25          # learning curve
  python train.py --model tfidf_logreg --set semi_supervised.enabled=true  # pseudo-labelling
  python train.py --model ft_roberta_base --device cuda                 # fine-tune on an NVIDIA GPU
  python train.py --model lm_gpt2 --device mps                           # Apple silicon
  python train.py --model tfidf_logreg --quick                             # 2k-example smoke test
  python train.py --config path/to/custom.yaml

--model takes the name of any file in configs/models/ (without .yaml).

Test predictions (`final.strategy` in the config):
  refit        : after CV, refit on all labeled data and predict test (default; cheap models)
  fold_average : average the test predictions of the CV fold models (no extra training;
                 default for fine-tuned transformers)

Every run writes outputs/runs/<run_name>/ (config.yaml, oof.csv, metrics.json,
test_pred.csv, unlabeled_pred.csv, status.json, model files),
outputs/submissions/<run_name>.csv, appends a row to experiments/results.csv
and refreshes the leaderboard table in AGENTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import shutil
import sys

# Apple-silicon friendliness: fall back to CPU for any op MPS lacks; quiet HF tokenizers.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.config import load_config, resolve, set_data_dir
from src.data import load_competition_data, load_csv, load_external, make_splits, subsample, subsample_n
from src.leaderboard import update_agents_md
from src.models import build_model, get_model_class
from src.preprocessing import preprocess
from src.utils import (append_result, auroc, binary_metrics, ensure_dir, free_torch_memory,
                       git_user, make_run_name, seed_everything, to_json, torch_runtime_info)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="config name in configs/models/ (e.g. tfidf_logreg, lm_gpt2)")
    ap.add_argument("--config", help="explicit config file instead of configs/models/<model>.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. model.params.C=8 cv.n_folds=3")
    ap.add_argument("--name", help="optional tag appended to the run name")
    ap.add_argument("--run-name", help="exact run name (used by run_all.py)")
    ap.add_argument("--notes", default="", help="free-text note stored in experiments/results.csv")
    ap.add_argument("--cv-only", action="store_true", help="skip test predictions + submission")
    ap.add_argument("--no-cv", action="store_true", help="skip CV, only fit on all data + predict test")
    ap.add_argument("--quick", action="store_true", help="use train_starter_2000.csv and 3 folds")
    ap.add_argument("--no-log", action="store_true", help="don't append to experiments/results.csv")
    ap.add_argument("--device", help="auto | cuda | cuda:<i> | mps | cpu (overrides config `device:`)")
    ap.add_argument("--data-dir", help="folder with the Kaggle CSVs (default: $HOA_DATA_DIR, ./data, common/data.py)")
    return ap.parse_args(argv)


def model_factory(cfg):
    mcfg = dict(cfg["model"])
    params = dict(mcfg.get("params") or {})
    accepted = inspect.signature(get_model_class(mcfg["name"]).__init__).parameters
    if "seed" in accepted:
        params.setdefault("seed", cfg["seed"])
    if "device" in accepted:  # only torch-based models take a device
        params.setdefault("device", cfg.get("device", "auto"))
    mcfg["params"] = params
    return lambda: build_model(mcfg)


def release(model) -> None:
    if hasattr(model, "release"):
        model.release()
    free_torch_memory()


def _make(make, ckpt_dir):
    model = make()
    if ckpt_dir is not None and hasattr(model, "set_checkpoint_dir"):
        model.set_checkpoint_dir(ckpt_dir)
    return model


def fit_model(make, cfg, X, y, X_val=None, y_val=None, pools=None, unlabeled_texts=None, ckpt_dir=None):
    """Fit one model; with semi_supervised.enabled, do pseudo-label rounds:
    fit -> score the pool -> add the most confident predictions -> refit.
    ckpt_dir: per-epoch checkpoint directory for models that support it (fine-tunes)."""
    model = _make(make, ckpt_dir).fit(X, y, X_val, y_val, unlabeled_texts=unlabeled_texts)
    ss = cfg["semi_supervised"]
    if not ss.get("enabled") or not pools:
        return model
    pool = [t for name in ss.get("pool", ["unlabeled"]) for t in pools[name]]
    for r in range(ss.get("rounds", 1)):
        p = model.predict_proba(pool)
        k = int(len(pool) * ss["confident_frac"] / 2)
        order = np.argsort(p)
        neg, pos = order[:k], order[-k:]
        X_aug = list(X) + [pool[i] for i in pos] + [pool[i] for i in neg]
        y_aug = np.concatenate([y, np.ones(k, int), np.zeros(k, int)])
        print(f"  pseudo-label round {r + 1}: +{2 * k} texts "
              f"(AI thr {p[pos[0]]:.3f}, human thr {p[neg[-1]]:.3f})")
        release(model)
        model = _make(make, ckpt_dir / f"round_{r + 1}" if ckpt_dir else None)
        model.fit(X_aug, y_aug, X_val, y_val, unlabeled_texts=unlabeled_texts)
    return model


def fold_fingerprint(cfg, va) -> str:
    """Identifies a fold's result: everything that affects training (not output paths / device) + its split."""
    keep = {k: v for k, v in cfg.items() if k not in ("output", "device")}
    h = hashlib.md5(json.dumps(keep, sort_keys=True, default=str).encode("utf-8"))
    h.update(np.asarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()


def load_fold(fold_dir, f, fingerprint, need):
    """A fold finished by an earlier attempt of this run (same run name), or None."""
    meta_path, arr_path = fold_dir / f"fold_{f}.json", fold_dir / f"fold_{f}.npz"
    if not (meta_path.exists() and arr_path.exists()):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except json.JSONDecodeError:  # killed while writing it -> the fold is redone
        return None
    arrs = dict(np.load(arr_path))
    if meta.get("fingerprint") != fingerprint or any(k not in arrs for k in need):
        return None
    return meta, arrs


def save_fold(fold_dir, f, meta, arrs) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    tmp = fold_dir / f"fold_{f}.tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, fold_dir / f"fold_{f}.npz")
    write_json(fold_dir / f"fold_{f}.json", meta)  # written last: its presence marks the fold as done


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run(args) -> dict:
    set_data_dir(args.data_dir)
    overrides = list(args.set)
    if args.quick:
        overrides += ["data.train_file=data/train_starter_2000.csv", "cv.n_folds=3"]
    if args.device:
        overrides.append(f"device={args.device}")
    cfg = load_config(args.model, args.config, overrides)
    cfg.setdefault("final", {}).setdefault("strategy", "refit")
    model_name = cfg["model"]["name"]
    config_name = args.model or os.path.splitext(os.path.basename(args.config))[0]
    dev = "cpu"
    if "device" in inspect.signature(get_model_class(model_name).__init__).parameters:
        from src.utils import device as resolve_device
        dev = resolve_device(cfg["model"].get("params", {}).get("device") or cfg.get("device"))
        print(f"Torch device: {dev}")  # fails here, before any output is written, if unavailable
    seed_everything(cfg["seed"])
    run_name = args.run_name or make_run_name(config_name, args.name or ("quick" if args.quick else None))
    run_dir = ensure_dir(f"{cfg['output']['runs_dir']}/{run_name}")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    write_json(run_dir / "status.json", {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"Run: {run_name}\n{yaml.safe_dump({'model': cfg['model']}, sort_keys=False)}")
    t_start = time.time()

    try:
        metrics = _run(args, cfg, run_name, run_dir, model_name, config_name, dev)
    except BaseException as e:
        write_json(run_dir / "status.json", {"status": "failed", "error": repr(e),
                                             "traceback": traceback.format_exc()})
        raise
    metrics["total_time_s"] = round(time.time() - t_start, 1)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "status.json", {"status": "ok", "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "cv_auc": metrics.get("cv_auc")})
    if not cfg["output"].get("keep_checkpoints", False):
        shutil.rmtree(resolve(cfg["output"].get("checkpoints_dir", "outputs/checkpoints")) / run_name,
                      ignore_errors=True)

    if not args.no_log and "cv_auc" in metrics:
        append_result(cfg, {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"), "run_name": run_name, "model": model_name,
            "config": config_name, "author": git_user(), "cv_scheme": cfg["cv"]["scheme"],
            "n_folds": len(metrics["fold_aucs"]), "cv_auc": round(metrics["cv_auc"], 5),
            "cv_auc_std": round(metrics["cv_auc_std"], 5),
            "fold_aucs": " ".join(f"{a:.4f}" for a in metrics["fold_aucs"]),
            "tpr_at_1fpr": round(metrics["oof_metrics"]["tpr_at_1fpr"], 4),
            "train_frac": cfg["data"]["train_frac"],
            "semi_supervised": cfg["semi_supervised"]["enabled"],
            "external_files": " ".join(cfg["data"]["external_files"] or []),
            "preprocessing": to_json({k: v for k, v in cfg["preprocessing"].items() if v}),
            "params": to_json(cfg["model"].get("params")),
            "device": metrics.get("gpu_name", dev), "train_time_s": metrics["total_time_s"],
            "notes": ("[quick] " if args.quick else "") + args.notes,
            "eval_auc": "",  # column kept for the older rows of the log; no run scores extra labelled sets now
        })
        update_agents_md(resolve(cfg["output"]["results_file"]))
        print(f"Logged to {cfg['output']['results_file']} (AGENTS.md leaderboard refreshed)")
    return metrics


def _run(args, cfg, run_name, run_dir, model_name, config_name, dev) -> dict:
    # ------------------------------------------------------------ data
    dfs = load_competition_data(cfg)
    ext = load_external(cfg)
    pp = cfg["preprocessing"]
    train_texts = preprocess(dfs["train"]["text"], pp, "preprocess train")
    test_texts = preprocess(dfs["test"]["text"], pp, "preprocess test")
    unl_texts = preprocess(dfs["unlabeled"]["text"], pp, "preprocess unlabeled")
    ext_texts = preprocess(ext["text"], pp, "preprocess external") if len(ext) else []
    ext_y = ext["label"].to_numpy(int) if len(ext) else np.array([], int)
    y = dfs["train"]["label"].to_numpy(int)
    pools = {"unlabeled": unl_texts, "test": test_texts}
    unsup_texts = unl_texts + test_texts  # for unsupervised steps (e.g. vocabulary), never labels
    print(f"train {len(train_texts)} | external {len(ext_texts)} | unlabeled {len(unl_texts)} | test {len(test_texts)}")

    make = model_factory(cfg)
    metrics = {"run_name": run_name, "model": model_name, "config": config_name, "device": dev,
               "cv_scheme": cfg["cv"]["scheme"], "final_strategy": cfg["final"]["strategy"]}

    # Feature-extractor models (LM perplexity, embeddings): compute + cache features for
    # every text once, so the pretrained network is loaded a single time.
    probe = make()
    if hasattr(probe, "precompute"):
        t0 = time.time()
        probe.precompute(list(dict.fromkeys(train_texts + list(ext_texts) + test_texts + unl_texts)))
        metrics["feature_time_s"] = round(time.time() - t0, 1)
    del probe

    fold_avg = cfg["final"]["strategy"] == "fold_average" and not args.no_cv and not args.cv_only
    test_fold_preds, unl_fold_preds = [], []
    out_cfg = cfg["output"]
    ckpt_root = resolve(out_cfg.get("checkpoints_dir", "outputs/checkpoints")) / run_name
    fold_dir = run_dir / "folds"

    # ------------------------------------------------------------ cross-validation
    if not args.no_cv:
        splits = make_splits(train_texts, y, cfg["cv"], cfg["seed"], ids=dfs["train"]["id"].to_numpy())
        fixed = cfg["cv"]["scheme"] == "fixed"
        train_pred = np.full(len(y), np.nan)  # fixed scheme: in-sample predictions on the training rows
        oof = np.full(len(y), np.nan)
        fold_of = np.full(len(y), -1)
        fold_aucs, fold_times, fit_times, histories = [], [], [], []
        need = ["test", "unl"] if fold_avg else []
        for f, (tr, va) in enumerate(tqdm(splits, desc=f"CV ({cfg['cv']['scheme']})")):
            t0 = time.time()
            fp = fold_fingerprint(cfg, va)
            done = load_fold(fold_dir, f, fp, need) if out_cfg.get("resume_folds", True) else None
            if done is not None:  # finished by an earlier attempt of this run (run_all --resume)
                meta, arrs = done
                oof[va], fold_of[va] = arrs["oof"], f
                if fixed and "train" in arrs:
                    train_pred = arrs["train"]
                if fold_avg:
                    test_fold_preds.append(arrs["test"])
                    unl_fold_preds.append(arrs["unl"])
                fold_aucs.append(meta["auc"])
                fold_times.append(meta["time_s"])
                fit_times.append(meta.get("fit_s", meta["time_s"]))
                histories.append(meta.get("history", []))
                tqdm.write(f"  fold {f + 1}/{len(splits)}: AUROC {meta['auc']:.4f}  (done in an earlier attempt)")
                continue
            seed_everything(cfg["seed"] + f)  # a fold's randomness doesn't depend on which folds were skipped
            tr = subsample(tr, y, cfg["data"]["train_frac"], cfg["seed"] + f)
            tr = subsample_n(tr, y, cfg["data"].get("train_n"), cfg["seed"])
            X_tr = [train_texts[i] for i in tr] + list(ext_texts)
            y_tr = np.concatenate([y[tr], ext_y])
            X_va = [train_texts[i] for i in va]
            model = fit_model(make, cfg, X_tr, y_tr, X_va, y[va], pools, unsup_texts,
                              ckpt_dir=ckpt_root / f"fold_{f}")
            fit_s = round(time.time() - t0, 1)
            oof[va] = model.predict_proba(X_va)
            fold_of[va] = f
            arrs = {"oof": oof[va]}
            if fixed:
                train_pred[tr] = model.predict_proba([train_texts[i] for i in tr])
                arrs["train"] = train_pred
            if fold_avg:
                test_fold_preds.append(model.predict_proba(test_texts))
                unl_fold_preds.append(model.predict_proba(unl_texts))
                arrs.update(test=test_fold_preds[-1], unl=unl_fold_preds[-1])
                if out_cfg.get("save_fold_models", False):
                    model.save(run_dir / f"fold_{f}")
            histories.append(list(getattr(model, "history", None) or []))
            release(model)
            del model
            fold_aucs.append(auroc(y[va], oof[va]))
            fold_times.append(round(time.time() - t0, 1))
            fit_times.append(fit_s)
            save_fold(fold_dir, f, {"fingerprint": fp, "auc": fold_aucs[-1], "time_s": fold_times[-1], "fit_s": fit_s,
                                    "n_train": len(X_tr), "history": histories[-1]}, arrs)
            if not out_cfg.get("keep_checkpoints", False):  # the fold's predictions are on disk now
                shutil.rmtree(ckpt_root / f"fold_{f}", ignore_errors=True)
            tqdm.write(f"  fold {f + 1}/{len(splits)}: AUROC {fold_aucs[-1]:.4f}  "
                       f"(n_train={len(X_tr)}, {fold_times[-1]:.0f}s)")
        if any(histories):
            metrics["epoch_history"] = histories
            per_epoch = {}
            for h in histories:
                for e in h:
                    if "val_auc" in e:
                        per_epoch.setdefault(e["epoch"], []).append(e["val_auc"])
            metrics["epoch_val_auc_mean"] = {k: round(float(np.mean(v)), 5) for k, v in sorted(per_epoch.items())}
            print("Mean fold-validation AUROC per epoch: " +
                  ", ".join(f"{k}: {v:.4f}" for k, v in metrics["epoch_val_auc_mean"].items()))
        if np.isnan(oof[fold_of >= 0]).any():
            raise RuntimeError("NaN in out-of-fold predictions -- the model produced invalid scores")
        seen = fold_of >= 0
        cv_auc = auroc(y[seen], oof[seen])
        metrics.update(cv_auc=cv_auc, cv_auc_mean_of_folds=float(np.mean(fold_aucs)),
                       cv_auc_std=float(np.std(fold_aucs)), fold_aucs=fold_aucs, fold_times_s=fold_times,
                       oof_metrics=binary_metrics(y[seen], oof[seen]), n_oof=int(seen.sum()))
        pd.DataFrame({"id": dfs["train"]["id"], "label": y, "fold": fold_of, "pred": oof}).to_csv(
            run_dir / "oof.csv", index=False)
        if fixed:  # cv_auc above is the validation AUROC on the 2,000 fixed rows
            fit_rows = ~np.isnan(train_pred)
            metrics.update(val_auc=cv_auc, train_auc=auroc(y[fit_rows], train_pred[fit_rows]),
                           n_train=int(fit_rows.sum()), fit_time_s=fit_times[0])
            pd.DataFrame({"id": dfs["train"]["id"], "label": y, "pred": train_pred})[fit_rows].to_csv(
                run_dir / "train_pred.csv", index=False)
            print(f"Fixed split: train AUROC {metrics['train_auc']:.4f} (n={fit_rows.sum()}), "
                  f"val AUROC {cv_auc:.4f} (n={len(splits[0][1])}), fit {fit_times[0]:.0f}s")
        print(f"\nCV AUROC ({cfg['cv']['scheme']}, {len(splits)} fold(s)): {cv_auc:.4f}  "
              f"[folds: {', '.join(f'{a:.4f}' for a in fold_aucs)}]  "
              f"TPR@1%FPR {metrics['oof_metrics']['tpr_at_1fpr']:.3f}")

    # ------------------------------------------------------------ test predictions + submission
    if not args.cv_only:
        t0 = time.time()
        if fold_avg:
            print(f"\nTest predictions = mean of {len(test_fold_preds)} fold models")
            p_test, p_unl = np.mean(test_fold_preds, 0), np.mean(unl_fold_preds, 0)
        else:
            print("\nFitting final model on all labeled data ...")
            X_all = list(train_texts) + list(ext_texts)
            y_all = np.concatenate([y, ext_y])
            model = fit_model(make, cfg, X_all, y_all, pools=pools, unlabeled_texts=unsup_texts,
                              ckpt_dir=ckpt_root / "final")
            if cfg["output"].get("save_models", True):
                model.save(run_dir)
            p_test = model.predict_proba(test_texts)
            p_unl = model.predict_proba(unl_texts)
            release(model)
        metrics["final_time_s"] = round(time.time() - t0, 1)
        sub = pd.DataFrame({"id": dfs["test"]["id"], "label": np.clip(p_test, 0, 1)})
        n_test = len(dfs["test"])
        if not (len(sub) == n_test and sub["id"].is_unique and sub["label"].between(0, 1).all()
                and sub["label"].notna().all()):
            raise RuntimeError("Invalid test predictions (wrong length, duplicate ids, NaN or out of [0,1])")
        sub.to_csv(run_dir / "test_pred.csv", index=False)
        pd.DataFrame({"id": dfs["unlabeled"]["id"], "label": p_unl}).to_csv(run_dir / "unlabeled_pred.csv", index=False)
        sub_path = ensure_dir(cfg["output"]["submissions_dir"]) / f"{run_name}.csv"
        sub.to_csv(sub_path, index=False)
        metrics["test_pred_mean"] = float(sub["label"].mean())
        metrics["test_pred_std"] = float(sub["label"].std())
        print(f"Submission written: {sub_path}")

    metrics.update(torch_runtime_info())
    return metrics


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
