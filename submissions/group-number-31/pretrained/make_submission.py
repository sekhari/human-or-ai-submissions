"""Make a Kaggle submission file from one or more trained runs.

  python make_submission.py --run <run_name>                       # one model
  python make_submission.py --run <A> <B> <C> --weights 2 1 1      # weighted rank-average blend
  python make_submission.py --run <A> --cached                     # reuse <run>/test_pred.csv (no model loading)
  python make_submission.py --check outputs/submissions/<file>.csv # only validate an existing file

A run is a directory in outputs/runs/ (or its name). By default each run's SAVED MODEL
re-predicts data/test.csv with the run's own preprocessing: the refit model for
`final.strategy: refit` runs, the mean of the fold models for `fold_average` runs (fine-tunes;
saved since 2026-10-30, `output.save_fold_models`). Runs without a saved model fall back
to the test predictions stored at training time (identical numbers, just not recomputed).

Output: outputs/submissions/<name>.csv (id,label), validated against
data/sample_submission.csv (3000 rows, same ids, labels in [0, 1], no NaN). For a
blend, the same weights are applied to the runs' out-of-fold predictions so you
see its CV AUROC before uploading. Record the Kaggle score afterwards with
  python scripts/record_kaggle_score.py --run outputs/submissions/<name>.csv --score <x>
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.config import load_config, read_yaml, set_data_dir  # noqa: E402
from src.data import load_csv  # noqa: E402
from src.ensemble import cdf_mean, load_runs, rank_mean  # noqa: E402
from src.inference import find_run, has_model, predict_run, validate_submission  # noqa: E402
from src.utils import auroc, ensure_dir, tpr_at_fpr  # noqa: E402


def check(path, sample_file) -> bool:
    problems = validate_submission(pd.read_csv(path), sample_file)
    print(f"{path}: " + ("OK - ready to upload" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems)))
    return not problems


def test_predictions(run: Path, test: pd.DataFrame, cached: bool, device: str | None) -> pd.Series:
    stored = run / "test_pred.csv"
    if cached or not has_model(run):
        if not stored.exists():
            raise FileNotFoundError(f"{run.name}: no saved model and no test_pred.csv")
        why = "--cached" if cached else "no saved model"
        print(f"  {run.name}: using stored test predictions ({why})")
        return pd.read_csv(stored).set_index("id")["label"].reindex(test["id"])
    print(f"  {run.name}: predicting {len(test)} test passages with the saved model ...")
    pred = pd.Series(predict_run(run, test["text"], device), index=test["id"].to_numpy())
    if stored.exists():  # sanity check: the reloaded model should reproduce the training-time predictions
        old = pd.read_csv(stored).set_index("id")["label"].reindex(pred.index)
        print(f"    matches training-time test_pred.csv: max |diff| {np.abs(old - pred).max():.2e}, "
              f"Spearman {spearmanr(old, pred).statistic:.5f}")
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", help="run directories or names in outputs/runs/")
    ap.add_argument("--weights", nargs="+", type=float, help="blend weights, one per run (default: equal)")
    ap.add_argument("--name", default=None, help="output file name (default: the run name, or blend_<runs>)")
    ap.add_argument("--blend", choices=["cdf", "rank"], default="cdf",
                    help="cdf (default): map each model's test scores through its OOF-score CDF; "
                         "rank: rank within test")
    ap.add_argument("--cached", action="store_true", help="use stored test predictions instead of the saved models")
    ap.add_argument("--device", default=None, help="auto | cuda | cuda:<i> | mps | cpu (torch models only)")
    ap.add_argument("--check", default=None, help="only validate this submission CSV")
    ap.add_argument("--data-dir", default=None, help="folder with the Kaggle CSVs (default: $HOA_DATA_DIR, ./data, common/data.py)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)

    cfg = load_config("tfidf_logreg")  # default.yaml paths only
    sample_file = cfg["data"]["sample_submission_file"]
    if args.check:
        sys.exit(0 if check(args.check, sample_file) else 1)
    if not args.run:
        ap.error("--run is required (or --check <file>)")
    if args.weights and len(args.weights) != len(args.run):
        ap.error(f"{len(args.weights)} weights for {len(args.run)} runs")

    runs = [find_run(r, cfg["output"]["runs_dir"]) for r in args.run]
    test = load_csv(cfg["data"]["test_file"])
    print(f"{len(runs)} run(s):")
    for r in runs:
        m = read_yaml(r / "metrics.json") if (r / "metrics.json").exists() else {}
        print(f"  {r.name}: CV AUROC {m.get('cv_auc', float('nan')):.4f}")
    preds = pd.DataFrame({r.name: test_predictions(r, test, args.cached, args.device) for r in runs})

    if len(runs) == 1:
        label = preds.iloc[:, 0].to_numpy()
    else:  # weighted average on a common scale (AUROC only cares about order)
        O, _, y, names = load_runs(runs)
        T = preds.copy()
        T.columns = names
        if args.blend == "cdf":
            oof, label = cdf_mean(O, T, names, args.weights)
        else:
            oof, _ = rank_mean(O, T, names, args.weights)
            _, label = rank_mean(T, T, names, args.weights)
        print(f"Blend ({args.blend}) out-of-fold AUROC {auroc(y, oof):.4f}, TPR@1%FPR {tpr_at_fpr(y, oof, 0.01):.3f} "
              f"on {len(y)} rows (best single {max(auroc(y, O[n]) for n in names):.4f})")
        label = np.asarray(label)

    name = args.name or (runs[0].name if len(runs) == 1 else "blend_" + "+".join(r.name.split("__")[-1] for r in runs))
    out = ensure_dir(cfg["output"]["submissions_dir"]) / f"{name}.csv"
    sub = pd.DataFrame({"id": test["id"], "label": np.clip(label, 0, 1)})
    sub.to_csv(out, index=False)
    print(f"Wrote {out}")
    sys.exit(0 if check(out, sample_file) else 1)


if __name__ == "__main__":
    main()
