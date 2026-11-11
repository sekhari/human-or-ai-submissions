"""Load a saved run and predict any CSV with a `text` column (default: test.csv).

  python predict.py --run outputs/runs/<run_name>
  python predict.py --run outputs/runs/<run_name> --input data/unlabeled.csv --output preds.csv
  python predict.py --run outputs/runs/<run_name> --device mps   # e.g. model trained on CUDA, used on a Mac

Runs that used `final.strategy: fold_average` are predicted with the mean of
their fold models, which exist only if they were trained with
`output.save_fold_models=true`. Their test predictions are already in
<run>/test_pred.csv either way.
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

import pandas as pd  # noqa: E402

from src.config import read_yaml, set_data_dir  # noqa: E402
from src.data import load_csv  # noqa: E402
from src.inference import find_run, model_dirs, predict_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/runs/<run_name> or just <run_name>")
    ap.add_argument("--input", default=None, help="CSV with id,text (default: config test_file)")
    ap.add_argument("--output", default=None, help="where to write id,label (default: <run>/predictions.csv)")
    ap.add_argument("--device", default=None, help="auto | cuda | cuda:<i> | mps | cpu (torch models only)")
    ap.add_argument("--data-dir", default=None, help="folder with the Kaggle CSVs (default: $HOA_DATA_DIR, ./data, common/data.py)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)

    run = find_run(args.run)
    cfg = read_yaml(run / "config.yaml")
    df = load_csv(args.input or cfg["data"]["test_file"])
    n_models = len(model_dirs(run))
    out = pd.DataFrame({"id": df["id"], "label": predict_run(run, df["text"], args.device)})
    out_path = Path(args.output) if args.output else run / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions ({n_models} model(s)) to {out_path}")


if __name__ == "__main__":
    main()
