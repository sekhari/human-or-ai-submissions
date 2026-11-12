"""Distance-weighted KNN baseline over the 23 stylometric features.

Evaluates k = 15, 35 and 75 on the shared Group 26 validation split and writes
`results/knn_comparison.csv`.

Protocol:
  * 8,000 training passages, 2,000 validation passages, taken from
    `group26_validation_ids.txt`.
  * `StandardScaler` fitted on the training features only, then applied to
    validation.
  * Euclidean distance, `weights="distance"`.
  * Training AUROC uses self-excluded neighbours, so a passage never votes for
    itself. Reporting it any other way would give a meaningless ~1.0.

Run from `submissions/group-number-26/`:

    python experiments/knn_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Make the package modules importable when run as a script from either
# submissions/group-number-26/ or the repository root.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import config  # noqa: E402
import data_loading as dl  # noqa: E402
import features  # noqa: E402
import models  # noqa: E402


def run(k_values=config.KNN_K_GRID, train_path: Path | None = None,
        ids_path: Path | None = None) -> pd.DataFrame:
    rows = dl.load_labeled(train_path)
    train, validation = dl.split_labeled(rows, ids_path)
    models.assert_no_leakage(
        [r["id"] for r in train], [r["id"] for r in validation]
    )

    start = perf_counter()
    raw_train = features.featurize([r["text"] for r in train])
    raw_val = features.featurize([r["text"] for r in validation])
    featurize_seconds = perf_counter() - start

    scaler = models.fit_scaler(raw_train)          # training rows only
    x_train = scaler.transform(raw_train)
    x_val = scaler.transform(raw_val)
    y_train = np.array([int(r["label"]) for r in train])
    y_val = np.array([int(r["label"]) for r in validation])

    print(f"featurized {len(train):,} training and {len(validation):,} "
          f"validation passages in {featurize_seconds:.2f}s "
          f"({features.N_FEATURES} features each)")
    print(f"{'k':<5}{'train AUROC*':>14}{'val AUROC':>12}"
          f"{'fit s':>9}{'predict s':>12}")

    records = []
    for k in k_values:
        model = models.build_knn(k)

        t0 = perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = perf_counter() - t0

        t0 = perf_counter()
        val_scores = models.knn_scores(model, x_val)
        predict_seconds = perf_counter() - t0

        t0 = perf_counter()
        train_scores = models.knn_self_excluded_train_scores(model)
        train_score_seconds = perf_counter() - t0

        train_auroc = float(roc_auc_score(y_train, train_scores))
        val_auroc = float(roc_auc_score(y_val, val_scores))

        records.append({
            "model": "distance_weighted_knn",
            "k": int(k),
            "weights": "distance",
            "metric": "euclidean",
            "n_features": features.N_FEATURES,
            "n_train": len(train),
            "n_val": len(validation),
            "train_auroc_self_excluded": round(train_auroc, 6),
            "val_auroc": round(val_auroc, 6),
            "fit_seconds": round(fit_seconds, 4),
            "val_predict_seconds": round(predict_seconds, 4),
            "train_score_seconds": round(train_score_seconds, 4),
            "featurize_seconds": round(featurize_seconds, 3),
            "split_source": config.VALIDATION_IDS.name,
            "split_seed": config.SPLIT_SEED,
        })
        print(f"{k:<5}{train_auroc:>14.4f}{val_auroc:>12.4f}"
              f"{fit_seconds:>9.3f}{predict_seconds:>12.3f}")

    print("*each training passage is excluded from its own vote "
          "(predict_proba(None)); the scaler is fitted on all 8,000 "
          "training rows")

    df = pd.DataFrame(records)
    best = df.sort_values(["val_auroc", "k"], ascending=[False, True]).iloc[0]
    print(f"\nbest: k={int(best['k'])}, validation AUROC {best['val_auroc']:.4f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=None,
                        help="Path to train.csv (default: autodetected).")
    parser.add_argument("--validation-ids", type=Path, default=None,
                        help="Path to group26_validation_ids.txt.")
    parser.add_argument("--k", type=int, nargs="+", default=list(config.KNN_K_GRID))
    parser.add_argument("--out", type=Path,
                        default=config.RESULTS_DIR / "knn_comparison.csv")
    args = parser.parse_args()

    df = run(args.k, args.train, args.validation_ids)

    config.ensure_results_dir()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
