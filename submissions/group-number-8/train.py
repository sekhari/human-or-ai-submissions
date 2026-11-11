"""Train and evaluate the group-number-8 detectors.

Report mode holds out the fixed validation split and writes results/metrics.json.
Submit mode fits ENS_top2 on all labeled rows and writes submission.csv.
"""

from __future__ import annotations

import argparse
import joblib
from pathlib import Path

from pipeline.config import SPLIT_SEED
from pipeline.data import load_split
from pipeline.experiments import fit_ensembles, run_report
from pipeline.models import submission_frame


def _data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    if args.train is None:
        raise SystemExit("Pass --data-dir or --train")
    return args.train.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Human-or-AI detectors")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None, help="train.csv; parent is the data dir")
    parser.add_argument("--unlabeled", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--mode", choices=("report", "submit"), default="report")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--metrics-out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--figure-out", type=Path, default=Path("figures/learning_curve.pdf"))
    parser.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    data_dir = _data_dir(args)
    train_path = args.train or (data_dir / "train.csv")
    unlabeled_path = args.unlabeled or (data_dir / "unlabeled.csv")
    test_path = args.test or (data_dir / "test.csv")
    train_df = load_split(train_path, labeled=True)
    unlabeled_df = load_split(unlabeled_path, labeled=False)

    if args.mode == "report":
        run_report(train_df, unlabeled_df, args.metrics_out, args.figure_out, seed=args.seed)
        return

    test_df = load_split(test_path, labeled=False)
    labels, bundle = fit_ensembles(train_df, unlabeled_df, test_df)
    top2 = submission_frame(test_df["id"], labels["ens_top2"])
    top3 = submission_frame(test_df["id"], labels["ens_top3"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    top2.to_csv(args.out, index=False)
    top3_path = args.out.with_name("submission_ens_top3.csv")
    top3.to_csv(top3_path, index=False)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print(f"Wrote {args.out} ({len(top2)} rows), {top3_path} ({len(top3)} rows), {args.model_out}")


if __name__ == "__main__":
    main()
