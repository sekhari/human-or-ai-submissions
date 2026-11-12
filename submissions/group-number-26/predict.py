"""Kaggle prediction entrypoint for the Group 26 submission.

Thin wrapper around `team_knn.py`, which holds Kevin's implementation of the
selected model: uniform-vote KNN with k = 15 over the 23 stylometric features.
This module contains **no modelling logic of its own**. It normalises the
command-line arguments and calls `team_knn.main()`, so the fitted model, the
feature pipeline and the written probabilities are byte-for-byte what
`team_knn.py --submission` already produces.

What the underlying pipeline does: fits the selected configuration on **all
10,000** labeled passages (no holdout at this stage) and writes an `id,label`
CSV with exactly **3,000** rows in `test.csv` order, where `label` is P(AI).

Run from the repository root or from this directory:

    python predict.py --out submission.csv
    python submissions/group-number-26/predict.py --out submission.csv

The competition CSVs are located through the shared configuration, so
`GROUP26_DATA_DIR` and `--data` both work. The underlying script refuses to
overwrite an existing file, so remove or rename an old submission first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import config  # noqa: E402
import features  # noqa: E402
import team_knn  # noqa: E402
from course_models import default_data_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the selected uniform-vote KNN (k=15) on all 10,000 labeled "
            "passages and write the Kaggle submission CSV."
        ),
        epilog=(
            f"Selected model: k={team_knn.K} uniform-vote KNN over "
            f"{features.N_FEATURES} stylometric features. All modelling code "
            "lives in team_knn.py."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=config.DEFAULT_SUBMISSION,
        help=(
            "Output CSV path "
            f"(default: {config.DEFAULT_SUBMISSION.name}). Must not exist."
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Directory holding the competition CSVs (default: autodetected, "
             "honours GROUP26_DATA_DIR).",
    )
    parser.add_argument(
        "--feature-set", choices=("all", "scalar"), default="all",
        help="'all' = the 23 features (reported model, default); 'scalar' = "
             "the 10 scalar features (shift-robust variant, see improvement.md).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    argv = ["team_knn.py", "--submission", str(args.out)]
    argv += ["--data", str(args.data if args.data else default_data_dir())]
    argv += ["--feature-set", args.feature_set]

    print(f"[predict.py] delegating to team_knn.py: {' '.join(argv[1:])}",
          flush=True)
    sys.argv = argv
    team_knn.main()
    print(f"[predict.py] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
