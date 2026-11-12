"""Training and validation entrypoint for the Group 26 submission.

Thin wrapper around `team_knn.py`, which holds Kevin's implementation of the
selected model: uniform-vote KNN with k = 15 over the 23 stylometric features.
This module deliberately contains **no modelling logic of its own**. It
normalises the command-line arguments and then calls `team_knn.main()`, so the
training, validation split, hyperparameters and metrics are exactly the ones
`team_knn.py` already produces. Delegating rather than re-implementing means
the two entrypoints cannot drift apart.

The course template asks every submission for a `train.py`; this provides that
name without duplicating or altering the pipeline behind it.

Run from the repository root or from this directory:

    python train.py --out results/validation_run
    python submissions/group-number-26/train.py --out results/validation_run

The competition CSVs are located through the shared configuration, so
`GROUP26_DATA_DIR` and `--data` both work. The underlying script refuses to
overwrite an existing output directory, so pick a fresh `--out` on each run.
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
from course_models import SEED, default_data_dir  # noqa: E402

DEFAULT_OUT = config.RESULTS_DIR / "validation_run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected uniform-vote KNN (k=15) on the shared "
            "Group 26 validation split and write the report metrics."
        ),
        epilog=(
            f"Validation split seed {SEED}; the selected model is "
            f"k={team_knn.K} uniform-vote KNN over {features.N_FEATURES} "
            "stylometric features. All modelling code lives in team_knn.py."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=(
            "New directory for metrics.json and validation_scores.csv "
            f"(default: {DEFAULT_OUT}). Must not already exist."
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Directory holding the competition CSVs (default: autodetected, "
             "honours GROUP26_DATA_DIR).",
    )
    parser.add_argument(
        "--validation-ids", type=Path, default=None,
        help=f"Shared validation ID file (default: {config.VALIDATION_IDS.name}).",
    )
    parser.add_argument(
        "--partner-logistic", type=Path, default=None,
        help="Optional partner probability CSV to score on the same IDs.",
    )
    parser.add_argument(
        "--partner-svm", type=Path, default=None,
        help="Optional partner decision-score CSV to score on the same IDs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    argv = ["team_knn.py", "--out", str(args.out)]
    argv += ["--data", str(args.data if args.data else default_data_dir())]
    argv += ["--validation-ids",
             str(args.validation_ids if args.validation_ids
                 else config.VALIDATION_IDS)]
    if args.partner_logistic:
        argv += ["--partner-logistic", str(args.partner_logistic)]
    if args.partner_svm:
        argv += ["--partner-svm", str(args.partner_svm)]

    print(f"[train.py] delegating to team_knn.py: {' '.join(argv[1:])}",
          flush=True)
    sys.argv = argv
    team_knn.main()


if __name__ == "__main__":
    main()
