"""Example prediction entrypoint (replace with your pipeline)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Kaggle submission.csv")
    parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    # TODO: load model, predict P(AI) for each test id, write submission.csv
    print(f"[example] would predict on {args.test} using {args.model}")
    print(f"[example] would write {args.out}")
    raise SystemExit(
        "Replace this example script with your real prediction code before submitting."
    )


if __name__ == "__main__":
    main()
