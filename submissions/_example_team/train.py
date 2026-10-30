"""Example training entrypoint (replace with your pipeline)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Human-or-AI detector")
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    args = parser.parse_args()

    # TODO: load data, fit model, save artifact
    print(f"[example] would train on {args.train} with seed={args.seed}")
    print(f"[example] would write model to {args.model_out}")
    raise SystemExit(
        "Replace this example script with your real training code before submitting."
    )


if __name__ == "__main__":
    main()
