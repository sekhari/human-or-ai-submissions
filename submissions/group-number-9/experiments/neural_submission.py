"""Refit a frozen neural experiment on all labeled data; never upload to Kaggle.

Usage:
    python -m experiments.neural_submission
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.make_submission import check_submission
from src.data import load_split
from src.neural_models import fitted_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "results/neural_comparison/refined/selected.json")
    parser.add_argument("--model", default="combined_mlp")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/neural_combined_mlp.csv")
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    matches = [m for m in selected["models"] if m["name"] == args.model]
    if len(matches) != 1:
        raise SystemExit(f"Expected one configuration for {args.model}; found {len(matches)}.")
    if args.output.exists():
        raise SystemExit(f"Preserving existing {args.output}; choose a different --output.")
    config = matches[0]
    train, test = load_split("train"), load_split("test")
    model = fitted_pipeline(config["representation"], config["classifier"], config["params"],
                            selected["components"], selected["seed"])
    with threadpool_limits(limits=2):
        model.fit(train, train.label.astype(int))
        predictions = model.predict_proba(test)[:, 1]
    sub = pd.DataFrame({"id": test.id, "label": predictions})
    check_submission(sub, test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    manifest = {"configuration": config, "svd_components": selected["components"],
                "seed": selected["seed"], "training_rows": len(train),
                "prediction_rows": len(sub), "selection_path": str(args.selection.relative_to(ROOT)) if args.selection.is_relative_to(ROOT) else str(args.selection),
                "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                "csv_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "uploaded": False}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote and checked {args.output}: {len(sub)} probabilities. No upload performed.")


if __name__ == "__main__":
    main()
