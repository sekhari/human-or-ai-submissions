"""Refit a tuned model on all 10,000 labeled rows and write a checked Kaggle submission.

Usage (after experiments.model_comparison):
    python -m experiments.make_submission --model char_lr
    # writes submissions/<model>.csv

Rank-average ensemble of experiments.robustness variants:
    python -m experiments.make_submission --ensemble masked_lr_tuned style_tuned --name ensemble_masked_style
    # writes submissions/<name>.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.robustness import VARIANTS, fit_and_score, rank_average
from src.data import load_split
from src.models import MODELS, SEED, build_tuned, scores, to_probability

ROOT = Path(__file__).resolve().parents[1]


def check_submission(sub, test):
    assert list(sub.columns) == ["id", "label"], f"bad columns: {list(sub.columns)}"
    assert len(sub) == len(test) == 3000, f"bad row count: {len(sub)}"
    assert (sub.id.values == test.id.values).all(), "ids differ from test.csv"
    assert sub.id.is_unique, "duplicate ids"
    assert sub.label.notna().all() and sub.label.between(0, 1).all(), "labels outside [0, 1]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODELS))
    group.add_argument("--ensemble", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--name", help="output file stem for --ensemble")
    args = ap.parse_args()
    if args.ensemble and not args.name:
        ap.error("--ensemble requires --name")

    train, test = load_split("train"), load_split("test")
    if args.ensemble:
        parts = [fit_and_score(v, train, {"test": test})[1]["test"] for v in args.ensemble]
        labels, stem, params = rank_average(parts), args.name, {v: VARIANTS[v] for v in args.ensemble}
    else:
        params = json.loads((ROOT / "results" / "model_comparison" / "best_params.json").read_text())[args.model]
        spec = MODELS[args.model]
        model = build_tuned(args.model, params).fit(spec.inputs(train), train.label.astype(int).values)
        labels, stem = to_probability(scores(model, spec.inputs(test))), args.model
    sub = pd.DataFrame({"id": test.id, "label": labels})
    check_submission(sub, test)

    out = ROOT / "submissions" / f"{stem}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} (seed={SEED}, params={params})")
    print(sub.label.describe().round(4).to_string())


if __name__ == "__main__":
    main()
