"""Record the public-leaderboard score Kaggle gave a submission, then refresh
the leaderboard table in AGENTS.md.

  python scripts/record_kaggle_score.py --run 20261028-103845_tfidf_logreg --score 0.9312
  python scripts/record_kaggle_score.py --run outputs/submissions/<session>__best.csv --score 0.95
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve  # noqa: E402
from src.leaderboard import update_agents_md  # noqa: E402
from src.utils import file_lock  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name (folder name under outputs/runs)")
    ap.add_argument("--score", required=True, type=float)
    ap.add_argument("--results", default="experiments/results.csv")
    args = ap.parse_args()
    path = resolve(args.results)
    run = Path(args.run).name.removesuffix(".csv")
    with file_lock(path):
        df = pd.read_csv(path, dtype=str)
        if run not in set(df["run_name"]):
            # submissions written by run_all ensembles are named <session>__best
            df = pd.concat([df, pd.DataFrame([{"run_name": run, "model": "submission", "config": run}])],
                           ignore_index=True)
            print(f"{run} was not in {path.name}; added a row for it")
        df.loc[df["run_name"] == run, "kaggle_public"] = str(args.score)
        df.to_csv(path, index=False)
    update_agents_md(path)
    print(f"Recorded Kaggle public {args.score} for {run}")


if __name__ == "__main__":
    main()
