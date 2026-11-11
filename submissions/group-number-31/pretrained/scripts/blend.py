"""Blend several finished runs into one submission.

  python scripts/blend.py --runs outputs/runs/<runA> outputs/runs/<runB> [--weights 2 1]
  python scripts/blend.py --runs outputs/runs/<A> outputs/runs/<B> outputs/runs/<C> --method greedy
  python scripts/blend.py --runs ... --method stack

Methods: rank (weighted rank average, default) | greedy (forward selection) | stack (logistic regression).
Runs with different CV layouts are fine: the blend is evaluated on the rows where
every run has an out-of-fold prediction. The blend is logged to experiments/results.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve  # noqa: E402
from src.ensemble import greedy, load_runs, rank_mean, stack_lr  # noqa: E402
from src.leaderboard import update_agents_md  # noqa: E402
from src.utils import append_result, auroc, ensure_dir, git_user, make_run_name, tpr_at_fpr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, help="only for --method rank")
    ap.add_argument("--method", choices=["rank", "greedy", "stack"], default="rank")
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    O, T, y, names = load_runs([resolve(r) for r in args.runs])
    print(f"{len(names)} runs, {len(y)} common OOF rows")
    for n in names:
        print(f"  {n}: {auroc(y, O[n]):.4f}")
    if args.method == "rank":
        oof, test = rank_mean(O, T, names, args.weights)
        cv, desc = auroc(y, oof), " + ".join(f"{n}*{w}" for n, w in zip(names, args.weights or [1] * len(names)))
    elif args.method == "greedy":
        cv, oof, test, w = greedy(O, T, y)
        desc = " + ".join(f"{n}*{v:.2f}" for n, v in w.items())
    else:
        cv, oof, test, w = stack_lr(O, T, y)
        desc = " + ".join(f"{n}*{v:.2f}" for n, v in w.items())
    print(f"Blend ({args.method}) AUROC: {cv:.4f}   [{desc}]")

    cfg = load_config("tfidf_logreg")  # only for output paths
    run_name = make_run_name(f"blend_{args.method}", args.name)
    out_dir = ensure_dir(f"{cfg['output']['runs_dir']}/{run_name}")
    sub = pd.DataFrame({"id": T.index, "label": test})
    sub.to_csv(out_dir / "test_pred.csv", index=False)
    sub_path = ensure_dir(cfg["output"]["submissions_dir"]) / f"{run_name}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission written: {sub_path}")
    if not args.no_log:
        append_result(cfg, {"timestamp": run_name[:15], "run_name": run_name, "model": "blend",
                            "config": f"blend_{args.method}", "author": git_user(),
                            "cv_scheme": "common-OOF", "cv_auc": round(cv, 5),
                            "tpr_at_1fpr": round(tpr_at_fpr(y, oof, 0.01), 4), "train_frac": 1.0,
                            "semi_supervised": False, "notes": args.notes, "params": desc})
        update_agents_md(resolve(cfg["output"]["results_file"]))


if __name__ == "__main__":
    main()
