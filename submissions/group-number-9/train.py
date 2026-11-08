"""Train and evaluate on the labeled validation protocol.

Default: compare the final model and its alternatives. Each method is scored by leave-one-domain-out
on the 8,000-row training partition and on the fixed 2,000-row validation set (splits/val_ids.csv).

--full reruns every report experiment in order. It overwrites files under results/, and retrains the
transformer several times, so expect several hours on a laptop.

Usage:
    python train.py          # writes results/improve_lm/
    python train.py --full
"""
import argparse
import subprocess
import sys

FINAL = ["experiments.extract_neural", "experiments.improve_lm"]
FULL = [
    ["src.eda"],
    ["experiments.model_comparison"],
    ["experiments.learning_curve"],
    ["experiments.robustness"],
    ["experiments.unlabeled"],
    ["experiments.extract_neural"],
    ["experiments.stacking"],
    ["experiments.finetune", "--jobs", "val", "rare", "oof8k", "test", "oof10k"],
    ["experiments.stack_transformer"],
    ["experiments.improve"],
    ["experiments.improve_lm"],
    ["experiments.rank10_validation"],
    ["experiments.error_analysis"],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    for step in FULL if args.full else [[m] for m in FINAL]:
        print("==>", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True)


if __name__ == "__main__":
    main()
