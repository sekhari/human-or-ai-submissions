"""Write a checked Kaggle submission with one command.

Computes any missing pretrained-model features (about 70 minutes the first time on Apple silicon,
cached under cache/ afterwards), fits the chosen model on all 10,000 labeled rows, and scores test.csv.

Usage:
    python predict.py --out submission.csv                       # final model
    python predict.py --method medium_bag --out submission.csv   # another experiments.improve_lm method
"""
import argparse
from pathlib import Path

from experiments import extract_neural, improve_lm

FINAL_METHOD = "all_blend"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(improve_lm.METHODS), default=FINAL_METHOD)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()
    extract_neural.main()
    improve_lm.submit(args.method, args.out)


if __name__ == "__main__":
    main()
