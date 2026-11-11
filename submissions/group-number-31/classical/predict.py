"""Write submission.csv from a model saved by `python train.py --model <name> --final`.

  python predict.py --model tfidf_lr [--out submission.csv] [--data-dir PATH]

Loads artifacts/<name>/ (git-ignored; re-create it with train.py --final), predicts P(AI) for every
test.csv id, validates the file against sample_submission.csv and writes it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from data import load_csv  # noqa: E402
from models import get_model_class  # noqa: E402
from validate_submission import problems  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    art = HERE / "artifacts" / args.model
    if not (art / "params.json").exists():
        sys.exit(f"{art} not found - run `python train.py --model {args.model} --final` first")
    reg = json.loads((art / "params.json").read_text(encoding="utf-8"))["model"]
    model = get_model_class(reg).load(art)
    test = load_csv("test", args.data_dir)
    sub = pd.DataFrame({"id": test["id"], "label": np.clip(model.predict_proba(test["text"].tolist()), 0, 1)})
    bad = problems(sub, load_csv("sample_submission", args.data_dir))
    if bad:
        sys.exit("invalid submission: " + "; ".join(bad))
    sub.to_csv(args.out, index=False, lineterminator="\n", float_format="%.6f")
    print(f"wrote {args.out} ({len(sub)} rows, validated)")


if __name__ == "__main__":
    main()
