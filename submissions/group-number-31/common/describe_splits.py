"""Observable profile of each data split (no labels used beyond train/val class shares).

  python common/describe_splits.py [--data-dir PATH]

Writes common/results/split_profile.csv: rows, AI share (labelled splits), verse / short-line / prose
shares by metrics.line_form(), passages with a blank line, with zero-width characters, and mean words.
The reports cite it for how test's line structure differs from the labelled data.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from data import load_csv, train_val_split
from metrics import line_form, word_count

ZW = re.compile("[​‌‍⁠﻿]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    tr, va = train_val_split(args.data_dir)
    rows = []
    for name, d in [("train (8,000)", tr), ("validation (2,000)", va), ("unlabeled", load_csv("unlabeled", args.data_dir)),
                    ("test", load_csv("test", args.data_dir))]:
        form = d["text"].map(line_form)
        rows.append({"split": name, "rows": len(d), "ai_share": d["label"].mean() if "label" in d else "",
                     "verse": int((form == "verse").sum()), "verse_share": (form == "verse").mean(),
                     "short_lines_share": (form == "short lines (lists, recipes)").mean(),
                     "prose_share": (form == "prose").mean(), "blank_line_share": d["text"].str.contains("\n\n").mean(),
                     "zero_width_passages": int(d["text"].str.contains(ZW).sum()),
                     "mean_words": word_count(d["text"]).mean()})
    out = Path(__file__).resolve().parent / "results" / "split_profile.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n", float_format="%.4g")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
