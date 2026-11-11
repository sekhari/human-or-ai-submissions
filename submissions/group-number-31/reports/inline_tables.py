"""Fill the generated blocks of reports/overleaf/kaggle_report.tex in place, so the report is one self-contained file
(upload the reports/overleaf/ folder to Overleaf: no \\input, no figure files; the one plot is pgfplots).

  python reports/inline_tables.py      # after the track scripts, compare_tracks.py and build_tables.py

A block is the lines between a start marker and an end marker:

  % <<gen file:common/tables/six_models_q12.tex          -> that file's contents
  % <<gen curve:classical/results/experiments/learning_curve.csv:stylo_gbm:val_auroc
                                                          -> pgfplots coordinates (train_rows, column)
  % gen>>

Paths are relative to the team folder. The script only copies what the track scripts generated; it computes
nothing, so every number in the report still comes from a results file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TEAM = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent / "overleaf" / "kaggle_report.tex"
BLOCK = re.compile(r"^(?P<indent>[ \t]*)% <<gen (?P<key>\S+)\n.*?^[ \t]*% gen>>$", re.M | re.S)


def render(key: str) -> str:
    kind, _, arg = key.partition(":")
    if kind == "file":
        return (TEAM / arg).read_text(encoding="utf-8").strip()
    if kind == "curve":
        path, model, col = arg.rsplit(":", 2)
        d = pd.read_csv(TEAM / path)
        d = d[d["model"] == model]
        return " ".join(f"({int(r.train_rows)},{float(r[col]):.4f})" for _, r in d.iterrows())
    raise ValueError(f"unknown block kind in {key!r}")


def main():
    text = REPORT.read_text(encoding="utf-8")
    n = 0

    def fill(m: re.Match) -> str:
        nonlocal n
        n += 1
        ind = m["indent"]
        return f"{ind}% <<gen {m['key']}\n{render(m['key'])}\n{ind}% gen>>"

    text = BLOCK.sub(fill, text)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print(f"filled {n} blocks in {REPORT.relative_to(TEAM)}")


if __name__ == "__main__":
    main()
