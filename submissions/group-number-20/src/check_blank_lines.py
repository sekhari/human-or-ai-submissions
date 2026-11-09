# Check the blank-line (\n\n) signal in train / unlabeled / test.
# Usage (inside submissions/group-number-20/):
#   python src/check_blank_lines.py --train data/train.csv --unlabeled data/unlabeled.csv --test data/test.csv
import argparse
import pandas as pd


def find_text_col(df):
    # Use "text" if it exists, otherwise the longest string column
    if "text" in df.columns:
        return "text"
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def has_blank_line(s, normalize=True):
    # normalize=False is the same rule as the pipeline (raw "\n\n")
    # normalize=True also counts Windows blank lines ("\r\n\r\n")
    s = s.fillna("").astype(str)
    if normalize:
        s = s.str.replace("\r\n", "\n", regex=False)
    return s.str.contains("\n\n", regex=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--unlabeled", default=None)
    ap.add_argument("--test", default=None)
    args = ap.parse_args()

    # Part 1: train, blank line vs label
    tr = pd.read_csv(args.train)
    col = find_text_col(tr)
    raw = has_blank_line(tr[col], normalize=False)
    print(f"train rows: {len(tr)}, text column: {col}")
    print(f"raw \\n\\n (pipeline rule): {raw.sum()} rows, human={int((raw & (tr.label == 0)).sum())}")
    tr["blank"] = has_blank_line(tr[col])
    print("below: \\r\\n counted as a line break too")
    print(f"rows with \\n\\n: {tr['blank'].sum()} ({tr['blank'].mean():.1%})")
    print("\ncrosstab (rows = has \\n\\n, cols = label):")
    print(pd.crosstab(tr["blank"], tr["label"], margins=True))

    # Step 2: how much of each class has blank lines
    for lab in sorted(tr["label"].unique()):
        sub = tr[tr["label"] == lab]
        print(f"label={lab}: {sub['blank'].mean():.1%} have \\n\\n")

    # Part 2: compare rate with unlabeled / test (no labels there)
    for name, path in [("unlabeled", args.unlabeled), ("test", args.test)]:
        if path is None:
            continue
        df = pd.read_csv(path)
        b = has_blank_line(df[find_text_col(df)])
        print(f"\n{name} rows: {len(df)}, with \\n\\n: {b.sum()} ({b.mean():.1%})")


if __name__ == "__main__":
    main()
