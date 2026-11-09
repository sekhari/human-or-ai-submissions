"""Step 0 - data check. Prints per-file counts of suspicious characters and word-length stats.

Usage: python data_check.py [--data-dir data]
"""
import argparse
import collections
import csv
import os
import statistics
import unicodedata

csv.field_size_limit(10**9)

ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "᠎", "‎", "‏"}
# Unusual whitespace: any Unicode space separator other than ASCII space, plus tab/CR/LF/other controls
NORMAL_WS = {" "}


def is_unusual_ws(ch):
    if ch in NORMAL_WS:
        return False
    return ch.isspace() or unicodedata.category(ch) in ("Zs", "Zl", "Zp")


def is_lookalike_script(ch):
    if ord(ch) < 128:
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return name.startswith("CYRILLIC") or name.startswith("GREEK")


def read_texts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row["text"] for row in csv.DictReader(f)]


def report(path):
    texts = read_texts(path)
    n = len(texts)
    c = dict(non_ascii=0, zero_width=0, unusual_ws=0, lookalike=0)
    word_lens = []
    for t in texts:
        if any(ord(ch) > 127 for ch in t):
            c["non_ascii"] += 1
        if any(ch in ZERO_WIDTH for ch in t):
            c["zero_width"] += 1
        if any(is_unusual_ws(ch) for ch in t if ch != "\n"):
            c["unusual_ws"] += 1
        if any(is_lookalike_script(ch) for ch in t):
            c["lookalike"] += 1
        word_lens.extend(len(w) for w in t.split())
    print(f"== {os.path.basename(path)} ({n} passages) ==")
    for k, v in c.items():
        print(f"  {k:<12} {v:>6}  ({100 * v / max(n, 1):.2f}%)")
    if word_lens:
        s = sorted(word_lens)
        print(
            f"  word length: mean={statistics.mean(s):.2f} median={statistics.median(s)} "
            f"std={statistics.pstdev(s):.2f} min={s[0]} p99={s[int(0.99 * (len(s) - 1))]} max={s[-1]}"
        )
    print()


def by_label_report(path):
    """train.csv only: suspicious-character rates and top non-ASCII chars per label."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checks = {
        "non_ascii": lambda t: any(ord(ch) > 127 for ch in t),
        "zero_width": lambda t: any(ch in ZERO_WIDTH for ch in t),
        "unusual_ws": lambda t: any(is_unusual_ws(ch) for ch in t if ch != "\n"),
        "lookalike": lambda t: any(is_lookalike_script(ch) for ch in t),
    }
    labels = sorted({r["label"] for r in rows})
    groups = {l: [r["text"] for r in rows if r["label"] == l] for l in labels}
    print("== train.csv by label ==")
    print(f"  {'feature':<12}" + "".join(f"  label {l}: n / rate".ljust(26) for l in labels))
    for name, fn in checks.items():
        cells = []
        for l in labels:
            k = sum(fn(t) for t in groups[l])
            cells.append(f"  {k:>5} / {100 * k / len(groups[l]):6.2f}%".ljust(26))
        print(f"  {name:<12}" + "".join(cells))
    print("  (label sizes: " + ", ".join(f"{l}={len(groups[l])}" for l in labels) + ")")
    for l in labels:
        cnt = collections.Counter(ch for t in groups[l] for ch in t if ord(ch) > 127)
        print(f"\n  Top 10 non-ASCII chars, label {l}:")
        for ch, k in cnt.most_common(10):
            print(f"    {k:>7}  U+{ord(ch):04X}  {ch!r}  {unicodedata.name(ch, '<unnamed>')}")
    print()


def char_rate_report(data_dir):
    """Passage-level rates (share of passages containing the char) for selected chars."""
    chars = ["…", "—", "½"]

    def fmt(texts):
        return "  ".join(
            f"{c} {sum(c in t for t in texts):>5} ({100 * sum(c in t for t in texts) / len(texts):5.2f}%)"
            for c in chars
        )

    print("== passage-level rates for … — ½ ==")
    with open(os.path.join(data_dir, "train.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for l in sorted({r["label"] for r in rows}):
        texts = [r["text"] for r in rows if r["label"] == l]
        print(f"  train label {l} (n={len(texts)}): {fmt(texts)}")
    for name in ("unlabeled.csv", "test.csv"):
        texts = read_texts(os.path.join(data_dir, name))
        print(f"  {name:<19}(n={len(texts)}): {fmt(texts)}")
    counts = [t.count("…") for r in rows if r["label"] == "1" for t in [r["text"]]]
    with_ell = [c for c in counts if c > 0]
    print(f"  train label 1: {len(with_ell)} distinct passages contain '…'; "
          f"max count in one passage = {max(counts)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    for name in ("train.csv", "unlabeled.csv", "test.csv"):
        report(os.path.join(args.data_dir, name))
    by_label_report(os.path.join(args.data_dir, "train.csv"))
    char_rate_report(args.data_dir)


if __name__ == "__main__":
    main()
