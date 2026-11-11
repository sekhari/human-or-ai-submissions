"""Data loading, inspection, and the shared train/validation split.

Every model must obtain its data through ``load_split`` so that all three
approaches train and validate on exactly the same examples.

CLI:
    python -m src.data inspect      # validate CSVs, report duplicates
    python -m src.data split        # create splits/split_seed42.json (once)
    python -m src.data verify       # check the committed split against local data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

TEAM_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TEAM_ROOT / "data"
SPLIT_PATH = TEAM_ROOT / "splits" / "split_seed42.json"
RESULTS_DIR = TEAM_ROOT / "results"

SEED = 42
N_VAL = 2000
EXPECTED_ROWS = {
    "train.csv": 10_000,
    "unlabeled.csv": 3_000,
    "test.csv": 3_000,
    "train_starter_2000.csv": 2_000,
}

# Column names are detected from these candidates (case-insensitive) because the
# exact headers must be confirmed against the real Kaggle files. If detection is
# wrong or ambiguous, pass an explicit ``Columns`` object instead.
TEXT_CANDIDATES = ("text", "passage", "essay", "content", "generation")
LABEL_CANDIDATES = ("label", "generated", "target", "is_ai", "y")
ID_CANDIDATES = ("id", "row_id", "passage_id", "text_id")


@dataclass(frozen=True)
class Columns:
    text: str
    label: str | None
    id: str | None


@dataclass
class LabeledData:
    """Labeled examples in canonical order (sorted by ID if present, else file order)."""

    df: pd.DataFrame
    columns: Columns
    ids: np.ndarray
    texts: list[str]
    labels: np.ndarray
    source_sha256: str


@dataclass
class SplitData:
    train_ids: np.ndarray
    train_texts: list[str]
    train_labels: np.ndarray
    val_ids: np.ndarray
    val_texts: list[str]
    val_labels: np.ndarray
    seed: int


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_column(df: pd.DataFrame, candidates: Sequence[str], kind: str, required: bool) -> str | None:
    by_lower = {c.lower(): c for c in df.columns}
    found = list(dict.fromkeys(by_lower[c.lower()] for c in candidates if c.lower() in by_lower))
    if len(found) > 1:
        raise ValueError(f"Ambiguous {kind} column: {found}. Pass Columns(...) explicitly.")
    if not found:
        if required:
            raise ValueError(
                f"No {kind} column found among {list(df.columns)}. Pass Columns(...) explicitly."
            )
        return None
    return found[0]


def detect_columns(df: pd.DataFrame, require_label: bool = True) -> Columns:
    return Columns(
        text=_pick_column(df, TEXT_CANDIDATES, "text", required=True),
        label=_pick_column(df, LABEL_CANDIDATES, "label", required=require_label),
        id=_pick_column(df, ID_CANDIDATES, "id", required=False),
    )


def read_csv(path: Path | str) -> pd.DataFrame:
    """Read a competition CSV without altering text.

    ``keep_default_na=False`` stops passages such as "NA" or "null" from being
    silently converted to NaN. Empty strings are treated as missing by callers.
    """
    return pd.read_csv(path, keep_default_na=False)


def load_labeled(path: Path | str = DATA_DIR / "train.csv", columns: Columns | None = None) -> LabeledData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download the Kaggle data into {DATA_DIR}/ (see README).")
    df = read_csv(path)
    cols = columns or detect_columns(df, require_label=True)

    texts = df[cols.text].astype(str)
    if (texts.str.strip() == "").any():
        raise ValueError(f"{path.name}: {(texts.str.strip() == '').sum()} empty text values")
    labels = pd.to_numeric(df[cols.label], errors="raise")
    if not set(labels.unique()) <= {0, 1}:
        raise ValueError(f"{path.name}: labels must be 0/1, found {sorted(labels.unique())}")

    if cols.id is not None:
        if df[cols.id].duplicated().any():
            raise ValueError(f"{path.name}: duplicate values in ID column '{cols.id}'")
        df = df.sort_values(cols.id, kind="mergesort").reset_index(drop=True)
        ids = df[cols.id].to_numpy()
    else:
        df = df.reset_index(drop=True)
        ids = np.arange(len(df))  # row position in the file is the stable ID

    return LabeledData(
        df=df,
        columns=cols,
        ids=ids,
        texts=df[cols.text].astype(str).tolist(),
        labels=pd.to_numeric(df[cols.label]).to_numpy(dtype=np.int64),
        source_sha256=file_sha256(path),
    )


def load_unlabeled(path: Path | str, columns: Columns | None = None) -> tuple[np.ndarray, list[str]]:
    """Return (ids, texts) for unlabeled.csv or test.csv, preserving file order."""
    df = read_csv(path)
    cols = columns or detect_columns(df, require_label=False)
    ids = df[cols.id].to_numpy() if cols.id else np.arange(len(df))
    return ids, df[cols.text].astype(str).tolist()


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def make_split(data: LabeledData, seed: int = SEED, n_val: int = N_VAL) -> dict[str, Any]:
    positions = np.arange(len(data.labels))
    train_pos, val_pos = train_test_split(
        positions, test_size=n_val, stratify=data.labels, random_state=seed, shuffle=True
    )
    train_pos, val_pos = np.sort(train_pos), np.sort(val_pos)
    split = {
        "seed": seed,
        "n_total": int(len(positions)),
        "n_train": int(len(train_pos)),
        "n_val": int(len(val_pos)),
        "id_column": data.columns.id,
        "canonical_order": f"sorted by '{data.columns.id}'" if data.columns.id else "file row order",
        "source_sha256": data.source_sha256,
        "train_index": train_pos.tolist(),
        "val_index": val_pos.tolist(),
        "train_ids": _to_json_list(data.ids[train_pos]),
        "val_ids": _to_json_list(data.ids[val_pos]),
    }
    verify_split(split, data)
    return split


def _to_json_list(values: np.ndarray) -> list:
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def verify_split(split: dict[str, Any], data: LabeledData, balance_tol: float = 0.01) -> None:
    """Assert the split is complete, disjoint, stratified, and matches the data."""
    train_pos, val_pos = np.asarray(split["train_index"]), np.asarray(split["val_index"])
    n = len(data.labels)
    assert len(train_pos) == split["n_train"] and len(val_pos) == split["n_val"], "size mismatch"
    assert len(train_pos) + len(val_pos) == n, f"split covers {len(train_pos) + len(val_pos)} of {n} rows"
    assert len(set(train_pos)) == len(train_pos) and len(set(val_pos)) == len(val_pos), "repeated indices"
    assert not set(train_pos) & set(val_pos), "train/validation overlap"
    assert not set(map(str, split["train_ids"])) & set(map(str, split["val_ids"])), "train/validation ID overlap"
    assert [str(i) for i in data.ids[train_pos]] == [str(i) for i in split["train_ids"]], "train IDs do not match data"
    assert [str(i) for i in data.ids[val_pos]] == [str(i) for i in split["val_ids"]], "val IDs do not match data"
    overall = data.labels.mean()
    for name, pos in (("train", train_pos), ("val", val_pos)):
        frac = data.labels[pos].mean()
        assert abs(frac - overall) <= balance_tol, f"{name} AI fraction {frac:.4f} vs overall {overall:.4f}"


def save_split(split: dict[str, Any], path: Path | str = SPLIT_PATH, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. The shared split must not change; use --force only if the team agrees.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")
    return path


def load_split(
    data: LabeledData | None = None,
    path: Path | str = SPLIT_PATH,
    strict_hash: bool = True,
) -> SplitData:
    """Load the shared split. This is the entry point every model should use."""
    data = data or load_labeled()
    split = json.loads(Path(path).read_text())
    if strict_hash and split["source_sha256"] != data.source_sha256:
        raise ValueError(
            "train.csv differs from the file used to create the split "
            f"(expected sha256 {split['source_sha256'][:12]}..., got {data.source_sha256[:12]}...). "
            "Re-download train.csv from Kaggle."
        )
    verify_split(split, data)
    tr, va = np.asarray(split["train_index"]), np.asarray(split["val_index"])
    return SplitData(
        train_ids=data.ids[tr],
        train_texts=[data.texts[i] for i in tr],
        train_labels=data.labels[tr],
        val_ids=data.ids[va],
        val_texts=[data.texts[i] for i in va],
        val_labels=data.labels[va],
        seed=split["seed"],
    )


# --------------------------------------------------------------------------- #
# Inspection / duplicates
# --------------------------------------------------------------------------- #
def normalize_for_dedup(text: str) -> str:
    """Normalization used ONLY for duplicate detection, never for model inputs."""
    return re.sub(r"\s+", " ", text).strip().lower()


def find_near_duplicates(texts: Sequence[str], threshold: float = 0.9, chunk: int = 1000) -> list[tuple[int, int, float]]:
    """Return (i, j, cosine) pairs with i < j and TF-IDF cosine >= threshold."""
    if len(texts) < 2:
        return []
    X = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    pairs = []
    for start in range(0, X.shape[0], chunk):
        sims = (X[start : start + chunk] @ X.T).toarray()
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if i < c:
                pairs.append((int(i), int(c), round(float(sims[r, c]), 4)))
    return pairs


def describe_file(path: Path, require_label: bool) -> dict[str, Any]:
    df = read_csv(path)
    cols = detect_columns(df, require_label=require_label)
    texts = df[cols.text].astype(str)
    lengths = texts.str.split().str.len()
    norm = texts.map(normalize_for_dedup)
    info: dict[str, Any] = {
        "rows": len(df),
        "expected_rows": EXPECTED_ROWS.get(path.name),
        "columns": list(df.columns),
        "detected": cols.__dict__,
        "missing_or_empty": {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns},
        "word_count": {k: float(v) for k, v in lengths.describe().round(1).items()},
        "exact_duplicate_texts": int(texts.duplicated().sum()),
        "normalized_duplicate_texts": int(norm.duplicated().sum()),
    }
    if cols.id:
        info["duplicate_ids"] = int(df[cols.id].duplicated().sum())
    if cols.label:
        info["class_counts"] = {str(k): int(v) for k, v in df[cols.label].value_counts().sort_index().items()}
    return info


def inspect_data(data_dir: Path = DATA_DIR, near_dup_threshold: float = 0.9) -> dict[str, Any]:
    report: dict[str, Any] = {"files": {}, "issues": []}
    present = {}
    for name in EXPECTED_ROWS:
        path = data_dir / name
        if not path.exists():
            report["issues"].append(f"missing file: {name}")
            continue
        present[name] = path
        info = describe_file(path, require_label=name.startswith("train"))
        report["files"][name] = info
        if info["rows"] != info["expected_rows"]:
            report["issues"].append(f"{name}: {info['rows']} rows, expected {info['expected_rows']}")
        if any(info["missing_or_empty"].values()):
            report["issues"].append(f"{name}: missing/empty values {info['missing_or_empty']}")
        if info["normalized_duplicate_texts"]:
            report["issues"].append(f"{name}: {info['normalized_duplicate_texts']} duplicate texts (normalized)")

    norm_sets = {n: set(read_csv(p).pipe(lambda d: d[detect_columns(d, False).text]).astype(str).map(normalize_for_dedup))
                 for n, p in present.items()}
    if "train.csv" in norm_sets and "train_starter_2000.csv" in norm_sets:
        missing = len(norm_sets["train_starter_2000.csv"] - norm_sets["train.csv"])
        report["starter_texts_not_in_train"] = missing
        if missing:
            report["issues"].append(f"{missing} starter texts are not in train.csv (expected subset)")
    for other in ("unlabeled.csv", "test.csv"):
        if "train.csv" in norm_sets and other in norm_sets:
            overlap = len(norm_sets["train.csv"] & norm_sets[other])
            report[f"train_overlap_with_{other}"] = overlap
            if overlap:
                report["issues"].append(f"{overlap} train texts also appear in {other}")

    if "train.csv" in present:
        data = load_labeled(present["train.csv"])
        pairs = find_near_duplicates(data.texts, threshold=near_dup_threshold)
        report["train_near_duplicate_pairs"] = len(pairs)
        report["train_near_duplicate_examples"] = [
            {"id_a": _to_json_list(data.ids[[i]])[0], "id_b": _to_json_list(data.ids[[j]])[0], "cosine": s,
             "labels": [int(data.labels[i]), int(data.labels[j])]}
            for i, j, s in pairs[:20]
        ]
        if SPLIT_PATH.exists():
            split = json.loads(SPLIT_PATH.read_text())
            val = set(split["val_index"])
            cross = [(i, j) for i, j, _ in pairs if (i in val) != (j in val)]
            report["near_duplicate_pairs_crossing_split"] = len(cross)
            if cross:
                report["issues"].append(f"{len(cross)} near-duplicate pairs straddle train/val (leakage risk)")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect", help="validate CSVs and report duplicates")
    p_inspect.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p_inspect.add_argument("--threshold", type=float, default=0.9, help="near-duplicate cosine threshold")
    p_split = sub.add_parser("split", help="create the shared stratified split")
    p_split.add_argument("--force", action="store_true", help="overwrite an existing split file")
    sub.add_parser("verify", help="verify the saved split against local train.csv")
    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        report = inspect_data(args.data_dir, args.threshold)
        out = RESULTS_DIR / "data_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str))
        print(f"\nSaved report to {out}")
    elif args.cmd == "split":
        data = load_labeled()
        path = save_split(make_split(data), overwrite=args.force)
        print(f"Wrote {path}")
        _print_split_summary(load_split(data))
    elif args.cmd == "verify":
        _print_split_summary(load_split())
        print("Split verified: sizes, disjointness, ID alignment, class balance, file hash.")


def _print_split_summary(s: SplitData) -> None:
    for name, y in (("train", s.train_labels), ("val", s.val_labels)):
        print(f"{name:5s}: n={len(y):5d}  human={int((y == 0).sum()):5d}  ai={int((y == 1).sum()):5d}")


if __name__ == "__main__":
    main()
