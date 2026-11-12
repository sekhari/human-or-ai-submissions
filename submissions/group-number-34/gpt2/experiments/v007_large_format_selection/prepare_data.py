"""Create raw-text classification folds and an unlabeled adaptation split.

CSV parsing preserves every original text value, including line endings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split


RAW_TEXT_POLICY = {
    "name": "exact_parsed_csv_text",
    "version": 1,
    "csv_encoding": "utf-8-sig",
    "newline_translation": False,
    "encoding_repair": False,
    "unicode_normalization": False,
    "whitespace_normalization": False,
    "stripping": False,
    "augmentation": False,
    "adaptation_overlap_rule": "Exclude exact text matches against all labeled rows; retain the first occurrence of each remaining unlabeled text.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv_exact(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if (not columns or len(columns) != len(set(columns))
            or any(None in row or any(value is None for value in row.values()) for row in rows)):
        raise ValueError(f"Malformed CSV: {path}")
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return columns, rows


def validate_frame(path: Path, expected_columns: set[str], *, labels=False, text=False):
    columns, rows = read_csv_exact(path)
    if set(columns) != expected_columns:
        raise ValueError(f"Expected exactly {sorted(expected_columns)} columns in {path}.")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not identifier.strip() for identifier in ids):
        raise ValueError(f"IDs must be nonempty and unique in {path}.")
    if text and any(row["text"] == "" for row in rows):
        raise ValueError(f"Text must not be empty in {path}; whitespace itself is preserved.")
    if labels and set(row["label"] for row in rows) != {"0", "1"}:
        raise ValueError(f"Both exact binary labels 0 and 1 are required in {path}.")
    return columns, rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        # QUOTE_ALL also protects isolated carriage returns inside raw fields.
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, identifiers: list[str]) -> None:
    write_csv(path, ["id"], [{"id": value} for value in identifiers])


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def class_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def csv_identity(path: Path, rows: list[dict]) -> dict:
    result = {"path": str(path.resolve()), "sha256": file_sha256(path),
              "rows": len(rows), "parsed_rows_sha256": value_sha256(rows)}
    if "text" in rows[0]:
        result["id_text_values_sha256"] = value_sha256([[row["id"], row["text"]] for row in rows])
    return result


def verify_existing(output_dir: Path, identity: dict) -> dict:
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    if manifest.get("preparation_identity") != identity:
        raise FileExistsError("Existing prepared data use different inputs, source, or settings. Choose a fresh output directory.")
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Existing prepared artifact failed its frozen hash check: {relative}")
    return manifest


def prepare(args) -> dict:
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    validation_path = args.validation_ids.resolve()
    if args.seed != 42:
        raise ValueError("This experiment fixes the preparation seed at 42.")
    if output_dir == data_dir or output_dir in data_dir.parents:
        raise ValueError("The prepared output directory must not contain or replace the raw input directory.")
    train_columns, train = validate_frame(data_dir / "train.csv", {"id", "text", "label"}, labels=True, text=True)
    test_columns, test = validate_frame(data_dir / "test.csv", {"id", "text"}, text=True)
    unlabeled_columns, unlabeled = validate_frame(data_dir / "unlabeled.csv", {"id", "text"}, text=True)
    sample_columns, sample = validate_frame(data_dir / "sample_submission.csv", {"id", "label"})
    _, validation_ids = validate_frame(validation_path, {"id"})
    outer_ids = [row["id"] for row in validation_ids]
    train_by_id = {row["id"]: row for row in train}
    if not set(outer_ids).issubset(train_by_id):
        raise ValueError("The fixed validation ID list contains IDs absent from train.csv.")
    if set(row["id"] for row in sample) != set(row["id"] for row in test):
        raise ValueError("The sample submission IDs must equal the test IDs exactly.")
    outer_set = set(outer_ids)
    fixed = [train_by_id[identifier] for identifier in outer_ids]
    development = [row for row in train if row["id"] not in outer_set]
    if not args.allow_small_data:
        if (len(train) != 10000 or class_counts(train) != {"0": 5000, "1": 5000}
                or len(test) != 3000 or len(unlabeled) != 3000):
            raise ValueError("Expected 10,000 balanced labeled rows and 3,000 rows in each test/unlabeled file.")
        if len(fixed) != 2000 or class_counts(fixed) != {"0": 1000, "1": 1000}:
            raise ValueError("The original fixed validation set must contain 2,000 rows, 1,000 per class.")
        if len(development) != 8000 or class_counts(development) != {"0": 4000, "1": 4000}:
            raise ValueError("The development set must contain 8,000 rows, 4,000 per class.")
    if (len(development) % 5 or min(class_counts(development).values(), default=0) < 5
            or set(class_counts(development)) != {"0", "1"} or set(class_counts(fixed)) != {"0", "1"}):
        raise ValueError("Development must support five equal stratified folds; development and fixed validation need both classes.")

    sources = {filename: csv_identity(data_dir / filename, rows) for filename, rows in (
        ("train.csv", train), ("test.csv", test), ("unlabeled.csv", unlabeled), ("sample_submission.csv", sample))}
    sources["validation_ids.csv"] = csv_identity(validation_path, validation_ids)
    identity = {"seed": args.seed, "allow_small_data": args.allow_small_data,
                "source_sha256": {name: value["sha256"] for name, value in sources.items()},
                "preparation_source_sha256": file_sha256(Path(__file__).resolve()),
                "sklearn_version": sklearn.__version__, "raw_text_policy": RAW_TEXT_POLICY}
    if output_dir.exists():
        if (output_dir / "split_manifest.json").is_file():
            manifest = verify_existing(output_dir, identity)
            print(json.dumps({"event": "completed_preparation_verified", "output_dir": str(output_dir)}), flush=True)
            return manifest
        if any(output_dir.iterdir()):
            raise FileExistsError("The prepared output directory must be empty or contain a verified completed preparation.")

    labeled_text_ids = defaultdict(list)
    for row in train:
        labeled_text_ids[row["text"]].append(row["id"])
    retained, exclusions, first_unlabeled = [], [], {}
    for row in unlabeled:
        text = row["text"]
        if text in labeled_text_ids:
            exclusions.append({"id": row["id"], "reason": "exact_text_matches_labeled",
                "matched_labeled_ids": json.dumps(labeled_text_ids[text], separators=(",", ":")),
                "duplicate_of_unlabeled_id": "", "text_sha256": text_sha256(text)})
        elif text in first_unlabeled:
            exclusions.append({"id": row["id"], "reason": "duplicate_unlabeled_text",
                "matched_labeled_ids": "[]", "duplicate_of_unlabeled_id": first_unlabeled[text],
                "text_sha256": text_sha256(text)})
        else:
            first_unlabeled[text] = row["id"]
            retained.append(row)
    if len(retained) < 10:
        raise ValueError("At least 10 unique unlabeled passages without labeled-text overlaps are required for adaptation.")
    lm_validation_count = math.ceil(len(retained) * 0.1)
    lm_train_indices, lm_validation_indices = train_test_split(
        list(range(len(retained))), test_size=lm_validation_count, shuffle=True, random_state=args.seed)
    lm_train = [retained[index] for index in sorted(lm_train_indices)]
    lm_validation = [retained[index] for index in sorted(lm_validation_indices)]
    lm_train_ids = [row["id"] for row in lm_train]
    lm_validation_ids = [row["id"] for row in lm_validation]
    assert set(row["text"] for row in lm_train).isdisjoint(row["text"] for row in lm_validation)
    assert set(row["text"] for row in retained).isdisjoint(labeled_text_ids)

    folds, inner_coverage = [], []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_indices, inner_indices) in enumerate(splitter.split(development, [row["label"] for row in development])):
        fit = [development[int(index)] for index in fit_indices]
        inner = [development[int(index)] for index in inner_indices]
        fit_ids, inner_ids = [row["id"] for row in fit], [row["id"] for row in inner]
        if (len(inner) != len(development) // 5 or set(fit_ids) & set(inner_ids)
                or (set(fit_ids) | set(inner_ids)) & outer_set):
            raise ValueError(f"Invalid classification fold boundary: {fold}")
        inner_coverage.extend(inner_ids)
        folds.append({"fold": fold, "train_ids_csv": f"folds/fold_{fold}_train_ids.csv",
                      "validation_ids_csv": f"folds/fold_{fold}_validation_ids.csv",
                      "train_ids": fit_ids, "validation_ids": inner_ids,
                      "train_rows": len(fit), "validation_rows": len(inner),
                      "train_class_counts": class_counts(fit), "validation_class_counts": class_counts(inner)})
    if len(inner_coverage) != len(set(inner_coverage)) or set(inner_coverage) != set(row["id"] for row in development):
        raise ValueError("The five inner validation folds must cover development exactly once.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output_dir.name + ".preparing-", dir=output_dir.parent))
    try:
        # Keep untouched byte copies when the entire original table is retained.
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copyfile(data_dir / name, staged / name)
        shutil.copyfile(validation_path, staged / "fixed_validation_ids.csv")
        write_csv(staged / "development.csv", train_columns, development)
        write_csv(staged / "fixed_validation.csv", train_columns, fixed)
        write_csv(staged / "unlabeled_development.csv", unlabeled_columns, retained)
        write_ids(staged / "unlabeled_validation_ids.csv", lm_validation_ids)
        write_ids(staged / "unlabeled_train_ids.csv", lm_train_ids)
        write_csv(staged / "unlabeled_exclusions.csv", ["id", "reason", "matched_labeled_ids", "duplicate_of_unlabeled_id", "text_sha256"], exclusions)
        for fold in folds:
            write_ids(staged / fold["train_ids_csv"], fold["train_ids"])
            write_ids(staged / fold["validation_ids_csv"], fold["validation_ids"])
        preserved = {}
        for name, expected in (("train.csv", train), ("test.csv", test), ("sample_submission.csv", sample),
                ("development.csv", development), ("fixed_validation.csv", fixed), ("unlabeled_development.csv", retained)):
            _, actual = read_csv_exact(staged / name)
            if actual != expected:
                raise ValueError(f"Raw parsed values changed while writing {name}.")
            preserved[name] = {"rows": len(actual), "parsed_rows_sha256": value_sha256(actual),
                               "all_retained_values_identical": True}
        # Guard against an input file changing while preparation was running.
        for source in sources.values():
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"Input changed during preparation: {source['path']}")
        fixed_texts = set(row["text"] for row in fixed)
        development_texts = set(row["text"] for row in development)
        audit = {"raw_text_policy": RAW_TEXT_POLICY, "sources": sources,
                 "parsed_value_preservation": preserved,
                 "text_transformations_applied": 0,
                 "labeled_exact_duplicate_text_groups": sum(len(ids) > 1 for ids in labeled_text_ids.values()),
                 "fixed_development_shared_exact_texts": len(fixed_texts & development_texts),
                 "fixed_ids_preserved_in_source_order": True,
                 "adaptation_source_rows": len(unlabeled), "adaptation_retained_rows": len(retained),
                 "adaptation_excluded_rows": len(exclusions),
                 "adaptation_exclusion_reasons": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
                 "adaptation_labeled_exact_text_overlap_after_exclusion": 0,
                 "adaptation_train_validation_exact_text_overlap": 0,
                 "adaptation_train_rows": len(lm_train), "adaptation_validation_rows": len(lm_validation),
                 "adaptation_validation_fraction_rule": "ceil(0.10 * unique eligible unlabeled rows)",
                 "classification_fixed_holdout_used_for_adaptation": False,
                 "note": "Labeled text is read only for exact-overlap exclusion; no labeled text or labels enter language-model adaptation."}
        write_json(staged / "raw_audit.json", audit)
        manifest = {"schema_version": 1, "seed": args.seed, "raw_text_policy": RAW_TEXT_POLICY,
            "preparation_identity": identity, "sources": sources,
            "fixed_validation": {"csv": "fixed_validation.csv", "ids_csv": "fixed_validation_ids.csv",
                "ids": outer_ids, "rows": len(fixed), "class_counts": class_counts(fixed),
                "ids_source_sha256": sources["validation_ids.csv"]["sha256"]},
            "development": {"csv": "development.csv", "ids": [row["id"] for row in development],
                "rows": len(development), "class_counts": class_counts(development)},
            "labeled_all": {"csv": "train.csv", "ids": [row["id"] for row in train],
                "rows": len(train), "class_counts": class_counts(train)},
            "test": {"csv": "test.csv", "sample_submission": "sample_submission.csv", "rows": len(test)},
            "folds": folds,
            "adaptation": {"csv": "unlabeled_development.csv", "train_ids_csv": "unlabeled_train_ids.csv",
                "validation_ids_csv": "unlabeled_validation_ids.csv", "train_ids": lm_train_ids,
                "validation_ids": lm_validation_ids, "retained_rows": len(retained),
                "train_rows": len(lm_train), "validation_rows": len(lm_validation),
                "excluded_rows": len(exclusions), "exclusions_csv": "unlabeled_exclusions.csv"},
            "files": {str(path.relative_to(staged)): file_sha256(path)
                      for path in sorted(staged.rglob("*")) if path.is_file()}}
        write_json(staged / "split_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staged, output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(json.dumps({"event": "raw_preparation_complete", "output_dir": str(output_dir),
                      "development_rows": len(development), "fixed_validation_rows": len(fixed),
                      "adaptation_train_rows": len(lm_train), "adaptation_validation_rows": len(lm_validation),
                      "adaptation_excluded_rows": len(exclusions), "text_transformations_applied": 0}), flush=True)
    return manifest


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--validation_ids", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_small_data", action="store_true", help="Use small integration fixtures.")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(arguments())
