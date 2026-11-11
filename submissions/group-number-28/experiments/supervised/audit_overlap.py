"""ID-only, exact lexical overlap audit; no model fitting or split modification.

Run: python -m experiments.supervised.audit_overlap --output results/supervised/RUN
The prefix filter is complete for set Jaccard >= the frozen threshold. Candidate
pairs are verified using unhashed 5-token tuples, avoiding hash-collision matches.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

from human_ai.data import ROOT, load_partitions, raw_path, read_csv, sha256, write_csv


TOKEN_PATTERN = r"[^\W\d_]+"
THRESHOLD = 0.8
SHINGLE_SIZE = 5


def tokens(text):
    return re.findall(TOKEN_PATTERN, text.lower(), flags=re.UNICODE)


def shingles(sequence):
    return {tuple(sequence[i:i + SHINGLE_SIZE])
            for i in range(max(0, len(sequence) - SHINGLE_SIZE + 1))}


def hashed_shingles(sequence):
    return {int.from_bytes(hashlib.blake2b("\x1f".join(item).encode(), digest_size=8).digest(), "big")
            for item in shingles(sequence)}


def category(left, right):
    return "__".join(sorted((left, right)))


def run(output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "tokenization": "lowercase Unicode letters only; digits and punctuation ignored",
        "shingle_size": SHINGLE_SIZE, "jaccard_threshold": THRESHOLD,
        "candidate_method": "global document-frequency-ordered prefix filter, followed by exact tuple-set Jaccard",
        "prefix_length": "n-ceil(0.8*n)+1; length ratio must be at least 0.8",
        "exact_normalization": "same tokenization joined with one space",
        "source_sha256": {name: sha256(raw_path(name)) for name in ("train.csv", "unlabeled.csv", "test.csv")},
        "script_sha256": sha256(__file__),
        "fitting_performed": False,
        "limitations": [
            "Detects substantial contiguous lexical overlap, not paraphrases, shared topics, domains, or generators.",
            "Global frequencies are used only to accelerate an exact audit, never to fit a predictive model.",
            "The 0.8 threshold is a strict duplicate screen, not an estimated source-group boundary.",
        ],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    train, validation = load_partitions()
    partitions = {"train": train, "validation": validation,
                  "unlabeled": read_csv(raw_path("unlabeled.csv")),
                  "test": read_csv(raw_path("test.csv"))}
    records = []
    tokenized = []
    hashed = []
    exact_buckets = defaultdict(list)
    frequencies = Counter()
    for partition, rows in partitions.items():
        for row in rows:
            index = len(records)
            sequence = tokens(row["text"])
            tokenized.append(sequence)
            current = hashed_shingles(sequence)
            hashed.append(current)
            frequencies.update(current)
            records.append({"partition": partition, "id": row["id"],
                            "label": row.get("label", "")})
            digest = hashlib.sha256(" ".join(sequence).encode()).hexdigest()
            exact_buckets[digest].append(index)
        print(f"Indexed {partition}: {len(rows)} rows", flush=True)

    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    def pair_record(i, j, **extra):
        a, b = records[i], records[j]
        return {"partition_a": a["partition"], "id_a": a["id"],
                "partition_b": b["partition"], "id_b": b["id"],
                "label_a": a["label"], "label_b": b["label"],
                "label_conflict": bool(a["label"] in ("0", "1") and b["label"] in ("0", "1")
                                       and a["label"] != b["label"]), **extra}

    exact_pairs = []
    for bucket in exact_buckets.values():
        for offset, i in enumerate(bucket):
            for j in bucket[offset + 1:]:
                # Verify the normalized text as well as its digest.
                if tokenized[i] == tokenized[j]:
                    exact_pairs.append(pair_record(i, j))

    postings = defaultdict(list)
    near_pairs = []
    candidates_verified = 0
    # Stable global ordering is required for the complete prefix-filter theorem.
    for i, current in enumerate(hashed):
        n = len(current)
        if n == 0:
            continue
        prefix = sorted(current, key=lambda value: (frequencies[value], value))[:n - math.ceil(THRESHOLD * n) + 1]
        candidates = set()
        for value in prefix:
            candidates.update(postings[value])
        for j in candidates:
            previous = hashed[j]
            m = len(previous)
            if min(n, m) < THRESHOLD * max(n, m):
                continue
            candidates_verified += 1
            intersection = len(current & previous)
            similarity = intersection / (n + m - intersection)
            if similarity + 1e-12 < THRESHOLD:
                continue
            a, b = shingles(tokenized[i]), shingles(tokenized[j])
            exact_similarity = len(a & b) / len(a | b)
            if exact_similarity + 1e-12 >= THRESHOLD:
                near_pairs.append(pair_record(i, j, jaccard=exact_similarity,
                                              n_shingles_a=len(a), n_shingles_b=len(b)))
                union(i, j)
        for value in prefix:
            postings[value].append(i)
        if (i + 1) % 2000 == 0:
            print(f"Compared {i + 1} rows; {len(near_pairs)} near pairs", flush=True)

    pair_fields = ["partition_a", "id_a", "partition_b", "id_b", "label_a", "label_b", "label_conflict"]
    write_csv(output / "exact_pairs.csv", pair_fields, exact_pairs)
    write_csv(output / "near_pairs.csv", pair_fields + ["jaccard", "n_shingles_a", "n_shingles_b"], near_pairs)
    grouped = defaultdict(list)
    for i in range(len(records)):
        grouped[find(i)].append(i)
    nontrivial = [indices for indices in grouped.values() if len(indices) > 1]
    component_rows = []
    for component, indices in enumerate(nontrivial):
        for i in indices:
            component_rows.append({"component": component, "size": len(indices), **records[i]})
    write_csv(output / "components.csv", ["component", "size", "partition", "id", "label"], component_rows)

    def summaries(pairs):
        counts = Counter(category(row["partition_a"], row["partition_b"]) for row in pairs)
        affected = defaultdict(set)
        for row in pairs:
            key = category(row["partition_a"], row["partition_b"])
            affected[key].add((row["partition_a"], row["id_a"]))
            affected[key].add((row["partition_b"], row["id_b"]))
        return {"pairs": len(pairs), "pairs_by_boundary": dict(counts),
                "affected_records_by_boundary": {key: len(value) for key, value in affected.items()},
                "label_conflicts": sum(row["label_conflict"] for row in pairs)}

    result = {
        "counts": {key: len(value) for key, value in partitions.items()},
        "exact_normalized": summaries(exact_pairs), "near_duplicate": summaries(near_pairs),
        "candidate_pairs_verified": candidates_verified,
        "nontrivial_components": len(nontrivial),
        "component_size_distribution": dict(Counter(map(len, nontrivial))),
        "largest_component": max(map(len, grouped.values())),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "No hidden labels or domain/generator memberships are inferred by this audit.",
        "limitations": plan["limitations"],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
