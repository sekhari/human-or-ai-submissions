"""Compare a clean-kernel rerun of notebooks 01-17 against the committed records.

Usage: uv run --locked python tools/compare_reruns.py /path/to/rerun/checkout
The rerun checkout is a separate worktree of the same commit. This script only reads it
and writes one safe summary to results/18_final_checks/<utc>-rerun.json.
"""
import hashlib, json, math, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERUN = Path(sys.argv[1]).resolve()
# Timing and identity fields always change between runs, so they are not compared.
SKIP = ("seconds", "timing", "utc", "run_id", "history")


def walk(a, b, path=""):
    """Yield (path, kind, value) for each difference. kind is 'numeric' or 'other'."""
    if any(s in path.lower() for s in SKIP):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield f"{path}/{key}", "other", "missing key"
            else:
                yield from walk(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path, "other", "length"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            yield path, "numeric", abs(a - b)
    elif a != b:
        yield path, "other", "value"


def completed(folder, root):
    rows = []
    for p in sorted((root / "results" / folder).glob("2*Z.json")):
        record = json.loads(p.read_text())
        if record.get("status") == "completed":
            rows.append((p, record))
    return rows


committed_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RERUN, text=True).strip()
rows = []
for folder in sorted(p.name for p in (ROOT / "results").iterdir() if p.is_dir() and p.name[:2].isdigit() and int(p.name[:2]) <= 17):
    old = completed(folder, ROOT)
    old_names = {p.name for p in (ROOT / "results" / folder).glob("*")}
    new = [(p, r) for p, r in completed(folder, RERUN) if p.name not in old_names]
    failed = [p.name for p in (RERUN / "results" / folder).glob("2*Z*.json")
              if p.name not in old_names and json.loads(p.read_text()).get("status") == "failed"]
    if not old or not new:
        rows.append({"notebook": folder, "status": "no new completed record", "failed_records": len(failed)})
        continue
    (old_path, a), (new_path, b) = old[-1], new[-1]
    diffs = list(walk(a.get("metrics"), b.get("metrics"), "metrics"))
    numeric = [v for _, k, v in diffs if k == "numeric"]
    largest = sorted(((v, p) for p, k, v in diffs if k == "numeric"), reverse=True)[:3]
    other = [p for p, k, _ in diffs if k == "other"]
    rows.append({"notebook": folder, "status": "rerun completed", "failed_records": len(failed),
                 "committed_record": old_path.name, "rerun_record": new_path.name,
                 "validation_auroc_committed": a["metrics"].get("validation_auroc"),
                 "validation_auroc_rerun": b["metrics"].get("validation_auroc"),
                 "metrics_identical": not diffs, "numeric_diffs": len(numeric),
                 "max_abs_numeric_diff": max(numeric) if numeric else 0.0,
                 "largest_numeric_diffs": [{"path": p, "abs_diff": v} for v, p in largest],
                 "other_diffs": len(other), "other_diff_paths": other[:5],
                 "config_identical": a.get("config_sha256") == b.get("config_sha256"),
                 "data_split_identical": a["data_sha256"] == b["data_sha256"] and a["split_sha256"] == b["split_sha256"],
                 "runtime_committed_s": round(a["runtime_seconds"], 1), "runtime_rerun_s": round(b["runtime_seconds"], 1)})

summary = {"source": f"clean-kernel rerun at commit {committed_sha[:7]}", "rerun_commit": committed_sha,
           "utc": datetime.now(timezone.utc).isoformat(), "rule": "latest completed committed record vs newest rerun record; "
           "metrics compared recursively; keys containing " + ", ".join(SKIP) + " skipped", "notebooks": rows}
out = ROOT / "results" / "18_final_checks" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-rerun.json"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
for row in rows:
    print({k: row[k] for k in row if k in ("notebook", "status", "metrics_identical", "max_abs_numeric_diff", "other_diffs")})
print("Saved", out.relative_to(ROOT), hashlib.sha256(out.read_bytes()).hexdigest()[:12])
