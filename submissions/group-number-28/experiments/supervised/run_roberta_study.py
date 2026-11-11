"""Execute the seven frozen encoder fits sequentially in python_course.

No score-dependent stopping, tuning or export. Native logs and every process
attempt are retained. Completed fit artifacts are not rewritten on continuation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(study, resume):
    if Path(sys.executable).parent != Path("/opt/anaconda3/envs/python_course/bin"):
        raise RuntimeError("Use the user-selected python_course interpreter")
    plan_path = study / "plan.json"
    plan = json.loads(plan_path.read_text())
    source_names = [
        "experiments/supervised/train_roberta.py", "experiments/supervised/evaluate_roberta.py",
        "experiments/supervised/prepare_roberta.py",
        "experiments/supervised/run_roberta_study.py", "experiments/supervised/benchmark_roberta_mps.py",
        "experiments/supervised/crop_evaluation.py", "experiments/supervised/q5_sensitivity.py",
        "human_ai/data.py", "data/splits/manifest.json", "docs/roberta_context_protocol_20261109.md",
    ]
    source_hashes = {name: sha(ROOT / name) for name in source_names}
    provenance_path = study / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["source_sha256"] != source_hashes or provenance["plan_sha256"] != sha(plan_path):
            raise RuntimeError("Executing sources or frozen plan changed; preserve existing study")
    else:
        for name in source_names:
            target = study / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        save(provenance_path, {"source_sha256": source_hashes, "plan_sha256": sha(plan_path),
                               "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                               "python": sys.executable, "before_first_neural_fit": True,
                               "independent_verifier": "Separately versioned zero-fit post-run audit; not a model-training dependency"})
    logs = study / "logs"
    logs.mkdir(exist_ok=True)
    receipt_path = study / "execution_receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    for fit in plan["fits"]:
        key = fit["key"]
        directory = study / "fits" / key
        completed = directory / "completion.json"
        if completed.exists():
            if not resume:
                raise FileExistsError(f"Preserve completed fit: {key}; use --resume for remaining fits")
            if json.loads(completed.read_text()).get("status") not in ("complete", "completed"):
                raise RuntimeError(f"Invalid completion marker: {key}")
            print(json.dumps({"skipped_completed_fit": key}), flush=True)
            continue
        attempt = 1 + sum(r["fit_key"] == key for r in receipts)
        stdout = logs / f"{key}.attempt{attempt}.stdout.log"
        stderr = logs / f"{key}.attempt{attempt}.stderr.log"
        if stdout.exists() or stderr.exists():
            raise FileExistsError("Preserve previous native logs")
        command = [sys.executable, "-u", "-m", "experiments.supervised.train_roberta",
                   "--study-dir", str(study), "--fit-key", key]
        if directory.exists():
            if not resume:
                raise FileExistsError(f"Preserve incomplete fit: {key}; use --resume")
            command.append("--resume")
        environment = os.environ.copy()
        environment.update(PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        print(json.dumps({"starting_fit": key, "attempt": attempt, "command": command}), flush=True)
        started = time.perf_counter()
        save(study / "status.json", {"status": "running", "fit_key": key, "attempt": attempt,
                                     "started_at_utc": datetime.now(timezone.utc).isoformat()})
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
            for line in process.stdout:
                out.write(line)
                out.flush()
                print(line.rstrip(), flush=True)
            code = process.wait()
        receipt = {"fit_key": key, "attempt": attempt, "command": command, "return_code": code,
                   "process_wall_seconds": time.perf_counter() - started,
                   "stdout_path": str(stdout.relative_to(ROOT)), "stderr_path": str(stderr.relative_to(ROOT)),
                   "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        receipts.append(receipt)
        save(receipt_path, receipts)
        if code or not completed.exists():
            print(stderr.read_text()[-8000:], flush=True)
            save(study / "status.json", {"status": "mechanical_failure", "fit_key": key, "attempt": attempt,
                                         "return_code": code, "no_quality_selection": True})
            raise RuntimeError(f"Mechanical failure; preserve artifacts and recover identical recipe: {key}")
        print(json.dumps({"completed_fit": key, "process_wall_seconds": receipt["process_wall_seconds"]}), flush=True)
    save(study / "training_completion.json", {"status": "complete", "fits": len(plan["fits"]),
                                               "quality_evaluation_pending": True,
                                               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                               "automatic_submission": False})
    save(study / "status.json", {"status": "training_complete", "fits": len(plan["fits"])})
    print(json.dumps({"all_seven_fits_complete": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.study_dir.resolve(), args.resume)
