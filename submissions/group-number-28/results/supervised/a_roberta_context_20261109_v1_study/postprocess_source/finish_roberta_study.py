"""Run only frozen, zero-fit analysis and the declared post-hoc inference after all primary fits."""
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

ROOT = Path.cwd().resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if Path(sys.executable).parent != Path("/opt/anaconda3/envs/python_course/bin"):
        raise RuntimeError("python_course is required")
    sources = ["experiments/supervised/evaluate_roberta.py", "experiments/supervised/verify_roberta.py",
               "experiments/supervised/report_roberta.py", "experiments/supervised/diagnose_roberta_prefix.py",
               "experiments/supervised/train_roberta.py", "experiments/supervised/crop_evaluation.py", "human_ai/data.py"]
    hashes = {name: sha(ROOT / name) for name in sources}
    source_dir = study / "postprocess_source"
    source_dir.mkdir(exist_ok=False)
    for name in sources:
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(__file__, source_dir / "finish_roberta_study.py")
    save(study / "postprocess_execution_plan.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
        "controller_sha256": sha(__file__), "primary_plan_sha256": sha(study / "plan.json"),
        "exploratory_plan_sha256": sha(study / "exploratory_prefix_whitespace_plan.json"),
        "new_model_fits": 0, "wait_for": "All7 completed primary fits and successful final native process receipts",
        "stages": ["paired_analysis", "full_independent_audit", "primary_report", "declared_prefix_whitespace_inference"],
        "not_a_training_dependency": True})
    print(json.dumps({"postprocessing": "waiting_for_all_primary_fits"}), flush=True)
    while not (study / "training_completion.json").exists():
        status = json.loads((study / "status.json").read_text())
        if status.get("status") == "mechanical_failure":
            raise RuntimeError("Primary training failed; preserve postprocessing and recover explicitly")
        time.sleep(20)
    plan = json.loads((study / "plan.json").read_text())
    receipts = json.loads((study / "execution_receipts.json").read_text())
    for fit in plan["fits"]:
        done = json.loads((study / "fits" / fit["key"] / "completion.json").read_text())
        attempts = [r for r in receipts if r["fit_key"] == fit["key"]]
        if done["status"] != "complete" or not attempts or attempts[-1]["return_code"] != 0:
            raise RuntimeError("Primary completion or native receipt is incomplete")
    commands = [
        ("paired_analysis", ["-m", "experiments.supervised.evaluate_roberta", "--study-dir", str(study)]),
        ("full_independent_audit", ["-m", "experiments.supervised.verify_roberta", "--study-dir", str(study), "--output", str(study / "FINAL_AUDIT.json")]),
        ("primary_report", ["-m", "experiments.supervised.report_roberta", "--study-dir", str(study), "--output", str(ROOT / "docs/roberta_context_results_20261109.md"), "--verification", str(study / "FINAL_AUDIT.json")]),
        ("declared_prefix_whitespace_inference", ["-m", "experiments.supervised.diagnose_roberta_prefix", "--study-dir", str(study), "--fit-key", "canonical_8000", "--output", str(study / "exploratory_prefix_whitespace")]),
    ]
    logs = study / "logs"
    stage_receipts = []
    env = os.environ.copy()
    env.update(PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for key, args in commands:
        for name, expected in hashes.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Frozen postprocessing source changed: " + name)
        command = [sys.executable, "-u"] + args
        out_path = logs / (key + ".stdout.log")
        err_path = logs / (key + ".stderr.log")
        print(json.dumps({"starting_stage": key}), flush=True)
        started = time.perf_counter()
        with out_path.open("x") as out, err_path.open("x") as err:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        receipt = {"stage": key, "command": command, "return_code": completed.returncode,
                   "process_wall_seconds": time.perf_counter() - started,
                   "stdout_path": str(out_path.relative_to(ROOT)), "stderr_path": str(err_path.relative_to(ROOT)),
                   "stdout_sha256": sha(out_path), "stderr_sha256": sha(err_path),
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        stage_receipts.append(receipt)
        save(study / "postprocess_execution_receipts.json", stage_receipts)
        print(json.dumps(receipt), flush=True)
        if completed.returncode:
            print(err_path.read_text()[-6000:], flush=True)
            raise RuntimeError("Postprocessing stage failed: " + key)
    save(study / "pipeline_completion.json", {"status": "complete", "primary_fits": 7,
        "new_fits_in_postprocessing": 0, "all_primary_analysis_and_declared_exploratory_inference_complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_export": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": repr(error)}), flush=True)
        raise
