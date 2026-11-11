from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path.cwd()))
from human_ai.data import sha256

root = Path.cwd()
run_id = "a_roberta_20261109_v1_raw_full10000"
folder = root / "results/supervised" / run_id
command = [sys.executable, "-u", "-m", "experiments.supervised.export_roberta",
    "--source-study", str(root / "results/supervised/a_roberta_context_20261109_v1_study"),
    "--run-id", run_id, "--output", str(root / "submissions" / (run_id + ".csv"))]
stdout_path = root / "tmp" / (run_id + ".stdout.log")
stderr_path = root / "tmp" / (run_id + ".stderr.log")
environment = os.environ.copy()
environment.update(PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
started = time.perf_counter()
print(json.dumps({"starting_full_label_refit": run_id, "expected_optimizer_updates": 1250}), flush=True)
with stdout_path.open("x") as out, stderr_path.open("x") as err:
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
    for line in process.stdout:
        out.write(line)
        out.flush()
        print(line.rstrip(), flush=True)
    code = process.wait()
if folder.exists():
    logs = folder / "logs"
    logs.mkdir(exist_ok=True)
    for path in [stdout_path, stderr_path]:
        shutil.copy2(path, logs / path.name)
    receipt = {"command": command, "return_code": code, "process_wall_seconds": time.perf_counter()-started,
               "stdout_path": str((logs/stdout_path.name).relative_to(root)), "stderr_path": str((logs/stderr_path.name).relative_to(root)),
               "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path),
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (folder / "execution_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    shutil.copy2(__file__, folder / "source/export_execution.py")
if code:
    print(stderr_path.read_text()[-8000:], flush=True)
    raise SystemExit(code)
print(json.dumps({"full_label_refit_and_submission_complete": True}), flush=True)
