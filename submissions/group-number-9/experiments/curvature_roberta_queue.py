"""Export the first candidate promptly, then continue the offline ensemble queue."""
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.curvature_roberta import CACHE, OUT, ROOT, child, export_option, run, setup, sha, write


def main():
    setup()
    registration = {"protocol_sha256": sha(OUT / "protocol.json"),
        "sources": {p: sha(ROOT / p) for p in ["experiments/curvature_roberta_queue.py",
                                                "experiments/curvature_roberta_audit.py"]}}
    path = OUT / "queue_registration.json"
    if path.exists():
        assert json.loads(path.read_text()) == registration
    write(path, registration)
    try:
        value = json.loads((OUT / "single_validation.json").read_text())
        if value["export_eligible"]:
            write(OUT / "status.json", {"status": "running", "stage": "preparing_first_submission"})
            deadline = time.monotonic() + 8 * 60 * 60
            for epoch in range(1, value["epoch"] + 1):
                child("train", "full", 42, epoch, deadline)
            child("score", "full", 42, value["epoch"], deadline)
            candidate = export_option("single")
            subprocess.run([sys.executable, "-m", "experiments.curvature_roberta_audit", "--partial"],
                           cwd=ROOT, check=True)
            write(OUT / "first_submission_ready.json", {"path": candidate,
                "sha256": sha(ROOT / candidate), "validated": True, "ensemble_continues": True,
                "validation_auroc": value["validation_auroc"], "uploaded_to_kaggle": False})
            print("FIRST SUBMISSION READY: " + candidate, flush=True)
        run()
        subprocess.run([sys.executable, "-m", "experiments.curvature_roberta_audit"], cwd=ROOT, check=True)
        write(OUT / "queue_status.json", {"status": "complete", "audit_passed": True})
    except BaseException as error:
        write(OUT / "queue_status.json", {"status": "failed_or_interrupted", "error": repr(error),
            "resume": "caffeinate -i .venv/bin/python -u -m experiments.curvature_roberta_queue"})
        raise


if __name__ == "__main__":
    main()
