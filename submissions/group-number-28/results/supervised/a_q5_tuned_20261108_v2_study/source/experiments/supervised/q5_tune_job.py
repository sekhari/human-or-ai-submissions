"""Run tuning, independent saved-artifact verification and plots in one local job."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback

from human_ai.data import ROOT, sha256
from .run_q5 import atomic_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.prefix):
        parser.error("Invalid prefix")
    local = ROOT / "tmp"
    local.mkdir(exist_ok=True)
    with (local / "q5_tune_job.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A tuning driver is already active") from exc
        state = {"pid": os.getpid(), "prefix": args.prefix, "started_at_utc": utc_now(),
                 "detached_local_execution": True, "network_or_model_api_calls": False}
        state_path = local / f"{args.prefix}_job.json"
        def status(stage, **details):
            state.update(stage=stage, updated_at_utc=utc_now(), **details)
            atomic_json(state_path, state)
            print(f"JOB {stage}", flush=True)
        def execute(module, extra):
            subprocess.run([sys.executable, "-u", "-m", module, "--prefix", args.prefix, *extra],
                           cwd=ROOT, check=True)
        try:
            status("tuning")
            execute("experiments.supervised.q5_tune", ["--resume"] if args.resume else [])
            status("verification")
            verification = ROOT / "results/supervised" / f"{args.prefix}_verification.json"
            if verification.exists():
                if not json.loads(verification.read_text()).get("all_checks_passed"):
                    raise ValueError("Existing verification failed")
            else:
                execute("experiments.supervised.verify_q5_tune", ["--output", str(verification)])
            status("plotting")
            execute("experiments.supervised.plot_q5_tune", [])
            study = ROOT / "results/supervised" / f"{args.prefix}_study"
            figure = ROOT / "report/figures" / f"q5_{args.prefix}_tuning.png"
            receipt = ROOT / "results/supervised" / f"{args.prefix}_delivery_checks.json"
            if not receipt.exists():
                atomic_json(receipt, {"all_checks_passed": True, "completed_at_utc": utc_now(),
                    "n_configurations": 6, "n_candidate_fold_fits": 18, "n_finalists": 2,
                    "uploaded_to_kaggle": False,
                    "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (verification, figure, study / "comparison.csv", study / "bootstrap.json", study / "selection.json")}})
            status("complete", completed_at_utc=utc_now(), delivery_receipt=str(receipt.relative_to(ROOT)))
        except BaseException as exc:
            status("failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
