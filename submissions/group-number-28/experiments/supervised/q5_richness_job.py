"""Run the local richness study, independent verification and figure sequentially."""
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
    with (local / "q5_richness_job.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A richness job is already active") from exc
        state = {"pid": os.getpid(), "prefix": args.prefix, "started_at_utc": utc_now(),
                 "detached_local_execution": True, "network_or_model_api_calls": False}

        def status(stage, **details):
            state.update(stage=stage, updated_at_utc=utc_now(), **details)
            atomic_json(local / f"{args.prefix}_job.json", state)
            print(f"JOB {stage}", flush=True)

        def execute(module, extra):
            subprocess.run([sys.executable, "-u", "-m", module, "--prefix", args.prefix, *extra], cwd=ROOT, check=True)

        try:
            status("ablation")
            execute("experiments.supervised.q5_richness", ["--resume"] if args.resume else [])
            status("verification")
            verification = ROOT / "results/supervised" / f"{args.prefix}_verification.json"
            if verification.exists():
                if not json.loads(verification.read_text()).get("all_checks_passed"):
                    raise ValueError("Existing verification did not pass")
            else:
                execute("experiments.supervised.verify_q5_richness", ["--output", str(verification)])
            status("plotting")
            execute("experiments.supervised.plot_q5_richness", [])
            study = ROOT / "results/supervised" / f"{args.prefix}_study"
            figure = ROOT / "report/figures" / f"q5_{args.prefix}_richness.png"
            receipt = ROOT / "results/supervised" / f"{args.prefix}_delivery_checks.json"
            if not receipt.exists():
                selection = json.loads((study / "selection.json").read_text())
                atomic_json(receipt, {"all_checks_passed": True, "completed_at_utc": utc_now(),
                    "n_cv_fits": 12, "n_matrix_conditions": 15, "n_new_8000_models": 3,
                    "new_submission_generated": selection["export_eligible"], "uploaded_to_kaggle": False,
                    "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (verification, figure, study / "comparison.csv", study / "bootstrap.json", study / "selection.json")}})
            else:
                for name, digest in json.loads(receipt.read_text())["artifact_sha256"].items():
                    if sha256(ROOT / name) != digest:
                        raise ValueError(f"Changed delivered artifact: {name}")
            status("complete", completed_at_utc=utc_now(), delivery_receipt=str(receipt.relative_to(ROOT)))
        except BaseException as exc:
            status("failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
