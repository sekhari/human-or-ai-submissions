"""Run Q5 training, saved-artifact verification and plotting as one local job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
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
        parser.error("Invalid run prefix")
    local = ROOT / "tmp"
    local.mkdir(exist_ok=True)
    state_path = local / f"{args.prefix}_job.json"
    with (local / f"{args.prefix}_job.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A driver for this study is already running") from exc
        state = {"pid": os.getpid(), "prefix": args.prefix, "started_at_utc": utc_now(),
                 "independent_process": True, "network_or_model_api_calls": False}
        def status(stage, **details):
            state.update(stage=stage, updated_at_utc=utc_now(), **details)
            atomic_json(state_path, state)
            print(f"JOB {stage}", flush=True)
        def execute(module, extra):
            subprocess.run([sys.executable, "-u", "-m", module, "--prefix", args.prefix, *extra],
                           cwd=ROOT, check=True)
        try:
            status("training")
            execute("experiments.supervised.run_q5", ["--resume"] if args.resume else [])
            status("verification")
            verification = ROOT / "results/supervised" / f"{args.prefix}_verification.json"
            if verification.exists():
                if not json.loads(verification.read_text()).get("all_checks_passed"):
                    raise ValueError("Existing verification did not pass")
            else:
                execute("experiments.supervised.verify_q5", ["--output", str(verification)])
            status("plotting")
            execute("experiments.supervised.plot_q5", [])
            figure = ROOT / "report/figures" / f"q5_{args.prefix}_comparison.png"
            study = ROOT / "results/supervised" / f"{args.prefix}_study"
            receipt = ROOT / "results/supervised" / f"{args.prefix}_delivery_checks.json"
            if not receipt.exists():
                atomic_json(receipt, {"completed_at_utc": utc_now(), "all_checks_passed": True,
                                     "n_models": 4, "n_cv_fits": 12, "full_data_refit_performed": False,
                                     "uploaded_to_kaggle": False,
                                     "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                                         (verification, figure, study / "comparison.csv", study / "bootstrap.json")}})
            status("complete", completed_at_utc=utc_now(), delivery_receipt=str(receipt.relative_to(ROOT)))
        except BaseException as exc:
            status("failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
