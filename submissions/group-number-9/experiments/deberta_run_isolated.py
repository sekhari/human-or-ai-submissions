"""Run each DeBERTa epoch in a fresh process, restoring full training state.

This releases MPS backend caches between epochs without changing the model,
data, optimizer schedule, or training example counts. Large state stays ignored.

    python -m experiments.deberta_run_isolated
"""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/deberta_v3_mac_isolated")
    parser.add_argument("--weights", type=Path, default=ROOT / ".cache/deberta_v3_mac_isolated")
    parser.add_argument("--submission-path", type=Path, default=ROOT / "submissions/deberta_v3_mac_isolated_8000.csv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "experiments.deberta_finetune", "--device", args.device,
               "--precision", args.precision, "--batch-size", "4", "--gradient-accumulation", "4",
               "--eval-batch-size", "8", "--resume", "--one-epoch-per-process",
               "--output", str(args.output), "--weights", str(args.weights),
               "--submission-path", str(args.submission_path)]
    completed = {"tune": args.output / "selection.json", "fit": args.weights / "fit/final/config.json",
                 "evaluate": args.output / "evaluation.json", "submit": args.submission_path}
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1",
               PYTORCH_ENABLE_MPS_FALLBACK="1")
    child = None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for stage in ("prepare", "smoke", "tune", "fit", "evaluate", "submit"):
        if stage in completed and completed[stage].exists():
            print(f"Reusing completed stage: {stage}", flush=True)
            continue
        while True:
            print(f"Starting stage {stage} in a fresh process.", flush=True)
            child = subprocess.Popen(command + ["--stage", stage], cwd=ROOT, env=env)
            status = child.wait()
            if status == 75:
                print("Completed epoch saved; releasing GPU caches and starting the next process.", flush=True)
                continue
            if status:
                raise SystemExit(status)
            break
    print("All training/evaluation stages completed; no Kaggle upload performed.", flush=True)


if __name__ == "__main__":
    main()
