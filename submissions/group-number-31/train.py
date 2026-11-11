"""Team training entry point (same shape as the course's _example_team/train.py).

  python train.py --model stylo_gbm                 # fit on the 8,000 training rows, score the fixed 2,000 val rows
  python train.py --model stylo_gbm --final         # refit on all 10,000 rows, write the Kaggle CSV
  python train.py --model blend_b                   # retrain a pretrained blend's members (GPU, hours)

Models (see MODELS.md):
  classical, CPU-friendly:  tfidf_lr (C1), stylo_gbm (C2), transformer_scratch (C3, GPU optional),
                            linear_svm, nbsvm, rf_stylo, ensemble_c123
  pretrained, GPU 12 GB:    blend_a, blend_b, blend_c   (5-fold CV runs, seed 42)

This script only forwards to the track scripts: classical/train.py and pretrained/run_all.py.
Data: put train.csv, test.csv, unlabeled.csv and sample_submission.csv in ./data/ or pass --data-dir.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parent
SEED = 20261023
CLASSICAL = ["tfidf_lr", "stylo_gbm", "transformer_scratch", "linear_svm", "nbsvm", "rf_stylo", "ensemble_c123"]
# members of the three pretrained blends (weights in predict.py BLENDS)
BLEND_MEMBERS = {
    "blend_a": ["binoculars_qwen05b", "ft_distilroberta", "ft_modernbert_base", "lm_gpt2_medium", "lm_gpt2_nostyle",
                "ft_roberta_base", "lm_gpt2"],
    "blend_b": ["lm_gpt2", "lm_gpt2_medium", "binoculars_qwen05b", "binoculars_qwen15b"],
    "blend_c": ["lm_gpt2", "lm_gpt2_medium", "binoculars_qwen05b", "binoculars_qwen15b", "ft_distilroberta"],
}


def run(cmd: list[str], cwd: Path) -> int:
    print("+ (cd " + str(cwd.relative_to(TEAM)) + ") " + " ".join(cmd), flush=True)
    return subprocess.call([sys.executable, *cmd], cwd=cwd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=CLASSICAL + list(BLEND_MEMBERS))
    ap.add_argument("--final", action="store_true", help="classical: refit on 10,000 rows and write the Kaggle CSV")
    ap.add_argument("--seed", type=int, default=SEED, help="classical models (the pretrained runs use seed 42)")
    ap.add_argument("--data-dir", default=None, help="folder with the Kaggle CSVs (default: ./data, $HOA_DATA_DIR)")
    args = ap.parse_args()
    data = ["--data-dir", str(Path(args.data_dir).resolve())] if args.data_dir else []

    if args.model in CLASSICAL:
        cmd = ["train.py", "--model", args.model, "--seed", str(args.seed), *data] + (["--final"] if args.final else [])
        raise SystemExit(run(cmd, TEAM / "classical"))

    # Pretrained blends: 5-fold CV runs of the `blends` profile (seed 42), the same jobs as the uploaded
    # runs. Afterwards build the CSV with pretrained/make_submission.py (MODELS.md has the exact command).
    cmd = ["run_all.py", "--profile", "blends", "--only", ",".join(BLEND_MEMBERS[args.model]), *data]
    raise SystemExit(run(cmd, TEAM / "pretrained"))


if __name__ == "__main__":
    main()
