"""Team prediction entry point (same shape as the course's _example_team/predict.py): writes submission.csv.

  python predict.py --model stylo_gbm --out submission.csv              # from the model saved by train.py --final
  python predict.py --model blend_b --out submission.csv                # from the saved pretrained runs (GPU)
  python predict.py --model blend_b --committed --out submission.csv    # copy the file we uploaded, no compute

Models: tfidf_lr, stylo_gbm, transformer_scratch (+ the other classical models after train.py --final),
blend_a, blend_b, blend_c. See MODELS.md for what each one is.

Blends re-predict test.csv with the saved 5-fold runs behind the uploaded files. Those runs are 0.3-1.7 GB per
fold, so they are not in git. They live in the research repo's outputs/runs/, or wherever --runs-dir points.
Run against them, this reproduces the uploaded files exactly (pretrained/results/reproduction.csv).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parent
RESEARCH_RUNS = TEAM.parents[2] / "AI-vs-Human-Text-Cornell-Tech-Competition-" / "outputs" / "runs"
# the runs and weights of the three uploaded files (pretrained/scripts/report_experiments.py: BLENDS, ORIGINAL_RUNS)
RUN = {m: f"20261030-135404_ood__{m}" for m in
       ["lm_gpt2", "lm_gpt2_medium", "binoculars_qwen05b", "binoculars_qwen15b", "ft_roberta_base",
        "ft_distilroberta", "ft_modernbert_base"]}
RUN.update(lm_gpt2_nostyle="20261031_var__lm_gpt2_nostyle")
# weights from pretrained/results/blend_selection/selection.json (train.csv labels only)
BLENDS = {
    "blend_a": ("final_A_topic_greedy", {"binoculars_qwen05b": 4, "ft_distilroberta": 2, "ft_modernbert_base": 2,
                                         "lm_gpt2_medium": 1, "lm_gpt2_nostyle": 1, "ft_roberta_base": 1,
                                         "lm_gpt2": 1}),
    "blend_b": ("final_B_lm4", {"lm_gpt2": 1, "lm_gpt2_medium": 1, "binoculars_qwen05b": 1, "binoculars_qwen15b": 1}),
    "blend_c": ("final_C_lm4_distil", {"lm_gpt2": 1, "lm_gpt2_medium": 1, "binoculars_qwen05b": 1,
                                       "binoculars_qwen15b": 1, "ft_distilroberta": 1}),
}


def committed_file(model: str) -> Path:
    if model in BLENDS:
        files = sorted((TEAM / "pretrained" / "kaggle_submissions").glob(f"{BLENDS[model][0]}.csv"))
    else:
        files = sorted((TEAM / "classical" / "kaggle_submissions").glob(f"final_?_{model}.csv"))
    if not files:
        raise SystemExit(f"no committed Kaggle file for {model}")
    return files[-1]  # latest upload letter = current version (e.g. final_G_tfidf_lr over final_E_tfidf_lr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--committed", action="store_true", help="copy the committed (uploaded) file instead of predicting")
    ap.add_argument("--data-dir", default=None, help="folder with the Kaggle CSVs (default: ./data, $HOA_DATA_DIR)")
    ap.add_argument("--runs-dir", default=None, help="blends: folder holding the saved runs (default: research repo)")
    ap.add_argument("--device", default=None, help="blends: cuda | cuda:<i> | mps | cpu")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    data = ["--data-dir", str(Path(args.data_dir).resolve())] if args.data_dir else []

    if args.committed:
        src = committed_file(args.model)
        shutil.copyfile(src, out)
        print(f"copied {src.relative_to(TEAM)} -> {out}")
        return

    if args.model not in BLENDS:
        cmd = [sys.executable, "predict.py", "--model", args.model, "--out", str(out), *data]
        raise SystemExit(subprocess.call(cmd, cwd=TEAM / "classical"))

    name, weights = BLENDS[args.model]
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else RESEARCH_RUNS
    runs = [str(runs_dir / RUN[m]) for m in weights]
    missing = [r for r in runs if not Path(r).is_dir()]
    if missing:
        raise SystemExit("saved runs not found (use --runs-dir, or --committed):\n  " + "\n  ".join(missing))
    cmd = [sys.executable, "make_submission.py", "--run", *runs, "--weights", *map(str, weights.values()),
           "--name", f"{name}_repredicted", *data] + (["--device", args.device] if args.device else [])
    code = subprocess.call(cmd, cwd=TEAM / "pretrained")
    if code == 0:
        shutil.copyfile(TEAM / "pretrained" / "outputs" / "submissions" / f"{name}_repredicted.csv", out)
        print(f"wrote {out}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
