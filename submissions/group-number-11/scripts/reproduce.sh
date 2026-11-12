#!/usr/bin/env bash
# Run from the repository root with the project environment activated.
set -euo pipefail
data_dir="${1:-cornell-tech-cs-5785-combined-2026-ai-vs-human-text}"
team_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$team_dir/src"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python "$src_dir/train.py" --data "$data_dir" --final
python "$src_dir/robustness.py" --data "$data_dir"
python "$src_dir/improve.py" --data "$data_dir" --final
python "$src_dir/analyze.py" --data "$data_dir"
python "$src_dir/predict.py" --test "$data_dir/test.csv" \
  --sample "$data_dir/sample_submission.csv" --out "$team_dir/output/submission_baseline.csv"
python "$src_dir/verify.py" --data "$data_dir"
python "$src_dir/build_report.py"
if command -v tectonic >/dev/null 2>&1; then
  (cd "$team_dir/report" && tectonic main.tex)
  python "$src_dir/package.py"
fi
