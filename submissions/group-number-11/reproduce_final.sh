#!/usr/bin/env bash
# Reproduce the medium_v4 candidate from supplied data, without hidden labels.
set -euo pipefail
data_dir="${1:-cornell-tech-cs-5785-combined-2026-ai-vs-human-text}"
team_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="$team_dir/src"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
bash "$team_dir/scripts/reproduce_neural.sh" "$data_dir" --no-report
python "$src_dir/lm_features.py" --data "$data_dir" \
  --output "$team_dir/artifacts/optimization/lm_medium" --model-id openai-community/gpt2-medium \
  --revision 6dcaa7a952f72f9298047fd5137cd6e4f05f41da \
  --max-tokens 128 --canonicalize --datasets train test
python "$src_dir/search_language_models.py" --data "$data_dir" --lm-dirs lm lm_medium --name language_both
python "$src_dir/select_language.py" --data "$data_dir" --search language_both --name language_medium --final
python "$src_dir/predict_optimized.py" --test "$data_dir/test.csv" --sample "$data_dir/sample_submission.csv" \
  --model "$team_dir/artifacts/optimization/language_medium.joblib" --out "$team_dir/output/submission_language_medium.csv"
python "$src_dir/ensemble_candidates.py" --data "$data_dir" --first neural_v3 --second language_medium --name medium_v4
python "$team_dir/predict_final.py" --data "$data_dir" --out "$team_dir/output/submission.csv"
python "$team_dir/verify_final.py" --data "$data_dir"
if [[ "${2:-}" != "--no-report" ]]; then
  python "$src_dir/analyze.py" --data "$data_dir"
  python "$src_dir/analyze_optimized.py" --data "$data_dir" --candidate language_v2 --search language_small
  python "$src_dir/analyze_optimized.py" --data "$data_dir" --candidate language_medium --search language_both --train-cache-only
  python "$src_dir/analyze_final.py" --data "$data_dir" --candidate medium_v4
  python "$src_dir/build_report.py"
  if command -v tectonic >/dev/null 2>&1; then
    (cd "$team_dir/report" && tectonic main.tex)
    python "$src_dir/package.py"
    python "$src_dir/package_delivery.py"
  fi
fi
