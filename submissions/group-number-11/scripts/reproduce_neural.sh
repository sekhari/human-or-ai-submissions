#!/usr/bin/env bash
# Reproduce the 0.98861 public neural_v3 hybrid from the repository root.
set -euo pipefail
data_dir="${1:-cornell-tech-cs-5785-combined-2026-ai-vs-human-text}"
team_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$team_dir/src"
bash "$team_dir/scripts/reproduce_optimized.sh" "$data_dir" --no-report
python "$src_dir/train_transformer.py" --data "$data_dir" --epochs 2 \
  --max-length 192 --batch-size 16 --train-last 4 --max-steps 750 --max-seconds 99999
python "$src_dir/blend_transformer.py" --data "$data_dir" --weight .1 --method logit --name neural_v3
if [[ "${2:-}" != "--no-report" ]]; then
  python "$src_dir/analyze.py" --data "$data_dir"
  python "$src_dir/analyze_optimized.py" --data "$data_dir" --candidate language_v2 --search language_small
  active_candidate="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"])' "$team_dir/results/optimization/current_best.json")"
  if [[ "$active_candidate" == "neural_v3" ]]; then
    python "$src_dir/analyze_final.py" --data "$data_dir" --candidate neural_v3
    python "$src_dir/build_report.py"
    if command -v tectonic >/dev/null 2>&1; then
      (cd "$team_dir/report" && tectonic main.tex)
      python "$src_dir/package.py"
    fi
  else
    python "$src_dir/analyze_final.py" --data "$data_dir" --candidate neural_v3 --no-report
  fi
fi
