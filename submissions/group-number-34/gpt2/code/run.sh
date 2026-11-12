#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${1:?Usage: bash run.sh DATA_DIRECTORY OUTPUT_DIRECTORY}"
output_dir="${2:?Usage: bash run.sh DATA_DIRECTORY OUTPUT_DIRECTORY}"
python_bin="${PYTHON_BIN:-python3}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$python_bin" -u "$project_dir/train.py" \
  --train_csv "$data_dir/train.csv" \
  --val_ids "$project_dir/val_ids.csv" \
  --test_csv "$data_dir/test.csv" \
  --sample_submission "$data_dir/sample_submission.csv" \
  --output_dir "$output_dir" \
  --epochs 3 --max_length 512 --batch_size 4 \
  --eval_batch_size 4 --gradient_accumulation 4 \
  --save_every_steps 50 --device cuda
