#!/bin/bash
# Install torch/transformers into the repo .venv and compute GPT-2 features (macOS, uses MPS GPU).
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/yuchen/lm_features
LOG=results/yuchen/lm_features/run.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) start"
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch transformers
python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
run() { caffeinate -i python -u submissions/Group_Number_19/yuchen_lm_features.py "$@"; }
if ! run --limit 20 --splits test; then
  echo "HF download failed; retrying via hf-mirror.com"; export HF_ENDPOINT=https://hf-mirror.com
  run --limit 20 --splits test
fi
run
echo "=== $(date) DONE"
