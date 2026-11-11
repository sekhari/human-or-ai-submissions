#!/bin/bash
# GPT-2 token-probability features on one GPU (Unicorn / SLURM).
# Submit from ~/hai:   cd ~/hai && sbatch gpt2_feats.sbatch
# Optional partition:  sbatch -p <partition> gpt2_feats.sbatch
#SBATCH -J gpt2feat
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o gpt2feat_%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)  $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
# self-contained Python 3.11 env via uv (no dependency on the cluster's python/modules)
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
UV=$HOME/.local/bin/uv
[ -d venv ] || $UV venv venv --python 3.11
source venv/bin/activate
$UV pip install -q torch transformers pandas numpy
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
for M in gpt2 gpt2-medium; do
  python -u yuchen_lm_features.py --model $M --data data --out out/$M --batch-size ${BS:-32} --device cuda
done
echo "DONE $(date)"
