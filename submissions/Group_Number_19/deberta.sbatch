#!/bin/bash
# Fine-tune DeBERTa-v3-base with leave-one-cluster-out CV + a full fit (one bf16 GPU).
# Submit from ~/hai:  sbatch -p gpu --gres=gpu:nvidia_rtx_a6000:1 deberta.sbatch
#SBATCH -J deberta
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o deberta_%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)  $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
UV=$HOME/.local/bin/uv
source venv128/bin/activate                   # created by bino_feats.sbatch (torch cu128)
$UV pip install -q --torch-backend=cu128 sentencepiece protobuf tiktoken scikit-learn
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -u yuchen_deberta.py --data data --clusters clusters.csv --out out/${TAG:-deberta_v3_base} ${ARGS}
echo "DONE $(date)"
