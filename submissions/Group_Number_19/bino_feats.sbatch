#!/bin/bash
# Qwen3.5-2B Base/Instruct Binoculars features on one bf16 GPU (>= 24 GB).
# Submit from ~/hai:  sbatch -p gpu --gres=gpu:nvidia_rtx_a6000:1 bino_feats.sbatch
#SBATCH -J binofeat
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o binofeat_%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)  $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
UV=$HOME/.local/bin/uv
# torch built for CUDA 12.8 runs on drivers >= 12.8 (rush-compute-02 has 12.8; cu132 wheels need newer)
[ -d venv128 ] || $UV venv venv128 --python 3.11
source venv128/bin/activate
# one resolve with --torch-backend so later installs cannot swap torch back to the PyPI cu13x build
$UV pip install -q --torch-backend=cu128 torch "transformers>=5" accelerate pandas numpy
$UV pip install -q --torch-backend=cu128 flash-linear-attention || echo "fla not installed; using the torch fallback"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
OUT=out/${TAG:-qwen35_2b}
python -u yuchen_bino_features.py --data data --out $OUT --batch-size ${BS:-8} --limit 16 ${MODELS}
python -u yuchen_bino_features.py --data data --out $OUT --batch-size ${BS:-8} ${MODELS}
echo "DONE $(date)"
