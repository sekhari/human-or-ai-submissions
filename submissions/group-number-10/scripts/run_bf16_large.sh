#!/bin/bash
# DeBERTa-v3-large validation run in bf16, identical to the fp32 baseline (val 0.99982) except for --bf16.
# Measures whether bf16 keeps quality and how much faster it is. Queued after run_pseudo.sh; never two jobs at once.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_pseudo.sh" >/dev/null; do sleep 30; done
echo "BF16_START $(date)"
.venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' experiments.distilbert_train \
  --model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 --bf16 \
  --seed 42 --notes "bf16 vs fp32 baseline" || { echo "BF16_ABORT"; exit 1; }
echo "BF16_DONE $(date)"
