#!/bin/bash
# Self-training with confident pseudo-labels on unlabeled.csv (test.csv is never used), DeBERTa-v3-large fp32.
#   1) pseudo-label with the 8k-split model (never trained on validation) -> validation experiment
#   2) pseudo-label with the 4 full-data seeds -> full-data model for Kaggle
# One job at a time; each run caps PyTorch at 85% of device memory (clean CUDA OOM instead of a kernel OOM kill).
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
P=deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512
COMMON="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1"
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
while pgrep -f "[d]istilbert_(train|full)" >/dev/null; do sleep 30; done
echo "PL_START $(date)"
capped experiments.pseudo_label --name large8k --checkpoints checkpoints/deberta/${P}_seed42 \
  || { echo "PL_ABORT pseudo-labeling (8k) failed"; exit 1; }
capped experiments.pseudo_label --name largefull4 --checkpoints checkpoints/full/${P}_seed{42,43,44,45}_full \
  || { echo "PL_ABORT pseudo-labeling (full) failed"; exit 1; }
echo "PL_LABELS_DONE $(date)"
capped experiments.distilbert_train $COMMON --seed 42 --extra-train results/pseudo/large8k.csv \
  --notes "self-training: + confident unlabeled.csv pseudo-labels (8k large seed42, t=0.02)" \
  || { echo "PL_ABORT val run failed"; exit 1; }
echo "PL_VAL_DONE $(date)"
# Validation is saturated (baseline 0.99982), so only stop on a clear drop.
.venv/bin/python - <<"EOP" || { echo "PL_STOP val dropped below 0.9990, skipping full-data run"; exit 0; }
import json, sys
auc = json.load(open("results/metrics/deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42_pl-large8k.json"))["val_auc"]
print(f"pseudo-label val {auc:.5f} (baseline 0.99982)", flush=True)
sys.exit(0 if auc >= 0.9990 else 1)
EOP
capped experiments.distilbert_full $COMMON --seeds 42 --extra-train results/pseudo/largefull4.csv
echo "PL_DONE $(date)"
