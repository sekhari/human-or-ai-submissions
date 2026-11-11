#!/bin/bash
# Overnight queue (starts after Phase A): DeBERTa validation runs, then full-data (10k) seed ensembles.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1   # models are cached; skips ~80 s of stalled hub lookups per process
PY=.venv/bin/python
LARGE=models/deberta-v3-large
while pgrep -f run_phase_a.sh >/dev/null; do sleep 30; done
echo "OVERNIGHT_START $(date)"
T="$PY -u -m experiments.distilbert_train --notes overnight --newline-marker [NL] --seed 42"
$T --model-name microsoft/deberta-v3-base --lr 2e-5 --epochs 3 --warmup-ratio 0.1
$T --model-name $LARGE --lr 1e-5 --epochs 3 --warmup-ratio 0.1
echo "VAL_RUNS_DONE $(date)"
F="$PY -u -m experiments.distilbert_full --newline-marker [NL] --seeds 42 43 44"
$F --model-name microsoft/deberta-v3-base --lr 2e-5 --epochs 3 --warmup-ratio 0.1
$F --lr 3e-5 --epochs 4
# DeBERTa-large full runs only if its validation AUROC is within noise of base or better (guards against non-convergence).
if $PY - <<"EOP"
import json, sys
m = lambda n: json.load(open(f"results/metrics/{n}.json"))["val_auc"]
b = m("deberta-v3-base_nl_wu0.1_lr2e-05_ep3_bs16_len512_seed42")
l = m("deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42")
print(f"val base {b:.4f} large {l:.4f}", flush=True)
sys.exit(0 if l >= b - 0.002 else 1)
EOP
then $F --model-name $LARGE --lr 1e-5 --epochs 3 --warmup-ratio 0.1
else echo "skip DeBERTa-large full runs"; fi
echo "OVERNIGHT_DONE $(date)"
