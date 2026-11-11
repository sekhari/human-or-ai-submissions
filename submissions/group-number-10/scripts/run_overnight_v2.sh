#!/bin/bash
# Revised overnight queue. DeBERTa-base turned out ~4.5x slower per step than DistilBERT, so every
# full-data run is started only if its estimated duration fits before DEADLINE.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
LARGE=models/deberta-v3-large
DEADLINE=$(date -d "today 09:40" +%s)
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$DEADLINE" ] || { echo "skip (would pass deadline): $2"; return 1; }; }
while kill -0 2531708 2>/dev/null; do sleep 30; done   # DeBERTa-base validation run from run_overnight.sh
echo "V2_START $(date)"
$PY -u -m experiments.distilbert_train --notes overnight --newline-marker [NL] --seed 42 \
    --model-name $LARGE --lr 1e-5 --epochs 3 --warmup-ratio 0.1
echo "VAL_RUNS_DONE $(date)"
F="$PY -u -m experiments.distilbert_full --newline-marker [NL]"
FB="$F --model-name microsoft/deberta-v3-base --lr 2e-5 --epochs 3 --warmup-ratio 0.1"
FL="$F --model-name $LARGE --lr 1e-5 --epochs 3 --warmup-ratio 0.1"
FD="$F --lr 3e-5 --epochs 4"
if $PY - <<"EOP"
import json, sys
m = lambda n: json.load(open(f"results/metrics/{n}.json"))["val_auc"]
b = m("deberta-v3-base_nl_wu0.1_lr2e-05_ep3_bs16_len512_seed42")
l = m("deberta-v3-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42")
print(f"val base {b:.4f} large {l:.4f}", flush=True)
sys.exit(0 if l >= b - 0.002 else 1)
EOP
then
  for s in 42 43; do fits 115 "large full seed $s" && $FL --seeds $s; done
  fits 40 "distilbert full x3" && $FD --seeds 42 43 44
  for s in 42 43 44; do fits 45 "base full seed $s" && $FB --seeds $s; done
else
  echo "skip DeBERTa-large full runs"
  for s in 42 43 44; do fits 45 "base full seed $s" && $FB --seeds $s; done
  fits 40 "distilbert full x3" && $FD --seeds 42 43 44
fi
echo "OVERNIGHT_DONE $(date)"
