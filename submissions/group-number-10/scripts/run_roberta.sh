#!/bin/bash
# RoBERTa-large, queued after the DeBERTa full-data runs (never two jobs at once on the Spark).
# Two validation runs (with / without the [NL] marker; RoBERTa's byte-level BPE keeps newlines natively),
# then a full-data model with whichever scored higher on validation.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
M=models/roberta-large
COMMON="--model-name $M --lr 1e-5 --epochs 3 --warmup-ratio 0.1"
while pgrep -f "[r]un_full_deberta.sh|[r]un_full_large43.sh" >/dev/null; do sleep 30; done
echo "ROBERTA_START $(date)"
$PY -u -m experiments.distilbert_train --notes roberta $COMMON --newline-marker [NL] --seed 42
$PY -u -m experiments.distilbert_train --notes roberta $COMMON --seed 42
echo "VAL_DONE $(date)"
NL=$($PY - <<"EOP"
import json
m = lambda n: json.load(open(f"results/metrics/{n}.json"))["val_auc"]
nl, plain = m("roberta-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42"), m("roberta-large_wu0.1_lr1e-05_ep3_bs16_len512_seed42")
print(f"val nl {nl:.4f} plain {plain:.4f}", flush=True)
print("--newline-marker [NL]" if nl >= plain else "")
EOP
)
echo "$NL" | head -1
$PY -u -m experiments.distilbert_full $COMMON $(echo "$NL" | tail -1) --seeds 42
echo "ROBERTA_DONE $(date)"
