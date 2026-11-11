#!/bin/bash
# 1) Learning curve on the 8k split (validation fixed): DistilBERT+[NL] and DeBERTa-v3-large at 1k / 2k / 4k rows
#    (8k points already exist). 2) Recipe-holdout experiment with DeBERTa-v3-large (holdout vs control, 7k rows).
# DeBERTa runs use bf16 if the bf16 test matched fp32 (val >= 0.9995). Queued after run_lr_sweep.sh; one job at a time.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_pseudo.sh|[r]un_bf16_large.sh|[r]un_lr_sweep.sh" >/dev/null; do sleep 30; done
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
PREC=""
.venv/bin/python - <<"EOP" && PREC="--bf16"
import json, sys
try:
    auc = json.load(open("results/metrics/deberta-v3-large_nl_wu0.1_bf16_lr1e-05_ep3_bs16_len512_seed42.json"))["val_auc"]
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if auc >= 0.9995 else 1)
EOP
DISTIL="--lr 3e-5 --epochs 4 --newline-marker [NL]"
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 $PREC"
echo "CURVE_START $(date) deberta precision=${PREC:-fp32}"
for n in 1000 2000 4000; do
  capped experiments.distilbert_train $DISTIL --train-size $n --no-checkpoint --notes "learning curve" \
    || echo "CURVE_FAIL distilbert n=$n"
done
for n in 1000 2000 4000; do
  capped experiments.distilbert_train $LARGE --train-size $n --no-checkpoint --notes "learning curve" \
    || echo "CURVE_FAIL large n=$n"
  echo "CURVE_RUN_DONE large n=$n $(date)"
done
echo "CURVE_DONE $(date)"
for c in holdout control; do
  capped experiments.genre_holdout $LARGE --condition $c --train-size 7000 || echo "HOLDOUT_FAIL $c"
  echo "HOLDOUT_RUN_DONE $c $(date)"
done
echo "HOLDOUT_DONE $(date)"
