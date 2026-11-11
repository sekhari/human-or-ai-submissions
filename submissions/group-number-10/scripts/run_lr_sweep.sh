#!/bin/bash
# DeBERTa-v3-large learning-rate sweep on the 8k split: 5e-6 and 2e-5 (1e-5 already run in fp32 and bf16).
# Uses bf16 if the bf16 test matched fp32 (val >= 0.9995), else fp32. Queued after run_bf16_large.sh.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_pseudo.sh|[r]un_bf16_large.sh" >/dev/null; do sleep 30; done
PREC=""
.venv/bin/python - <<"EOP" && PREC="--bf16"
import json, sys
try:
    auc = json.load(open("results/metrics/deberta-v3-large_nl_wu0.1_bf16_lr1e-05_ep3_bs16_len512_seed42.json"))["val_auc"]
except FileNotFoundError:
    print("bf16 test missing -> fp32", flush=True); sys.exit(1)
print(f"bf16 test val {auc:.5f} (fp32 0.99982) -> {'bf16' if auc >= 0.9995 else 'fp32'}", flush=True)
sys.exit(0 if auc >= 0.9995 else 1)
EOP
echo "LR_START $(date) precision=${PREC:-fp32}"
for lr in 5e-6 2e-5; do
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' experiments.distilbert_train \
    --model-name models/deberta-v3-large --lr $lr --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 $PREC \
    --seed 42 --notes "lr sweep" || echo "LR_FAIL lr=$lr"
  echo "LR_RUN_DONE lr=$lr $(date)"
done
echo "LR_DONE $(date)"
