#!/bin/bash
# DeBERTa-v2-xlarge (900M) in bf16, queued after the DeBERTa-large seeds (never two jobs at once).
# Every run caps PyTorch at 85% of device memory so an overrun is a clean CUDA OOM in this job,
# not a kernel OOM kill on the shared unified memory.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
M=models/deberta-v2-xlarge
COMMON="--model-name $M --newline-marker [NL] --lr 5e-6 --epochs 3 --warmup-ratio 0.1 --bf16"
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
while pgrep -f "[r]un_roberta.sh|[r]un_full_large4445.sh" >/dev/null; do sleep 30; done
echo "XL_START $(date)"
# 1) Benchmark: 40 steps without gradient checkpointing; fall back to checkpointing if it runs out of memory.
GC=""
if capped experiments.distilbert_train $COMMON --epochs 1 --limit 640 --no-checkpoint --notes xl-bench; then
  echo "XL_BENCH no-gc ok"
else
  GC="--grad-checkpointing"
  capped experiments.distilbert_train $COMMON --epochs 1 --limit 640 --no-checkpoint --notes xl-bench $GC \
    && echo "XL_BENCH gc ok" || { echo "XL_ABORT benchmark failed with checkpointing too"; exit 1; }
fi
# 2) Validation run on the 8k split.
capped experiments.distilbert_train $COMMON $GC --seed 42 --notes xlarge || { echo "XL_ABORT val run failed"; exit 1; }
echo "XL_VAL_DONE $(date)"
# 3) Full-data seeds only if validation is in DeBERTa-large's league (0.9998); otherwise stop for review.
.venv/bin/python - <<"EOP" || { echo "XL_STOP val below 0.9995, skipping full-data runs"; exit 0; }
import json, sys
auc = json.load(open("results/metrics/deberta-v2-xlarge_nl_wu0.1_bf16_lr5e-06_ep3_bs16_len512_seed42.json"))["val_auc"]
print(f"xlarge val {auc:.5f}", flush=True)
sys.exit(0 if auc >= 0.9995 else 1)
EOP
capped experiments.distilbert_full $COMMON $GC --seeds 42 43
echo "XL_DONE $(date)"
