#!/bin/bash
# Learning curve, extra points for Q2.1: DeBERTa-v3-large + [NL] at 500 and 5,000 rows of the 8k split.
# Same settings as run_curve_holdout.sh (fp32, lr 1e-5, 3 epochs, warmup 0.1); validation set unchanged.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
systemctl --user is-active -q llm.service && { echo "CURVE2_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1"
echo "CURVE2_START $(date)"
for n in 500 5000; do
  capped experiments.distilbert_train $LARGE --train-size $n --no-checkpoint --notes "learning curve" \
    || echo "CURVE2_FAIL large n=$n"
  echo "CURVE2_RUN_DONE large n=$n $(date)"
done
echo "CURVE2_DONE $(date)"
