#!/bin/bash
# ELECTRA-large + [NL] on the 8k split, seed 42, same recipe as DeBERTa-large (lr 1e-5, 3 epochs, warmup 0.1, fp32).
# Question: is it strong enough (~0.9997) and different enough from DeBERTa-large for a blend to help?
# Queued after the line-break stress test (one job at a time). Smoke test first.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_format_stress.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "ELECTRA_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
E="--model-name models/electra-large-discriminator --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1"
echo "ELECTRA_START $(date)"
capped experiments.distilbert_train $E --limit 64 --epochs 1 --no-checkpoint || { echo "ELECTRA_FAIL smoke $(date)"; exit 1; }
echo "ELECTRA_SMOKE_DONE $(date)"
capped experiments.distilbert_train $E --seed 42 --notes "ELECTRA-large: blend partner for DeBERTa-large?" \
  || echo "ELECTRA_FAIL $(date)"
echo "ELECTRA_DONE $(date)"
