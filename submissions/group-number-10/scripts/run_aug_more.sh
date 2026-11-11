#!/bin/bash
# Overnight follow-ups to file 13 (DeBERTa-large + --layout-aug 0.5, 0.99037 public). Queued after run_layout_aug08.sh.
#   1) aug 0.5, full 10k, seeds 44 + 45 -> averaged locally with file 13's seeds 42 + 43 = 4-seed file
#   2) aug 0.5, full 10k, 4 epochs instead of 3, seeds 42 + 43 -> does the harder (augmented) data need longer training?
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_layout_aug08.sh|[r]un_layout_aug.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "MORE_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
BASE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --warmup-ratio 0.1 --layout-aug 0.5"
echo "MORE_START $(date)"
capped experiments.distilbert_full $BASE --epochs 3 --seeds 44 45 || echo "MORE_FAIL seeds44-45 $(date)"
echo "MORE_SEEDS_DONE $(date)"
capped experiments.distilbert_full $BASE --epochs 4 --seeds 42 43 || echo "MORE_FAIL ep4 $(date)"
echo "MORE_DONE $(date)"
