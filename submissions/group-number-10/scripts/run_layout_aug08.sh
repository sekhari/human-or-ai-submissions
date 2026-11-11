#!/bin/bash
# Stronger layout augmentation for DeBERTa-large: 80% of training passages get a random line layout each epoch.
# File 13 (--layout-aug 0.5) scored 0.99037 public vs. 0.98524 without augmentation; this tests whether more helps.
# Straight to full 10k, seeds 42 + 43 (same seeds as files 5 and 13), so the Kaggle comparison is like for like.
set -u
cd ~/aml-kaggle/team-10
export HF_HUB_OFFLINE=1
while pgrep -f "[r]un_layout_aug.sh|[r]un_format_stress.sh|[r]un_electra.sh" >/dev/null; do sleep 30; done
systemctl --user is-active -q llm.service && { echo "AUG8_ABORT llm.service is running $(date)"; exit 1; }
capped() {  # capped <module> <args...>
  .venv/bin/python -u -c 'import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.85)
sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)' "$@"
}
LARGE="--model-name models/deberta-v3-large --lr 1e-5 --newline-marker [NL] --epochs 3 --warmup-ratio 0.1 --layout-aug 0.8"
echo "AUG8_START $(date)"
capped experiments.distilbert_train $LARGE --limit 64 --epochs 1 --no-checkpoint || { echo "AUG8_FAIL smoke $(date)"; exit 1; }
capped experiments.distilbert_full $LARGE --seeds 42 43 || { echo "AUG8_FAIL full $(date)"; exit 1; }
echo "AUG8_DONE $(date)"
